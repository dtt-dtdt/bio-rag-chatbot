"""
뉴스 청크 + 특허 임베딩 → ChromaDB 저장

[저장 구조]  store/chroma 폴더 안에 컬렉션 2개
  - patents : 특허 1건 = 1개 (ID = 출원번호)
      본문     = 발명의명칭 + 요약 (+ 핵심키워드, 키워드 파일이 있으면)
      메타데이터 = 분야(바이오의약품/바이오헬스케어), 출원번호, 등록번호, 출원인, 최종권리자,
                  IPC, CPC, 법적상태, 출원일/등록일/공개일(문자 + 숫자), 분야태그(있으면)
  - news    : 뉴스 청크 1개 = 1개 (ID = 청크ID)
      본문     = "[제목] ...\n" + 청크본문 (안내 문구 제거 후)
      메타데이터 = 기사ID, 청크번호, 청크수, 제목, 원문URL, 매체명, 수집키워드, 발행일(문자 + 숫자)
  ※ 날짜는 '2024-03-05'(표시용)와 20240305(숫자, 기간 필터용) 두 가지로 저장

[실행 방법 - Jupyter]
  1) pip install openai chromadb pandas python-dotenv
  2) .env 파일에 OPENAI_API_KEY=sk-... (노트북과 같은 폴더)
  3) 입력 파일 3개를 같은 폴더에 두기 (아래 '설정' 참고)
  4) 처음에는 SAMPLE_N = 200 으로 시험 → 마지막의 검색 테스트 결과 확인
     → 괜찮으면 SAMPLE_N = None 으로 바꿔 전체 실행
  ※ 중간에 멈춰도 다시 실행하면 이미 저장된 ID는 건너뜀 (비용 중복 없음)
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd

# ============================== 설정 ==============================
import config

BASE_DIR = config.ROOT

NEWS_FILE = config.NEWS_CHUNKS
PATENT_FILES = {  # 파일 : 분야 이름
    config.PATENT_PHARMA: "바이오의약품",
    config.PATENT_HEALTHCARE: "바이오헬스케어",
}
# indexer/keywords.py 결과 파일이 있으면 자동으로 사용 (없으면 무시)
KEYWORD_FILES = [config.KEYWORD_PHARMA, config.KEYWORD_HEALTHCARE]

DB_DIR = config.CHROMA_DIR
EMBED_MODEL = "text-embedding-3-small"   # 특허·뉴스 반드시 같은 모델
BATCH = 20                               # 한 번에 임베딩할 개수 (작을수록 공용 한도의 빈틈에 들어가기 쉬움)
TPM_LIMIT = 1_000_000                    # 계정의 분당 토큰 한도 (오류 메시지의 'Limit' 값)
TPM_SAFETY = 0.5                         # 한도의 50%까지만 사용 (키를 여러 명이 같이 쓰는 경우 대비)
MAX_RETRY = 30                           # 한도 오류 시 최대 재시도 횟수
SAMPLE_N = 200                           # 시험용 (컬렉션별 개수). 전체 실행은 None
# 백그라운드 실행(indexer/background.py)에서는 환경변수 EMBED_SAMPLE_N=all 로 전체 실행
if os.getenv("EMBED_SAMPLE_N"):
    SAMPLE_N = None if os.getenv("EMBED_SAMPLE_N").lower() == "all" else int(os.getenv("EMBED_SAMPLE_N"))

REMOVE_BOILERPLATE = True                # 뉴스 본문 끝 사이트 안내 문구 제거
BOILERPLATE_RE = re.compile(r"\s*(기사를 더 보고 싶다면\??|기사 섹션 분류 안내)\s*")
# ==================================================================


# ---------------------------- 공통 ----------------------------
def to_date(s: str) -> tuple[str, int]:
    """'2022.02.28' / '20220228' / '2022-02-28T11:43:00' → ('2022-02-28', 20220228). 없으면 ('', 0)"""
    digits = re.sub(r"\D", "", str(s))[:8]
    if len(digits) != 8:
        return "", 0
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}", int(digits)


def make_clients():
    from openai import OpenAI
    import chromadb
    try:
        from dotenv import load_dotenv
        load_dotenv(config.ENV_FILE)
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY가 없습니다. project3 폴더의 .env 파일에 넣어주세요: {config.ENV_FILE}")
    return OpenAI(), chromadb.PersistentClient(path=str(DB_DIR))


_usage: list[tuple[float, int]] = []   # 최근 60초 동안 보낸 (시각, 토큰 수)
_tok_per_char = 1.0                    # 글자당 토큰 비율 (실제 사용량으로 계속 보정)


def _wait_for_budget(est_tokens: int):
    """최근 60초 사용량 + 이번 요청이 한도(80%)를 넘으면 넘지 않을 때까지 기다림"""
    budget = TPM_LIMIT * TPM_SAFETY
    while True:
        now = time.time()
        _usage[:] = [(t, n) for t, n in _usage if now - t < 60]
        used = sum(n for _, n in _usage)
        if used + est_tokens <= budget or not _usage:
            return
        time.sleep(max(0.5, 60 - (now - _usage[0][0])))


def embed(openai_client, texts: list[str]) -> list[list[float]]:
    global _tok_per_char
    chars = sum(len(t) for t in texts)
    for attempt in range(MAX_RETRY):
        _wait_for_budget(int(chars * _tok_per_char) + 1)
        try:
            r = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
            used = r.usage.total_tokens
            _usage.append((time.time(), used))
            _tok_per_char = 0.8 * _tok_per_char + 0.2 * (used / max(chars, 1))
            return [d.embedding for d in r.data]
        except Exception as e:  # 요청 한도·일시 오류 → 기다렸다 재시도
            if attempt == MAX_RETRY - 1:
                raise
            m = re.search(r"try again in ([\d.]+)\s*(ms|s)", str(e))
            suggested = (float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 1) if m else 0
            # 안내 시간과 '점점 길어지는 대기(최대 60초)' 중 긴 쪽 → 다른 사용자가 한도를 쓰고 있어도 버팀
            wait = max(suggested, min(60, 3 * (2 ** attempt)))
            print(f"    재시도 {attempt + 1}/{MAX_RETRY - 1} ({type(e).__name__}) - {wait:.1f}초 대기")
            time.sleep(wait)


def upsert_collection(openai_client, chroma, name: str, ids, docs, metas):
    col = chroma.get_or_create_collection(name, metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL})
    if col.metadata and col.metadata.get("embed_model") not in (None, EMBED_MODEL):
        raise RuntimeError(f"'{name}' 컬렉션은 {col.metadata['embed_model']} 모델로 만들어졌습니다. 모델을 섞을 수 없어요.")

    # 이미 저장된 ID는 건너뜀
    existing = set()
    for i in range(0, len(ids), 5000):
        existing |= set(col.get(ids=ids[i:i + 5000], include=[])["ids"])
    todo = [i for i, x in enumerate(ids) if x not in existing]
    chars = sum(len(docs[i]) for i in todo)
    print(f"\n[{name}] 대상 {len(ids):,}개 / 이미 저장 {len(existing):,}개 / 새로 임베딩 {len(todo):,}개 (약 {chars:,}자)")

    for b in range(0, len(todo), BATCH):
        idx = todo[b:b + BATCH]
        vecs = embed(openai_client, [docs[i] for i in idx])
        col.add(ids=[ids[i] for i in idx], documents=[docs[i] for i in idx],
                metadatas=[metas[i] for i in idx], embeddings=vecs)
        done = b + len(idx)
        if done % (BATCH * 20) == 0 or done == len(todo):
            print(f"  진행 {done:,}/{len(todo):,}")
    print(f"  '{name}' 컬렉션 총 {col.count():,}개")
    return col, [ids[i] for i in todo]   # 컬렉션, 이번에 새로 넣은 ID


# ---------------------------- 특허 ----------------------------
def build_patents():
    frames = []
    for path, field in PATENT_FILES.items():
        d = pd.read_csv(path, dtype=str, keep_default_na=False)
        d["분야"] = field
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("출원번호")

    # 특허 파일에 이미 키워드 칼럼이 있으면 그대로 쓰고, 키워드 파일이 있으면 그 값으로 채움 (칼럼 이름 충돌 방지)
    kw_cols = ["분야태그", "핵심키워드"]
    for c in kw_cols:
        if c not in df.columns:
            df[c] = ""
    kw_frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in KEYWORD_FILES if p.exists()]
    kw_frames = [k[["출원번호", *[c for c in kw_cols if c in k.columns]]] for k in kw_frames]
    if kw_frames:
        kw = pd.concat(kw_frames).drop_duplicates("출원번호").rename(columns={c: f"{c}_kw" for c in kw_cols})
        df = df.merge(kw, on="출원번호", how="left")
        for c in kw_cols:
            if f"{c}_kw" in df.columns:
                new = df[f"{c}_kw"].fillna("")
                df[c] = new.where(new != "", df[c])
                df = df.drop(columns=f"{c}_kw")
    df = df.fillna("")
    print(f"특허 {len(df):,}건 (키워드 있는 특허 {(df['핵심키워드'] != '').sum():,}건)")

    if SAMPLE_N:
        df = pd.concat([g.sample(min(len(g), SAMPLE_N // 2), random_state=42) for _, g in df.groupby("분야")])

    ids, docs, metas = [], [], []
    for r in df.itertuples(index=False):
        text = f"{r.발명의명칭}\n{r.요약}".strip()
        if r.핵심키워드:
            text += f"\n핵심키워드: {r.핵심키워드}"
        meta = {"분야": r.분야, "출원번호": r.출원번호, "등록번호": r.등록번호, "발명의명칭": r.발명의명칭,
                "출원인": r.출원인, "최종권리자": r.최종권리자, "법적상태": r.법적상태,
                "IPC": r.IPC분류[:500], "CPC": r.CPC분류[:500], "분야태그": r.분야태그}
        for col, key in [("출원일자", "출원일"), ("등록일자", "등록일"), ("공개일자", "공개일")]:
            meta[key], meta[key + "_num"] = to_date(getattr(r, col))
        ids.append(r.출원번호); docs.append(text); metas.append(meta)
    return ids, docs, metas


# ---------------------------- 뉴스 ----------------------------
def build_news():
    df = pd.read_csv(NEWS_FILE, dtype=str, keep_default_na=False)
    if SAMPLE_N:
        df = df.sample(min(len(df), SAMPLE_N), random_state=42)

    ids, docs, metas, removed = [], [], [], 0
    for r in df.itertuples(index=False):
        body = r.청크본문
        if REMOVE_BOILERPLATE:
            new = BOILERPLATE_RE.sub(" ", body).strip()
            removed += new != body.strip()
            body = new
        if len(body) < 10:  # 안내 문구만 남았던 청크
            continue
        date_s, date_n = to_date(r.발행일)
        ids.append(r.청크ID)
        docs.append(f"[제목] {r.제목}\n{body}")
        metas.append({"기사ID": r.기사ID, "청크번호": int(r.청크번호), "청크수": int(r.청크수), "제목": r.제목,
                      "원문URL": r.원문URL, "매체명": r.매체명, "수집키워드": r.수집키워드,
                      "발행일": date_s, "발행일_num": date_n})
    print(f"뉴스 청크 {len(df):,}개 → 임베딩 대상 {len(ids):,}개 (안내 문구 제거 {removed:,}개)")
    return ids, docs, metas


# ---------------------------- 검색 테스트 ----------------------------
def search(question: str, k: int = 3, patent_where: dict | None = None, news_where: dict | None = None):
    """예: search("항체 바이오시밀러 위탁생산", patent_where={"분야": "바이오의약품"},
                  news_where={"발행일_num": {"$gte": 20250101}})"""
    openai_client, chroma = make_clients()
    q = embed(openai_client, [question])[0]
    for name, where, title_key in [("patents", patent_where, "발명의명칭"), ("news", news_where, "제목")]:
        col = chroma.get_collection(name)
        res = col.query(query_embeddings=[q], n_results=k, where=where)
        print(f"\n[{name}] '{question}'")
        for doc_id, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
            date = meta.get("등록일") or meta.get("발행일", "")
            print(f"  ({1 - dist:.3f}) {doc_id} | {date} | {meta[title_key][:50]}")


def run():
    openai_client, chroma = make_clients()
    upsert_collection(openai_client, chroma, "patents", *build_patents())
    news_col, new_news_ids = upsert_collection(openai_client, chroma, "news", *build_news())

    # 새로 넣은 뉴스를 노이즈 분류 모델로 표시 (indexer/noise_news.py와 store/noise/news_bio_classifier.joblib가 있을 때만)
    if new_news_ids:
        try:
            from indexer import noise_news as news_noise
            news_noise.tag_new_chunks(news_col, new_news_ids)
        except ImportError:
            print("  (indexer/noise_news.py 파일이 없어 노이즈 분류는 건너뛰었어요)")
    print(f"\n저장 위치: {DB_DIR}")
    search("항체 의약품 위탁생산 공정")


if __name__ == "__main__":
    run()
