"""
검색기 1차 평가 하네스 — Hit@3 측정 + 뉴스 청킹 설정 비교

[하는 일]
  1. 평가 질문 자동 생성 (GPT)
     - 뉴스: 기사 2,000건 표본에서 150건을 골라, 기사의 '무작위 위치 발췌'로 질문 생성 → 정답 = 그 기사
       (기사 앞·중간·뒤 어디서 나온 질문인지도 기록 → 청킹이 뒷부분 내용을 잘 찾는지 확인)
     - 특허: 100건을 골라 명칭·요약으로 질문 생성 → 정답 = 그 특허 (제목 같은 패밀리 특허도 정답 인정)
  2. 청킹 설정 4개로 표본 2,000건을 각각 임베딩 (별도 폴더 chroma_eval — 본 DB는 건드리지 않음)
  3. 측정
     - [청킹 비교]  설정별 Hit@1 / Hit@3 / Hit@5 / MRR, 발췌 위치별 Hit@3
     - [1차 기준선] 본 DB(store/chroma) 전체로 예전 벡터 검색과 같은 방식(벡터 검색 + 같은 기사 묶기)의 Hit@3
  4. 결과 저장: eval 폴더 (eval_report.md, eval_results.csv, eval_details.csv)

[용어]
  Hit@3 : 정답 문서가 검색 상위 3개 안에 들어온 질문의 비율
  MRR   : 정답이 몇 위에 나왔는지 평균 (1위=1, 2위=0.5, 3위=0.33 …). 높을수록 정답이 위에 나옴

[실행 - Jupyter]  같은 폴더에 원본 뉴스 CSV, _final_ 특허 CSV 2개, chroma_db, .env 필요
  셀에 붙여 실행 → 약 30~40분 (공용 키 한도에 따라 다름), 약 $0.3
  중간에 멈춰도 다시 실행하면 이어서 진행 (질문·임베딩 모두 저장됨)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# ============================== 설정 ==============================
import config

BASE_DIR = config.ROOT

RAW_NEWS_FILE = config.NEWS_RAW
PATENT_FILES = config.PATENT_FILES
MAIN_DB_DIR = config.CHROMA_DIR      # 본 DB (읽기만 함)
EVAL_DB_DIR = config.CHROMA_EVAL_DIR    # 평가용 DB (청킹 설정별)
OUT_DIR = config.EVAL_RESULTS_DIR

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
POOL_N = 2000         # 청킹 비교용 기사 표본 수
N_NEWS_Q = 150        # 뉴스 평가 질문 수
N_PATENT_Q = 100      # 특허 평가 질문 수
SEED = 42

# 청킹 설정: 컬렉션 이름(영문) → (표시 이름, 최대 크기, 겹침).  크기 None = 자르지 않음
CONFIGS = {
    "c1_none": ("① 자르지 않음", None, 0),
    "c2_500":  ("② 500자/겹침50", 500, 50),
    "c3_1000": ("③ 1000자/겹침150 (현재)", 1000, 150),
    "c4_1500": ("④ 1500자/겹침200", 1500, 200),
}
MAX_EMBED_CHARS = 6000   # '자르지 않음'에서 임베딩 한도(약 8천 토큰)를 넘지 않도록 앞부분만 사용
BATCH = 50
TPM_LIMIT, TPM_SAFETY, MAX_RETRY = 1_000_000, 0.5, 30
BOILERPLATE_RE = re.compile(r"\s*(기사를 더 보고 싶다면\??|기사 섹션 분류 안내)\s*")
# ==================================================================

_clients: dict = {}


def get_clients():
    if not _clients:
        from openai import OpenAI
        import chromadb
        try:
            from dotenv import load_dotenv
            load_dotenv(config.ENV_FILE)
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"OPENAI_API_KEY가 없습니다. project3 폴더의 .env 파일에 넣어주세요: {config.ENV_FILE}")
        _clients["openai"] = OpenAI()
        _clients["eval"] = chromadb.PersistentClient(path=str(EVAL_DB_DIR))
        _clients["main"] = chromadb.PersistentClient(path=str(MAIN_DB_DIR)) if MAIN_DB_DIR.exists() else None
    return _clients


# ---------------------------- 공통 도구 ----------------------------
def article_id(url: str, title: str, date: str) -> str:   # indexer/chunk.py와 동일 → 본 DB의 기사ID와 일치
    return hashlib.sha1(f"{url}|{title}|{date}".encode("utf-8")).hexdigest()[:12]


SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|(?<=다\.)(?=\S)")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", BOILERPLATE_RE.sub(" ", str(text))).strip()


def split_sentences(text: str, max_len: int) -> list[str]:
    out = []
    for s in (x.strip() for x in SENT_SPLIT.split(text) if x and x.strip()):
        while len(s) > max_len:
            cut = s.rfind(" ", 0, max_len)
            cut = cut if cut > max_len // 2 else max_len
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def chunk_text(text: str, size: int | None, overlap: int) -> list[str]:
    """indexer/chunk.py와 같은 방식. 기준값만 크기에 비례 (자르지 않는 기준 = 크기×1.2, 마지막 최소 = 크기×0.3)"""
    if size is None:
        return [text[:MAX_EMBED_CHARS]]
    if len(text) <= int(size * 1.2):
        return [text]
    sents = split_sentences(text, size)
    chunks, cur, new = [], [], 0
    for s in sents:
        if cur and new and len(" ".join(cur)) + 1 + len(s) > size:
            chunks.append(cur)
            keep, tot = [], 0
            for x in reversed(cur):
                if tot + len(x) + 1 > overlap:
                    break
                keep.insert(0, x)
                tot += len(x) + 1
            cur, new = keep, 0
            if len(" ".join(cur)) + 1 + len(s) > size:
                cur = []
        cur.append(s)
        new += 1
    if cur and new:
        chunks.append(cur)
    if len(chunks) >= 2 and len(" ".join(chunks[-1])) < int(size * 0.3):
        prev, last = chunks[-2], chunks[-1]
        n = next((k for k in range(min(len(prev), len(last)), 0, -1) if prev[-k:] == last[:k]), 0)
        chunks = chunks[:-2] + [prev + last[n:]]
    return [" ".join(c) for c in chunks]


_usage: list[tuple[float, int]] = []


def embed(texts: list[str]) -> list[list[float]]:
    """분당 토큰 한도를 넘지 않게 조절하며 임베딩"""
    oa = get_clients()["openai"]
    est = sum(len(t) for t in texts)
    for attempt in range(MAX_RETRY):
        while True:
            now = time.time()
            _usage[:] = [(t, n) for t, n in _usage if now - t < 60]
            if not _usage or sum(n for _, n in _usage) + est <= TPM_LIMIT * TPM_SAFETY:
                break
            time.sleep(max(0.5, 60 - (now - _usage[0][0])))
        try:
            r = oa.embeddings.create(model=EMBED_MODEL, input=texts)
            _usage.append((time.time(), getattr(getattr(r, "usage", None), "total_tokens", est)))
            return [d.embedding for d in r.data]
        except Exception as e:
            if attempt == MAX_RETRY - 1:
                raise
            m = re.search(r"try again in ([\d.]+)\s*(ms|s)", str(e))
            wait = max((float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 1) if m else 0,
                       min(60, 3 * 2 ** attempt))
            print(f"    재시도 {attempt + 1} ({type(e).__name__}) {wait:.0f}초 대기")
            time.sleep(wait)


def _retryable(e: Exception) -> bool:
    """한도(429)·서버 오류(5xx)·연결 오류만 재시도. 키·모델 이름 오류 등은 바로 알려줌"""
    status = getattr(e, "status_code", None)
    return status is None or status == 429 or status >= 500


def chat_json(prompt: str) -> dict:
    oa = get_clients()["openai"]
    for attempt in range(MAX_RETRY):
        try:
            r = oa.chat.completions.create(
                model=CHAT_MODEL, temperature=0.3, response_format={"type": "json_object"}, timeout=60,
                # JSON 모드는 지시문에 'json'이라는 단어가 반드시 있어야 함
                messages=[{"role": "system", "content": "Respond only with a valid JSON object."},
                          {"role": "user", "content": prompt}])
            return json.loads(r.choices[0].message.content)
        except json.JSONDecodeError:
            return {}
        except Exception as e:
            if not _retryable(e) or attempt == MAX_RETRY - 1:
                print(f"    ❌ GPT 호출 실패: {type(e).__name__}: {str(e)[:200]}")
                raise
            m = re.search(r"try again in ([\d.]+)\s*(ms|s)", str(e))
            wait = max((float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 1) if m else 0,
                       min(60, 3 * 2 ** attempt))
            print(f"    재시도 {attempt + 1} ({type(e).__name__}) {wait:.0f}초 대기")
            time.sleep(wait)


# ---------------------------- 1. 데이터·질문 ----------------------------
def load_pool() -> pd.DataFrame:
    df = pd.read_csv(RAW_NEWS_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["기사ID"] = [article_id(u, t, d) for u, t, d in zip(df["원문URL"], df["제목"], df["발행일"])]
    df["본문"] = df["본문전문"].map(clean)
    df = df[df["본문"].str.len() >= 200].drop_duplicates("기사ID")
    return df.sample(min(POOL_N, len(df)), random_state=SEED).reset_index(drop=True)


NEWS_Q_PROMPT = """너는 검색 시스템 평가용 질문을 만드는 사람이다.
[기사 제목] {title}
[발췌] {passage}

규칙:
1. 위 발췌 내용으로 답할 수 있는 구체적인 질문 1개를 만든다.
2. 실제 사용자가 챗봇에 물어볼 법한 자연스러운 한국어로 쓴다.
3. 제목 문장을 그대로 베끼지 않는다. (회사명·핵심 용어는 써도 된다)
4. '이 기사', '발췌', '위 내용' 같은 표현은 쓰지 않는다.
출력: {{"question": "..."}}"""

PATENT_Q_PROMPT = """너는 검색 시스템 평가용 질문을 만드는 사람이다.
[특허 명칭] {title}
[요약] {abstract}

규칙:
1. 이 특허를 찾고 싶은 사람이 물어볼 법한 질문 1개를 만든다. (예: "~하는 기술 특허 있어?")
2. 명칭을 그대로 베끼지 말고, 기능·용도·대상 중심으로 쓴다.
3. 자연스러운 한국어로 쓴다.
출력: {{"question": "..."}}"""


def make_questions(pool: pd.DataFrame) -> list[dict]:
    path = config.EVAL_SET_DIR / "eval_questions.json"
    if path.exists():
        qs = json.loads(path.read_text(encoding="utf-8"))
        print(f"평가 질문 불러옴: {len(qs)}개 ({path.name})")
        return qs

    OUT_DIR.mkdir(exist_ok=True)
    partial = OUT_DIR / "eval_questions_partial.jsonl"   # 하나씩 저장 → 중간에 멈춰도 이어서
    rng = random.Random(SEED)
    tasks = []
    for r in pool.sample(min(N_NEWS_Q, len(pool)), random_state=SEED).itertuples():
        sents = split_sentences(r.본문, 1000)
        i = rng.randrange(len(sents))
        passage, j = "", i
        while j < len(sents) and len(passage) < 300:
            passage += sents[j] + " "
            j += 1
        start = r.본문.find(sents[i])
        pos = start / max(len(r.본문), 1)
        tasks.append({"종류": "뉴스", "정답": r.기사ID, "정답제목": r.제목,
                      "위치": "앞" if pos < 1 / 3 else ("중간" if pos < 2 / 3 else "뒤"),
                      "prompt": NEWS_Q_PROMPT.format(title=r.제목, passage=passage.strip())})

    pat = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in PATENT_FILES]).drop_duplicates("출원번호")
    pat = pat[pat["요약"].str.strip() != ""]
    for r in pat.sample(min(N_PATENT_Q, len(pat)), random_state=SEED).itertuples():
        tasks.append({"종류": "특허", "정답": r.출원번호, "정답제목": r.발명의명칭, "위치": "",
                      "prompt": PATENT_Q_PROMPT.format(title=r.발명의명칭, abstract=r.요약[:1500])})

    for i, t in enumerate(tasks):
        t["qid"] = f"q{i:03d}"
    done = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["qid"]] = rec
    todo = [t for t in tasks if t["qid"] not in done]
    print(f"평가 질문 생성: 전체 {len(tasks)}개 / 이미 생성 {len(done)}개 / 새로 생성 {len(todo)}개 (GPT)")

    # 첫 1개를 먼저 호출해서 키·모델 문제를 바로 확인
    if todo:
        print("  연결 확인 중 (첫 질문 1개)…")

    def work(t):
        q = chat_json(t["prompt"]).get("question", "").strip()
        return {**{k: v for k, v in t.items() if k != "prompt"}, "질문": q}

    lock = __import__("threading").Lock()
    with partial.open("a", encoding="utf-8") as f:
        if todo:
            rec = work(todo[0])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            done[rec["qid"]] = rec
            print(f"  ✅ 연결 정상. 예시 질문: {rec['질문'][:60]}")
        with ThreadPoolExecutor(4) as ex:
            for n, rec in enumerate(ex.map(work, todo[1:]), 2):
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                    done[rec["qid"]] = rec
                if n % 25 == 0 or n == len(todo):
                    print(f"  진행 {n}/{len(todo)}")

    qs = [done[t["qid"]] for t in tasks if t["qid"] in done and done[t["qid"]]["질문"]]
    path.write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"  저장: {path.name} ({len(qs)}개)")
    return qs


def question_vectors(qs: list[dict]) -> dict[str, list[float]]:
    path = OUT_DIR / "question_vecs.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [q for q in qs if q["qid"] not in cache]
    for b in range(0, len(todo), BATCH):
        part = todo[b:b + BATCH]
        for q, v in zip(part, embed([q["질문"] for q in part])):
            cache[q["qid"]] = v
    path.write_text(json.dumps(cache))
    return cache


# ---------------------------- 2. 청킹 설정별 임베딩 ----------------------------
def build_configs(pool: pd.DataFrame) -> dict:
    stats = {}
    for name, (label, size, overlap) in CONFIGS.items():
        ids, docs, metas = [], [], []
        for r in pool.itertuples():
            for k, c in enumerate(chunk_text(r.본문, size, overlap), 1):
                ids.append(f"{r.기사ID}-{k:02d}")
                docs.append(f"[제목] {r.제목}\n{c}")
                metas.append({"기사ID": r.기사ID})
        col = get_clients()["eval"].get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        have = set()
        for i in range(0, len(ids), 5000):
            have |= set(col.get(ids=ids[i:i + 5000], include=[])["ids"])
        todo = [i for i, x in enumerate(ids) if x not in have]
        print(f"\n[{label}] 청크 {len(ids):,}개 / 새로 임베딩 {len(todo):,}개")
        for b in range(0, len(todo), BATCH):
            idx = todo[b:b + BATCH]
            col.add(ids=[ids[i] for i in idx], documents=[docs[i] for i in idx],
                    metadatas=[metas[i] for i in idx], embeddings=embed([docs[i] for i in idx]))
            if (b // BATCH) % 20 == 0:
                print(f"  진행 {b + len(idx):,}/{len(todo):,}")
        lens = [len(d) for d in docs]
        stats[name] = {"청크수": len(ids), "평균청크길이": round(float(np.mean(lens)))}
    return stats


# ---------------------------- 3. 측정 ----------------------------
def rank_news(col, vec, gold, n=40) -> int | None:
    """뉴스: 같은 기사 청크는 한 번만 세어 '기사 순위'로 정답 위치 반환"""
    r = col.query(query_embeddings=[vec], n_results=n, include=["metadatas"])
    seen = []
    for m in r["metadatas"][0]:
        if m["기사ID"] not in seen:
            seen.append(m["기사ID"])
    return seen.index(gold) + 1 if gold in seen else None


def rank_patent(col, vec, gold, gold_title, n=20) -> int | None:
    """특허: 제목 같은 패밀리 특허는 한 건으로 세고, 그것도 정답으로 인정"""
    r = col.query(query_embeddings=[vec], n_results=n, include=["metadatas"])
    seen = []
    for _id, m in zip(r["ids"][0], r["metadatas"][0]):
        t = m.get("발명의명칭", _id)
        if t not in seen:
            seen.append(t)
        if _id == gold or t == gold_title:
            return seen.index(t) + 1
    return None


def metrics(ranks: list) -> dict:
    n = len(ranks)
    hit = lambda k: sum(1 for x in ranks if x and x <= k) / n if n else 0
    return {"질문수": n, "Hit@1": hit(1), "Hit@3": hit(3), "Hit@5": hit(5),
            "MRR": sum(1 / x for x in ranks if x and x <= 10) / n if n else 0}


def run():
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)
    pool = load_pool()
    print(f"기사 표본: {len(pool):,}건")
    qs = make_questions(pool)
    vecs = question_vectors(qs)
    stats = build_configs(pool)

    rows, details = [], []
    news_q = [q for q in qs if q["종류"] == "뉴스"]
    pat_q = [q for q in qs if q["종류"] == "특허"]

    # [청킹 비교] 표본 2,000건
    for name, (label, _, _) in CONFIGS.items():
        col = get_clients()["eval"].get_collection(name)
        ranks = [rank_news(col, vecs[q["qid"]], q["정답"]) for q in news_q]
        row = {"구분": "청킹 비교", "대상": "뉴스", "설정": label, "문서풀": f"기사 {len(pool):,}건",
               **stats[name], **metrics(ranks)}
        for p in ("앞", "중간", "뒤"):
            sub = [r for r, q in zip(ranks, news_q) if q["위치"] == p]
            row[f"Hit@3({p})"] = metrics(sub)["Hit@3"]
        rows.append(row)
        details += [{"설정": label, **{k: q[k] for k in ("qid", "종류", "위치", "질문", "정답제목")}, "정답순위": r}
                    for q, r in zip(news_q, ranks)]

    # [1차 기준선] 본 DB 전체
    main = get_clients()["main"]
    if main:
        col = main.get_collection("news")
        ranks = [rank_news(col, vecs[q["qid"]], q["정답"]) for q in news_q]
        rows.append({"구분": "1차 기준선", "대상": "뉴스", "설정": "③ 현재 (본 DB 전체)",
                     "문서풀": f"청크 {col.count():,}개", **metrics(ranks)})
        details += [{"설정": "본DB-뉴스", **{k: q[k] for k in ("qid", "종류", "위치", "질문", "정답제목")}, "정답순위": r}
                    for q, r in zip(news_q, ranks)]
        col = main.get_collection("patents")
        ranks = [rank_patent(col, vecs[q["qid"]], q["정답"], q["정답제목"]) for q in pat_q]
        rows.append({"구분": "1차 기준선", "대상": "특허", "설정": "청킹 없음 (본 DB 전체)",
                     "문서풀": f"특허 {col.count():,}건", **metrics(ranks)})
        details += [{"설정": "본DB-특허", **{k: q[k] for k in ("qid", "종류", "위치", "질문", "정답제목")}, "정답순위": r}
                    for q, r in zip(pat_q, ranks)]
    else:
        print("⚠️ 본 DB(store/chroma)가 없어 1차 기준선 측정은 건너뛰었어요.")

    res = pd.DataFrame(rows)
    shown = res.copy().astype(object)
    for c in shown.columns:
        if c.startswith("Hit@"):
            shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif c == "MRR":
            shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        elif c in ("청크수", "평균청크길이", "질문수"):
            shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    res.to_csv(OUT_DIR / "eval_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(details).to_csv(OUT_DIR / "eval_details.csv", index=False, encoding="utf-8-sig")

    cols = ["구분", "대상", "설정", "문서풀", "청크수", "평균청크길이", "질문수", "Hit@1", "Hit@3", "Hit@5", "MRR",
            "Hit@3(앞)", "Hit@3(중간)", "Hit@3(뒤)"]
    cols = [c for c in cols if c in shown.columns]
    shown = shown[cols].where(shown[cols].notna(), "")
    table = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    table += "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in shown.itertuples(index=False))
    (OUT_DIR / "eval_report.md").write_text(
        f"# 검색기 1차 평가\n\n- 측정 시각: {time.strftime('%Y-%m-%d %H:%M')}\n- 임베딩 모델: {EMBED_MODEL} / 질문 생성: {CHAT_MODEL}\n"
        f"- 평가 질문: 뉴스 {len(news_q)}개(기사 무작위 위치 발췌로 생성), 특허 {len(pat_q)}개(명칭·요약으로 생성)\n"
        f"- 정답 기준: 뉴스는 같은 기사(청크 무관), 특허는 같은 출원번호 또는 같은 제목(패밀리)\n\n" + table, encoding="utf-8")

    print("\n" + "=" * 70)
    print(shown.to_string(index=False))
    print("=" * 70)
    print(f"저장: eval/eval_report.md, eval_results.csv, eval_details.csv ({(time.time() - t0) / 60:.0f}분)")
    return res


if __name__ == "__main__":
    run()
