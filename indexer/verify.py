"""
벡터 DB 적재 검증

[검증 항목]
  1. 개수 일치      : 원본 CSV 수 = DB 수
  2. 누락·초과 ID   : 원본에 있는데 DB에 없는 ID / DB에만 있는 ID
  3. 메타데이터     : 필터에 쓰는 값(분야, 날짜 등)이 채워져 있는지
  4. 키워드 반영    : 특허 본문의 핵심키워드, 메타데이터의 분야태그
  5. 모델·벡터      : 임베딩 모델 이름, 벡터 차원(1536), 비어 있는(0) 벡터 없는지
  6. 색인 자기검색  : 저장된 벡터로 검색하면 자기 자신이 1위로 나오는지 (API 비용 없음)
  7. 질문 검색 테스트: 제목으로 검색하면 그 문서가 상위 5위 안에 나오는지 (API 20회, 1원 미만)

[실행]  셀에 붙여 실행 → 화면에 ✅/❌ 표 + verify_report.md 저장 (보고서·발표 자료용)
"""
from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path

import pandas as pd

# ============================== 설정 ==============================
import config

BASE_DIR = config.ROOT

DB_DIR = config.CHROMA_DIR
PATENT_FILES = config.PATENT_FILES
NEWS_FILE = config.NEWS_CHUNKS
EMBED_MODEL = "text-embedding-3-small"
EXPECTED_DIM = 1536
SAMPLE = 200            # 벡터·색인 검사에 쓸 표본 수 (컬렉션별)
API_TEST_N = 10         # 질문 검색 테스트 수 (컬렉션별). 0이면 건너뜀
BOILERPLATE_RE = re.compile(r"\s*(기사를 더 보고 싶다면\??|기사 섹션 분류 안내)\s*")  # indexer/embed.py와 동일
REPORT = config.REPORT_DIR / "verify_report.md"
# ==================================================================

rows = []  # (구분, 항목, 결과, 상세)


def check(group, item, ok, detail=""):
    mark = "✅" if ok is True else ("⚠️" if ok is None else "❌")
    rows.append((group, item, mark, detail))
    print(f"  {mark} {item}: {detail}")


def get_all(col, include):
    """컬렉션 전체를 나눠서 읽기"""
    ids, metas, docs, off = [], [], [], 0
    while True:
        r = col.get(limit=5000, offset=off, include=include)
        if not r["ids"]:
            break
        ids += r["ids"]
        metas += r.get("metadatas") or [None] * len(r["ids"])
        docs += r.get("documents") or [None] * len(r["ids"])
        off += len(r["ids"])
    return ids, metas, docs


def verify_collection(col, name, source_ids, meta_checks, title_key, group_key=None, openai_client=None):
    print(f"\n[{name}]")
    ids, metas, docs = get_all(col, ["metadatas", "documents"])
    db, src = set(ids), set(source_ids)

    # 1·2. 개수와 ID
    check(name, "개수 일치", len(db) == len(src), f"원본 {len(src):,} / DB {len(db):,}")
    missing, extra = src - db, db - src
    check(name, "누락 ID", not missing, f"{len(missing):,}개" + (f" 예: {list(missing)[:3]}" if missing else ""))
    check(name, "초과 ID", not extra, f"{len(extra):,}개" + (f" 예: {list(extra)[:3]}" if extra else ""))

    # 3. 메타데이터
    md = pd.DataFrame(metas)
    for key, rule, desc, warn_only in meta_checks:
        if key not in md.columns:
            check(name, f"메타 '{key}'", False, "칼럼 없음")
            continue
        ok_rate = md[key].map(rule).mean()
        ok = True if ok_rate == 1 else (None if warn_only else False)
        check(name, f"메타 '{key}' ({desc})", ok, f"{ok_rate:.1%} 채워짐")

    # 5. 모델·벡터
    used = (col.metadata or {}).get("embed_model")
    check(name, "임베딩 모델", used == EMBED_MODEL, f"{used}")
    random.seed(42)
    got = col.get(ids=random.sample(ids, min(SAMPLE, len(ids))), include=["embeddings"])
    sample_ids, emb = got["ids"], got["embeddings"]   # 요청한 순서와 다르게 올 수 있어서, 돌려받은 순서 그대로 사용
    dims = {len(e) for e in emb}
    zero = sum(1 for e in emb if not any(abs(v) > 1e-9 for v in e))
    check(name, "벡터 차원", dims == {EXPECTED_DIM}, f"{dims} (표본 {len(emb)}개)")
    check(name, "빈 벡터", zero == 0, f"{zero}개")

    # 6. 색인 자기검색 (저장된 벡터로 검색 → 자기 자신 또는 같은 내용이 1위)
    hit = 0
    for _id, e in zip(sample_ids, emb):
        r = col.query(query_embeddings=[list(e)], n_results=1, include=["distances"])
        hit += r["ids"][0][0] == _id or r["distances"][0][0] < 1e-6
    check(name, "색인 자기검색", hit == len(sample_ids), f"{hit}/{len(sample_ids)} 1위")

    # 7. 질문 검색 테스트 (제목으로 검색 → 상위 5위 안)
    if openai_client and API_TEST_N:
        by_id = dict(zip(ids, metas))
        test_ids = sample_ids[:API_TEST_N]
        titles = [by_id[i][title_key] for i in test_ids]
        vecs = [d.embedding for d in openai_client.embeddings.create(model=EMBED_MODEL, input=titles).data]
        hit = 0
        for _id, v in zip(test_ids, vecs):
            r = col.query(query_embeddings=[v], n_results=5, include=["metadatas"])
            if group_key:  # 뉴스: 같은 기사의 다른 청크가 나와도 정답
                hit += by_id[_id][group_key] in {m[group_key] for m in r["metadatas"][0]}
            else:          # 특허: 자기 자신 또는 제목이 같은 패밀리 특허
                hit += _id in r["ids"][0] or by_id[_id][title_key] in {m[title_key] for m in r["metadatas"][0]}
        check(name, "제목 검색 상위5", hit >= 0.8 * len(test_ids), f"{hit}/{len(test_ids)}")

    return ids, metas, docs


def run():
    import chromadb
    rows.clear()
    t0 = time.time()
    client = chromadb.PersistentClient(path=str(DB_DIR))

    openai_client = None
    if API_TEST_N:
        try:
            from openai import OpenAI
            from dotenv import load_dotenv
            load_dotenv(config.ENV_FILE)
            openai_client = OpenAI() if os.getenv("OPENAI_API_KEY") else None
        except ImportError:
            pass

    # ---- 원본 ID ----
    pat = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in PATENT_FILES])
    pat_ids = pat["출원번호"].drop_duplicates().tolist()
    news = pd.read_csv(NEWS_FILE, dtype=str, keep_default_na=False, usecols=["청크ID", "청크본문"])
    keep = news["청크본문"].map(lambda b: len(BOILERPLATE_RE.sub(" ", b).strip()) >= 10)  # indexer/embed.py가 건너뛰는 청크 제외
    news_ids = news.loc[keep, "청크ID"].tolist()

    filled = lambda v: v not in (None, "", 0)
    pos = lambda v: isinstance(v, int) and v > 0

    # ---- 특허 ----
    ids, metas, docs = verify_collection(
        client.get_collection("patents"), "patents", pat_ids,
        [("분야", lambda v: v in ("바이오의약품", "바이오헬스케어"), "의약품/헬스케어", False),
         ("등록일_num", pos, "기간 필터용", False),
         ("출원일_num", pos, "기간 필터용", False),
         ("공개일_num", pos, "빈 칸은 정상(조기 등록)", True),
         ("출원인", filled, "", True)],
        title_key="발명의명칭", openai_client=openai_client)
    kw = sum("핵심키워드:" in (d or "") for d in docs)
    tag = sum(filled(m.get("분야태그")) for m in metas)
    check("patents", "핵심키워드 반영", kw == len(ids) or None, f"{kw:,}/{len(ids):,}건")
    check("patents", "분야태그 반영", tag == len(ids) or None, f"{tag:,}/{len(ids):,}건")

    # ---- 뉴스 ----
    verify_collection(
        client.get_collection("news"), "news", news_ids,
        [("발행일_num", pos, "기간 필터용", False),
         ("원문URL", filled, "출처 표시용", False),
         ("기사ID", filled, "기사 묶기용", False),
         ("매체명", filled, "", True)],
        title_key="제목", group_key="기사ID", openai_client=openai_client)

    if not openai_client and API_TEST_N:
        print("\n  ⚠️ OpenAI 키가 없어 '제목 검색 상위5' 테스트는 건너뛰었어요.")

    # ---- 결과 ----
    df = pd.DataFrame(rows, columns=["컬렉션", "항목", "결과", "상세"])
    n_fail, n_warn = (df["결과"] == "❌").sum(), (df["결과"] == "⚠️").sum()
    verdict = "✅ 적재 검증 통과" if n_fail == 0 else f"❌ 실패 {n_fail}개 항목 확인 필요"
    print(f"\n{verdict} (경고 {n_warn}개, {time.time() - t0:.0f}초)")
    table = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    table += "".join("| " + " | ".join(str(v).replace("|", "/") for v in r) + " |\n" for r in df.itertuples(index=False))
    REPORT.write_text(
        f"# 벡터 DB 적재 검증 보고서\n\n- 검증 시각: {time.strftime('%Y-%m-%d %H:%M')}\n- DB 위치: {DB_DIR}\n"
        f"- 결과: {verdict} (경고 {n_warn}개)\n\n" + table, encoding="utf-8")
    print(f"보고서 저장: {REPORT.name}")
    return df


if __name__ == "__main__":
    run()
