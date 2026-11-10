"""
python -m indexer <단계>  →  인덱싱 배치 실행 (project3 폴더에서)

  chunk     뉴스 본문을 문장 단위 청크로 자름          → store/processed/news_chunks.csv          (API 없음)
  keywords  특허 분야태그·핵심키워드 생성 (LLM)        → store/processed/kipris_*_keywords.csv     (API 사용)
  embed     임베딩 → ChromaDB 저장 (이어서 처리됨)     → store/chroma/                             (API 사용)
  noise     뉴스·특허 노이즈(바이오 무관) 판정 (LLM)   → store/noise/  (DB 반영은 apply 단계)      (API 사용)
  apply     노이즈 판정을 DB 메타데이터에 반영          → store/chroma/                             (API 없음)
  verify    적재 검증 (개수·메타데이터·벡터·검색)       → report/verify_report.md                   (API 약 20회)

BM25 색인(store/bm25/)은 처음 검색할 때 rag/retriever.py가 자동으로 만들어요.
"""
import sys

STEPS = ("chunk", "keywords", "embed", "noise", "apply", "verify")


def main(step: str):
    if step == "chunk":
        from indexer import chunk; chunk.run()
    elif step == "keywords":
        from indexer import keywords; keywords.run()
    elif step == "embed":
        from indexer import embed; embed.run()
    elif step == "noise":
        from indexer import noise_news, noise_patent
        noise_news.run(); noise_patent.run()
    elif step == "apply":
        from indexer import noise_news, noise_patent
        noise_news.apply_to_db(); noise_patent.apply_to_db()
    elif step == "verify":
        from indexer import verify; verify.run()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
