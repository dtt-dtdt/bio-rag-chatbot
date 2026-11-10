"""
경로 설정 — 모든 모듈이 이 파일의 경로를 씀 (폴더 위치를 바꿀 때는 여기만 고치면 됨)

project3/
├── .env                 API 키 (OPENAI_API_KEY). GitHub에 올리지 않음
├── store/raw/           원천 데이터 (뉴스 CSV, 키프리스 특허 CSV)
├── store/processed/     가공 데이터 (뉴스 청크, 특허 키워드)
├── store/chroma/        벡터 DB (ChromaDB)
├── store/bm25/          키워드 검색(BM25) 색인
├── store/noise/         노이즈(바이오 무관 문서) 판정 결과·분류 모델
├── eval/eval_set/       평가 질문
└── eval/results/        평가 결과
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"

# ③ 저장 계층
STORE_DIR = ROOT / "store"
RAW_DIR = STORE_DIR / "raw"
PROCESSED_DIR = STORE_DIR / "processed"
CHROMA_DIR = STORE_DIR / "chroma"
CHROMA_EVAL_DIR = STORE_DIR / "chroma_eval"      # 청킹 설정 비교용 DB (eval/harness.py가 만듦)
BM25_DIR = STORE_DIR / "bm25"
NOISE_DIR = STORE_DIR / "noise"

# 원천 데이터 (파일 이름은 받은 그대로)
NEWS_RAW = RAW_DIR / "04_바이오헬스_묶음_바이오헬스케어_2021-10-16~2026-10-16_중복제거.csv"
PATENT_PHARMA = RAW_DIR / "kipris_pharma_final_20261104.csv"           # 키프리스: 바이오의약품
PATENT_HEALTHCARE = RAW_DIR / "kipris_healthcare_final_20261104.csv"   # 키프리스: 바이오헬스케어
PATENT_FILES = [PATENT_PHARMA, PATENT_HEALTHCARE]

# 가공 데이터
NEWS_CHUNKS = PROCESSED_DIR / "news_chunks.csv"
KEYWORD_PHARMA = PROCESSED_DIR / "kipris_pharma_keywords.csv"
KEYWORD_HEALTHCARE = PROCESSED_DIR / "kipris_healthcare_keywords.csv"

# 평가
EVAL_SET_DIR = ROOT / "eval" / "eval_set"
EVAL_RESULTS_DIR = ROOT / "eval" / "results"

REPORT_DIR = ROOT / "report"
