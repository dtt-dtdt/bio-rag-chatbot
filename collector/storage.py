"""
원천 데이터 저장 위치와 불러오기

[원천 데이터]  store/raw/ 에 받은 파일 그대로 둠 (GitHub에는 올리지 않음)
  - 뉴스 : 제공받은 바이오헬스 뉴스 묶음 CSV (2021-10-16 ~ 2026-10-16, 중복 제거본 32,168건)
           칼럼: 제목, 본문전문, 원문URL, 발행일, 매체명, 문서종류, 수집키워드
  - 특허 : 키프리스(kipris.or.kr)에서 바이오의약품·바이오헬스케어 검색식으로 내려받은 등록 특허를
           합치고 중복을 없앤 CSV 2개 (합계 8,790건)
           주요 칼럼: 발명의명칭, IPC분류, CPC분류, 출원번호, 출원일자, 출원인, 등록번호, 등록일자, 요약, 최종권리자

[사용]
  from collector import storage
  storage.check()            # 파일이 있는지, 몇 건인지
  news = storage.load_news()
"""
from __future__ import annotations

import pandas as pd

import config

NEWS_COLUMNS = ["제목", "본문전문", "원문URL", "발행일", "매체명", "수집키워드"]
PATENT_COLUMNS = ["발명의명칭", "출원번호", "출원인", "등록일자", "요약"]


def load_news() -> pd.DataFrame:
    return pd.read_csv(config.NEWS_RAW, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def load_patents() -> pd.DataFrame:
    """특허 CSV 2개를 합침. '분야' 칼럼에 바이오의약품/바이오헬스케어 표시"""
    parts = []
    for path, field in ((config.PATENT_PHARMA, "바이오의약품"), (config.PATENT_HEALTHCARE, "바이오헬스케어")):
        d = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        parts.append(d.assign(분야=field))
    return pd.concat(parts, ignore_index=True)


def check() -> bool:
    """원천 파일이 제자리에 있는지, 필요한 칼럼이 있는지 확인"""
    ok = True
    for name, path, cols in (("뉴스", config.NEWS_RAW, NEWS_COLUMNS),
                             ("특허(바이오의약품)", config.PATENT_PHARMA, PATENT_COLUMNS),
                             ("특허(바이오헬스케어)", config.PATENT_HEALTHCARE, PATENT_COLUMNS)):
        if not path.exists():
            print(f"❌ {name}: 파일이 없어요 → {path.relative_to(config.ROOT)}")
            ok = False
            continue
        d = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        miss = [c for c in cols if c not in d.columns]
        print(f"{'✅' if not miss else '⚠️'} {name}: {len(d):,}건" + (f" (빠진 칼럼: {', '.join(miss)})" if miss else ""))
        ok = ok and not miss
    return ok
