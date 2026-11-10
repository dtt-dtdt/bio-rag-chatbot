"""
실사용 점검 2 — smoke_test.py와 같은 방식, 질문만 새로 15개

[왜]  첫 번째 점검 질문(알테오젠·셀트리온·한미약품 등)에 맞춰 검색기를 고쳐 왔기 때문에,
      처음 보는 질문에서도 잘 되는지 확인 (같은 질문만 계속 보면 그 질문에만 맞춰질 수 있음)
[방식]  smoke_test.py의 run()·summarize()를 그대로 쓰고, 질문과 저장 파일 이름만 바꿈
        - 질문 유형 5가지 × 3개 (회사명+이벤트, 회사명+기술, 개념, DB에 없는 회사, 기간)
        - 결과: eval/smoke_test2.md (보고서), eval/smoke_test2_review.csv (판정 칸 O/△/X)
        - 첫 번째 점검 파일(smoke_test.md, smoke_test_review.csv)은 건드리지 않음
[사용 - Jupyter]
  from eval import smoke_test2
  smoke_test2.run()          # 검색 + 시간 측정 (질문 임베딩 15번, 비용 거의 없음)
  # eval/smoke_test2_review.csv 의 '판정' 칸을 채워 저장한 뒤
  smoke_test2.summarize()    # 기준 ⑤ 판정 (기간 유형 포함하려면 summarize(include_period=True))
"""
from __future__ import annotations

from eval import smoke_test as base

REVIEW = base.OUT_DIR / "smoke_test2_review.csv"
REPORT = base.OUT_DIR / "smoke_test2.md"
TOP_K = base.TOP_K          # 첫 번째 점검과 같게 5개 (챗봇 답변은 8개를 쓰지만, 비교를 위해 기준을 맞춤)

# (유형, 질문, 좋은 결과의 모습) — 첫 번째 점검과 겹치지 않는 회사·주제
QUESTIONS = [
    ("회사명+이벤트", "유한양행 렉라자 기술수출 성과", "유한양행 렉라자(레이저티닙) 관련 기사"),
    ("회사명+이벤트", "삼성바이오로직스 CDMO 수주 실적", "삼성바이오로직스 수주 기사"),
    ("회사명+이벤트", "SK바이오사이언스 백신 해외 계약", "SK바이오사이언스 백신 계약 기사"),
    ("회사명+기술", "녹십자 혈액제제 특허", "녹십자 출원 특허 (혈액제제·면역글로불린 등)"),
    ("회사명+기술", "종근당 신약 후보물질 특허", "종근당 출원 특허"),
    ("회사명+기술", "대웅제약 나보타 해외 진출", "대웅제약 나보타 수출·허가 기사"),
    ("개념", "CAR-T 세포치료제 기술", "CAR-T 관련 특허·기사"),
    ("개념", "디지털 치료기기 허가 현황", "디지털 치료기기(DTx) 허가 기사"),
    ("개념", "항체약물접합체 링커 기술 특허", "ADC 링커 관련 특허"),
    ("DB에 없는 회사", "리가켐바이오 ADC 특허", "특허 DB에 이 회사 특허 없음. ADC 특허면 △, 무관한 특허면 X (뉴스는 관련 기사면 O)"),
    ("DB에 없는 회사", "하늘별바이오 신약 임상 결과", "가상의 회사. 관련 결과가 없어야 정상 (엉뚱한 결과는 X)"),
    ("DB에 없는 회사", "지씨셀 세포치료제 특허", "특허 DB에 이 회사 특허 없음. 세포치료제 특허면 △, 무관한 특허면 X (뉴스는 관련 기사면 O)"),
    ("기간 (참고)", "2023년 바이오시밀러 FDA 허가", "2023년 바이오시밀러 허가 기사"),
    ("기간 (참고)", "2025년 비대면 진료 제도 변화", "2025년 비대면 진료 기사"),
    ("기간 (참고)", "2022년 국내 제약사 기술수출 규모", "2022년 기술수출 기사"),
]


class _Swap:
    """smoke_test.py의 설정을 잠시 바꿨다가 원래대로 돌려놓음 (첫 번째 점검에 영향 없게)"""
    def __enter__(self):
        self.saved = {k: getattr(base, k) for k in ("QUESTIONS", "REVIEW", "REPORT", "TOP_K")}
        prev = base.REVIEW.with_name("smoke_test_review_prev.csv")      # 첫 번째 점검의 백업 파일 보호
        self.prev = (prev, prev.read_bytes()) if prev.exists() else (prev, None)
        base.QUESTIONS, base.REVIEW, base.REPORT, base.TOP_K = QUESTIONS, REVIEW, REPORT, TOP_K
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(base, k, v)
        path, data = self.prev
        new_prev = path.read_bytes() if path.exists() else None
        if new_prev is not None and new_prev != data:                  # 이번 실행이 덮어썼으면
            REVIEW.with_name("smoke_test2_review_prev.csv").write_bytes(new_prev)
            if data is None:
                path.unlink()
            else:
                path.write_bytes(data)


def run():
    with _Swap():
        return base.run()


def summarize(include_period: bool = False):
    with _Swap():
        return base.summarize(include_period)


if __name__ == "__main__":
    run()
