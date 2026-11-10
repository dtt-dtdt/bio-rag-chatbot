"""
리트리버 실사용 점검(기준 ⑤) + 속도 측정(기준 ⑥)

[무엇을 하나]
  1. 실제로 물어볼 법한 질문 15개(5가지 유형)를 하이브리드 리트리버로 검색
  2. 질문마다 특허 상위 5개 + 뉴스 상위 5개를 보고서와 검토표(CSV)로 저장
  3. 질문마다 걸린 시간을 '질문 임베딩(OpenAI)'과 '검색(DB+BM25)'으로 나눠 측정

[확정 기준]
  ⑤ 상위 5개에 '명백히 엉뚱한' 결과가 없을 것  → 검토표의 '판정' 칸에 사람이 O / △ / X 를 적고 summarize() 실행
  ⑥ 질문 1개당 2초 이내 (색인을 만든 뒤 기준, 처음 한 번 준비 시간은 제외)

[사용 - Jupyter]
  from eval import smoke_test; smoke_test.run()          # 검색 + 시간 측정 (API: 질문 임베딩 15번, 비용 거의 없음)
  # eval/smoke_test_review.csv 를 엑셀로 열어 '판정' 칸 채우고 저장 (O=관련, △=애매, X=명백히 엉뚱)
  smoke_test.summarize()                       # 기준 ⑤ 판정
"""
from __future__ import annotations

import importlib
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

import config

BASE_DIR = config.ROOT

OUT_DIR = config.EVAL_RESULTS_DIR
REVIEW = OUT_DIR / "smoke_test_review.csv"
REPORT = OUT_DIR / "smoke_test.md"
TOP_K = 5
TIME_LIMIT = 2.0        # 기준 ⑥: 질문 1개당 초

# (유형, 질문, 좋은 결과의 모습)
QUESTIONS = [
    ("회사명+이벤트", "알테오젠 최근 기술수출", "알테오젠 기술수출 기사, 알테오젠 히알루로니다제 특허"),
    ("회사명+이벤트", "셀트리온 미국 FDA 허가", "셀트리온 FDA 승인 기사"),
    ("회사명+이벤트", "한미약품 기술이전 계약", "한미약품 라이선스 아웃 기사"),
    ("회사명+기술", "셀트리온 항체 특허", "셀트리온 출원 특허"),
    ("회사명+기술", "한미약품 비만 치료제", "한미약품 GLP-1 계열 특허·기사"),
    ("회사명+기술", "알테오젠 히알루로니다제 피하주사 제형", "알테오젠 히알루로니다제 특허·키트루다SC 기사"),
    ("개념", "면역항암제 내성 극복 기술", "면역관문억제제 내성 관련 특허·기사"),
    ("개념", "암세포와 T세포를 연결하는 이중항체", "CD3 이중특이항체 특허"),
    ("개념", "원격의료 규제 현황", "비대면 진료 제도 기사"),
    ("개념", "mRNA 백신 플랫폼 기술", "mRNA·지질나노입자 특허·기사"),
    ("DB에 없는 회사", "에이비엘바이오 이중항체 특허", "특허 DB에 이 회사 특허 없음. 이중항체 특허면 △, 무관한 특허면 X"),
    ("DB에 없는 회사", "가나다바이오 기술수출", "가상의 회사. 관련 결과가 없어야 정상 (엉뚱한 결과는 X)"),
    ("기간 (셀프쿼리 전, 참고)", "2024년 ADC 기술수출 동향", "2024년 ADC 기사 (지금은 기간 필터가 없어 실패 예상)"),
    ("기간 (셀프쿼리 전, 참고)", "2022년 코로나 백신 국산화", "2022년 국산 백신 기사"),
    ("기간 (셀프쿼리 전, 참고)", "2025년 GLP-1 비만약 개발", "2025년 GLP-1 기사·특허"),
]


def _load_retriever():
    """항상 파일의 기본 설정으로 다시 불러옴 (평가 후 가중치가 1:1로 바뀐 상태 방지)"""
    sys.modules.pop("rag.retriever", None)
    rt = importlib.import_module("rag.retriever")
    return rt


def _chat_model(rt) -> str:
    """챗봇이 실제로 쓰는 답변 모델 (rag/chain.py 기준)"""
    try:
        from rag import chain
        return chain.CHAT_MODEL
    except Exception:
        return "알 수 없음 (rag/chain.py를 불러오지 못함)"


def run():
    rt = _load_retriever()
    OUT_DIR.mkdir(exist_ok=True)
    setting = (f"가중치 벡터 {rt.W_VEC:g} : BM25 {rt.W_BM25:g} / 노이즈 필터 {rt.NOISE_FILTER} / "
               f"회사명 필터 {getattr(rt, 'COMPANY_FILTER', '없음')} / 챗봇 답변 모델 {_chat_model(rt)}")
    print("설정:", setting)

    # 준비: 색인 불러오기 + 연결 (시간 측정에서 제외)
    t0 = time.perf_counter()
    rt.get_index("patents"); rt.get_index("news")
    rt.retrieve("준비", q_vec=rt._embed("준비"))     # 첫 검색에만 드는 준비 시간도 제외
    prep = time.perf_counter() - t0
    print(f"준비(색인·연결): {prep:.1f}초 — 시간 측정에서 제외\n")

    rows, times = [], []
    for n, (kind, q, expect) in enumerate(QUESTIONS, 1):
        t1 = time.perf_counter()
        vec = rt._embed(q)
        t2 = time.perf_counter()
        res = rt.retrieve(q, q_vec=vec, k_patent=TOP_K, k_news=TOP_K)
        t3 = time.perf_counter()
        times.append({"번호": n, "유형": kind, "질문": q, "임베딩(초)": t2 - t1, "검색(초)": t3 - t2, "합계(초)": t3 - t1})
        comp = next((x.get("회사명필터") for x in res if x.get("회사명필터")), None)
        print(f"[{n:2d}] {q}  ({t3 - t1:.2f}초{', 회사명 필터: ' + ','.join(comp) if comp else ''})")

        rank = {"특허": 0, "뉴스": 0}
        for x in res:
            rank[x["종류"]] += 1
            m = x["메타"]
            who = rt.owner_text(m) if x["종류"] == "특허" and hasattr(rt, "owner_text") else m.get("매체명", "")
            rows.append({"번호": n, "유형": kind, "질문": q, "좋은 결과": expect, "종류": x["종류"],
                         "순위": rank[x["종류"]], "날짜": str(x["날짜"])[:10], "제목": x["제목"],
                         "출원인/매체": who[:40], "경로": x["검색경로"],
                         "벡터순위": x["벡터순위"] or "", "BM25순위": x["BM25순위"] or "",
                         "출처": x["출처"], "판정": ""})
        if not res:
            rows.append({"번호": n, "유형": kind, "질문": q, "좋은 결과": expect, "종류": "-", "순위": 0,
                         "날짜": "", "제목": "(결과 없음)", "출원인/매체": "", "경로": "", "벡터순위": "",
                         "BM25순위": "", "출처": "", "판정": ""})

    review = pd.DataFrame(rows)
    if REVIEW.exists():   # 이미 적어 둔 판정은 지우지 않음 → 같은 질문·문서면 판정을 옮겨 줌
        old = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
        keep = {(r["질문"], r["종류"], r["출처"]): r["판정"] for _, r in old.iterrows() if r["판정"].strip()}
        review["판정"] = [keep.get((r["질문"], r["종류"], r["출처"]), "") for _, r in review.iterrows()]
        old.to_csv(REVIEW.with_name("smoke_test_review_prev.csv"), index=False, encoding="utf-8-sig")
    review.to_csv(REVIEW, index=False, encoding="utf-8-sig")

    # ---------- 기준 ⑥: 속도 ----------
    t = pd.DataFrame(times)
    total = t["합계(초)"].tolist()
    over = t[t["합계(초)"] > TIME_LIMIT]
    speed_ok = over.empty
    speed = (f"- 질문 {len(t)}개, 합계 평균 {statistics.mean(total):.2f}초 / 중앙값 {statistics.median(total):.2f}초 / "
             f"최대 {max(total):.2f}초\n"
             f"- 그중 임베딩(OpenAI 호출) 평균 {t['임베딩(초)'].mean():.2f}초, 검색(DB+BM25) 평균 {t['검색(초)'].mean():.2f}초\n"
             f"- {TIME_LIMIT:g}초 넘은 질문: {len(over)}개 → 기준 ⑥ {'통과' if speed_ok else '미달'}")
    print("\n[기준 ⑥ 속도]\n" + speed)

    # ---------- 보고서 ----------
    lines = [f"# 리트리버 실사용 점검 (기준 ⑤·⑥)\n", f"- 측정: {time.strftime('%Y-%m-%d %H:%M')}",
             f"- 설정: {setting}", f"- 준비(색인·연결) {prep:.1f}초는 측정에서 제외\n",
             "## 기준 ⑥ 속도\n", speed, "",
             "| 번호 | 유형 | 질문 | 임베딩 | 검색 | 합계 |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['번호']} | {r['유형']} | {r['질문']} | {r['임베딩(초)']:.2f} | {r['검색(초)']:.2f} | "
              f"{r['합계(초)']:.2f} |" for _, r in t.iterrows()]
    lines += ["", "## 기준 ⑤ 검색 결과 (판정은 smoke_test_review.csv에 적기)\n"]
    for n, (kind, q, expect) in enumerate(QUESTIONS, 1):
        lines.append(f"### {n}. [{kind}] {q}\n- 좋은 결과: {expect}\n")
        g = review[review["번호"] == n]
        lines += ["| 종류 | 순위 | 날짜 | 제목 | 출원인/매체 | 경로 |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['종류']} | {r['순위']} | {r['날짜']} | {str(r['제목'])[:50]} | {r['출원인/매체']} | {r['경로']} |"
                  for _, r in g.iterrows()]
        lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n저장: {REPORT.relative_to(BASE_DIR)} (보고서), {REVIEW.relative_to(BASE_DIR)} (판정 칸을 채워 주세요)")
    return review, t


def summarize(include_period: bool = False):
    """검토표의 판정(O/△/X)으로 기준 ⑤ 판정. 기본은 '기간' 유형 제외 (셀프쿼리 전이라 참고용)"""
    d = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
    d["판정"] = d["판정"].str.strip().str.upper()
    if not include_period:
        d = d[~d["유형"].str.startswith("기간")]
    empty = d[(d["판정"] == "") & (d["제목"] != "(결과 없음)")]
    if len(empty):
        print(f"⚠️ 판정이 비어 있는 줄이 {len(empty)}개 있어요. 채운 뒤 다시 실행해 주세요.")
    rows = []
    for (n, q, kind), g in d.groupby(["번호", "질문", "유형"], sort=False):
        rows.append({"번호": int(n), "유형": kind, "질문": q,
                     "특허 X": int(((g["종류"] == "특허") & (g["판정"] == "X")).sum()),
                     "뉴스 X": int(((g["종류"] == "뉴스") & (g["판정"] == "X")).sum()),
                     "O": int((g["판정"] == "O").sum()), "△": int((g["판정"] == "△").sum())})
    s = pd.DataFrame(rows).sort_values("번호")
    s["통과"] = (s["특허 X"] + s["뉴스 X"] == 0).map({True: "O", False: "X"})
    print(s.to_string(index=False))
    fails = (s["통과"] == "X").sum()
    print(f"\n질문 {len(s)}개 중 엉뚱한 결과가 있는 질문 {fails}개 → 기준 ⑤ {'통과' if fails == 0 else '미달'}"
          f"{'' if include_period else ' (기간 유형 제외)'}")
    return s


if __name__ == "__main__":
    run()
