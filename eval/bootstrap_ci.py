"""
부트스트랩 신뢰구간 — 리트리버 설정 간 차이가 '우연'인지 확인 (API 비용 없음)

[원리]
  1. 같은 질문으로 두 설정을 짝지어 비교 (대응 비교: 질문 난이도 차이를 없앰)
  2. 질문 목록에서 '중복을 허용해 같은 개수'를 무작위로 다시 뽑아 차이를 계산 → 2,000번 반복
  3. 2,000개 차이 중 가운데 95% 범위 = 95% 신뢰구간
     · 구간이 0을 포함하지 않으면 → 차이가 우연이 아님 (확실히 낫다/나쁘다)
     · 0을 포함하면 → 이 질문 수로는 '비슷하다'로 봐야 함
  + 갈린 질문 수: 기준만 맞춘 질문 / 상대만 맞춘 질문 (직관적인 확인용)

[입력]  eval/results/eval_retriever_details.csv  (eval/retriever_eval.py 최신 버전 실행 시 생성)
[결과]  eval/bootstrap_ci.md, bootstrap_ci.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config

BASE_DIR = config.ROOT

DETAILS = config.EVAL_RESULTS_DIR / "eval_retriever_details.csv"
OUT_DIR = config.EVAL_RESULTS_DIR
REFERENCE = "하이브리드 벡터1:BM25 2 + 노이즈"   # 기준 설정 (최종 후보)
N_BOOT = 2000
SEED = 42
GOOD_ONLY = True        # 좋은 질문만 (노이즈 필터 비교를 공정하게)


def boot_diff(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    """a - b 의 평균 차이와 95% 신뢰구간 (같은 질문끼리 짝지어 다시 뽑기)"""
    n = len(a)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "기준이 확실히 나음"
    if hi < 0:
        return "상대가 확실히 나음"
    return "비슷함 (우연 범위)"


def run():
    if not DETAILS.exists():
        raise FileNotFoundError(f"{DETAILS} 가 없어요. 최신 eval/retriever_eval.py를 실행하면 만들어져요 "
                                "(판정은 저장된 걸 써서 몇 분이면 돼요).")
    d = pd.read_csv(DETAILS)
    if GOOD_ONLY:
        d = d[d["좋은질문"].astype(str) == "True"]
    d["hit"] = d["맞춤(정답급)"].astype(str).eq("True").astype(float)
    d["p3"] = d["쓸모있는수"].astype(float) / 3
    rng = np.random.default_rng(SEED)

    rows = []
    for (qset, kind), g in d.groupby(["질문세트", "대상"]):
        settings = list(dict.fromkeys(g["설정"]))
        if REFERENCE not in settings:
            continue
        ref = g[g["설정"] == REFERENCE].set_index("qid")
        for other in settings:
            if other == REFERENCE:
                continue
            o = g[g["설정"] == other].set_index("qid")
            common = ref.index.intersection(o.index)
            for metric, name in (("hit", "Hit@3"), ("p3", "P@3")):
                a, b = ref.loc[common, metric].values, o.loc[common, metric].values
                mean, lo, hi = boot_diff(a, b, rng)
                row = {"질문세트": qset, "대상": kind, "지표": name, "비교 상대": other, "질문수": len(common),
                       "기준 점수": a.mean(), "상대 점수": b.mean(), "차이(기준-상대)": mean,
                       "95% 신뢰구간": f"[{lo * 100:+.1f}, {hi * 100:+.1f}]%p", "판정": verdict(lo, hi)}
                if metric == "hit":
                    row["기준만 맞춤"] = int(((a == 1) & (b == 0)).sum())
                    row["상대만 맞춤"] = int(((a == 0) & (b == 1)).sum())
                rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    shown = res.copy().astype(object)
    for c in ("기준 점수", "상대 점수"):
        shown[c] = shown[c].map(lambda v: f"{v:.1%}")
    shown["차이(기준-상대)"] = shown["차이(기준-상대)"].map(lambda v: f"{v * 100:+.1f}%p")
    for c in ("기준만 맞춤", "상대만 맞춤"):
        if c in shown:
            shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else int(v))

    hit = shown[shown["지표"] == "Hit@3"].drop(columns="지표")
    p3 = shown[shown["지표"] == "P@3"].drop(columns=["지표", "기준만 맞춤", "상대만 맞춤"], errors="ignore")

    def md(df):
        cols = list(df.columns)
        return ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
                + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False)))

    head = (f"- 기준: {REFERENCE} / {'좋은 질문만' if GOOD_ONLY else '전체 질문'} / 부트스트랩 {N_BOOT}회, 95% 신뢰구간\n"
            "- 구간이 0을 포함하지 않으면 우연이 아닌 차이, 포함하면 '비슷함'")
    (OUT_DIR / "bootstrap_ci.md").write_text(
        f"# 설정 간 차이의 신뢰구간\n\n{head}\n\n## Hit@3 (정답급)\n\n{md(hit)}\n## P@3\n\n{md(p3)}", encoding="utf-8")

    pd.set_option("display.width", 250)
    print(head + "\n\n[Hit@3]\n" + hit.to_string(index=False) + "\n\n[P@3]\n" + p3.to_string(index=False))
    print("\n저장: eval/bootstrap_ci.md, bootstrap_ci.csv")
    return res


if __name__ == "__main__":
    run()
