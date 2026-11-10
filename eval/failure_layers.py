"""
실패 원인 계층 분석 — 실패한 질문마다 원인이 데이터·전처리·인덱싱·검색·생성 중 어느 계층인지 판정

[입력]  eval/results/<태그>/answer_eval_details_<태그>.csv (채점 결과), answer_runs_<태그>.jsonl (실제 검색 자료)
[판정 순서]  (앞 계층의 문제가 먼저. 정답 문서가 정해진 '단순 사실' 질문은 추가 검색으로 확인)
  정답 문서가 있는 질문
    1. 정답 문서가 DB에 없음                                    → 데이터
    2. 정답 문서가 검색됨 + 답변에 근거 없는 내용               → 생성 (근거 없는 내용)
       정답 문서가 검색됨 + 답변이 틀리거나 부족                → 생성 (답변 부정확)
    3. 정답 문서가 노이즈(바이오 무관)로 표시돼 검색에서 빠짐   → 전처리 (노이즈 기준)
    4. 후보를 200개까지 넓혀도 정답 문서가 안 나옴              → 인덱싱
    5. 같은 내용 기사가 자리를 차지해 밀려남                     → 전처리 (중복)
       (정답 순위 ≤ 사용 개수 + 같은 내용 중복 수)
    6. 200위 안에는 있지만 사용 개수 밖                          → 검색 (순위 낮음)
    ※ 질문 분석이 주제 밖으로 분류해 검색 자체를 안 함          → 검색 (질문 분석)
  정답 문서가 없는 질문 (거절 점검·실사용·설계 질문)
    - 근거 없는 내용·거절해야 할 질문에 답함                    → 생성 (근거 없는 내용)
    - 불완전·질문 오해·출처 오류 등                              → 생성 (답변 부정확)
    - 검색 실패·근거 부족 거절                                   → 데이터 (추정: 해당 주제 문서 부재)
[비용]  정답 문서가 검색되지 않은 질문만 검색어 임베딩 1번씩 (몇 원)
[실행]
  from eval import failure_layers as fl
  d = fl.analyze("v5")        # → eval/results/v5/failure_layers.md, failure_layers.csv
"""
from __future__ import annotations

import json
import time

import pandas as pd

import config
from rag import retriever as rt

RESULTS_ROOT = config.EVAL_RESULTS_DIR
DEEP = 200                     # 추가 검색 때 후보 수 (평소 50)
DUP_TITLE_SIM, DUP_DAYS = 0.5, 3   # 같은 내용 기사: 제목 단어 50% 이상 겹침 + 보도일 3일 이내

# 계층 → (증상, 대응)  — 평가 설계 문서의 표 그대로
LAYERS = {
    "데이터": ("검색 결과에 정답 문서가 포함되지 않음 (DB에 없음)", "해당 주제 문서 부재 → 공공 문서 추가 수집"),
    "전처리": ("검색 결과 상위가 동일 내용으로 채워짐 / 노이즈 표시로 제외", "중복 처리·노이즈 판정 기준 재설정"),
    "인덱싱": ("정답 문서가 존재하나 검색되지 않음 (후보 200위 밖)", "청크 크기, 제목 포함 여부 점검"),
    "검색": ("검색은 되었으나 순위가 낮음", "top_k 조정, Reranker, 하이브리드 가중치"),
    "생성 (답변 부정확)": ("근거는 정확하나 답변이 부정확함", "프롬프트 수정"),
    "생성 (근거 없는 내용)": ("근거 없는 내용을 생성함", "거절 지시 강화, 유사도 임계값 적용"),
    "판단 보류": ("규칙으로 정할 수 없음", "사례를 직접 확인"),
}
GEN_WRONG = {"불완전", "질문 오해", "출처 오류", "한쪽만 답함", "날짜 오류", "상충 근거 누락", "기타"}
GEN_UNGROUNDED = {"환각", "거절 실패"}


def _load(tag: str):
    folder = RESULTS_ROOT / tag
    det = pd.read_csv(folder / f"answer_eval_details_{tag}.csv", dtype=str, keep_default_na=False)
    runs = {}
    for line in (folder / f"answer_runs_{tag}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            runs[r["key"]] = r["out"]
    qs = {q["qid"]: q for q in json.loads((config.EVAL_SET_DIR / "eval_questions.json").read_text(encoding="utf-8"))}
    return folder, det, runs, qs


def _same_event(a: dict, b: dict) -> bool:
    try:
        days = abs((pd.Timestamp(str(a["날짜"])[:10]) - pd.Timestamp(str(b["날짜"])[:10])).days)
    except ValueError:
        return False
    if days > DUP_DAYS:
        return False
    wa, wb = set(rt.tokenize(a["제목"])), set(rt.tokenize(b["제목"]))
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= DUP_TITLE_SIM


def _dup_count(sources: list[dict]) -> int:
    """검색 자료(뉴스) 중 앞 자료와 같은 내용인 자료 수"""
    news = [s for s in sources if s["종류"] == "뉴스"]
    return sum(1 for i, s in enumerate(news) if any(_same_event(s, p) for p in news[:i]))


def _gold(kind: str, gold: str) -> dict:
    idx = rt.get_index("news" if kind == "뉴스" else "patents")
    if kind == "뉴스":
        metas = [m for m in idx.metas if m.get("기사ID") == gold]
    else:
        metas = [idx.metas[idx.id_pos[gold]]] if gold in idx.id_pos else []
    return {"in_db": bool(metas), "noise": any(m.get("바이오관련") is False for m in metas)}


def _deep_rank(kind: str, gold: str, title: str, query: str, cond: dict):
    """후보를 DEEP개로 넓혀 같은 조건으로 다시 검색했을 때 정답 문서의 순위 (같은 종류 안, 없으면 None)"""
    target = "news" if kind == "뉴스" else "patent"
    k = {"k_news": DEEP, "k_patent": 0} if target == "news" else {"k_news": 0, "k_patent": DEEP}
    keep = {key: cond[key] for key in ("date_from", "date_to", "field", "companies") if cond.get(key)}
    old, rt.CANDIDATES = rt.CANDIDATES, DEEP
    try:
        res = rt.retrieve(query, target=target, **keep, **k)
    finally:
        rt.CANDIDATES = old
    for i, x in enumerate(res, 1):
        if (kind == "뉴스" and x["메타"].get("기사ID") == gold) or (kind == "특허" and (x["id"] == gold or x["제목"] == title)):
            return i
    return None


def classify(row: pd.Series, out: dict, q: dict | None) -> dict:
    ftype, grounded = row["실패유형"], row["근거성"]
    ungrounded = grounded in ("no", "partly") or ftype in GEN_UNGROUNDED
    analysis = out.get("analysis") or {}
    res = {"계층": "판단 보류", "근거": "", "정답순위(200위까지)": "", "사용개수": "", "중복기사": ""}

    if q is None:                                   # 정답 문서가 정해지지 않은 질문
        if ungrounded:
            res.update(계층="생성 (근거 없는 내용)", 근거=f"실패 유형 '{ftype}', 근거성 {grounded}")
        elif ftype in GEN_WRONG:
            res.update(계층="생성 (답변 부정확)", 근거=f"실패 유형 '{ftype}'")
        elif ftype in ("검색 실패", "과잉 거절"):
            res.update(계층="데이터", 근거="정답 문서가 없는 질문에서 자료를 찾지 못함 → 주제 문서 부재로 추정 (확인 필요)")
        return res

    kind, gold, title = q["종류"], q["정답"], q["정답제목"]
    docs = [s["doc"] for s in out.get("sources", [])]
    g = _gold(kind, gold)
    if not g["in_db"]:
        return {**res, "계층": "데이터", "근거": "정답 문서가 DB에 없음"}
    if gold in docs:
        if ungrounded:
            return {**res, "계층": "생성 (근거 없는 내용)", "근거": f"정답 문서는 검색됨, 근거성 {grounded}"}
        return {**res, "계층": "생성 (답변 부정확)", "근거": f"정답 문서는 검색됨, 실패 유형 '{ftype}'"}
    if analysis and analysis.get("category") not in (None, "ok", "advice"):
        return {**res, "계층": "검색", "근거": f"질문 분석이 '{analysis.get('category')}'로 분류해 검색하지 않음"}
    if g["noise"]:
        return {**res, "계층": "전처리", "근거": "정답 문서가 노이즈(바이오 무관)로 표시돼 검색에서 제외"}

    cond = out.get("적용조건", {}) or {}
    used = int(cond.get("k_news" if kind == "뉴스" else "k_patent", 8) or 8)
    query = analysis.get("search_query") or row["질문"]
    rank = _deep_rank(kind, gold, title, query, cond)
    dups = _dup_count(out.get("sources", [])) if kind == "뉴스" else 0
    res.update({"정답순위(200위까지)": rank or f"{DEEP}위 밖", "사용개수": used, "중복기사": dups})
    if rank is None:
        return {**res, "계층": "인덱싱", "근거": f"후보 {DEEP}개 안에도 정답 문서가 없음 (검색어: {query})"}
    if dups and rank <= used + dups:
        return {**res, "계층": "전처리", "근거": f"같은 내용 기사 {dups}건이 자리를 차지 (정답 {rank}위, 사용 {used}개)"}
    if rank <= used:      # 후보를 넓히면 들어오지만 실제 검색(후보 CANDIDATES개)에서는 빠짐
        return {**res, "계층": "검색", "근거": f"후보 {DEEP}개 기준 {rank}위인데 실제 검색(후보 {rt.CANDIDATES}개)에선 빠짐 → 후보 수 영향"}
    return {**res, "계층": "검색", "근거": f"정답 {rank}위 — 답변에 쓰는 {used}개 밖"}


def analyze(tag: str = "v5") -> pd.DataFrame:
    t0 = time.time()
    folder, det, runs, qs = _load(tag)
    fail = det[det["실패"] == "True"]
    print(f"[{tag}] 실패 {len(fail)}개의 원인 계층을 판정해요 (정답 문서가 검색 안 된 질문은 후보 {DEEP}개로 다시 검색)")
    rows = []
    for _, r in fail.iterrows():
        c = classify(r, runs.get(r["key"], {}), qs.get(r["key"]))
        rows.append({"key": r["key"], "그룹": r["그룹"], "질문": r["질문"], "실패유형": r["실패유형"], **c,
                     "채점이유": r["채점이유"]})
        print(f"  {r['key']:5s} {r['실패유형']:8s} → {c['계층']}  ({c['근거'][:60]})")
    d = pd.DataFrame(rows)
    d.to_csv(folder / "failure_layers.csv", index=False, encoding="utf-8-sig")

    order = list(LAYERS)
    cnt = d["계층"].value_counts().reindex(order).fillna(0).astype(int) if len(d) else pd.Series(dtype=int)
    summary = pd.DataFrame({"원인 계층": cnt.index, "건수": cnt.values,
                            "증상": [LAYERS[k][0] for k in cnt.index], "대응": [LAYERS[k][1] for k in cnt.index],
                            "예": [", ".join(d[d["계층"] == k]["key"].head(5)) for k in cnt.index]})
    summary = summary[summary["건수"] > 0]

    def md(df):
        return ("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
                + "".join("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in x) + " |\n"
                          for x in df.itertuples(index=False)))
    top = summary.sort_values("건수", ascending=False).head(1)
    (folder / "failure_layers.md").write_text(
        f"# 실패 원인 계층 — {tag}\n\n- 실행: {time.strftime('%Y-%m-%d %H:%M')} / 실패 {len(d)}개 / "
        f"전체 {len(det)}개\n- 다음 실험 후보: **{top['원인 계층'].iloc[0] if len(top) else '-'}** 계층"
        f"{' → ' + top['대응'].iloc[0] if len(top) else ''} (한 번에 변수 하나만 바꿔 재측정)\n\n"
        f"## 계층별\n\n{md(summary)}\n## 질문별\n\n"
        + md(d[["key", "그룹", "실패유형", "계층", "근거", "정답순위(200위까지)", "사용개수", "중복기사"]]), encoding="utf-8")
    print("\n" + summary[["원인 계층", "건수", "대응"]].to_string(index=False))
    print(f"\n저장: eval/results/{tag}/failure_layers.md, failure_layers.csv ({time.time() - t0:.0f}초)")
    return d


if __name__ == "__main__":
    import sys
    analyze(sys.argv[1] if len(sys.argv) > 1 else "v5")
