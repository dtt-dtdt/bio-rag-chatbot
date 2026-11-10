"""
검색 실패 진단 — 정답 문서가 DB에 있는데 챗봇이 못 찾은 질문을 하나씩 조사해서 원인을 분류

[입력]  eval/answer_eval_details_v2.csv (실패유형 = 검색 실패), eval/answer_runs_v2.jsonl (실제 쓴 검색어·조건),
        eval/eval_questions.json (정답 문서), eval/question_vecs.json (원래 질문 벡터)
[확인]  질문마다
  ① 정답 문서가 노이즈로 표시됐나?            → 그렇다면 '노이즈 필터' (필터를 꺼도 순위가 낮으면 메모에 표시)
  ② 질문 분석이 넣은 기간 밖의 문서인가?        → 그렇다면 '기간 필터'
  ③ 원래 질문으로 검색하면 찾아지나?           → 그렇다면 '검색어(질문 분석)'
  ④ 벡터·BM25 중 한쪽에선 상위인가?            → 그렇다면 '결합 비율'
  ⑤ 상위 50위 안에는 있나?                    → 그렇다면 '순위 밀림 (비슷한 문서가 많음)'
  ⑥ 아무 데서도 안 보이나?                     → '표현 차이' (약칭·동의어 등. 질문 단어 중 정답 문서에 없는 단어를 보여 줌)
  ※ 챗봇은 특허·뉴스를 각각 상위 5개씩 쓰므로 '찾음' = 같은 종류에서 5위 안
[비용]  검색어 임베딩만 질문당 1번 (16번 정도 → 거의 무료). 답변·채점 LLM은 부르지 않음
[실행 - Jupyter]
  from eval import diagnose_misses as dm
  d = dm.run()
[결과]  eval/miss_diagnosis.csv, eval/miss_diagnosis.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from rag import retriever as rt

import config

BASE_DIR = config.ROOT

OUT_DIR = config.EVAL_RESULTS_DIR
VERSION = "v2"
DEEP = 200                 # 진단할 때만 후보를 넓게 (평소 50) → 50위 밖 순위도 확인
CAUSES = ["노이즈 필터", "기간 필터", "검색어(질문 분석)", "결합 비율", "순위 밀림", "표현 차이"]


def _rank(res: list[dict], kind: str, gold: str, gold_title: str):
    for i, x in enumerate(res, 1):
        if kind == "뉴스" and x["메타"].get("기사ID") == gold:
            return i
        if kind == "특허" and (x["id"] == gold or x["제목"] == gold_title):
            return i
    return None


def _search(q, kind, gold, title, mode, noise, vec, dates):
    t = "news" if kind == "뉴스" else "patent"
    k = {"k_news": DEEP, "k_patent": 0} if t == "news" else {"k_news": 0, "k_patent": DEEP}
    res = rt.retrieve(q, target=t, mode=mode, noise_filter=noise, q_vec=vec, **dates, **k)
    return _rank(res, kind, gold, title)


def _gold_info(kind: str, gold: str) -> dict:
    """정답 문서의 노이즈 표시·날짜·검색용 단어"""
    idx = rt.get_index("news" if kind == "뉴스" else "patents")
    if kind == "뉴스":
        pos = [i for i, m in enumerate(idx.metas) if m.get("기사ID") == gold]
        date_key = "발행일"
    else:
        pos = [idx.id_pos[gold]] if gold in idx.id_pos else []
        date_key = "등록일"
    if not pos:
        return {"DB에 있음": False, "노이즈표시": None, "날짜": "", "단어": set()}
    metas = [idx.metas[i] for i in pos]
    words = set()
    for i in pos:
        words |= set(rt.tokenize(idx.docs[i] + ("\n" + rt.owner_text(idx.metas[i]) if kind == "특허" else "")))
    return {"DB에 있음": True, "노이즈표시": any(m.get("바이오관련") is False for m in metas),
            "날짜": str(metas[0].get(date_key, ""))[:10], "단어": words}


def _in_period(date: str, f: dict) -> bool:
    if not date:
        return True
    d = int(date.replace("-", "")[:8] or 0)
    lo, hi = rt._date_num(f.get("date_from")), rt._date_num(f.get("date_to"), end=True)
    return (not lo or d >= lo) and (not hi or d <= hi)


def run():
    t0 = time.time()
    det = pd.read_csv(OUT_DIR / VERSION / f"answer_eval_details_{VERSION}.csv", dtype=str, keep_default_na=False)
    qs = {q["qid"]: q for q in json.loads((config.EVAL_SET_DIR / "eval_questions.json").read_text(encoding="utf-8"))}
    vecs = json.loads((OUT_DIR / "question_vecs.json").read_text())
    runs = {}
    for line in (OUT_DIR / VERSION / f"answer_runs_{VERSION}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            runs[r["key"]] = r["out"]

    miss = det[(det["실패유형"] == "검색 실패") & det["key"].isin(qs)]
    no_gold = det[(det["실패유형"] == "검색 실패") & ~det["key"].isin(qs)]
    print(f"검색 실패 {len(miss) + len(no_gold)}건 중 정답 문서가 있는 평가 질문 {len(miss)}건을 진단해요"
          + (f" (정답 문서가 없는 질문 {len(no_gold)}건은 제외: {', '.join(no_gold['key'])})" if len(no_gold) else ""))

    rt.get_index("patents"); rt.get_index("news")
    old_cand, rt.CANDIDATES = rt.CANDIDATES, DEEP
    rows = []
    try:
        for r in miss.itertuples():
            q = qs[r.key]
            kind, gold, title = q["종류"], q["정답"], q["정답제목"]
            run_out = runs.get(r.key, {})
            sq = (run_out.get("analysis") or {}).get("search_query") or q["질문"]
            used = run_out.get("적용조건", {}) or {}
            dates = {k: v for k, v in used.items() if k in ("date_from", "date_to")}
            g = _gold_info(kind, gold)
            sq_vec = rt._embed(sq) if sq != q["질문"] else vecs[r.key]
            top = rt.K_NEWS if kind == "뉴스" else rt.K_PATENT

            rank_actual = _search(sq, kind, gold, title, "hybrid", True, sq_vec, dates)        # 챗봇이 실제로 한 검색
            rank_nonoise = _search(sq, kind, gold, title, "hybrid", False, sq_vec, dates)      # 노이즈 필터 끔
            rank_nodate = _search(sq, kind, gold, title, "hybrid", True, sq_vec, {}) if dates else rank_actual
            rank_orig = _search(q["질문"], kind, gold, title, "hybrid", True, vecs[r.key], dates)  # 원래 질문
            rank_vec = _search(sq, kind, gold, title, "vector", False, sq_vec, {})
            rank_bm = _search(sq, kind, gold, title, "bm25", False, None, {})
            missing = [w for w in dict.fromkeys(rt.tokenize(sq)) if w not in g["단어"]]

            ok = lambda x: x is not None and x <= top    # noqa: E731
            if not g["DB에 있음"]:
                cause = "표현 차이"
            elif g["노이즈표시"]:
                cause = "노이즈 필터"     # 표시돼 있으면 필터가 켜진 한 절대 못 찾음
            elif dates and not _in_period(g["날짜"], dates):
                cause = "기간 필터"
            elif ok(rank_orig):
                cause = "검색어(질문 분석)"
            elif ok(rank_vec) or ok(rank_bm):
                cause = "결합 비율"
            elif rank_actual is not None:
                cause = "순위 밀림"
            else:
                cause = "표현 차이"
            fmt = lambda x: "-" if x is None else str(x)   # noqa: E731
            rows.append({"key": r.key, "종류": kind, "질문": q["질문"], "검색어": sq, "정답제목": title,
                         "정답날짜": g["날짜"], "노이즈표시": {True: "예", False: "아니오", None: "?"}[g["노이즈표시"]],
                         "적용기간": f"{dates.get('date_from', '')}~{dates.get('date_to', '')}" if dates else "",
                         "실제순위": fmt(rank_actual), "노이즈끔": fmt(rank_nonoise), "기간뺌": fmt(rank_nodate),
                         "원래질문": fmt(rank_orig), "벡터만": fmt(rank_vec), "BM25만": fmt(rank_bm),
                         "정답에없는단어": ", ".join(missing[:8]), "원인": cause,
                         "메모": (f"노이즈 필터를 꺼도 {f'{rank_nonoise}위' if rank_nonoise else f'{DEEP}위 밖'} (검색 순위도 함께 개선 필요)"
                                if cause == "노이즈 필터" and not ok(rank_nonoise) else "")})
            print(f"  {r.key} {kind} | 실제 {fmt(rank_actual)}위 · 노이즈끔 {fmt(rank_nonoise)} · 원래질문 {fmt(rank_orig)} · "
                  f"벡터 {fmt(rank_vec)} · BM25 {fmt(rank_bm)} → {cause}")
    finally:
        rt.CANDIDATES = old_cand

    d = pd.DataFrame(rows)
    d.to_csv(OUT_DIR / "miss_diagnosis.csv", index=False, encoding="utf-8-sig")
    cnt = d["원인"].value_counts().reindex(CAUSES).fillna(0).astype(int) if len(d) else pd.Series(dtype=int)

    def md(df):
        cols = list(df.columns)
        return ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
                + "".join("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in x) + " |\n"
                          for x in df.itertuples(index=False)))
    guide = {"노이즈 필터": "정답 문서가 노이즈로 표시돼 검색에서 빠짐 → 노이즈 판정 기준 점검",
             "기간 필터": "질문 분석이 넣은 기간 밖의 문서 → '최근' 해석(RECENT_MONTHS) 점검",
             "검색어(질문 분석)": "원래 질문으론 찾는데 다듬은 검색어로 못 찾음 → 질문 분석 프롬프트 점검",
             "결합 비율": "벡터나 BM25 한쪽에선 상위인데 합치면 밀림 → 가중치·후보 수 점검",
             "순위 밀림": "후보 안에는 있지만 비슷한 문서가 많아 5위 밖 → 중복 기사 묶기·재정렬 검토",
             "표현 차이": "어디서도 상위에 없음 → 약칭·동의어(정답에 없는 단어 참고) 처리 검토"}
    summary = pd.DataFrame({"원인": cnt.index, "건수": cnt.values, "의미 → 고칠 곳": [guide[c] for c in cnt.index]})
    show_cols = ["key", "종류", "질문", "검색어", "실제순위", "노이즈끔", "원래질문", "벡터만", "BM25만", "정답에없는단어", "원인", "메모"]
    (OUT_DIR / "miss_diagnosis.md").write_text(
        f"# 검색 실패 진단\n\n- 실행: {time.strftime('%Y-%m-%d %H:%M')} / 진단 {len(d)}건 / "
        f"순위는 같은 종류 안에서의 순위 (챗봇은 {rt.K_NEWS}위까지 사용, '-'는 {DEEP}위 밖)\n\n## 원인별\n\n{md(summary)}\n"
        f"## 질문별\n\n{md(d[show_cols]) if len(d) else '(없음)'}", encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print("\n" + summary.to_string(index=False))
    print(f"\n저장: eval/miss_diagnosis.md, miss_diagnosis.csv ({time.time() - t0:.0f}초)")
    return d


if __name__ == "__main__":
    run()
