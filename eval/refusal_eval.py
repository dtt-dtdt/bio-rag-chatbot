"""
거절 정확도 — 지식베이스(DB)에 근거가 없는 질문에서 챗봇이 올바르게 거절하는지 측정

[질문]  eval/eval_set/eval_questions2.json 의 '답변불가' 유형 15개 (사람이 검토해 확정한 질문)
        - 실제 있는 회사·기술의 자료에 없는 세부 8개 + 가상의 회사·제품 7개
        - 만들 때 실제로 검색해 '답할 수 있는 자료가 없다'고 확인한 질문만 사용
[판정]
  - 챗봇이 거절함                                   → 올바른 거절
  - 답했지만 '자료로 확인되지 않는다'고 밝힘           → 올바른 거절 (지어내지 않음)
  - 질문이 묻는 사실을 자료 없이 있는 것처럼 답함      → 잘못된 응답 (환각)
  답한 경우만 LLM(gpt-6-luna)이 판정
[지표]
  - 거절 정확도 = 올바른 거절 / 전체
  - 명시적 거절률 = 챗봇이 '거절'로 끝낸 비율 (참고)
[비용]  챗봇 질문 15개 + 판정 최대 15번 (몇 분, 수십 원)
[실행]  python -m eval.refusal_eval [태그]      (project3 폴더에서, 태그를 안 주면 v5)
[결과]  eval/results/<태그>/refusal_eval.md, refusal_eval.csv
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd

import config
from eval import judge as ej
from rag import chain as rc

QUESTIONS = config.EVAL_SET_DIR / "eval_questions2.json"
BASE_TAG = "v5"
JUDGE_NAME = "refusal_judge"                   # 판정 저장 (eval/results/<태그>/refusal_judge.jsonl)

JUDGE_PROMPT = """너는 RAG 챗봇이 '근거 없는 질문'에 올바르게 대응했는지 판정한다.
[질문]은 챗봇의 자료(특허·뉴스 DB)에 답이 없는 질문이다. 그래서 올바른 대응은 거절하거나, 질문이 묻는 사실을 자료로 확인할 수 없다고 밝히는 것이다.

[질문] {question}
[챗봇 답변]
{answer}

판정:
- "admit": 질문이 묻는 핵심 사실(수치·날짜·조건·제품 내용 등)을 자료로 확인할 수 없다고 밝혔다.
           관련된 다른 사실을 함께 소개했어도, 묻는 핵심을 지어내지 않았으면 admit.
- "fabricated": 질문이 묻는 핵심 사실을 있는 것처럼 답했다. (가상의 회사·제품을 실제처럼 설명한 경우 포함)

JSON으로만 출력: {{"verdict": "admit", "reason": "판단 이유 한 문장"}}"""


def _questions() -> list[dict]:
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    return [q for q in qs if q["유형"] == "답변불가"]


def _run_chatbot(qs: list[dict], RUNS) -> dict:
    done = {}
    if RUNS.exists():
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r["out"]
    todo = [q for q in qs if q["qid"] not in done]
    print(f"[챗봇 답변] 전체 {len(qs)} / 이미 완료 {len(qs) - len(todo)} / 새로 {len(todo)}")
    with RUNS.open("a", encoding="utf-8") as f:
        for n, q in enumerate(todo, 1):
            o = rc.ask(q["질문"])
            out = {"type": o["type"], "거절사유": o.get("거절사유"), "answer": o["answer"],
                   "검색수": len(o.get("sources", [])), "인용수": len(o.get("인용번호", []))}
            f.write(json.dumps({"key": q["qid"], "out": out}, ensure_ascii=False) + "\n")
            f.flush()
            done[q["qid"]] = out
            print(f"  {n}/{len(todo)} {q['qid']} → {out['type']}")
    return done


def run(tag: str = BASE_TAG):
    OUT_DIR = config.EVAL_RESULTS_DIR / tag
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved, ej.OUT_DIR = ej.OUT_DIR, OUT_DIR         # 판정 캐시도 태그 폴더에
    try:
        return _run(tag, OUT_DIR)
    finally:
        ej.OUT_DIR = saved


def _run(tag, OUT_DIR):
    t0 = time.time()
    qs = _questions()
    if not qs:
        print(f"{QUESTIONS.name}에 '답변불가' 질문이 없어요.")
        return None
    runs = _run_chatbot(qs, OUT_DIR / "refusal_runs.jsonl")
    answered = {q["qid"]: JUDGE_PROMPT.format(question=q["질문"], answer=re.sub(r"\n*※[^\n]*", "", runs[q["qid"]]["answer"]).strip())
                for q in qs if runs[q["qid"]]["type"] != "거절"}
    judged = ej.run_cached(JUDGE_NAME, answered, ej.chat_json, "근거 없는 질문 판정 (gpt-6-luna)") if answered else {}

    rows = []
    for q in qs:
        o, j = runs[q["qid"]], judged.get(q["qid"]) or {}
        if o["type"] == "거절":
            verdict, ok, why = "거절", True, f"챗봇 거절 ({o.get('거절사유')})"
        else:
            verdict = j.get("verdict", "미판정")
            ok, why = verdict == "admit", j.get("reason", "")
        rows.append({"key": q["qid"], "질문": q["질문"], "챗봇 결과": o["type"], "판정": verdict,
                     "올바른 거절": ok, "이유": why, "검색수": o["검색수"], "답변": o["answer"]})
    d = pd.DataFrame(rows)
    d.to_csv(OUT_DIR / "refusal_eval.csv", index=False, encoding="utf-8-sig")

    n = len(d)
    acc, explicit = d["올바른 거절"].mean(), (d["챗봇 결과"] == "거절").mean()
    admit, fab = (d["판정"] == "admit").sum(), (d["판정"] == "fabricated").sum()
    md = lambda df: ("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"   # noqa: E731
                     + "".join("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in r) + " |\n"
                               for r in df.itertuples(index=False)))
    (OUT_DIR / "refusal_eval.md").write_text(
        f"# 거절 정확도 (근거 없는 질문 {n}개) — {tag}\n\n- 실행: {time.strftime('%Y-%m-%d %H:%M')} / 답변 모델 {rc.CHAT_MODEL}\n"
        f"- 질문: eval/eval_set/eval_questions2.json의 '답변불가' 유형 (자료에 없는 세부 + 가상의 회사·제품)\n\n"
        f"| 지표 | 결과 |\n|---|---|\n| **거절 정확도** (거절 + 확인 불가 명시) | **{acc:.1%}** ({int(d['올바른 거절'].sum())}/{n}) |\n"
        f"| 명시적 거절률 (챗봇이 거절로 끝냄) | {explicit:.1%} |\n| 답했지만 확인 불가를 밝힘 | {admit}개 |\n"
        f"| 지어냄 (잘못된 응답) | {fab}개 |\n\n## 질문별\n\n"
        + md(d[["key", "질문", "챗봇 결과", "판정", "이유"]]), encoding="utf-8")
    print(f"\n[거절 정확도] {acc:.1%} ({int(d['올바른 거절'].sum())}/{n}) | 명시적 거절 {explicit:.1%} | "
          f"확인 불가 명시 {admit}개 | 지어냄 {fab}개")
    print(f"저장: eval/results/{tag}/refusal_eval.md, refusal_eval.csv ({time.time() - t0:.0f}초)")
    return d


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else BASE_TAG)
