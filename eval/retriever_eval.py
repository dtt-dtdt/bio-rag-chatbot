"""
리트리버 비교 평가 v2 — 원래 질문 + '표현을 바꾼 질문' 두 세트로, 검색 방식·가중치·노이즈 필터 비교

[왜 두 세트인가]
  평가 질문은 GPT가 문서를 보고 만들어서 원문 단어를 많이 따라 씀 → 단어 일치 검색(BM25)에 유리.
  실제 사용자는 다른 표현으로 물으므로(위탁생산↔CDMO), 고유명사·연도는 그대로 두고 나머지 단어를 바꾼
  질문 세트로도 측정해서 '어느 쪽 질문에도 강한' 설정을 고름.

[설정]  벡터(현재, 필터 없음) / 벡터+노이즈 / BM25+노이즈 / 하이브리드 1:1, 2:1, 1:2 (+노이즈)
[지표]  Hit@3 (엄격) · Hit@3 (정답급 v3) · MRR@3 · Precision@3 (상위 3개 중 쓸모 있는 문서 비율)
        · 노이즈@3 (뉴스 상위 3개 중 노이즈 기사 비율)
[재사용] eval_questions.json, question_vecs.json, judge_relevance_v3.jsonl, judge_quality_v3.jsonl
[새 비용] 질문 바꾸기 250회(gpt-6-luna) + 임베딩 250개 + 새로 검색된 문서 판정 (합쳐서 수십~수백 원)
[필요]  rag/retriever.py, eval/judge.py(v3), 원본 뉴스 CSV, news_noise.apply_to_db() 완료
[결과]  eval/eval_retriever.md (요약표 + 전체표), eval_retriever.csv
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

import config

BASE_DIR = config.ROOT

from eval import judge as ej    # 판정 함수·캐시 재사용
from rag import retriever as rt     # 하이브리드 리트리버

if not (getattr(ej, "VERSION", None) == "v3" and hasattr(ej, "judge_relevance")):
    raise RuntimeError("project3 폴더의 eval/judge.py가 예전 버전이에요. 최신(v3) 파일로 바꾸고 커널을 재시작한 뒤 "
                       "다시 실행해 주세요. (파일 안에 VERSION = \"v3\" 줄이 있어야 해요)")
if not hasattr(rt, "BM25Index"):
    raise RuntimeError("project3 폴더의 rag/retriever.py가 하이브리드 리트리버 파일이 아니에요. 최신 파일로 바꿔 주세요.")

OUT_DIR = config.EVAL_RESULTS_DIR
K = 3
SETTINGS = [  # (표시 이름, mode, 노이즈 필터, 벡터 가중치, BM25 가중치)
    ("벡터 (현재)", "vector", False, 1.0, 1.0),
    ("벡터 + 노이즈", "vector", True, 1.0, 1.0),
    ("BM25 + 노이즈", "bm25", True, 1.0, 1.0),
    ("하이브리드 1:1 + 노이즈", "hybrid", True, 1.0, 1.0),
    ("하이브리드 벡터2:BM25 1 + 노이즈", "hybrid", True, 2.0, 1.0),
    ("하이브리드 벡터1:BM25 2 + 노이즈", "hybrid", True, 1.0, 2.0),
]

PARA_PROMPT = """다음 질문을 같은 뜻의 다른 표현으로 다시 써라.
[질문] {q}

규칙:
1. 고유명사(회사·인물·기관·제품·행사·질환·약물 이름)와 연도·숫자는 그대로 둔다.
2. 그 밖의 핵심 단어는 다른 말로 바꾼다. (동의어, 풀어 쓰기, 약어↔전체 이름, 한국어↔영어)
   예) 위탁생산 → CDMO, 비대면 진료 → 원격의료, 출시하다 → 내놓다, 특허가 있나요 → 관련 기술 권리 있어?
3. 실제 사용자가 챗봇에 묻듯 짧고 자연스럽게 쓴다. (원래 질문보다 길지 않게)
4. 묻는 내용(답)이 바뀌면 안 된다.
JSON으로 출력: {{"question": "다시 쓴 질문"}}"""


def make_paraphrases(qs: list[dict]) -> tuple[list[dict], dict]:
    """표현을 바꾼 질문 세트와 그 벡터 (저장해 두고 재사용)"""
    q_path, v_path = config.EVAL_SET_DIR / "eval_questions_para.json", OUT_DIR / "question_vecs_para.json"
    if q_path.exists() and v_path.exists():
        para = json.loads(q_path.read_text(encoding="utf-8"))
        vecs = json.loads(v_path.read_text())
        if len(para) == len(qs) and all(p["qid"] in vecs for p in para):
            print(f"표현 바꾼 질문 불러옴: {len(para)}개")
            return para, vecs
    out = ej.run_cached("paraphrase_v1", {q["qid"]: q["질문"] for q in qs},
                        lambda q: ej.chat_json(PARA_PROMPT.format(q=q)), "질문 표현 바꾸기")
    para = []
    for q in qs:
        new_q = str(out.get(q["qid"], {}).get("question", "")).strip() or q["질문"]   # 실패하면 원래 질문
        para.append({**q, "질문": new_q, "원래질문": q["질문"]})
    print("표현 바꾼 질문 임베딩 중…")
    oa = rt._get_clients()["openai"]
    vecs = {}
    for i in range(0, len(para), 100):
        part = para[i:i + 100]
        r = oa.embeddings.create(model=rt.EMBED_MODEL, input=[p["질문"] for p in part])
        vecs.update({p["qid"]: d.embedding for p, d in zip(part, r.data)})
    q_path.write_text(json.dumps(para, ensure_ascii=False, indent=1), encoding="utf-8")
    v_path.write_text(json.dumps(vecs))
    return para, vecs


def word_overlap(a: str, b: str) -> float:
    ta, tb = set(rt.tokenize(a)), set(rt.tokenize(b))
    return len(ta & tb) / max(len(ta | tb), 1)


def run():
    t0 = time.time()
    qs = json.loads((config.EVAL_SET_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    vecs = json.loads((OUT_DIR / "question_vecs.json").read_text())
    para, para_vecs = make_paraphrases(qs)
    overlap = sum(word_overlap(p["원래질문"], p["질문"]) for p in para) / len(para)
    print(f"원래 질문과 바꾼 질문의 단어 겹침(평균): {overlap:.0%}  예) {para[0]['원래질문'][:40]} → {para[0]['질문'][:40]}")

    raw = pd.read_csv(ej.RAW_NEWS_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    art = {}
    for u, t, d, b in zip(raw["원문URL"], raw["제목"], raw["발행일"], raw["본문전문"]):
        a = hashlib.sha1(f"{u}|{t}|{d}".encode("utf-8")).hexdigest()[:12]
        art[a] = (t, re.sub(r"\s+", " ", ej.BOILERPLATE_RE.sub(" ", b)).strip()[:ej.JUDGE_CHARS])

    print("BM25 색인 준비…")
    rt.get_index("patents"); rt.get_index("news")
    if not any("바이오관련" in m for m in rt.get_index("news").metas[:50]):
        print("⚠️ 뉴스에 '바이오관련' 표시가 없어요. news_noise.apply_to_db()를 먼저 실행해 주세요.")

    sets = [("원래 질문", "", qs, vecs), ("표현 바꾼 질문", "p|", para, para_vecs)]
    tops, tasks, pat_text = {}, {}, {}
    for set_name, prefix, qlist, qv in sets:
        for label, mode, nf, wv, wb in SETTINGS:
            print(f"검색: [{set_name}] {label}")
            rt.W_VEC, rt.W_BM25 = wv, wb
            for q in qlist:
                kind = "news" if q["종류"] == "뉴스" else "patent"
                if kind == "patent" and nf and mode == "vector":
                    continue   # 특허는 노이즈 필터와 무관 → '벡터 (현재)'와 같음
                res = rt.retrieve(q["질문"], target=kind, k_news=K, k_patent=K, mode=mode, noise_filter=nf,
                                  q_vec=qv[q["qid"]])
                items = []
                for x in res:
                    if kind == "news":
                        doc, gold = x["메타"]["기사ID"], x["메타"]["기사ID"] == q["정답"]
                        noise = x["메타"].get("바이오관련") is False
                    else:
                        doc, gold, noise = x["id"], x["id"] == q["정답"] or x["제목"] == q["정답제목"], False
                        pat_text[doc] = (x["제목"], x["본문"].split("\n핵심키워드:")[0])
                    items.append((doc, gold, noise))
                    if not gold:
                        title, text = art.get(doc, ("", "")) if kind == "news" else pat_text[doc]
                        tasks[f"{prefix}{q['qid']}|{doc}"] = (q["질문"], kind, title, text)
                tops[(set_name, label, q["qid"])] = items
    rt.W_VEC, rt.W_BM25 = 1.0, 1.0

    rel = ej.run_cached(f"judge_relevance_{ej.VERSION}", tasks, ej.judge_relevance, "정답급 판정 (새 문서만)")

    good = set()
    qual_path = OUT_DIR / f"judge_quality_{ej.VERSION}.jsonl"
    if qual_path.exists():
        for line in qual_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["out"].get("specific") and r["out"].get("bio_related"):
                    good.add(r["key"])

    # 질문별 상세 (분석용): 어떤 설정에서 맞고 틀렸는지, 상위 3개가 무엇이었는지
    def title_of(kind_name, d):
        return art.get(d, ("",))[0] if kind_name == "뉴스" else pat_text.get(d, ("",))[0]
    details = []
    for set_name, prefix, qlist, _ in sets:
        for label, mode, nf, _, _ in SETTINGS:
            for q in qlist:
                items = tops.get((set_name, label, q["qid"]))
                if items is None:
                    continue
                oks = [bool(g or rel.get(f"{prefix}{q['qid']}|{d}", {}).get("final")) for d, g, _ in items]
                details.append({
                    "질문세트": set_name, "설정": label, "qid": q["qid"], "대상": q["종류"],
                    "좋은질문": q["qid"] in good, "질문": q["질문"], "정답": q["정답제목"],
                    "정답순위": next((i for i, (_, g, _) in enumerate(items, 1) if g), None),
                    "맞춤(정답급)": any(oks), "쓸모있는수": sum(oks), "노이즈수": sum(n for _, _, n in items),
                    **{f"{i}위": f"{'○' if ok else '×'}{'(노이즈)' if n else ''} {title_of(q['종류'], d)[:40]}"
                       for i, ((d, _, n), ok) in enumerate(zip(items, oks), 1)}})
    pd.DataFrame(details).to_csv(OUT_DIR / "eval_retriever_details.csv", index=False, encoding="utf-8-sig")

    rows = []
    for set_name, prefix, qlist, _ in sets:
        for label, mode, nf, _, _ in SETTINGS:
            for kind_name in ("뉴스", "특허"):
                if kind_name == "특허" and nf and mode == "vector":
                    continue
                base = [q for q in qlist if q["종류"] == kind_name]
                for subset_name, subset in (("전체 질문", base), ("좋은 질문만", [q for q in base if q["qid"] in good])):
                    if not subset:
                        continue
                    strict = hit = mrr = prec = noise = 0
                    for q in subset:
                        items = tops[(set_name, label, q["qid"])]
                        ok = [i for i, (d, g, _) in enumerate(items, 1)
                              if g or rel.get(f"{prefix}{q['qid']}|{d}", {}).get("final")]
                        strict += any(g for _, g, _ in items)
                        hit += bool(ok)
                        mrr += 1 / ok[0] if ok else 0
                        prec += len(ok) / K
                        noise += sum(n for _, _, n in items) / K
                    n = len(subset)
                    rows.append({"질문세트": set_name, "설정": label, "대상": kind_name, "질문": subset_name, "질문수": n,
                                 "Hit@3 (엄격)": strict / n, "Hit@3 (정답급)": hit / n, "MRR@3": mrr / n,
                                 "Precision@3": prec / n, "노이즈@3": (noise / n) if kind_name == "뉴스" else None})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "eval_retriever.csv", index=False, encoding="utf-8-sig")

    # 요약표: 좋은 질문만, 두 질문 세트를 나란히
    g = res[res["질문"] == "좋은 질문만"]
    summary = []
    for label, *_ in SETTINGS:
        row = {"설정": label}
        for kind_name in ("뉴스", "특허"):
            for set_name, short in (("원래 질문", "원래"), ("표현 바꾼 질문", "바꾼")):
                s = g[(g["설정"] == label) & (g["대상"] == kind_name) & (g["질문세트"] == set_name)]
                if len(s):
                    row[f"{kind_name} Hit@3 ({short})"] = s["Hit@3 (정답급)"].iloc[0]
                    row[f"{kind_name} P@3 ({short})"] = s["Precision@3"].iloc[0]
                    if kind_name == "뉴스":
                        row[f"뉴스 노이즈@3 ({short})"] = s["노이즈@3"].iloc[0]
        summary.append(row)
    sm = pd.DataFrame(summary)

    def fmt(df):
        df = df.copy().astype(object)
        for c in df.columns:
            if any(t in c for t in ("Hit@", "P@3", "Precision", "노이즈")):
                df[c] = df[c].map(lambda v: "" if v is None or pd.isna(v) else f"{v:.1%}")
            elif c.startswith("MRR"):
                df[c] = df[c].map(lambda v: f"{v:.3f}")
        return df

    def md(df):
        cols = list(df.columns)
        return ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
                + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False)))

    note = (f"- 단어 나누기: {rt._tok['name']} / 결합: RRF(k={rt.RRF_K}) / 후보 각 {rt.CANDIDATES}개 / 판정: {ej.JUDGE_MODEL} v3\n"
            f"- 표현 바꾼 질문: 고유명사·연도 유지, 나머지 단어 변경. 원래 질문과 단어 겹침 평균 {overlap:.0%}\n"
            f"- 요약표는 '좋은 질문만'(주체 분명 + 정답이 바이오 문서) 기준, Hit@3는 정답급 v3 기준")
    (OUT_DIR / "eval_retriever.md").write_text(
        f"# 리트리버 비교 v2\n\n- 측정 시각: {time.strftime('%Y-%m-%d %H:%M')}\n{note}\n\n"
        f"## 요약 (좋은 질문만)\n\n{md(fmt(sm))}\n## 전체 결과\n\n{md(fmt(res))}", encoding="utf-8")
    print("\n" + "=" * 100 + f"\n{note}\n\n[요약: 좋은 질문만]")
    print(fmt(sm).to_string(index=False))
    print("=" * 100)
    print(f"저장: eval/eval_retriever.md (요약 + 전체), eval_retriever.csv, eval_retriever_details.csv(질문별) "
          f"({(time.time() - t0) / 60:.0f}분)")
    return res


if __name__ == "__main__":
    run()
