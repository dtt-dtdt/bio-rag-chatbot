"""
특허 노이즈 점검 — 바이오의약품·바이오헬스케어와 무관한 특허(반도체·물류·일반 식품 등)를 찾아 표시

[왜]  실사용 점검에서 '원격의료 규제 현황'에 열분해 장치·물류 창고 특허가 나왔어요.
      키프리스 검색으로 모은 특허에 바이오와 무관한 특허가 섞여 있었기 때문이에요.

[방법]  뉴스 노이즈(indexer/noise_news.py)와 같은 '규칙 + LLM' 방식. 특허는 분류 코드(IPC/CPC)가 있어서 규칙이 더 정확해요.
  ① 규칙 구간
     - 바이오 확실 : 의약(A61K, 화장품 A61K 8 제외)·치료 효능(A61P)·의료기기(A61B·A61M 등)·
                     단백질(C07K)·세포·유전자(C12N)·진단(C12Q)·의료정보(G16H·G16B) 코드가 하나라도 있음
     - 화장품 / 식품 / 그 밖 : 위 코드가 없음 → 전부 LLM 판정
  ② LLM 판정 : 화장품·식품·그 밖 전부 + '바이오 확실' 표본 300건(규칙이 맞는지 확인용)
  ③ 최종 라벨 = LLM 판정이 있으면 LLM, 없으면 규칙
  apply_to_db()  ④ (결과 확인 후 따로 실행) 본 DB 특허에 '바이오관련'(True/False), '노이즈판정' 표시
                    ※ 지우지 않고 표시만. 원본 CSV·임베딩은 그대로. 검색에서만 제외(rag/retriever.py)

[실행 - Jupyter]
  from indexer import noise_patent; noise_patent.run()      # LLM 판정 약 2,200건 (약 20~40분, 1달러 안팎). 멈춰도 이어서 진행
  # noise/patent_noise_report.md 확인 후
  noise_patent.apply_to_db()
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config
from indexer import noise_news as nn   # GPT 호출(chat_json)·DB 연결·메타데이터 병합을 그대로 재사용

# ============================== 설정 ==============================
BASE_DIR = nn.BASE_DIR
OUT_DIR = config.NOISE_DIR
CACHE = OUT_DIR / "judge_patent_bio.jsonl"
WORKERS = 8
RULE_CHECK_N = 300          # '바이오 확실' 구간에서 LLM으로 확인할 표본 수
JUDGE_CHARS = 700           # 판정에 쓸 요약 앞부분 길이
SEED = 42

CORE_RE = re.compile(r"^(A61[PBMFNLHJGC]|C07K|C12N|C12Q|G16[HB])$")     # 이 중 하나라도 있으면 바이오 확실
FOOD = {"A23L", "A23V", "A23K", "A23B", "A23P", "A23F", "A23C", "A23G", "A23D", "A23J", "A21D", "C12G"}
# ==================================================================

JUDGE_PROMPT = """너는 특허를 분류하는 사람이다. 이 특허가 '바이오의약품·바이오헬스케어' 챗봇의 검색 대상에 맞는지 판단한다.
[발명의 명칭] {title}
[분류 코드] {codes}
[요약 앞부분] {text}

true (관련 있음):
- 의약품·바이오의약품(항체·단백질·세포·유전자·백신·미생물 치료제), 약물 전달, 의약 제형
- 질병의 예방·치료·진단, 의료기기, 의료 데이터·디지털헬스·원격진료
- 특정 질환·건강 기능(예: 혈당, 면역, 인지기능, 피부질환)의 개선 효능을 주장하는 조성물 (건강기능식품 포함)
- 의약·진단용 바이오 소재·공정 (세포 배양, 단백질 생산 등)
false (관련 없음):
- 일반 식품·음료·조리법, 미용 목적만의 화장품(보습·향·메이크업)
- 농업·사료·환경·일반 화학·소재·반도체·IT·물류·금융 등 의료와 무관한 기술
JSON으로 출력: {{"bio_related": true/false, "reason": "한 문장"}}"""


def _codes(meta: dict) -> list[tuple[str, str]]:
    s = f"{meta.get('IPC') or ''}|{meta.get('CPC') or ''}"
    return re.findall(r"([A-H]\d{2}[A-Z])\s*(\d+)", s)


def rule_tier(meta: dict) -> str:
    """'바이오 확실' | '화장품' | '식품' | '그 밖'"""
    cs = _codes(meta)
    subs = {a for a, _ in cs}
    if any(a == "A61K" and g != "8" for a, g in cs) or any(CORE_RE.match(a) for a in subs):
        return "바이오 확실"
    if any((a == "A61K" and g == "8") or a == "A61Q" for a, g in cs):
        return "화장품"
    if subs & FOOD:
        return "식품"
    return "그 밖"


def _all_patents(col):
    ids, docs, metas, off = [], [], [], 0
    while True:
        r = col.get(limit=2000, offset=off, include=["documents", "metadatas"])
        if not r["ids"]:
            break
        ids += r["ids"]; docs += r["documents"]; metas += r["metadatas"]
        off += len(r["ids"])
    return ids, docs, metas


def _task(doc: str, meta: dict) -> tuple[str, str, str]:
    body = doc.split("\n핵심키워드:")[0]
    title = meta.get("발명의명칭") or body.split("\n")[0]
    text = re.sub(r"\s+", " ", body[len(title):] if body.startswith(title) else body).strip()[:JUDGE_CHARS]
    codes = ", ".join(sorted({a for a, _ in _codes(meta)}))[:200]
    return title, codes, text


def judge_all(tasks: dict) -> dict:
    """tasks = {출원번호: (명칭, 코드, 요약)} → {출원번호: {"bio_related", "reason"}}  (하나씩 저장, 이어서 진행)"""
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec["out"]
    todo = [k for k in tasks if k not in done]
    print(f"\n[LLM 판정] 전체 {len(tasks):,} / 이미 판정 {len(tasks) - len(todo):,} / 새로 판정 {len(todo):,}")
    if not todo:
        return done

    def work(k):
        title, codes, text = tasks[k]
        return k, nn.chat_json(JUDGE_PROMPT.format(title=title, codes=codes, text=text))

    t0 = time.time()
    with CACHE.open("a", encoding="utf-8") as f:
        print("  연결 확인 중 (첫 판정 1개)…")
        k, out = work(todo[0])
        f.write(json.dumps({"key": k, "out": out}, ensure_ascii=False) + "\n")
        done[k] = out
        print(f"  ✅ 연결 정상 ({nn.JUDGE_MODEL})")
        with ThreadPoolExecutor(WORKERS) as ex:
            for n, (k, out) in enumerate(ex.map(work, todo[1:]), 2):
                f.write(json.dumps({"key": k, "out": out}, ensure_ascii=False) + "\n")
                f.flush()
                done[k] = out
                if n % 200 == 0 or n == len(todo):
                    print(f"  진행 {n:,}/{len(todo):,} ({time.time() - t0:.0f}초)")
    return done


def run():
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)
    col = nn.get_clients()["main"].get_collection("patents")
    ids, docs, metas = _all_patents(col)
    df = pd.DataFrame({"출원번호": ids, "발명의명칭": [m.get("발명의명칭", "") for m in metas],
                       "분야": [m.get("분야", "") for m in metas], "구간": [rule_tier(m) for m in metas]})
    print(f"특허 {len(df):,}건 →", df["구간"].value_counts().to_dict())

    # ② LLM 판정: 확실 구간 밖 전부 + 확실 구간 표본
    core = df[df["구간"] == "바이오 확실"]
    check_ids = list(df.loc[df["구간"] != "바이오 확실", "출원번호"]) + \
        list(core.sample(min(RULE_CHECK_N, len(core)), random_state=SEED)["출원번호"])
    by_id = {i: (d, m) for i, d, m in zip(ids, docs, metas)}
    judged = judge_all({i: _task(*by_id[i]) for i in check_ids})

    # ③ 최종 라벨
    df["LLM판정"] = df["출원번호"].map(lambda i: judged.get(i, {}).get("bio_related"))
    df["LLM근거"] = df["출원번호"].map(lambda i: judged.get(i, {}).get("reason", ""))
    df["바이오관련"] = df["LLM판정"].where(df["LLM판정"].notna(), df["구간"] == "바이오 확실").astype(bool)
    df["판정출처"] = np.where(df["LLM판정"].notna(), "LLM", "규칙")
    df.to_csv(OUT_DIR / "patent_labels.csv", index=False, encoding="utf-8-sig")

    # ---------- 보고서 ----------
    s = df[(df["구간"] == "바이오 확실") & df["LLM판정"].notna()]
    rule_ok = (s["LLM판정"] == True).mean() if len(s) else float("nan")   # noqa: E712
    tier = (df.groupby("구간")["바이오관련"].agg(전체="size", 바이오="sum")
            .assign(노이즈=lambda t: t["전체"] - t["바이오"]).reindex(["바이오 확실", "화장품", "식품", "그 밖"]).dropna())
    field = df.groupby("분야")["바이오관련"].agg(전체="size", 바이오="sum").assign(노이즈=lambda t: t["전체"] - t["바이오"])
    noise = df[~df["바이오관련"]]
    kept = df[df["바이오관련"] & (df["구간"] != "바이오 확실")]

    def md(t, index=True):
        t = t.reset_index() if index else t
        cols = list(t.columns)
        return ("| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n"
                + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in t.itertuples(index=False)))

    pick = lambda d, n: d.sample(min(n, len(d)), random_state=SEED)[["구간", "발명의명칭", "LLM근거"]]  # noqa: E731
    report = (f"# 특허 노이즈 점검 결과\n\n- 실행: {time.strftime('%Y-%m-%d %H:%M')} / 판정 모델: {nn.JUDGE_MODEL}\n"
              f"- 특허 {len(df):,}건 중 노이즈(바이오 무관) **{len(noise):,}건 ({len(noise) / len(df):.1%})**\n"
              f"- 규칙 '바이오 확실' 표본 {len(s)}건 중 LLM도 바이오로 판단: {rule_ok:.1%}\n\n"
              f"## 구간별\n\n{md(tier.astype(int))}\n## 분야별\n\n{md(field.astype(int))}\n"
              f"## 노이즈로 판정된 특허 (표본 30건)\n\n{md(pick(noise, 30), False)}\n"
              f"## 코드 규칙 밖인데 바이오로 남긴 특허 (표본 30건)\n\n{md(pick(kept, 30), False)}")
    (OUT_DIR / "patent_noise_report.md").write_text(report, encoding="utf-8")
    print(f"\n노이즈 {len(noise):,}건 ({len(noise) / len(df):.1%}) / 규칙 확인 일치율 {rule_ok:.1%}")
    print(tier.astype(int).to_string())
    print(f"\n저장: noise/patent_labels.csv, noise/patent_noise_report.md ({time.time() - t0:.0f}초)")
    print("→ 보고서를 확인한 뒤 patent_noise.apply_to_db()를 실행하면 본 DB에 표시돼요.")
    return df


def apply_to_db():
    labels = pd.read_csv(OUT_DIR / "patent_labels.csv", dtype=str, keep_default_na=False)
    col = nn.get_clients(need_openai=False)["main"].get_collection("patents")
    new = {r.출원번호: {"바이오관련": r.바이오관련 == "True", "노이즈판정": r.판정출처} for r in labels.itertuples()}
    nn._merge_update(col, list(new), new)
    n_noise = sum(not v["바이오관련"] for v in new.values())
    print(f"✅ 특허 {len(new):,}건에 표시 완료 (노이즈 {n_noise:,}건)")
    print("   rag/retriever.py를 다시 불러오면 노이즈 필터(NOISE_FILTER)가 특허에도 적용돼요.")


def tag_new_patents(col, new_ids: list[str]):
    """나중에 새 특허를 넣었을 때 쓰는 함수: 규칙 + (확실 구간 밖만) LLM 판정 후 표시"""
    if not new_ids:
        return
    r = col.get(ids=list(new_ids), include=["documents", "metadatas"])
    new = {}
    for i, d, m in zip(r["ids"], r["documents"], r["metadatas"]):
        if rule_tier(m) == "바이오 확실":
            new[i] = {"바이오관련": True, "노이즈판정": "규칙"}
        else:
            out = nn.chat_json(JUDGE_PROMPT.format(**dict(zip(("title", "codes", "text"), _task(d, m)))))
            new[i] = {"바이오관련": bool(out.get("bio_related", True)), "노이즈판정": "LLM"}
    nn._merge_update(col, list(new), new)
    print(f"  특허 노이즈 표시: 새 특허 {len(new):,}건 중 노이즈 {sum(not v['바이오관련'] for v in new.values()):,}건")


if __name__ == "__main__":   # background.start("indexer.noise_patent")로 백그라운드 실행할 때
    run()
