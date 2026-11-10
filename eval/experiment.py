"""
실험 관리 — 측정 → 실패 사례 확인 → 원인 계층 특정 → 단일 변수 변경 → 재측정

[규칙]
  - 한 번에 변수 하나만 바꿈. 기준(v5) 설정과 다른 값이 2개 이상이면 실행하지 않음
  - 실험마다 태그를 붙여 eval/results/<태그>/ 에 저장 (설정 config.json, 답변·채점, 비교 결과)
  - eval/results/experiments.csv·md 에 기록 → 리포트의 근거 자료
[순서 - Jupyter, project3 폴더에서]
  from eval import experiment as ex
  ex.status()                                   # 기준 설정, 바꿀 수 있는 변수, 지금까지의 실험
  ex.analyze("v5")                              # ② 실패 사례 → ③ 원인 계층 (eval/results/v5/failure_layers.md)
  ex.run_experiment("e01_k10", K_BOTH=10,       # ④ 변수 하나 변경 → ⑤ 재측정 → 기준과 비교 → 기록
                    note="검색 계층(순위 낮음) → 자료 수 8 → 10")
  ex.decide("e01_k10", "채택")                  # 결과를 보고 채택/기각 기록 (채택해도 코드는 직접 바꿔야 함)
[바꿀 수 있는 변수]  VARIABLES 참고. 청킹·임베딩은 DB를 다시 만들어야 해서 여기서 바꾸지 않음
  → 별도 DB로 indexer를 다시 돌린 뒤 config.CHROMA_DIR을 바꿔 측정하고, record_manual()로 기록
[비용]  실험 1번 = 질문 257개 답변 + 채점 (약 20~30분, 1~2달러 안팎) + 거절 정확도 15개
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd

import config
from rag import chain as rc
from rag import retriever as rt

RESULTS_ROOT = config.EVAL_RESULTS_DIR
REGISTRY = RESULTS_ROOT / "experiments.csv"
BASE_TAG = "v5"
BOOT_N, SEED = 2000, 42

# 변수 이름 → (계층, 설명, 바꾸는 곳)
VARIABLES = {
    "W_VEC": ("검색", "하이브리드 벡터 가중치", (rt, "W_VEC")),
    "W_BM25": ("검색", "하이브리드 BM25 가중치", (rt, "W_BM25")),
    "CANDIDATES": ("검색", "각 검색의 후보 수", (rt, "CANDIDATES")),
    "RRF_K": ("검색", "RRF 상수 k", (rt, "RRF_K")),
    "K_BOTH": ("검색", "답변에 쓰는 자료 수 (특허·뉴스 같이) = top_k", None),
    "K_PATENT": ("검색", "답변에 쓰는 특허 수", (rc, "K_PATENT")),
    "K_NEWS": ("검색", "답변에 쓰는 뉴스 수", (rc, "K_NEWS")),
    "COMPANY_FILTER": ("검색", "특허 회사명 필터", (rt, "COMPANY_FILTER")),
    "NOISE_FILTER": ("전처리", "노이즈(바이오 무관) 문서 제외", (rt, "NOISE_FILTER")),
    "ANALYZE": ("검색", "질문 분석(검색어·기간·회사명) 사용", (rc, "ANALYZE")),
    "BODY_CHARS": ("생성", "자료 1건당 LLM에 넘길 본문 길이", (rc, "BODY_CHARS")),
    "SYSTEM_PROMPT_FILE": ("생성", "답변 지시문 파일 (eval/prompts/*.txt, {instruction} 포함)", None),
}
BASELINE = {"W_VEC": 1.0, "W_BM25": 2.0, "CANDIDATES": 50, "RRF_K": 60, "K_BOTH": 8, "K_PATENT": 8, "K_NEWS": 8,
            "COMPANY_FILTER": True, "NOISE_FILTER": True, "ANALYZE": True, "BODY_CHARS": 1200, "SYSTEM_PROMPT_FILE": None}
NOT_HERE = {"CHUNK_SIZE": "청킹", "CHUNK_OVERLAP": "청킹", "EMBED_MODEL": "임베딩"}
COLUMNS = ["태그", "날짜", "기준", "계층", "변수", "기준값", "변경값", "가설", "정답", "근거 있음", "정답 문서 검색됨",
           "거절 적절", "실패 수", "거절 정확도", "고쳐진 질문", "새로 실패", "정답 차이 (95% CI)", "근거 차이 (95% CI)",
           "판정", "결정"]


# ---------------------------- 설정 ----------------------------
def current_settings() -> dict:
    s = {k: getattr(mod, attr) for k, (_, _, ref) in VARIABLES.items() if ref for mod, attr in [ref]}
    s["K_BOTH"] = rc.K_PATENT if rc.K_PATENT == rc.K_NEWS else None
    s["SYSTEM_PROMPT_FILE"] = _ACTIVE.get("prompt_file")
    return s


_ACTIVE: dict = {}


def _apply(name: str, value):
    """변수 하나를 바꾸고, 되돌리는 함수를 돌려줌"""
    if name == "K_BOTH":
        old = (rc.K_PATENT, rc.K_NEWS)
        rc.K_PATENT = rc.K_NEWS = int(value)
        return lambda: setattr(rc, "K_PATENT", old[0]) or setattr(rc, "K_NEWS", old[1])
    if name == "SYSTEM_PROMPT_FILE":
        from langchain_core.prompts import ChatPromptTemplate
        text = (config.ROOT / value).read_text(encoding="utf-8")
        if "{instruction}" not in text:
            raise ValueError("지시문 파일에 {instruction} 자리가 있어야 해요 (질문별 추가 지시가 들어가는 곳)")
        old = (rc.SYSTEM_PROMPT, rc.PROMPT)
        rc.SYSTEM_PROMPT = text
        rc.PROMPT = ChatPromptTemplate.from_messages([("system", text), ("human", rc.HUMAN_PROMPT)])
        _ACTIVE["prompt_file"] = value

        def undo():
            rc.SYSTEM_PROMPT, rc.PROMPT = old
            _ACTIVE.pop("prompt_file", None)
        return undo
    mod, attr = VARIABLES[name][2]
    old = getattr(mod, attr)
    setattr(mod, attr, type(old)(value) if not isinstance(old, bool) else bool(value))
    return lambda: setattr(mod, attr, old)


def _check_one_change(change: dict):
    if len(change) != 1:
        raise ValueError(f"한 번에 변수 하나만 바꿀 수 있어요. 지금 {len(change)}개: {list(change)}")
    (name, value), = change.items()
    if name in NOT_HERE:
        raise ValueError(f"{name}은(는) {NOT_HERE[name]} 변수라 DB를 다시 만들어야 해요. 별도 DB로 indexer를 돌린 뒤 "
                         "record_manual()로 기록해 주세요.")
    if name not in VARIABLES:
        raise ValueError(f"모르는 변수예요: {name}. 바꿀 수 있는 변수: {', '.join(VARIABLES)}")
    if value == BASELINE[name]:
        raise ValueError(f"{name}={value!r}는 기준값과 같아요. 바꿀 값을 주세요.")
    diff = {k: v for k, v in current_settings().items()
            if k in BASELINE and v != BASELINE[k] and not (k == "K_BOTH" and v is None)}
    if diff:
        raise RuntimeError(f"지금 설정이 기준(v5)과 이미 달라요: {diff}. 커널을 재시작해 주세요 (변수 2개가 섞이지 않게).")
    return name, value


# ---------------------------- 지표 ----------------------------
def _details(tag: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_ROOT / tag / f"answer_eval_details_{tag}.csv", dtype=str, keep_default_na=False)


def metrics(tag: str) -> dict:
    d = _details(tag)
    ev = d[d["그룹"].str.startswith("평가")]
    base = d[~d["그룹"].str.startswith("설계")]
    out = {"정답": (ev["정답성"] == "correct").mean(), "근거 있음": (ev["근거성"] == "yes").mean(),
           "정답 문서 검색됨": (ev["정답검색됨"] == "True").mean(), "거절 적절": (base["거절적절"] == "True").mean(),
           "실패 수": f"{int((base['실패'] == 'True').sum())}/{len(base)}"}
    ref = RESULTS_ROOT / tag / "refusal_eval.csv"
    if ref.exists():
        r = pd.read_csv(ref, dtype=str, keep_default_na=False)
        out["거절 정확도"] = (r["올바른 거절"] == "True").mean()
    return out


def _boot_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """같은 질문끼리 짝지어 (b - a) 평균 차이의 95% 신뢰구간"""
    rng = np.random.default_rng(SEED)
    diff = b - a
    idx = rng.integers(0, len(diff), size=(BOOT_N, len(diff)))
    boots = diff[idx].mean(axis=1)
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def compare(tag: str, base: str = BASE_TAG) -> dict:
    """기준과 질문별 비교 + 신뢰구간 → eval/results/<tag>/compare_vs_<base>.md"""
    a, b = _details(base), _details(tag)
    m = a.merge(b, on="key", suffixes=("_a", "_b"))
    m = m[~m["그룹_b"].str.startswith("설계")]
    fixed = m[(m["실패_a"] == "True") & (m["실패_b"] != "True")]
    broken = m[(m["실패_a"] != "True") & (m["실패_b"] == "True")]
    ev = m[m["그룹_b"].str.startswith("평가")]
    ci = {}
    for name, col, good in (("정답", "정답성", "correct"), ("근거", "근거성", "yes")):
        ci[name] = _boot_ci((ev[f"{col}_a"] == good).to_numpy(float), (ev[f"{col}_b"] == good).to_numpy(float))
    fail_ci = _boot_ci((m["실패_a"] == "True").to_numpy(float), (m["실패_b"] == "True").to_numpy(float))

    def fmt(c):
        mean, lo, hi = c
        sure = "확실" if lo > 0 or hi < 0 else "비슷 (우연과 구분 안 됨)"
        return f"{mean * 100:+.1f}%p [{lo * 100:+.1f}, {hi * 100:+.1f}] {sure}"
    verdict = ("개선" if ci["정답"][1] > 0 or ci["근거"][1] > 0 else "악화" if ci["정답"][2] < 0 or ci["근거"][2] < 0
               else "차이 없음")
    ma, mb = metrics(base), metrics(tag)
    rows = [{"지표": k, base: (f"{ma[k]:.1%}" if isinstance(ma.get(k), float) else ma.get(k, "-")),
             tag: (f"{mb[k]:.1%}" if isinstance(mb.get(k), float) else mb.get(k, "-"))} for k in mb]

    def md(df):
        return ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
                + "".join("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in x) + " |\n"
                          for x in df.itertuples(index=False)))
    cols = ["key", "그룹_b", "질문_b", "실패유형_a", "실패유형_b"]
    text = (f"# {tag} vs {base}\n\n- 비교: {time.strftime('%Y-%m-%d %H:%M')} / 같은 질문 {len(m)}개 (설계 질문 제외)\n"
            f"- 판정: **{verdict}** (단순 사실 164개의 정답·근거 차이 95% 신뢰구간 기준)\n\n## 지표\n\n{md(pd.DataFrame(rows))}\n"
            f"## 차이와 신뢰구간 (부트스트랩 {BOOT_N}회, 같은 질문끼리 짝지음)\n\n"
            f"| 지표 | 차이 [95% CI] |\n|---|---|\n| 정답 | {fmt(ci['정답'])} |\n| 근거 있음 | {fmt(ci['근거'])} |\n"
            f"| 실패율 (낮을수록 좋음) | {fmt(fail_ci)} |\n\n"
            f"## 고쳐진 질문 ({len(fixed)}개)\n\n{md(fixed[cols].rename(columns=lambda c: c.replace('_b', '').replace('_a', ' (기준)')))}\n"
            f"## 새로 실패한 질문 ({len(broken)}개)\n\n{md(broken[cols].rename(columns=lambda c: c.replace('_b', '').replace('_a', ' (기준)')))}")
    (RESULTS_ROOT / tag / f"compare_vs_{base}.md").write_text(text, encoding="utf-8")
    print(f"[{tag} vs {base}] 판정 {verdict} | 정답 {fmt(ci['정답'])} | 근거 {fmt(ci['근거'])} | "
          f"고쳐짐 {len(fixed)} · 새로 실패 {len(broken)}")
    print(f"저장: eval/results/{tag}/compare_vs_{base}.md")
    return {"판정": verdict, "고쳐진 질문": len(fixed), "새로 실패": len(broken),
            "정답 차이 (95% CI)": fmt(ci["정답"]), "근거 차이 (95% CI)": fmt(ci["근거"])}


# ---------------------------- 기록표 ----------------------------
def _registry() -> pd.DataFrame:
    if REGISTRY.exists():
        return pd.read_csv(REGISTRY, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def _save_registry(r: pd.DataFrame):
    r = r[COLUMNS]
    r.to_csv(REGISTRY, index=False, encoding="utf-8-sig")
    md = ("# 실험 기록\n\n측정 → 실패 사례 확인 → 원인 계층 특정 → 단일 변수 변경 → 재측정. 기준은 v5. "
          "지표는 단순 사실 질문 164개 기준(거절 적절·실패는 197개, 거절 정확도는 근거 없는 질문 15개).\n\n"
          "| " + " | ".join(COLUMNS) + " |\n|" + "---|" * len(COLUMNS) + "\n"
          + "".join("| " + " | ".join(str(v).replace("|", "/") for v in x) + " |\n" for x in r.itertuples(index=False)))
    REGISTRY.with_suffix(".md").write_text(md, encoding="utf-8")


def _record(row: dict):
    r = _registry()
    r = pd.concat([r[r["태그"] != row["태그"]], pd.DataFrame([{c: row.get(c, "") for c in COLUMNS}])], ignore_index=True)
    _save_registry(r)


def _pct(v):
    return f"{v:.1%}" if isinstance(v, float) else (v or "")


def decide(tag: str, decision: str):
    """실험 결과를 보고 '채택' 또는 '기각'을 기록 (채택한 설정은 rag/ 코드에 직접 반영해야 함)"""
    r = _registry()
    if tag not in set(r["태그"]):
        raise ValueError(f"기록표에 없는 태그예요: {tag}")
    r.loc[r["태그"] == tag, "결정"] = decision
    _save_registry(r)
    print(f"{tag}: {decision} 기록")


def record_manual(tag: str, layer: str, variable: str, base_value, new_value, note: str = ""):
    """청킹·임베딩처럼 DB를 다시 만들어야 하는 실험을, 측정을 마친 뒤 기록 (eval/results/<tag>/에 결과가 있어야 함)"""
    m = metrics(tag)
    _record({"태그": tag, "날짜": time.strftime("%Y-%m-%d"), "기준": BASE_TAG, "계층": layer, "변수": variable,
             "기준값": base_value, "변경값": new_value, "가설": note, **{k: _pct(v) for k, v in m.items()}, **compare(tag)})


# ---------------------------- 실험 실행 ----------------------------
def analyze(tag: str = BASE_TAG):
    """실패 사례 → 원인 계층 (eval/failure_layers.py)"""
    from eval import failure_layers
    return failure_layers.analyze(tag)


def run_experiment(tag: str, note: str = "", with_refusal: bool = True, **change):
    """변수 하나를 바꿔 재측정 → 기준과 비교 → 기록표에 추가"""
    from eval import refusal_eval, run_eval
    name, value = _check_one_change(change)
    folder = RESULTS_ROOT / tag
    cfg_path = folder / "config.json"
    if cfg_path.exists():
        saved = json.loads(cfg_path.read_text(encoding="utf-8"))
        if saved.get("변경") != {name: value}:
            raise ValueError(f"태그 {tag}는 이미 다른 실험({saved.get('변경')})에 쓰였어요. 다른 태그를 주세요.")
        print(f"ℹ️ {tag} 폴더가 있어서 저장된 결과에 이어서 진행해요")
    folder.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({"태그": tag, "기준": BASE_TAG, "변경": {name: value}, "계층": VARIABLES[name][0],
                                    "가설": note, "기준 설정": BASELINE, "날짜": time.strftime("%Y-%m-%d %H:%M")},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[실험 {tag}] {VARIABLES[name][1]}: {BASELINE[name]} → {value}  (나머지는 기준 v5 그대로)")
    undo = _apply(name, value)
    try:
        run_eval.run(tag)
        if with_refusal:
            refusal_eval.run(tag)
    finally:
        undo()
    m = metrics(tag)
    if with_refusal and not (RESULTS_ROOT / BASE_TAG / "refusal_eval.csv").exists():
        print(f"ℹ️ 기준 {BASE_TAG}의 거절 정확도가 없어요. python -m eval.refusal_eval 로 먼저 측정해 두면 비교할 수 있어요.")
    c = compare(tag)
    _record({"태그": tag, "날짜": time.strftime("%Y-%m-%d"), "기준": BASE_TAG, "계층": VARIABLES[name][0], "변수": name,
             "기준값": BASELINE[name], "변경값": value, "가설": note, **{k: _pct(v) for k, v in m.items()}, **c})
    print(f"기록: eval/results/experiments.md  → 결과를 보고 ex.decide('{tag}', '채택' 또는 '기각')")
    return c


def status():
    print("[기준]", BASE_TAG, "|", ", ".join(f"{k}={v}" for k, v in BASELINE.items()))
    print("\n[바꿀 수 있는 변수]")
    for k, (layer, desc, _) in VARIABLES.items():
        print(f"  {k:20s} {layer:4s} {desc}")
    r = _registry()
    print(f"\n[실험 기록] {len(r)}개 (eval/results/experiments.md)")
    if len(r):
        print(r[["태그", "계층", "변수", "변경값", "정답", "근거 있음", "판정", "결정"]].to_string(index=False))


def tags() -> list[str]:
    return sorted(p.name for p in RESULTS_ROOT.iterdir() if p.is_dir() and (p / f"answer_eval_details_{p.name}.csv").exists())

