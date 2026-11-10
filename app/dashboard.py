"""
④ 평가 결과 화면 — eval/results/<태그>/의 답변 평가 결과(v1~v7 + 이후 실험)를 표·차트로 보여 줌 (읽기만 함)

  - 최종 채택 버전(v5) 핵심 지표
  - 버전별 단순 사실 질문 164개 지표 변화 (같은 질문으로 비교)
  - 최종 버전의 실패 유형
  - 평가 설계 보완 질문(비교·시점·종합·답변 불가) 결과 (v6·v7 실험, 미채택)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config

FINAL = "v5"
VERSION_NOTE = {
    "v1": "첫 답변 체인",
    "v2": "채점 자료에 출원인·날짜 추가",
    "v3": "'최근' 처리, 자료 5개 → 8개",
    "v4": "회사명 필터 보완, 특허 없는 회사 안내",
    "v5": "안내는 특허 질문에만, 기술 용어 오인 방지 (최종)",
    "v6": "프롬프트 날짜·출처 2줄 + 평가 설계 질문 추가 (실험)",
    "v7": "비교·시점·종합 질문용 검색·답변 지시 (실험)",
}
DESIGN = ["비교", "시점", "종합", "답변불가"]


def _read(path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.DataFrame()


def _order(tag: str):
    """v1~v7을 먼저 번호순으로, 그 뒤 실험 태그를 이름순으로"""
    return (0, int(tag[1:]), "") if tag[:1] == "v" and tag[1:].isdigit() else (1, 0, tag)


@st.cache_data
def load_versions() -> dict[str, pd.DataFrame]:
    out = {}
    for folder in config.EVAL_RESULTS_DIR.iterdir():
        p = folder / f"answer_eval_details_{folder.name}.csv"
        if folder.is_dir() and p.exists():
            out[folder.name] = _read(p)
    return dict(sorted(out.items(), key=lambda kv: _order(kv[0])))


@st.cache_data
def load_notes() -> dict[str, str]:
    """실험 기록표(eval/results/experiments.csv)의 '변수: 기준값 → 변경값' (v1~v7은 VERSION_NOTE)"""
    notes = dict(VERSION_NOTE)
    path = config.EVAL_RESULTS_DIR / "experiments.csv"
    if path.exists():
        for r in _read(path).itertuples(index=False):
            if r.태그 not in notes:
                notes[r.태그] = f"{r.변수}: {r.기준값} → {r.변경값}" + (f" ({r.결정})" if r.결정 else "")
    return notes


def summary(versions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    notes = load_notes()
    rows = []
    for v, d in versions.items():
        ev = d[d["그룹"].str.startswith("평가")]
        base = d[~d["그룹"].str.startswith("설계")]          # 설계 질문(v6~)을 뺀 기존 평가셋
        rows.append({"버전": v, "변경": notes.get(v, ""),
                     "답변함": ev["결과종류"].isin(["답변", "조언 제한"]).mean(),
                     "정답": (ev["정답성"] == "correct").mean(),
                     "정답+부분": ev["정답성"].isin(["correct", "partial"]).mean(),
                     "근거 있음": (ev["근거성"] == "yes").mean(),
                     "정답 문서 검색됨": (ev["정답검색됨"] == "True").mean(),
                     "거절 적절": (base["거절적절"] == "True").mean(),
                     "실패 수": int((base["실패"] == "True").sum()), "질문 수": len(base)})
    return pd.DataFrame(rows).set_index("버전")


def render():
    st.title("평가 결과")
    versions = load_versions()
    if not versions:
        st.info("eval/results/<태그>/에 평가 결과가 없어요. `python -m eval.run_eval`로 평가를 먼저 실행해 주세요.")
        return
    s = summary(versions)
    pct = ["답변함", "정답", "정답+부분", "근거 있음", "정답 문서 검색됨", "거절 적절"]

    if FINAL in s.index:
        f = s.loc[FINAL]
        st.subheader(f"최종 버전 {FINAL}")
        st.caption("단순 사실 질문 164개 기준 (거절 적절·실패는 거절 점검·실사용 질문 포함 197개)")
        c = st.columns(4)
        c[0].metric("정답", f"{f['정답']:.1%}")
        c[1].metric("근거 있음", f"{f['근거 있음']:.1%}")
        c[2].metric("거절 적절", f"{f['거절 적절']:.1%}")
        c[3].metric("실패", f"{f['실패 수']} / {f['질문 수']}")

    st.subheader("버전·실험별 변화 (같은 질문 164개)")
    st.line_chart(s[["정답", "근거 있음", "정답 문서 검색됨"]] * 100, y_label="%")
    st.caption("v1·v4의 '근거 있음'이 낮은 것은 채점 쪽 문제예요 (v1: 채점 자료에 출원인 누락, v4: 안내 문구까지 채점).")
    st.dataframe(s.style.format({c: "{:.1%}" for c in pct}), width="stretch")

    if FINAL in versions:
        d = versions[FINAL]
        fail = d[d["실패"] == "True"]["실패유형"].value_counts()
        st.subheader(f"{FINAL} 실패 유형")
        st.bar_chart(fail, horizontal=True)
        layers = config.EVAL_RESULTS_DIR / FINAL / "failure_layers.csv"
        if layers.exists():
            st.subheader(f"{FINAL} 실패 원인 계층")
            st.caption("데이터 → 전처리 → 인덱싱 → 검색 → 생성 중 어디서 실패했는지 (eval/failure_layers.py)")
            st.bar_chart(_read(layers)["계층"].value_counts(), horizontal=True)
        with st.expander("실패한 질문 보기"):
            st.dataframe(d[d["실패"] == "True"][["key", "그룹", "질문", "실패유형", "채점이유"]], width="stretch")

    design = {v: d for v, d in versions.items() if d["그룹"].str.startswith("설계").any()}
    if design:
        st.subheader("평가 설계 보완 질문 (실험, 미채택)")
        st.caption("비교·시점·종합·답변 불가 각 15개. 채점 기준표(꼭 들어갈 요소)로 채점. 정답 = 요소 80% 이상 + 유형 항목 통과")
        rows = []
        for v, d in design.items():
            for t in DESIGN:
                x = d[d["그룹"] == f"설계-{t}"]
                if len(x):
                    rows.append({"버전": v, "유형": t, "질문 수": len(x),
                                 "정답": (x["정답성"] == "correct").mean() if t != "답변불가" else None,
                                 "요소 포함률": pd.to_numeric(x["요소포함률"], errors="coerce").mean() if t != "답변불가" else None,
                                 "거절 적절": (x["거절적절"] == "True").mean()})
        st.dataframe(pd.DataFrame(rows).style.format({"정답": "{:.1%}", "요소 포함률": "{:.1%}", "거절 적절": "{:.1%}"}, na_rep="-"),
                     width="stretch", hide_index=True)
