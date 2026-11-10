"""
④ 챗봇 대화 화면 — 질문 → 답변 + 출처(원문 확인) / 멀티턴 / 검색 조건

  - 가운데: 채팅. 이전 대화를 기억해서 '그 회사 특허는?' 같은 이어지는 질문도 이해 (멀티턴)
  - 답변 아래: 인용된 출처를 하나씩 펼쳐서 원문(본문·링크·출원번호) 확인
  - 왼쪽: 검색 조건(대상·분야·기간), 대화 초기화, 질문 분석 정보 보기
  (조회 전용: 검색 색인·DB를 읽기만 하고 바꾸지 않음)
"""
from __future__ import annotations

import re

import streamlit as st

HISTORY_TURNS = 3          # 질문 분석에 넘길 이전 대화 수 (질문+답변 한 쌍 = 1)
EXAMPLES = ["알테오젠 기술수출 현황 알려줘", "CD3 이중항체 특허 있어?", "2024년 ADC 기술수출 동향"]
TYPE_LABEL = {"답변": "", "조언 제한": "⚠️ 판단은 드리지 않고 자료 속 사실만 정리했어요",
              "거절": "🚫 답변하지 않은 질문이에요", "안내": ""}


@st.cache_resource(show_spinner="검색 색인을 불러오는 중… (처음 한 번, 수십 초)")
def load_chain():
    from rag import chain as rc
    rc.rt.get_index("patents")
    rc.rt.get_index("news")
    return rc


def md_safe(text: str) -> str:
    """Streamlit 마크다운에서 $(수식)와 ~(취소선)로 잘못 바뀌지 않게"""
    return text.replace("$", "\\$").replace("~", "\\~")


def date_ok(s: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d{2}(-(0[1-9]|1[0-2]))?", s))


def sidebar_filters() -> tuple[dict, bool]:
    """왼쪽: 검색 조건 → (retrieve 조건, 질문 분석 정보 보기 여부)"""
    with st.sidebar:
        st.header("검색 조건")
        target = st.radio("대상", ["자동", "특허+뉴스", "특허만", "뉴스만"], index=0,
                          help="자동: 질문을 보고 챗봇이 정해요 (예: '특허 있어?' → 특허만)")
        field = st.selectbox("특허 분야", ["전체", "바이오의약품", "바이오헬스케어"])
        c1, c2 = st.columns(2)
        date_from = c1.text_input("시작", placeholder="2024", help="YYYY 또는 YYYY-MM. 비우면 질문에서 자동으로 읽어요")
        date_to = c2.text_input("끝", placeholder="2025-06")
        show_analysis = st.toggle("질문 분석 정보 보기", value=False)
        st.divider()
        if st.button("대화 초기화", width="stretch"):
            st.session_state.messages = []
            st.rerun()
        st.caption("자료: 특허 8,790건, 뉴스 32,168건 (2021~2026). 답변은 검색된 자료만 근거로 하며, "
                   "투자·진단·법률 판단은 하지 않아요.")

    filters = {}
    if target != "자동":
        filters["target"] = {"특허+뉴스": "both", "특허만": "patent", "뉴스만": "news"}[target]
    if field != "전체":
        filters["field"] = field
    bad_dates = [d for d in (date_from.strip(), date_to.strip()) if d and not date_ok(d)]
    if date_from.strip() and not bad_dates:
        filters["date_from"] = date_from.strip()
    if date_to.strip() and not bad_dates:
        filters["date_to"] = date_to.strip()
    if bad_dates:
        st.sidebar.warning(f"기간 형식이 달라 무시했어요: {', '.join(bad_dates)} (예: 2024, 2024-03)")
    return filters, show_analysis


def render_sources(out: dict):
    cited = out.get("인용출처", [])
    if cited:
        st.caption(f"출처 {len(cited)}건 — 눌러서 원문 확인")
        for s in cited:
            who = s["출원인"] or s["매체"]
            with st.expander(f"[{s['번호']}] {s['종류']} · {s['날짜']} · {s['제목'][:60]}"):
                st.markdown(f"**{md_safe(s['제목'])}**")
                meta = [s["종류"], s["날짜"]] + ([f"출원인: {who}"] if s["종류"] == "특허" and who else [who] if who else [])
                st.caption(" | ".join(meta))
                if s["링크"]:
                    st.markdown(f"[원문 기사 열기]({s['링크']})")
                else:
                    st.caption(f"{s['출처']} — 키프리스(kipris.or.kr)에서 출원번호로 원문을 볼 수 있어요")
                st.write(s["본문"])
    others = [s for s in out.get("sources", []) if s["번호"] not in out.get("인용번호", [])]
    if others and out.get("type") in ("답변", "조언 제한"):
        with st.expander(f"그 밖에 검색된 자료 {len(others)}건 (답변에 쓰이지 않음)"):
            for s in others:
                st.markdown(f"- [{s['번호']}] {s['종류']} · {s['날짜']} · {md_safe(s['제목'][:70])}")


def render_assistant(out: dict, show_analysis: bool = False):
    label = TYPE_LABEL.get(out.get("type"), "")
    if label:
        st.caption(label)
    st.markdown(md_safe(out["answer"]))
    render_sources(out)
    if show_analysis and out.get("analysis"):
        a = out["analysis"]
        cond = out.get("적용조건", {})
        st.caption(f"질문 분석: 분류 {a['category']} · 검색어 '{a['search_query']}' · 대상 {cond.get('target', 'both')} · "
                   f"기간 {cond.get('date_from') or '-'}~{cond.get('date_to') or '-'}"
                   f"{' (0건이라 기간 뺌)' if out.get('기간완화') else ''} · "
                   f"분석 {out['분석초']:.1f}초 + 검색 {out['검색초']:.1f}초 + 답변 {out['답변초']:.1f}초")


def render():
    """챗봇 화면 전체 (main.py가 부름)"""
    filters, show_analysis = sidebar_filters()
    st.title("바이오 특허·뉴스 챗봇")
    st.caption("바이오의약품·바이오헬스케어 특허와 뉴스를 찾아, 출처와 함께 답해 드려요.")

    if "messages" not in st.session_state:
        st.session_state.messages = []      # [{"role": "user"|"assistant", "content": str, "out": dict|None}]

    rc = load_chain()

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and m.get("out"):
                render_assistant(m["out"], show_analysis)
            else:
                st.markdown(md_safe(m["content"]))

    clicked = None
    example_box = st.empty()                 # 첫 화면의 예시 질문 (질문하면 사라짐)
    if not st.session_state.messages:
        with example_box.container():
            st.write("이렇게 물어보세요:")
            cols = st.columns(len(EXAMPLES))
            for c, q in zip(cols, EXAMPLES):
                if c.button(q, width="stretch"):
                    clicked = q

    question = st.chat_input("질문을 입력하세요 (예: 알테오젠 기술수출 현황)") or clicked
    if question:
        example_box.empty()
        # 멀티턴: 최근 대화를 질문 분석에 넘겨 '그 회사', '그 기술' 같은 말을 실제 이름으로 바꿈
        past = st.session_state.messages[-2 * HISTORY_TURNS:]
        history = [(m["role"], m["content"]) for m in past]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(md_safe(question))
        with st.chat_message("assistant"):
            with st.spinner("질문을 분석하고 자료를 찾아 답변을 쓰는 중… (10초 안팎)"):
                try:
                    out = rc.ask(question, history=history, **filters)
                except Exception as e:      # API 키·연결 문제 등. 외부 오류 문구에는 키 일부가 섞일 수 있어 종류만 보여 줌
                    out = None
                    detail = str(e)[:200] if isinstance(e, RuntimeError) else type(e).__name__
                    st.error(f"답변을 만들지 못했어요: {detail}. 잠시 후 다시 시도해 주세요.")
            if out:
                render_assistant(out, show_analysis)
                st.session_state.messages.append({"role": "assistant", "content": out["answer"], "out": out})
