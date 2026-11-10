"""
바이오의약품·바이오헬스케어 RAG 챗봇 — Streamlit 시작 파일

[실행]  (터미널/Anaconda Prompt, project3 폴더에서)
  streamlit run app/main.py
  → 브라우저에서 http://localhost:8501 이 열려요. 끄려면 터미널에서 Ctrl + C
  ※ rag/ 코드를 고친 뒤에는 화면을 껐다 켜야 반영돼요 (검색 색인을 한 번만 불러 두기 때문)

[화면]  왼쪽 위에서 선택
  - 챗봇       : app/chat.py
  - 평가 결과  : app/dashboard.py (eval/results/의 평가 결과를 표·차트로)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:          # streamlit은 app/ 폴더만 경로에 넣으므로 project3 폴더를 추가
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import chat, dashboard  # noqa: E402

st.set_page_config(page_title="바이오 특허·뉴스 챗봇", page_icon="🧬", layout="centered")

page = st.sidebar.radio("화면", ["💬 챗봇", "📊 평가 결과"], label_visibility="collapsed",
                        index=0 if chat.data_ready() else 1)   # 데이터 없는 배포본은 평가 결과부터
st.sidebar.divider()
if page == "💬 챗봇":
    chat.render()
else:
    dashboard.render()
