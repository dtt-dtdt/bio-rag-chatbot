// API 서버 주소와 토큰 — 여기만 바꾸면 돼요.
// 내 PC에서 `python -m uvicorn api.server:app --port 8000` 실행 → `ngrok http 8000`이 준 https 주소를 넣기
export const API_BASE_URL = "https://YOUR-TUNNEL.ngrok-free.app";
// 서버 .env에 API_TOKEN을 넣었다면 같은 값. (브라우저에서 보이는 값이라 비밀번호가 아니라 '무단 호출 막기' 용도예요)
export const API_TOKEN = "";
// 질문 분석에 넘길 이전 대화 수 (질문+답변 한 쌍 = 1). 서버도 최대 3쌍만 씀
export const HISTORY_TURNS = 3;
