# Lovable로 화면 만들기

Lovable은 화면(React)만 만들어요. 답변은 내 PC의 API 서버(`api/server.py`)가 만들어요.

```
Lovable 화면 (브라우저) ──HTTPS──▶ ngrok 터널 ──▶ 내 PC: api/server.py ──▶ rag/chain.py ──▶ store/chroma, bm25
```

## 순서

1. **API 서버 켜기** (project3 폴더)
   ```powershell
   ..\.venv\Scripts\python.exe -m uvicorn api.server:app --port 8000
   ```
   브라우저에서 http://localhost:8000/api/health 가 `{"ok": true, "ready": true, ...}`이면 준비 완료.
2. **터널 열기** (새 터미널) — ngrok 설치·가입 후
   ```powershell
   ngrok http 8000
   ```
   `Forwarding https://xxxx.ngrok-free.app` 주소를 복사.
3. **Lovable 새 프로젝트**에 아래 '프롬프트'를 붙여 넣기.
4. 생성된 프로젝트에 `src/` 아래 파일 5개를 **같은 경로로** 넣기 (Lovable 코드 편집기 또는 GitHub 연동):
   `src/config.ts`, `src/lib/api.ts`, `src/lib/format.ts`, `src/hooks/useChat.ts`, `src/hooks/useEvalSummary.ts`
5. `src/config.ts`의 `API_BASE_URL`을 2번 주소로 바꾸기.

> 보안: OpenAI 키는 서버 `.env`에만 있어요. 터널 주소를 아는 사람은 누구나 질문할 수 있으니,
> 서버 `.env`에 `API_TOKEN=아무문자열`을 넣고 `src/config.ts`의 `API_TOKEN`에 같은 값을 넣어 두세요.
> 시연이 끝나면 터널을 꺼 주세요.

---

## 프롬프트 (Lovable에 붙여 넣기)

```
바이오의약품·바이오헬스케어 특허와 뉴스를 근거로 답하는 RAG 챗봇의 웹 화면을 만들어 줘.
답변은 외부 API 서버가 만들고, 이 앱은 화면만 담당해. 디자인(색·레이아웃·컴포넌트 스타일)은 자유롭게 해 줘.

[이미 있는 로직 파일 — 내용은 바꾸지 말고 그대로 import해서 써 줘]
- src/config.ts : API_BASE_URL, API_TOKEN, HISTORY_TURNS
- src/lib/api.ts : 타입(AskResponse, Source, EvalSummary 등)과 api.ask / api.evalSummary / api.health
- src/lib/format.ts : TYPE_LABEL, EXAMPLES, sourceHeading, sourceMeta, sourceLinkText, CITATION_RE, pct
- src/hooks/useChat.ts : useChat() → { messages, loading, send, reset, filterInput, setFilterInput, invalidDates }
- src/hooks/useEvalSummary.ts : useEvalSummary() → { data, final, loading, error }
OpenAI 등 외부 AI를 직접 부르거나 API 키를 앱에 넣지 마. 모든 요청은 src/lib/api.ts로만.

[페이지 1: 챗봇 "/"]
1. 채팅
   - 입력창에 질문 → useChat().send(질문). 보내는 동안 loading이면 "질문을 분석하고 자료를 찾아 답변을 쓰는 중… (10초 안팎)" 표시, 입력 막기.
   - messages를 순서대로 표시. role "user"는 질문, "assistant"는 답변.
   - 답변 본문(content)은 마크다운으로 렌더링. 문장 끝의 [1] [2] 같은 번호는 아래 출처 목록의 같은 번호와 연결해서,
     누르면 해당 출처가 펼쳐지거나 그 위치로 이동하게 해 줘 (CITATION_RE 사용).
   - response.type에 따라 TYPE_LABEL의 문구를 답변 위에 작은 안내로 표시 (빈 문자열이면 표시 안 함).
   - message.error가 있으면 오류 안내로 표시.
   - 대화가 없을 때는 EXAMPLES 3개를 버튼으로 보여 주고, 누르면 그 질문을 바로 send.
2. 출처(원문 확인 영역) — 답변 바로 아래
   - response.sources 중 cited가 true인 것만 "출처 N건 — 눌러서 원문 확인"으로 접어서 보여 줘.
   - 각 출처: 제목줄 sourceHeading(s), 보조 정보 sourceMeta(s), 펼치면 본문 s.body 전체.
     s.url이 있으면 새 탭으로 여는 링크("원문 기사 열기"), 없으면 sourceLinkText(s) 문구를 그대로 표시.
   - response.type이 "answer" 또는 "advice_limited"이고 cited가 false인 출처가 있으면
     "그 밖에 검색된 자료 N건 (답변에 쓰이지 않음)"으로 따로 접어서 제목줄만 목록으로.
3. 검색 조건 (사이드바 또는 설정 패널)
   - 대상: 자동 / 특허+뉴스 / 특허만 / 뉴스만 → filterInput.target = "auto" | "both" | "patent" | "news"
   - 특허 분야: 전체 / 바이오의약품 / 바이오헬스케어 → filterInput.field = "all" | "바이오의약품" | "바이오헬스케어"
   - 기간: 시작·끝 입력칸 (placeholder "2024", "2025-06"), 형식은 YYYY 또는 YYYY-MM → filterInput.dateFrom / dateTo
   - invalidDates가 있으면 "기간 형식이 달라 무시했어요: … (예: 2024, 2024-03)" 경고 표시.
   - "질문 분석 정보 보기" 토글: 켜면 각 답변 아래에
     "분류 {analysis.category} · 검색어 '{analysis.search_query}' · 대상 {analysis.target} · 기간 {date_from}~{date_to}
      (date_relaxed면 '(0건이라 기간 뺌)') · 분석 {timing.analyze_s}초 + 검색 {timing.search_s}초 + 답변 {timing.answer_s}초" 표시.
   - "대화 초기화" 버튼 → reset().
   - 안내 문구: "자료: 특허 8,790건, 뉴스 32,168건 (2021~2026). 답변은 검색된 자료만 근거로 하며, 투자·진단·법률 판단은 하지 않아요."

[페이지 2: 평가 결과 "/eval"]
- useEvalSummary()로 불러와. loading·error 상태 표시.
- 최종 버전(final, data.final_version): 정답 pct(final.correct), 근거 있음 pct(final.grounded),
  거절 적절 pct(final.refusal_ok), 실패 "{final.failures} / {final.questions}", 거절 정확도 pct(final.refusal_accuracy)를 카드로.
  설명: "단순 사실 질문 164개 기준 (거절 적절·실패는 197개, 거절 정확도는 근거 없는 질문 15개)".
- 버전·실험별 변화: data.versions를 tag 순서대로 x축, correct·grounded·gold_retrieved를 선 3개로 그린 차트(%).
  아래에 표: 버전(tag) | 바꾼 것(note) | 정답 | 근거 있음 | 정답 문서 검색됨 | 거절 적절 | 실패 | 결정(decision).
  설명: "v1·v4의 '근거 있음'이 낮은 것은 채점 쪽 문제예요."
- 최종 버전의 실패 유형(data.final.failure_types)과 실패 원인 계층(data.final.failure_layers)을 가로 막대 차트 2개로.

[공통]
- 상단 내비게이션: "챗봇" / "평가 결과".
- 앱 시작 때 api.health()를 한 번 불러서 실패하면 "API 서버에 연결하지 못했어요. 서버와 터널이 켜져 있는지 확인해 주세요." 배너.
- 모든 문구는 한국어.
```

## API 요약

| 메서드 | 경로 | 요청 | 응답 |
|---|---|---|---|
| GET | `/api/health` | - | `{ ok, ready, final_version, model, auth_required }` |
| POST | `/api/ask` | `{ question, history: [{role, content}], filters: {target?, field?, date_from?, date_to?} }` | `AskResponse` (src/lib/api.ts) |
| GET | `/api/eval/summary` | - | `EvalSummary` |

- 헤더: `Content-Type: application/json`, `ngrok-skip-browser-warning: true`, (토큰을 쓰면) `X-API-Key`
- 오류: `{ error, message }` + HTTP 400(잘못된 요청)·401(토큰)·500(서버 오류)
