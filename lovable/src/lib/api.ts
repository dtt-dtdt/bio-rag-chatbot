// RAG 챗봇 API 타입과 호출 함수 (서버: project3/api/server.py)
import { API_BASE_URL, API_TOKEN } from "@/config";

export type Target = "both" | "patent" | "news";
export type Field = "바이오의약품" | "바이오헬스케어";

export interface Filters {
  target?: Target;          // 없으면 질문을 보고 서버가 정함 ('특허 있어?' → patent)
  field?: Field;            // 특허 분야
  date_from?: string;       // "YYYY" 또는 "YYYY-MM"
  date_to?: string;
}

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export interface Source {
  number: number;           // 답변 속 [번호]
  kind: "patent" | "news";
  title: string;
  date: string;             // 특허 = 등록일, 뉴스 = 보도일
  url: string | null;       // 뉴스 원문 링크 (특허는 null → reference의 출원번호로 키프리스에서 확인)
  reference: string;        // 특허: "출원번호 …", 뉴스: 원문 URL
  applicant: string | null; // 특허 출원인
  media: string | null;     // 뉴스 매체
  body: string;             // 검색된 본문 (원문 확인 영역에 표시)
  cited: boolean;           // 답변이 실제로 인용했는지
}

export type AnswerType = "answer" | "advice_limited" | "refused" | "info";

export interface AskResponse {
  type: AnswerType;          // answer: 정상 답변 / advice_limited: 투자·진단·법률 판단은 빼고 사실만 / refused: 거절 / info: 인사 안내
  refusal_reason: string | null; // 근거부족, 검색결과없음, off_topic, injection, advice
  answer: string;            // 마크다운. 문장 끝 [1] [2]가 sources.number와 연결됨
  sources: Source[];
  analysis: {
    category: string;
    search_query: string;
    target: Target;
    date_from: string | null;
    date_to: string | null;
    date_relaxed: boolean;   // 기간 때문에 0건이라 기간을 빼고 다시 찾았는지
  } | null;
  timing: { analyze_s: number; search_s: number; answer_s: number; total_s: number };
}

export interface VersionSummary {
  tag: string;               // v1~v7, 실험 태그
  note: string;              // 바꾼 내용
  decision: string;          // 채택/기각 등
  correct: number;           // 0~1, 단순 사실 164개 정답률
  grounded: number;          // 근거 있음 (출처 정확성)
  gold_retrieved: number;    // 정답 문서 검색됨
  refusal_ok: number;        // 거절 적절성
  failures: number;
  questions: number;
  refusal_accuracy: number | null; // 거절 정확도 (근거 없는 질문 15개)
}

export interface EvalSummary {
  final_version: string;     // "v5"
  versions: VersionSummary[];
  final: { failure_types?: Record<string, number>; failure_layers?: Record<string, number> };
  generated_at: string;
}

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true", // ngrok 무료 터널의 안내 페이지를 건너뜀
  };
  if (API_TOKEN) headers["X-API-Key"] = API_TOKEN;
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, { ...init, headers: { ...headers, ...(init?.headers ?? {}) } });
  } catch {
    throw new ApiError("API 서버에 연결하지 못했어요. 서버와 터널이 켜져 있는지 확인해 주세요.", 0);
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    throw new ApiError(data?.message ?? `요청이 실패했어요 (HTTP ${res.status}).`, res.status);
  }
  return data as T;
}

export const api = {
  health: () => request<{ ok: boolean; ready: boolean; final_version: string; model: string | null; auth_required: boolean }>("/api/health"),
  ask: (question: string, history: HistoryMessage[] = [], filters: Filters = {}) =>
    request<AskResponse>("/api/ask", { method: "POST", body: JSON.stringify({ question, history, filters }) }),
  evalSummary: () => request<EvalSummary>("/api/eval/summary"),
};
