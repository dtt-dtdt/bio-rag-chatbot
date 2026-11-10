// 채팅 상태 관리: 멀티턴(이전 대화 전달), 검색 조건, 전송·초기화, 오류 처리
import { useCallback, useState } from "react";
import { api, ApiError, type AskResponse, type Filters, type HistoryMessage } from "@/lib/api";
import { HISTORY_TURNS } from "@/config";
import { isValidDate } from "@/lib/format";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;          // 사용자 질문 또는 답변 본문
  response?: AskResponse;   // 답변일 때 전체 응답 (출처·분석 정보)
  error?: string;           // 답변을 못 만들었을 때
}

export interface FilterInput {
  target: "auto" | Filters["target"];   // "auto"면 서버가 정함
  field: "all" | NonNullable<Filters["field"]>;
  dateFrom: string;
  dateTo: string;
}

export const DEFAULT_FILTERS: FilterInput = { target: "auto", field: "all", dateFrom: "", dateTo: "" };

/** 화면 입력 → API 조건. 형식이 틀린 기간은 무시하고 경고 목록으로 돌려줌 */
export function toFilters(input: FilterInput): { filters: Filters; invalidDates: string[] } {
  const filters: Filters = {};
  if (input.target && input.target !== "auto") filters.target = input.target;
  if (input.field !== "all") filters.field = input.field;
  const dates = [input.dateFrom.trim(), input.dateTo.trim()];
  const invalidDates = dates.filter((d) => d && !isValidDate(d));
  if (invalidDates.length === 0) {
    if (dates[0]) filters.date_from = dates[0];
    if (dates[1]) filters.date_to = dates[1];
  }
  return { filters, invalidDates };
}

const newId = () => Math.random().toString(36).slice(2);

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [filterInput, setFilterInput] = useState<FilterInput>(DEFAULT_FILTERS);
  const [loading, setLoading] = useState(false);

  const { filters, invalidDates } = toFilters(filterInput);

  const send = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || loading) return;
      // 멀티턴: 최근 대화(질문+답변 HISTORY_TURNS쌍)를 넘겨 '그 회사' 같은 말을 실제 이름으로 바꾸게 함
      const history: HistoryMessage[] = messages
        .filter((m) => !m.error)
        .slice(-2 * HISTORY_TURNS)
        .map((m) => ({ role: m.role, content: m.content }));
      setMessages((prev) => [...prev, { id: newId(), role: "user", content: q }]);
      setLoading(true);
      try {
        const res = await api.ask(q, history, filters);
        setMessages((prev) => [...prev, { id: newId(), role: "assistant", content: res.answer, response: res }]);
      } catch (e) {
        const msg = e instanceof ApiError ? e.message : "답변을 만들지 못했어요. 잠시 후 다시 시도해 주세요.";
        setMessages((prev) => [...prev, { id: newId(), role: "assistant", content: "", error: msg }]);
      } finally {
        setLoading(false);
      }
    },
    [messages, filters, loading],
  );

  const reset = useCallback(() => setMessages([]), []);

  return { messages, loading, send, reset, filterInput, setFilterInput, invalidDates };
}
