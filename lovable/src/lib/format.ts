// 화면 표시용 도우미 (디자인과 무관한 문구·변환만)
import type { AnswerType, Source } from "@/lib/api";

export const TYPE_LABEL: Record<AnswerType, string> = {
  answer: "",
  advice_limited: "판단은 드리지 않고 자료 속 사실만 정리했어요",
  refused: "답변하지 않은 질문이에요",
  info: "",
};

export const EXAMPLES = ["알테오젠 기술수출 현황 알려줘", "CD3 이중항체 특허 있어?", "2024년 ADC 기술수출 동향"];

const DATE_RE = /^(19|20)\d{2}(-(0[1-9]|1[0-2]))?$/;
export const isValidDate = (s: string) => DATE_RE.test(s.trim());

/** 출처 한 줄 제목: "[1] 특허 · 2024-11-26 · 제목" */
export const sourceHeading = (s: Source) =>
  `[${s.number}] ${s.kind === "patent" ? "특허" : "뉴스"} · ${s.date} · ${s.title}`;

/** 출처 보조 정보: 특허는 출원인, 뉴스는 매체 */
export const sourceMeta = (s: Source) =>
  s.kind === "patent" ? (s.applicant ? `출원인: ${s.applicant}` : "") : s.media ?? "";

/** 원문 확인 안내: 뉴스는 링크, 특허는 키프리스에서 출원번호로 검색 */
export const sourceLinkText = (s: Source) =>
  s.url ? "원문 기사 열기" : `${s.reference} — 키프리스(kipris.or.kr)에서 출원번호로 원문을 볼 수 있어요`;

/** 답변 속 [번호]를 찾아 출처와 연결할 때 사용 */
export const CITATION_RE = /\[(\d{1,2})\]/g;

export const pct = (v: number | null | undefined) => (v == null ? "-" : `${(v * 100).toFixed(1)}%`);
