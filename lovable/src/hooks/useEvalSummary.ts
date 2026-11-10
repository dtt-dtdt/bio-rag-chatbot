// 평가 결과 불러오기 (버전·실험별 지표, 최종 버전의 실패 유형·원인 계층)
import { useEffect, useState } from "react";
import { api, ApiError, type EvalSummary, type VersionSummary } from "@/lib/api";

export function useEvalSummary() {
  const [data, setData] = useState<EvalSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .evalSummary()
      .then(setData)
      .catch((e) => setError(e instanceof ApiError ? e.message : "평가 결과를 불러오지 못했어요."))
      .finally(() => setLoading(false));
  }, []);

  const final: VersionSummary | undefined = data?.versions.find((v) => v.tag === data.final_version);
  return { data, final, loading, error };
}
