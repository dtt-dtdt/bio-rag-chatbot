# 평가 하네스

개선은 아래 순서를 반복해요. **한 번에 변수 하나만** 바꾸고, 실험마다 태그를 붙여 `eval/results/<태그>/`에 저장해요.
`eval/results/experiments.md`(실험 기록표)가 리포트의 근거 자료예요.

```
① 측정 → ② 실패 사례 확인 → ③ 원인 계층 특정 → ④ 단일 변수 변경 → ⑤ 재측정 (→ ①)
```

| 단계 | 실행 (Jupyter, project3 폴더) | 결과 |
|---|---|---|
| ① 측정 | `run_eval.run("v5")`, `refusal_eval.run("v5")` | `results/v5/answer_eval_report_v5.md`, `refusal_eval.md` |
| ② 실패 사례 확인 | 보고서의 '실패 사례 분류'·'유형별 사례' | 같은 보고서 |
| ③ 원인 계층 특정 | `experiment.analyze("v5")` | `results/v5/failure_layers.md` |
| ④ 단일 변수 변경 + ⑤ 재측정 | `experiment.run_experiment("e01_k10", K_BOTH=10, note="…")` | `results/e01_k10/` + `compare_vs_v5.md` + 기록표 한 줄 |
| 결정 | `experiment.decide("e01_k10", "채택")` | 기록표의 '결정' 칸 |

```python
from eval import experiment as ex
ex.status()                     # 기준 설정, 바꿀 수 있는 변수, 지금까지의 실험
ex.analyze("v5")                # 실패 → 원인 계층
ex.run_experiment("e01_k10", K_BOTH=10, note="검색 계층(순위 낮음) → 자료 수 8 → 10")
ex.decide("e01_k10", "기각")
```

## 원인 계층과 대응

| 증상 | 원인 계층 | 판정 방법 (`failure_layers.py`) | 대응 → 바꿀 변수 |
|---|---|---|---|
| 검색 결과에 정답 문서가 포함되지 않음 | 데이터 | 정답 문서가 DB에 없음 | 해당 주제 문서 부재 → 공공 문서 추가 수집 (`collector`, `indexer` 다시 실행) |
| 검색 결과 상위가 동일 내용으로 채워짐 | 전처리 | 같은 내용 기사(제목 50% 이상 겹침·3일 이내)가 자리를 차지해 정답이 밀림, 또는 노이즈로 표시돼 제외 | 중복 처리·노이즈 기준 재설정 → `NOISE_FILTER` |
| 정답 문서가 존재하나 검색되지 않음 | 인덱싱 | 후보를 200개로 넓혀도 정답 문서가 없음 | 청크 크기, 제목 포함 여부 점검 → DB를 다시 만들어야 해서 `record_manual()`로 기록 |
| 검색은 되었으나 순위가 낮음 | 검색 | 200위 안에는 있지만 답변에 쓰는 개수 밖 | top_k 조정, Reranker, 하이브리드 → `K_BOTH`, `W_VEC`, `W_BM25`, `CANDIDATES` |
| 근거는 정확하나 답변이 부정확함 | 생성 | 정답 문서는 검색됐는데 틀리거나 부족 | 프롬프트 수정 → `SYSTEM_PROMPT_FILE`, `BODY_CHARS` |
| 근거 없는 내용을 생성함 | 생성 | 근거성 '일부·없음' 또는 환각·거절 실패 | 거절 지시 강화(`SYSTEM_PROMPT_FILE`), 유사도 임계값(아직 변수 없음) |

## 규칙

- **한 번에 변수 하나만 바꿔요.** 청킹, 임베딩, 프롬프트를 동시에 바꾸면 무엇 때문에 좋아졌는지 알 수 없어요.
  - `run_experiment`는 바꾸는 값이 2개 이상이거나, 이미 기준과 다른 설정이 섞여 있으면 실행하지 않아요.
- **기준은 v5예요.** 모든 실험은 v5 설정에서 변수 하나만 바꾼 상태로 측정하고, v5와 질문별로 비교해요.
- **차이가 우연인지 확인해요.** 같은 질문끼리 짝지은 부트스트랩 95% 신뢰구간이 0을 포함하지 않을 때만 '개선' 또는 '악화'로 판정해요.
  - 질문 2~6개 차이는 LLM이 실행할 때마다 생기는 흔들림과 구분되지 않는 경우가 많아요.
- **채택한 설정은 코드에 직접 반영해요.** `run_experiment`는 실험이 끝나면 설정을 v5로 되돌려요.
- **프롬프트 실험:** `eval/prompts/system_v5.txt`(현재 지시문)를 복사해 고친 뒤 `SYSTEM_PROMPT_FILE="eval/prompts/새파일.txt"`로 실행해요. `{instruction}` 자리는 지우지 마세요.

## 폴더

```
eval/
├── eval_set/               평가 질문 (단순 사실 250 + 표현 바꾼 250 + 설계 보완 60)
├── prompts/                프롬프트 실험용 지시문 파일
├── results/
│   ├── experiments.md/csv  실험 기록표
│   ├── v1/ … v7/           규칙 도입 전 버전 (v5 = 최종 기준)
│   ├── <실험 태그>/         config.json(바꾼 변수), 답변·채점, compare_vs_v5.md, failure_layers.md, refusal_eval.md
│   └── (검색 평가 결과)     eval_retriever.md, bootstrap_ci.md, eval_report_v3.md, 판정 캐시 등
├── run_eval.py             답변 품질 측정 (정답성·근거성·거절 적절성·실패 유형)
├── refusal_eval.py         거절 정확도 (근거 없는 질문 15개)
├── failure_layers.py       실패 원인 계층 판정
├── experiment.py           단일 변수 실험·비교·기록
├── harness.py · judge.py · retriever_eval.py · bootstrap_ci.py   검색 평가 (Hit@k, MRR)
├── diagnose_misses.py      (예전) 검색 실패 원인 진단 — v2 기준
├── smoke_test.py · smoke_test2.py   실사용 질문 점검·속도
└── design_questions.py     비교·시점·종합·답변 불가 질문 생성
```

## 측정 지표

| 구분 | 지표 | 산출 방식 | 파일 |
|---|---|---|---|
| 검색 | Hit@k | 상위 k개 결과에 정답(급) 문서가 포함된 질문의 비율 (k = 1·3·5) | `harness.py`, `retriever_eval.py` |
| 검색 | MRR@3 | 첫 정답 문서 순위의 역수 평균 (3위 밖은 0) | `retriever_eval.py` |
| 답변 | 정확도 | 정확·부분 정확·오답 3단계. LLM 채점 + 사람 표본 30개로 일치율 확인(`agreement`) | `run_eval.py` |
| 답변 | 출처 정확성 | 답변의 사실이 인용한 자료에 실제로 있는지 (근거성: 있음·일부·없음) | `run_eval.py` |
| 답변 | 거절 정확도 | 근거 없는 질문 15개 중 거절하거나 '확인되지 않는다'고 밝힌 비율 | `refusal_eval.py` |
