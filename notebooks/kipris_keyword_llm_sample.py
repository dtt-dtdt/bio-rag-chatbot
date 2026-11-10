# [보관용] 초기 탐색 스크립트 — 특허 키워드를 뽑는 방법(분류 코드·TF-IDF·LLM)을 비교할 때 사용.
# 실제 파이프라인은 indexer/keywords.py. 입력 엑셀 경로는 당시 gen_ai 폴더 기준이에요.

# 방법 3: LLM 구조화 출력으로 특허 키워드 추출 (샘플 10건)
# 실행 전: %pip install -U langchain-openai pandas openpyxl python-dotenv
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

SRC = "kipris_키워드_샘플10.xlsx"          # 방법 1·2 결과가 들어 있는 샘플 파일
DST = "kipris_키워드_샘플10_LLM.xlsx"      # 방법 3 결과를 추가해 새로 저장

df = pd.read_excel(SRC)


# 1) 받고 싶은 모양
class PatentKeywords(BaseModel):
    core_tech: list[str] = Field(description="핵심 기술 키워드 3~5개. 띄어쓰기가 있어도 하나의 기술 용어면 그대로 (예: '클릭 화학')")
    application: list[str] = Field(description="적용 분야·용도 1~3개 (예: '탈모 개선', '치매 조기 진단')")


# 2) 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "특허의 발명의 명칭과 요약을 읽고 키워드를 추출하세요.\n"
     "- '발명', '방법', '조성물', '효과'처럼 모든 특허에 나오는 일반 단어는 제외합니다.\n"
     "- 균주 번호(예: NK357), 수탁번호 같은 식별 코드는 키워드에 넣지 않습니다.\n"
     "- 요약에 없는 내용은 추측하지 마세요."),
    ("human", "발명의 명칭: {title}\n요약: {abstract}"),
])

# 3) 체인
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")
chain = prompt | llm.with_structured_output(PatentKeywords)

# 4) 10건을 동시에 5개씩 처리
inputs = [{"title": t, "abstract": a} for t, a in zip(df["발명의명칭"], df["요약"])]
results = chain.batch(inputs, config={"max_concurrency": 5})

# 5) 결과를 열로 추가하고 저장
df["핵심기술(LLM)"] = [", ".join(r.core_tech) for r in results]
df["적용분야(LLM)"] = [", ".join(r.application) for r in results]
df.to_excel(DST, index=False)

# 6) 방법 2와 나란히 비교 출력
for _, row in df.iterrows():
    print("■", row["발명의명칭"][:45])
    print("  TF-IDF :", row["키워드(TF-IDF)"])
    print("  LLM 기술:", row["핵심기술(LLM)"])
    print("  LLM 분야:", row["적용분야(LLM)"])

print("저장:", DST)
