# [보관용] 초기 탐색 스크립트 — 특허 키워드를 뽑는 방법(분류 코드·TF-IDF·LLM)을 비교할 때 사용.
# 실제 파이프라인은 indexer/keywords.py. 입력 엑셀 경로는 당시 gen_ai 폴더 기준이에요.

import re
import pandas as pd
from collections import Counter
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer

SRC = "20261103172104.xlsx"   # 키프리스 엑셀 파일 경로 (gen_ai 폴더 기준)
df = pd.read_excel(SRC)

# 방법 1: 분류 코드
def ipc_summary(value, n=4):
    codes = [c.strip() for c in str(value).split("|") if c.strip()]
    return Counter(c[:n] for c in codes)

IPC_KO = {
    "A61K": "의약·화장용 제제", "A61P": "치료 활성(약효)", "A61Q": "화장품 용도", "A61N": "전기·광·방사선 치료",
    "A61L": "의료용 재료·살균", "A23L": "식품 일반", "A23K": "동물 사료", "A23B": "식품 보존", "A23P": "식품 성형·가공",
    "A01N": "살생물제·보존", "A01P": "살생물 활성", "C12N": "미생물·효소·유전공학", "C12Q": "효소·핵산·미생물 측정",
    "C12R": "미생물 분류(부가)", "C08L": "고분자 조성물", "C08B": "다당류", "C08K": "고분자 배합제", "C08J": "고분자 가공",
    "C09D": "코팅 조성물", "C11D": "세제", "C02F": "수처리", "B01J": "화학·물리 공정(촉매 등)", "D01F": "인조섬유 제조",
    "G01N": "물질 분석·검사", "G01K": "온도 측정", "G01D": "측정 일반", "G06Q": "비즈니스 데이터 처리", "G06N": "AI·계산 모델",
    "G06F": "디지털 데이터 처리", "G08B": "경보 시스템", "G16H": "헬스케어 정보학", "G16B": "생물정보학", "G16C": "화학정보학",
}

def ipc_text(value):
    c = ipc_summary(value)
    return ", ".join(f"{k}({IPC_KO.get(k, '?')})" + (f"×{v}" if v > 1 else "") for k, v in c.most_common())

# 방법 2: 형태소 + TF-IDF
kiwi = Kiwi()
STOP = {"발명", "방법", "제공", "포함", "이용", "관련", "단계", "특징", "효과", "구성", "사용", "대상", "상기", "본원",
        "제조", "조성물", "용도", "개선", "가능", "기반", "이상", "경우", "대한", "통한", "위한", "또한", "따른", "형성", "유효",
        "성분", "함유", "kccm", "kctc", "kacc", "수탁", "번호", "기탁", "수행", "적어도", "하나", "제1", "제2", "실시", "예방", "치료", "목적", "기술", "분야", "동시", "시간", "제조방법", "신규"}

NOUN_TAGS = ("NNG", "NNP", "SL", "SN")

def nouns(text):
    """붙어 있는 명사 조각은 하나로 합침 (예: 루드+위기 → 루드위기, 마이크로+니들 → 마이크로니들)"""
    words, cur, end = [], "", -1
    for t in kiwi.tokenize(str(text)):
        if t.tag in NOUN_TAGS and t.start == end and cur:      # 앞 명사와 띄어쓰기 없이 붙어 있으면 이어 붙임
            cur += t.form
        else:
            if cur: words.append(cur)
            cur = t.form if t.tag in NOUN_TAGS else ""
        end = t.start + t.len
    if cur: words.append(cur)
    words = [w[:-1] if len(w) >= 4 and w.endswith("의") else w for w in words]   # "프로바이오틱스의" → "프로바이오틱스"
    return [w for w in words if len(w) > 1 and w.lower() not in STOP and not re.fullmatch(r"[\d\W]+", w)]

text = (df["발명의명칭"].fillna("") + " ") * 2 + df["요약"].fillna("")   # 명칭은 2번 넣어 가중치
vec = TfidfVectorizer(tokenizer=nouns, lowercase=False, token_pattern=None, ngram_range=(1, 1), sublinear_tf=True)
X = vec.fit_transform(text)            # 30건 전체로 학습 (IDF를 안정적으로)
words = vec.get_feature_names_out()

def top_k(i, k=6):
    row = X[i].toarray()[0]
    picked = []
    for j in row.argsort()[::-1]:
        if row[j] <= 0 or len(picked) >= k: break
        w = words[j]
        if any(w in p or p in w for p in picked):     # "마이크로니들"과 "마이크로니들 제조" 같은 겹침 제외
            continue
        picked.append(w)
    return ", ".join(picked)

s = df.head(10).copy()
s["분류(IPC 서브클래스)"] = s["IPC분류"].map(ipc_text)
s["키워드(TF-IDF)"] = [top_k(i) for i in range(10)]
out = s[["발명의명칭", "출원일자", "출원인", "IPC분류", "분류(IPC 서브클래스)", "키워드(TF-IDF)", "요약"]]
out.to_excel("kipris_키워드_샘플10.xlsx", index=False)
for _, r in out.iterrows():
    print("■", r["발명의명칭"][:45]); print("  분류:", r["분류(IPC 서브클래스)"]); print("  키워드:", r["키워드(TF-IDF)"])
