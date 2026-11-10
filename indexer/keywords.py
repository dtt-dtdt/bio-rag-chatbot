"""
특허 키워드 칼럼 생성 (규칙 + LLM 조합)

[만드는 칼럼]
  - 규칙후보태그 : IPC·CPC 코드로 뽑은 분야태그 후보 (검토용)
  - 분야태그     : LLM이 요약을 읽고 TAG_LIST 안에서 고른 1~3개 (없으면 '기타')
  - 핵심키워드   : LLM이 요약에서 뽑은 3~5개 → 코드로 다시 검사한 결과
  - 제외된키워드 : 코드 검사에서 걸러진 것과 그 이유 (검토용)

[키워드 규칙]  LLM 지시문에 넣고 + 코드로 한 번 더 검사
  1. 일반 단어 제외 : '발명', '방법', '조성물', '효과' 등 (GENERIC_WORDS)
  2. 식별 코드 제외 : 균주 번호(NK357 등), 수탁번호(KCTC 12345BP 등)
  3. 추측 금지     : 명칭·요약에 실제로 나온 단어로만 구성된 키워드만 남김

[실행 방법 - Jupyter]
  1) pip install openai pandas python-dotenv
  2) .env 파일에 OPENAI_API_KEY=sk-... 추가 (노트북과 같은 폴더)
  3) 아래 '설정'에서 INPUT_FILES 경로 확인 → 이 셀 전체 실행
  4) 처음에는 SAMPLE_N = 20 으로 결과 확인 → 괜찮으면 None 으로 바꿔 전체 실행

[중간 저장]
  결과는 1건마다 cache 파일(JSONL)에 저장됩니다. 중간에 멈춰도 다시 실행하면
  이미 처리한 출원번호는 건너뛰고 이어서 진행합니다(API 비용 중복 없음).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ============================== 설정 ==============================
import config

BASE_DIR = config.ROOT

INPUT_FILES = {  # 파일 경로 : 출력 파일 이름
    config.PATENT_PHARMA: config.KEYWORD_PHARMA.name,
    config.PATENT_HEALTHCARE: config.KEYWORD_HEALTHCARE.name,
}
SAMPLE_N = 20            # 시험용 건수. 전체 실행은 None
MODEL = "gpt-4o-mini"    # 사용할 OpenAI 모델
MAX_WORKERS = 4          # 동시 호출 수 (요청 한도 오류가 나면 줄이기)
CACHE_FILE = config.PROCESSED_DIR / "keyword_cache.jsonl"

TAG_LIST = [
    "항체", "세포치료", "유전자치료", "단백질·펩타이드", "백신", "마이크로바이옴",
    "저분자", "제형·전달", "진단·바이오마커", "디지털헬스·AI", "의료기기",
    "생산·정제공정", "품질관리", "기타",
]

# (태그, 포함할 코드 앞부분, 제외할 코드 앞부분)  ※ 제외 코드 앞에 '='를 붙이면 정확히 그 코드만 제외
TAG_RULES = [
    ("항체",            ("C07K16", "A61K39/395", "A61K47/68"), ()),   # A61K47/68 = 항체-약물 접합체
    ("세포치료",        ("A61K35/1", "A61K35/2", "A61K35/3", "A61K35/4", "A61K35/5", "C12N5"), ()),
    ("유전자치료",      ("A61K48", "C12N15/86"), ()),
    ("단백질·펩타이드", ("C07K14", "C07K19", "A61K38", "C07K7", "C07K5"), ()),
    ("백신",            ("A61K39",), ("A61K39/395", "=A61K39/00")),   # A61K39/00 = 항원·항체 포괄 코드
    ("마이크로바이옴",  ("A61K35/74", "C12N1/20", "C12R"), ()),
    ("저분자",          ("A61K31", "C07D", "C07C"), ()),
    ("제형·전달",       ("A61K9", "A61K47"), ("A61K47/68",)),
    ("진단·바이오마커", ("C12Q1/68", "G01N33"), ()),
    ("디지털헬스·AI",   ("G16H", "G16B", "G06N"), ()),
    ("의료기기",        ("A61B", "A61M", "A61N", "A61F"), ()),
    ("생산·정제공정",   ("C12M", "C12P21", "C07K1/", "B01D15"), ()),
]

# 규칙 1: 일반 단어 (키워드가 이 단어들'만'으로 이루어져 있으면 제외)
GENERIC_WORDS = {
    "발명", "본발명", "방법", "조성물", "효과", "용도", "제조방법", "제조", "장치", "시스템",
    "유효성분", "약학적", "약학", "조성", "건강기능식품", "식품", "화장료", "예방", "개선",
    "치료", "치료용", "단계", "포함", "제공", "특징", "이용", "상기", "기술", "또는", "및",
}

# 규칙 2: 식별 코드
ACCESSION_RE = re.compile(r"(KCTC|KCCM|KFCC|KACC|ATCC|DSM|NRRL|CGMCC|NCIMB)\s*-?\s*\d+[A-Z]*", re.I)
CODE_TOKEN_RE = re.compile(r"^[A-Za-z]{1,6}[-_]?[A-Za-z]{0,3}\d+[A-Za-z0-9-]*$")  # NK357, MG5159, OPT-A1 ...
STRAIN_CONTEXT = ("균주", "수탁", "기탁", "strain", "바실러스", "박테리움", "코커스", "마이세스",
                  "박터", "스트렙토", "락토", "비피도", "효모", "KCTC", "KCCM", "KFCC", "KACC", "ATCC")
# ==================================================================


# ---------------------------- 규칙 단계 ----------------------------
def parse_codes(s: str) -> list[str]:
    """'C07K 16/28(2006.01)|A61K 39/395(...)' → ['C07K16/28', 'A61K39/395']"""
    return [re.sub(r"\(.*?\)", "", c).replace(" ", "") for c in str(s).split("|") if c.strip()]


def rule_tags(ipc: str, cpc: str) -> list[str]:
    codes = parse_codes(ipc) + parse_codes(cpc)
    out = []
    for tag, inc, exc in TAG_RULES:
        exact = {e[1:] for e in exc if e.startswith("=")}
        prefix = tuple(e for e in exc if not e.startswith("="))
        for c in codes:
            excluded = c in exact or (prefix and c.startswith(prefix))
            if c.startswith(inc) and not excluded:
                out.append(tag)
                break
    return out


# ---------------------------- LLM 단계 ----------------------------
SYSTEM_PROMPT = f"""너는 바이오 특허 분류 전문가다. 주어진 특허의 명칭과 요약만 보고 JSON으로 답한다.

[분야태그]
- 아래 목록에서만 1~3개를 고른다: {", ".join(TAG_LIST)}
- '규칙후보태그'는 IPC/CPC 코드에서 나온 힌트다. 요약 내용과 맞는 것만 채택하고, 맞지 않으면 버린다.
- 바이오의약품·바이오헬스케어와 관련 없는 특허(화장품, 포장재, 일반 전자회로 등)는 ["기타"].

[핵심키워드] 3~5개
- 요약(또는 명칭)에 실제로 나온 표현을 그대로 사용한다. 요약에 없는 내용은 절대 추가하지 않는다.
- '발명', '방법', '조성물', '효과', '용도', '유효성분', '예방', '치료' 같은 일반 단어는 단독 키워드로 쓰지 않는다.
- 균주 번호(예: NK357, MG5159)와 수탁번호(예: KCTC 12345BP)는 넣지 않는다. 균주는 종 이름만 쓴다.
- 유전자·단백질·약물 이름(예: CD3, HER2, FL118)은 기술 내용이므로 넣어도 된다.
- 각 키워드는 2~15자 내외의 명사구로 쓴다.

출력 형식: {{"tags": ["..."], "keywords": ["..."]}}"""


def make_client():
    from openai import OpenAI
    try:
        from dotenv import load_dotenv
        load_dotenv(config.ENV_FILE)
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY가 없습니다. project3 폴더의 .env 파일에 넣어주세요: {config.ENV_FILE}")
    return OpenAI()


def call_llm(client, title: str, abstract: str, candidates: list[str]) -> tuple[dict, int]:
    """LLM 호출. (결과 dict, 사용 토큰 수) 반환. 실패 시 3번까지 재시도."""
    user = json.dumps({"명칭": title, "요약": abstract, "규칙후보태그": candidates}, ensure_ascii=False)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
            )
            return json.loads(r.choices[0].message.content), r.usage.total_tokens
        except Exception as e:  # 요청 한도·일시 오류 → 잠시 후 재시도
            if attempt == 2:
                raise
            time.sleep(2 ** attempt * 3)
    return {}, 0


# ------------------------ 코드 검사 (규칙 1~3) ------------------------
def _norm(s: str) -> str:
    return re.sub(r"[\s·・,\-‐()\[\]'\"/]", "", str(s)).lower()


def _is_strain_code(token: str, source: str) -> bool:
    """코드 모양 토큰이 원문에서 균주·수탁 문맥에 쓰였는지 확인"""
    if not CODE_TOKEN_RE.match(token):
        return False
    for m in re.finditer(re.escape(token), source):
        ctx = source[max(0, m.start() - 20): m.end() + 20]
        if any(w.lower() in ctx.lower() for w in STRAIN_CONTEXT):
            return True
    return False


def clean_keywords(keywords: list, title: str, abstract: str) -> tuple[list[str], list[str]]:
    """(통과한 키워드, '키워드(이유)' 목록) 반환"""
    source = f"{title} {abstract}"
    source_n = _norm(source)
    kept, dropped = [], []
    for kw in keywords or []:
        kw = str(kw).strip()
        if not kw:
            continue
        # 규칙 2-a: 수탁번호는 키워드에서 지움
        if ACCESSION_RE.search(kw):
            kw2 = ACCESSION_RE.sub("", kw).strip(" ,()")
            if not kw2:
                dropped.append(f"{kw}(수탁번호)")
                continue
            kw = kw2
        # 규칙 2-b: 균주 번호 토큰은 지움 (예: '락토바실러스 플란타룸 MG5159' → '락토바실러스 플란타룸')
        tokens = [t for t in re.split(r"\s+", kw) if t]
        code_tokens = [t for t in tokens if _is_strain_code(t, source)]
        if code_tokens:
            tokens = [t for t in tokens if t not in code_tokens]
            if not tokens:
                dropped.append(f"{kw}(균주번호)")
                continue
            kw = " ".join(tokens)
        # 규칙 1: 일반 단어로만 구성
        parts = [p for p in re.split(r"[\s·,/]+", kw) if p]
        if all(_norm(p) in GENERIC_WORDS for p in parts):
            dropped.append(f"{kw}(일반단어)")
            continue
        # 규칙 3: 원문에 없는 단어가 있으면 제외 (2글자 이상 단어 기준)
        missing = [p for p in parts if len(_norm(p)) >= 2 and _norm(p) not in source_n]
        if missing:
            dropped.append(f"{kw}(원문에없음:{'/'.join(missing)})")
            continue
        if kw not in kept:
            kept.append(kw)
    return kept, dropped


def clean_tags(tags: list) -> list[str]:
    out = [t for t in (tags or []) if t in TAG_LIST and t != "기타"][:3]
    return out or ["기타"]


# ---------------------------- 실행 ----------------------------
_lock = threading.Lock()


def load_cache() -> dict:
    cache = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["출원번호"]] = rec
    return cache


def process_row(client, row) -> dict:
    cand = rule_tags(row["IPC분류"], row["CPC분류"])
    result, tokens = call_llm(client, row["발명의명칭"], row["요약"], cand)
    kws, dropped = clean_keywords(result.get("keywords", []), row["발명의명칭"], row["요약"])
    rec = {
        "출원번호": row["출원번호"],
        "규칙후보태그": ", ".join(cand),
        "분야태그": ", ".join(clean_tags(result.get("tags", []))),
        "핵심키워드": ", ".join(kws),
        "제외된키워드": ", ".join(dropped),
        "_tokens": tokens,
    }
    with _lock, CACHE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def run():
    client = make_client()
    cache = load_cache()
    total_tokens = 0
    for in_path, out_name in INPUT_FILES.items():
        df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
        if SAMPLE_N:
            df = df.sample(min(SAMPLE_N, len(df)), random_state=42)
        todo = df[~df["출원번호"].isin(cache)]
        print(f"\n[{Path(in_path).name}] 대상 {len(df):,}건 / 이미 처리 {len(df) - len(todo):,}건 / 새로 처리 {len(todo):,}건")

        done, failed = 0, []
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            futures = {ex.submit(process_row, client, r): r["출원번호"] for _, r in todo.iterrows()}
            for fu in as_completed(futures):
                try:
                    rec = fu.result()
                    cache[rec["출원번호"]] = rec
                    total_tokens += rec["_tokens"]
                except Exception as e:
                    failed.append((futures[fu], type(e).__name__))
                done += 1
                if done % 100 == 0 or done == len(todo):
                    print(f"  진행 {done:,}/{len(todo):,}  (누적 토큰 {total_tokens:,})")

        cols = ["규칙후보태그", "분야태그", "핵심키워드", "제외된키워드"]
        add = pd.DataFrame([{k: cache[a][k] for k in ["출원번호", *cols]}
                            for a in df["출원번호"] if a in cache])
        out = df.drop(columns=[c for c in cols if c in df.columns]).merge(add, on="출원번호", how="left")
        out_path = config.PROCESSED_DIR / (f"SAMPLE_{out_name}" if SAMPLE_N else out_name)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  저장: {out_path.name}")
        if failed:
            print(f"  실패 {len(failed)}건 (다시 실행하면 이어서 처리): {failed[:5]}")

    print(f"\n이번 실행 사용 토큰: {total_tokens:,}")


if __name__ == "__main__":
    run()
