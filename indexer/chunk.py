"""
뉴스 본문 청킹 (문장 단위)

[규칙]
  - 본문이 SINGLE_MAX(1,200자) 이하 → 자르지 않고 청크 1개
  - 그보다 길면 문장 단위로 모아 CHUNK_MAX(1,000자) 이하 청크로 자름
  - 다음 청크 앞에 이전 청크의 마지막 문장들을 OVERLAP_MAX(150자) 이내로 겹침
  - 마지막 청크가 MIN_TAIL(300자)보다 짧으면 앞 청크에 합침
  - 문장 하나가 CHUNK_MAX보다 길면 공백 위치에서 강제로 나눔
  - 임베딩텍스트 = "[제목] ...\n" + 청크본문  (잘린 조각만 봐도 어떤 기사인지 알 수 있게)

[출력 칼럼]
  청크ID, 기사ID, 청크번호, 청크수, 제목, 청크본문, 임베딩텍스트, 글자수,
  원문URL, 발행일, 매체명, 수집키워드

[실행]  python -m indexer chunk   (project3 폴더에서)
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

# ============================== 설정 ==============================
import config

BASE_DIR = config.ROOT

INPUT_FILE = config.NEWS_RAW
OUTPUT_FILE = config.NEWS_CHUNKS

SINGLE_MAX = 1200   # 이 길이 이하 기사는 자르지 않음
CHUNK_MAX = 1000    # 청크 최대 길이
OVERLAP_MAX = 150   # 겹침 최대 길이
MIN_TAIL = 300      # 마지막 청크 최소 길이 (짧으면 앞 청크에 합침)
# ==================================================================

SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|(?<=다\.)(?=\S)")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s and s.strip()]
    out = []
    for s in sents:  # 너무 긴 문장은 공백 위치에서 나눔
        while len(s) > CHUNK_MAX:
            cut = s.rfind(" ", 0, CHUNK_MAX)
            cut = cut if cut > CHUNK_MAX // 2 else CHUNK_MAX
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def _overlap(sents: list[str]) -> list[str]:
    """이전 청크 끝에서 OVERLAP_MAX 이내의 문장들을 가져옴"""
    picked, total = [], 0
    for s in reversed(sents):
        if total + len(s) + 1 > OVERLAP_MAX:
            break
        picked.insert(0, s)
        total += len(s) + 1
    return picked


def chunk_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= SINGLE_MAX:
        return [text] if text else []

    sents = split_sentences(text)
    chunks: list[list[str]] = []
    cur: list[str] = []
    new_in_cur = 0  # 겹침이 아닌 새 문장 수
    for s in sents:
        cur_len = len(" ".join(cur))
        if cur and new_in_cur > 0 and cur_len + 1 + len(s) > CHUNK_MAX:
            chunks.append(cur)
            cur = _overlap(cur)
            new_in_cur = 0
            # 겹침 + 새 문장이 한도를 넘으면 겹침을 버림
            if len(" ".join(cur)) + 1 + len(s) > CHUNK_MAX:
                cur = []
        cur.append(s)
        new_in_cur += 1
    if cur and new_in_cur > 0:
        chunks.append(cur)

    texts = [" ".join(c) for c in chunks]
    # 마지막 청크가 너무 짧으면 앞 청크에 합침 (겹친 문장은 중복 제거)
    if len(texts) >= 2 and len(texts[-1]) < MIN_TAIL:
        prev, last = chunks[-2], chunks[-1]
        merged = prev + last[_count_overlap(prev, last):]
        texts = texts[:-2] + [" ".join(merged)]
    return texts


def _count_overlap(prev: list[str], last: list[str]) -> int:
    """last의 앞부분 중 prev의 끝부분과 겹치는 문장 수"""
    for n in range(min(len(prev), len(last)), 0, -1):
        if prev[-n:] == last[:n]:
            return n
    return 0


def article_id(url: str, title: str, date: str) -> str:
    return hashlib.sha1(f"{url}|{title}|{date}".encode("utf-8")).hexdigest()[:12]


def run():
    df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows = []
    for r in df.itertuples(index=False):
        aid = article_id(r.원문URL, r.제목, r.발행일)
        parts = chunk_text(r.본문전문)
        for i, c in enumerate(parts, 1):
            rows.append({
                "청크ID": f"{aid}-{i:02d}", "기사ID": aid, "청크번호": i, "청크수": len(parts),
                "제목": r.제목, "청크본문": c, "임베딩텍스트": f"[제목] {r.제목}\n{c}", "글자수": len(c),
                "원문URL": r.원문URL, "발행일": r.발행일, "매체명": r.매체명, "수집키워드": r.수집키워드,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    n_art = out["기사ID"].nunique()
    print(f"기사 {len(df):,}건 → 청크 {len(out):,}개 (기사ID {n_art:,}개)")
    print(f"자르지 않은 기사: {(out.groupby('기사ID')['청크수'].first() == 1).sum():,}건")
    print("청크 길이:", out["글자수"].describe(percentiles=[.1, .5, .9, .99]).round().to_dict())
    print(f"저장: {OUTPUT_FILE.name}")
    return out


if __name__ == "__main__":
    run()
