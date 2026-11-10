"""
평가 개선 v3 — 엄격한 정답급 판정 + 질문 품질 판정 (eval/harness.py 실행 결과를 재사용)

[v2 → v3 변경]
  - 판정 모델: gpt-4o-mini → gpt-6-luna (JUDGE_MODEL)
  - 근거 인용 의무화: 답이 되는 문장을 문서에서 '그대로' 인용해야 true.
    → 코드가 인용문이 실제 문서에 있는지 다시 확인. 없으면 false (지어낸 인용 차단)
  - 핵심 요소 확인: 질문의 주체·시점·대상을 먼저 뽑고, 문서에 '모두' 있을 때만 true
  - 특허 전용 기준: 질문이 묘사한 기술의 대상·기능·용도가 모두 일치해야 true
  - 질문 품질 판정도 같은 방식(핵심 요소 추출)으로 엄격하게
  - v2 결과(eval_results_v2.csv)가 있으면 나란히 비교

[재사용]  eval/eval_questions.json, eval/question_vecs.json, chroma_eval, chroma_db  (질문·임베딩 비용 없음)
[결과]    eval/eval_report_v3.md, eval_results_v3.csv, eval_judgments_v3.csv, eval_question_quality_v3.csv
[실행]    셀에 붙여 실행. 약 10~20분(추론형 모델이라 v2보다 느림). 중간에 멈춰도 이어서 진행
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# ============================== 설정 ==============================
import config

BASE_DIR = config.ROOT

RAW_NEWS_FILE = config.NEWS_RAW
MAIN_DB_DIR = config.CHROMA_DIR
EVAL_DB_DIR = config.CHROMA_EVAL_DIR
OUT_DIR = config.EVAL_RESULTS_DIR
JUDGE_MODEL = "gpt-6-luna"
VERSION = "v3"
K = 3
WORKERS = 6
MAX_RETRY = 30
JUDGE_CHARS = 3000

CONFIGS = {  # eval/harness.py와 같아야 함
    "c1_none": "① 자르지 않음",
    "c2_500": "② 500자/겹침50",
    "c3_1000": "③ 1000자/겹침150 (현재)",
    "c4_1500": "④ 1500자/겹침200",
}
BOILERPLATE_RE = re.compile(r"\s*(기사를 더 보고 싶다면\??|기사 섹션 분류 안내)\s*")
# ==================================================================

_clients: dict = {}
_unsupported: set[str] = set()   # 모델이 지원하지 않는 설정 (자동으로 빼고 재시도)
_lock = threading.Lock()


def get_clients():
    if not _clients:
        from openai import OpenAI
        import chromadb
        try:
            from dotenv import load_dotenv
            load_dotenv(config.ENV_FILE)
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"OPENAI_API_KEY가 없습니다. project3 폴더의 .env 파일에 넣어주세요: {config.ENV_FILE}")
        _clients.update(openai=OpenAI(), eval=chromadb.PersistentClient(path=str(EVAL_DB_DIR)),
                        main=chromadb.PersistentClient(path=str(MAIN_DB_DIR)))
    return _clients


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f￾￿]")


def _clean_text(s: str) -> str:
    """요청을 깨뜨릴 수 있는 제어 문자·깨진 글자(짝 없는 서로게이트) 제거"""
    s = _CTRL_RE.sub(" ", str(s))
    return s.encode("utf-8", "ignore").decode("utf-8", "ignore")


def chat_json(prompt: str) -> dict:
    oa = get_clients()["openai"]
    prompt = _clean_text(prompt)
    attempt, read_errors = 0, 0
    while attempt < MAX_RETRY:
        params = dict(model=JUDGE_MODEL, timeout=120,
                      messages=[{"role": "system", "content": "Respond only with a valid JSON object."},
                                {"role": "user", "content": prompt}])
        if "temperature" not in _unsupported:
            params["temperature"] = 0
        if "response_format" not in _unsupported:
            params["response_format"] = {"type": "json_object"}
        try:
            r = oa.chat.completions.create(**params)
            return _parse_json(r.choices[0].message.content)
        except Exception as e:
            status, msg = getattr(e, "status_code", None), str(e)
            if status == 400:  # 지원하지 않는 설정이면 빼고 다시
                bad = next((p for p in ("temperature", "response_format") if p in msg and p not in _unsupported), None)
                if bad:
                    with _lock:
                        if bad not in _unsupported:
                            _unsupported.add(bad)
                            print(f"    ℹ️ {JUDGE_MODEL}은(는) '{bad}' 설정을 지원하지 않아 빼고 진행해요")
                    continue
                if "reading your request" in msg and read_errors < 3:   # 일시적 오류일 수 있어 몇 번 더
                    read_errors += 1
                    time.sleep(3 * read_errors)
                    continue
            if (status is not None and status != 429 and status < 500) or attempt == MAX_RETRY - 1:
                print(f"    ❌ GPT 호출 실패: {type(e).__name__}: {msg[:200]}")
                if status == 404:
                    print(f"       모델 이름('{JUDGE_MODEL}')을 확인하거나, 이 키로 쓸 수 있는 모델인지 확인해 주세요.")
                raise
            m = re.search(r"try again in ([\d.]+)\s*(ms|s)", msg)
            wait = max((float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 1) if m else 0,
                       min(60, 3 * 2 ** attempt))
            print(f"    재시도 {attempt + 1} ({type(e).__name__}) {wait:.0f}초 대기")
            time.sleep(wait)
            attempt += 1
    return {}


def run_cached(name: str, tasks: dict, fn, label: str) -> dict:
    """tasks = {키: 입력}.  fn(입력) → 결과 dict.  jsonl에 하나씩 저장 → 다시 실행하면 이어서"""
    path = OUT_DIR / f"{name}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec["out"]
    todo = [k for k in tasks if k not in done]
    print(f"\n[{label}] 전체 {len(tasks):,} / 이미 판정 {len(tasks) - len(todo):,} / 새로 판정 {len(todo):,}")
    failed = []

    def safe(k):   # 한 건이 실패해도 나머지는 계속. 실패한 건은 저장 안 함 → 다시 실행하면 재시도
        try:
            return k, fn(tasks[k])
        except Exception as e:
            if getattr(e, "status_code", None) in (401, 403, 404):   # 키·권한·모델 문제는 전체 중단
                raise
            return k, None

    lock = threading.Lock()
    if todo:
        print("  연결 확인 중 (첫 판정 1개)…")
        k, first = safe(todo[0])          # 키·모델 문제면 여기서 멈춤
        if first is None:
            failed.append(k)
            print("  ⚠️ 첫 판정이 실패했지만 키·모델 문제는 아니라서 계속 진행해요")
        else:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "out": first}, ensure_ascii=False) + "\n")
            done[k] = first
            print(f"  ✅ 연결 정상 ({JUDGE_MODEL})")

    with path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        for n, (k, out) in enumerate(ex.map(safe, todo[1:]), 2):
            with lock:
                if out is None:
                    failed.append(k)
                else:
                    f.write(json.dumps({"key": k, "out": out}, ensure_ascii=False) + "\n")
                    f.flush()
                    done[k] = out
            if n % 50 == 0 or n == len(todo):
                print(f"  진행 {n:,}/{len(todo):,}" + (f" (건너뜀 {len(failed)}건)" if failed else ""))
    if failed:
        print(f"  ⚠️ 판정 실패 {len(failed)}건은 '판정 안 됨'(정답급 아님)으로 계산돼요. 다시 실행하면 이 건들만 재시도해요.")
    return done


# ---------------------------- 판정 지시문 ----------------------------
NEWS_PROMPT = """너는 검색 결과를 엄격하게 채점하는 평가자다.
[질문] {q}
[문서 제목] {title}
[문서 내용] {text}

다음 순서로 판단한다.
1. 질문의 핵심 요소를 뽑는다: 주체(회사·인물·기관·행사·제품), 시점(연도·날짜), 묻는 대상(무엇을 묻는가).
   질문에 없는 요소는 뽑지 않는다.
2. 각 요소가 문서에 '같은 것'으로 있는지 확인한다.
   - 다른 회사·인물·기관이면 불일치. (예: 거래소 ≠ 롯데그룹, 김영주 부의장 발언 ≠ 다른 사람 발언)
   - 다른 연도·회차의 행사면 불일치. (예: 2023 메디엑스포 ≠ 2024 메디엑스포)
3. 질문의 답이 되는 문장을 문서에서 '한 글자도 바꾸지 말고' 그대로 인용한다. 없으면 빈 문자열.
4. 모든 요소가 일치하고 인용이 있을 때만 answerable=true. 주제만 비슷하면 false.

JSON으로 출력:
{{"elements": ["요소1", "요소2"], "all_matched": true/false, "quote": "문서에서 그대로 인용한 문장", "answerable": true/false, "reason": "한 문장"}}"""

PATENT_PROMPT = """너는 특허 검색 결과를 엄격하게 채점하는 평가자다.
[질문] {q}
[특허 명칭] {title}
[특허 요약] {text}

다음 순서로 판단한다.
1. 질문이 찾는 기술의 핵심 요소를 뽑는다: 대상(무엇에 관한 것인가), 기능·방법(어떻게), 용도(무엇을 위해).
2. 이 특허가 그 요소를 '모두' 같은 의미로 다루는지 확인한다.
   - 대상이 다르면 불일치. (예: 피부 약물전달용 하이드로겔 ≠ 항암용 하이드로겔, 유방암 ≠ 암 일반)
   - 일부만 겹치거나 '관련 있음' 수준이면 불일치.
3. 일치를 보여주는 문장을 요약에서 '한 글자도 바꾸지 말고' 그대로 인용한다. 없으면 빈 문자열.
4. 모든 요소가 일치하고 인용이 있을 때만 answerable=true.

JSON으로 출력:
{{"elements": ["요소1", "요소2"], "all_matched": true/false, "quote": "요약에서 그대로 인용한 문장", "answerable": true/false, "reason": "한 문장"}}"""

QUALITY_PROMPT = """너는 검색 평가용 질문의 품질을 엄격하게 검사하는 사람이다.
[질문] {q}
[정답 문서 제목] {title}
[정답 문서 앞부분] {text}

1) specific: 정답 문서를 보지 않고 '질문만' 읽었을 때, 무엇을 찾는지 하나로 특정할 수 있는가?
   - 질문 안에 고유한 주체(회사명·인물명·기관명·행사명·제품명·지명)나 시점이 하나 이상 있어야 true.
   - "새로 건립된 체육관은 어떤 용도인가요?" → false (누구의, 어디 체육관인지 질문에 없음)
   - "이 회사의", "해당 기술은"처럼 문서를 봐야 알 수 있는 지시어가 있으면 false.
   - 특허 질문은 기술의 대상과 용도가 구체적으로 적혀 있으면 true.
2) bio_related: 정답 문서의 '주된 내용'이 바이오의약품·제약·바이오헬스케어·디지털헬스·의료 산업인가?
   - 스폰서나 참여 기업만 헬스케어 회사이고 주된 내용이 다른 분야면 false.
JSON으로 출력: {{"named_subject": "질문 속 주체(없으면 빈 문자열)", "specific": true/false, "bio_related": true/false, "reason": "한 문장"}}"""


def _norm(s: str) -> str:
    return re.sub(r"[\s\"'“”‘’.,·…\-]", "", str(s))


def judge_relevance(x) -> dict:
    """GPT 판정 + 인용문이 실제 문서에 있는지 코드로 재확인"""
    question, kind, title, text = x
    prompt = (NEWS_PROMPT if kind == "news" else PATENT_PROMPT).format(q=question, title=title, text=text)
    out = chat_json(prompt)
    quote = str(out.get("quote", "")).strip()
    source = _clean_text(f"{title} {text}")   # GPT에 보낸 것과 같은(정리된) 글로 비교
    quote_ok = bool(quote) and len(_norm(quote)) >= 8 and _norm(quote) in _norm(source)
    out["quote_verified"] = quote_ok
    out["final"] = bool(out.get("answerable")) and bool(out.get("all_matched")) and quote_ok
    return out


# ---------------------------- 검색 ----------------------------
def ranked_news(col, vec, n=40):
    r = col.query(query_embeddings=[vec], n_results=n, include=["metadatas"])
    seen = []
    for m in r["metadatas"][0]:
        if m["기사ID"] not in seen:
            seen.append(m["기사ID"])
    return seen


def ranked_patents(col, vec, n=20):
    r = col.query(query_embeddings=[vec], n_results=n, include=["metadatas", "documents"])
    seen_titles, out = [], []
    for _id, m, d in zip(r["ids"][0], r["metadatas"][0], r["documents"][0]):
        t = m.get("발명의명칭", _id)
        if t not in seen_titles:
            seen_titles.append(t)
            out.append((_id, t, d.split("\n핵심키워드:")[0]))   # 판정은 명칭+요약만 (키워드 줄 제외)
    return out


def run():
    t0 = time.time()
    qs = json.loads((config.EVAL_SET_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    qmap = {q["qid"]: q for q in qs}
    vecs = json.loads((OUT_DIR / "question_vecs.json").read_text())
    news_q = [q for q in qs if q["종류"] == "뉴스"]
    pat_q = [q for q in qs if q["종류"] == "특허"]
    c = get_clients()

    raw = pd.read_csv(RAW_NEWS_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    raw["기사ID"] = [hashlib.sha1(f"{u}|{t}|{d}".encode("utf-8")).hexdigest()[:12]
                    for u, t, d in zip(raw["원문URL"], raw["제목"], raw["발행일"])]
    art = {a: (t, re.sub(r"\s+", " ", BOILERPLATE_RE.sub(" ", b)).strip()[:JUDGE_CHARS])
           for a, t, b in zip(raw["기사ID"], raw["제목"], raw["본문전문"])}

    # 1) 설정별 검색 결과 (API 비용 없음)
    settings = {label: ("news", c["eval"].get_collection(name)) for name, label in CONFIGS.items()}
    settings["본 DB 전체 - 뉴스"] = ("news", c["main"].get_collection("news"))
    settings["본 DB 전체 - 특허"] = ("patent", c["main"].get_collection("patents"))
    ranks, rel_tasks, pat_docs = {}, {}, {}
    print("검색 결과 수집 중…")
    for label, (kind, col) in settings.items():
        for q in (news_q if kind == "news" else pat_q):
            v = vecs[q["qid"]]
            if kind == "news":
                order = ranked_news(col, v)
                gold_rank = order.index(q["정답"]) + 1 if q["정답"] in order else None
                top = order[:K]
            else:
                order = ranked_patents(col, v)
                gold_rank = next((i + 1 for i, (pid, t, _) in enumerate(order)
                                  if pid == q["정답"] or t == q["정답제목"]), None)
                top = [pid for pid, _, _ in order[:K]]
                for pid, t, d in order[:K]:
                    pat_docs[pid] = (t, d)
            ranks[(label, q["qid"])] = (gold_rank, top)
            for i, doc in enumerate(top, 1):
                if gold_rank == i:
                    continue
                title, text = art.get(doc, ("", "")) if kind == "news" else pat_docs[doc]
                rel_tasks[f"{q['qid']}|{doc}"] = (q["질문"], kind, title, text)

    # 2) 정답급 판정 (엄격)
    rel = run_cached(f"judge_relevance_{VERSION}", rel_tasks, judge_relevance, "정답급 판정 (엄격)")

    # 3) 질문 품질 판정 (엄격)
    def gold_text(q):
        if q["종류"] == "뉴스":
            return art.get(q["정답"], (q["정답제목"], ""))[1][:800]
        d = c["main"].get_collection("patents").get(ids=[q["정답"]], include=["documents"])["documents"]
        return (d[0].split("\n핵심키워드:")[0] if d else "")[:800]
    qual = run_cached(f"judge_quality_{VERSION}", {q["qid"]: q for q in qs},
                      lambda q: chat_json(QUALITY_PROMPT.format(q=q["질문"], title=q["정답제목"], text=gold_text(q))),
                      "질문 품질 판정 (엄격)")
    good = {qid for qid, o in qual.items() if o.get("specific") and o.get("bio_related")}

    # 4) 점수
    rows = []
    for label, (kind, col) in settings.items():
        qlist = news_q if kind == "news" else pat_q
        for subset_name, subset in (("전체 질문", qlist), ("좋은 질문만", [q for q in qlist if q["qid"] in good])):
            if not subset:
                continue
            strict = lenient = mrr = 0
            for q in subset:
                gold_rank, top = ranks[(label, q["qid"])]
                ok = [i for i, d in enumerate(top, 1) if gold_rank == i or rel.get(f"{q['qid']}|{d}", {}).get("final")]
                strict += gold_rank is not None and gold_rank <= K
                lenient += bool(ok)
                mrr += 1 / ok[0] if ok else 0
            n = len(subset)
            rows.append({"설정": label, "대상": "뉴스" if kind == "news" else "특허", "질문": subset_name, "질문수": n,
                         "Hit@3 (엄격)": strict / n, f"Hit@3 (정답급 {VERSION})": lenient / n,
                         f"MRR@3 (정답급 {VERSION})": mrr / n})
    res = pd.DataFrame(rows)

    # v2와 비교 (있으면)
    v2_path = OUT_DIR / "eval_results_v2.csv"
    if v2_path.exists():
        v2 = pd.read_csv(v2_path)[["설정", "질문", "Hit@3 (정답급 포함)"]].rename(
            columns={"Hit@3 (정답급 포함)": "Hit@3 (정답급 v2, 참고)"})
        res = res.merge(v2, on=["설정", "질문"], how="left")

    # 저장
    judg = []
    for key, o in rel.items():
        qid, doc = key.split("|", 1)
        q = qmap[qid]
        title = (art.get(doc, ("",))[0] if q["종류"] == "뉴스" else pat_docs.get(doc, ("",))[0])
        judg.append({"qid": qid, "종류": q["종류"], "질문": q["질문"], "원래 정답": q["정답제목"], "검색된 문서": title,
                     "핵심요소": ", ".join(map(str, o.get("elements", []))), "요소일치": o.get("all_matched"),
                     "GPT판정": o.get("answerable"), "인용확인": o.get("quote_verified"), "최종": o.get("final"),
                     "인용": o.get("quote", ""), "근거": o.get("reason", "")})
    res.to_csv(OUT_DIR / f"eval_results_{VERSION}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(judg).to_csv(OUT_DIR / f"eval_judgments_{VERSION}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"qid": k, "질문": qmap[k]["질문"], **v} for k, v in qual.items()]).to_csv(
        OUT_DIR / f"eval_question_quality_{VERSION}.csv", index=False, encoding="utf-8-sig")

    shown = res.copy().astype(object)
    for col_ in shown.columns:
        if col_.startswith("Hit@"):
            shown[col_] = shown[col_].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif col_.startswith("MRR"):
            shown[col_] = shown[col_].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    j = pd.DataFrame(judg)
    n_gpt = int(j["GPT판정"].fillna(False).astype(bool).sum()) if len(j) else 0
    n_final = int(j["최종"].fillna(False).astype(bool).sum()) if len(j) else 0
    summary = (f"- 판정 모델: {JUDGE_MODEL}" + (f" (지원 안 해서 뺀 설정: {', '.join(sorted(_unsupported))})" if _unsupported else "") + "\n"
               f"- 질문 품질: 전체 {len(qual)}개 중 주체 불명확 {sum(not o.get('specific') for o in qual.values())}개, "
               f"바이오 무관 {sum(not o.get('bio_related') for o in qual.values())}개 → 좋은 질문 {len(good)}개\n"
               f"- 정답급 판정: 정답 아닌 상위 문서 {len(rel):,}건 중 GPT '답할 수 있음' {n_gpt:,}건 → "
               f"인용·요소 확인 통과 {n_final:,}건")
    cols = list(shown.columns)
    table = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    table += "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in shown.itertuples(index=False))
    (OUT_DIR / f"eval_report_{VERSION}.md").write_text(
        f"# 검색기 평가 {VERSION} (엄격한 정답급 판정)\n\n- 측정 시각: {time.strftime('%Y-%m-%d %H:%M')}\n{summary}\n"
        f"- Hit@3 (엄격): 질문을 만든 원래 문서가 상위 3개 안\n"
        f"- Hit@3 (정답급 {VERSION}): 원래 문서 또는 '핵심 요소 모두 일치 + 문서에서 그대로 인용 가능'한 문서가 상위 3개 안\n\n"
        + table, encoding="utf-8")

    print("\n" + "=" * 90)
    print(summary)
    print(shown.to_string(index=False))
    print("=" * 90)
    print(f"저장: eval/eval_report_{VERSION}.md, eval_results_{VERSION}.csv, eval_judgments_{VERSION}.csv "
          f"({(time.time() - t0) / 60:.0f}분)")
    return res


if __name__ == "__main__":
    run()
