"""
평가 설계 보완 — 비교 · 시점 · 종합·판단 · 답변 불가 질문 만들기 (유형별 15개, 사람 검토 후 확정)

[왜]  기존 평가 질문 164개는 '문서 1건 → 질문 1개'로 만들어 모두 단순 사실 질문이에요.
      여러 문서를 합쳐야 하는 질문(비교·시점·종합)과, DB에 근거가 없는 질문(답변 불가)을 따로 만들어요.
[방식]
  1. 씨앗 문서를 무작위로 뽑고, 검색기로 비슷한 문서 6건을 붙여 '문서 묶음'을 만듦
  2. LLM이 묶음을 읽고 유형에 맞는 질문 + 채점 기준표(꼭 들어가야 할 요소)를 만듦. 만들 수 없으면 건너뜀
     - 비교   : 서로 다른 두 대상(회사·기관·기술)을 같은 측면에서 비교 → 요소에 양쪽 내용
     - 시점   : 날짜가 다른 문서로 변화·최신 정보를 묻는 질문 → 요소에 날짜
     - 종합   : 문서 사이에 입장·근거가 엇갈리는 쟁점 질문 → 요소에 양쪽 입장
     - 답변 불가: 묶음 주제와 관련 있어 보이지만 답이 어디에도 없는 질문
                  → 실제로 검색해서 LLM이 '답할 수 있는 자료가 있다'고 하면 버림
  3. eval/eval_set/eval_questions2_draft.csv 로 저장 → 사람이 '검토' 칸에 O/X (추천 칸 O = 유형별 앞 15개)
  4. confirm() → O 표시한 질문만 eval/eval_set/eval_questions2.json 으로 확정 (v6 이후 평가가 읽음 (최종 v5 평가에는 쓰지 않음))
[비용]  LLM 호출 약 130번(유형별 씨앗 30개 + 답변 불가 확인) + 임베딩 약 120번. 중간에 멈춰도 이어서 진행
[실행 - Jupyter]
  from eval import design_questions as eq
  eq.generate()      # 약 10~20분 → eval/eval_set/eval_questions2_draft.csv
  # CSV를 엑셀로 열어 '검토' 칸에 O(쓸 질문) / X(뺄 질문) 적고 저장
  eq.confirm()       # → eval/eval_set/eval_questions2.json
"""
from __future__ import annotations

import json
import random
import re
import time

import pandas as pd

import config
from eval import judge as ej      # chat_json(gpt-6-luna JSON 호출)·run_cached(이어하기) 재사용
from rag import retriever as rt

# ============================== 설정 ==============================
OUT_DIR = ej.OUT_DIR
DRAFT = config.EVAL_SET_DIR / "eval_questions2_draft.csv"
FINAL = config.EVAL_SET_DIR / "eval_questions2.json"
N_PER_TYPE = 15            # 유형별 확정할 질문 수
SEEDS_PER_TYPE = 30        # 유형별 씨앗 문서 수 (만들 수 없는 묶음이 있어서 넉넉하게)
NEIGHBORS = 6              # 씨앗에 붙일 비슷한 문서 수
DOC_CHARS = 1500           # LLM에 넘길 문서 1건 길이
MIN_CHARS = 600            # 이보다 짧은 문서는 씨앗으로 쓰지 않음
SEED = 7
PREFIX = {"비교": "c", "시점": "t", "종합": "j", "답변불가": "u"}
# 씨앗 문서 종류: 비교는 특허도 섞음 (두 출원인의 기술 비교)
SEED_PLAN = {"비교": {"news": 20, "patents": 10}, "시점": {"news": 30}, "종합": {"news": 30}, "답변불가": {"news": 30}}
# ==================================================================

COMMON = """너는 바이오의약품·바이오헬스케어 RAG 챗봇의 평가 질문을 만드는 사람이다.
아래 [문서 묶음]만 보고 '{kind}' 유형 질문 1개와 채점 기준표를 만든다.

[유형 설명]
{guide}

공통 규칙:
- 질문은 실제 사용자가 챗봇에 물을 법한 자연스러운 한국어 한 문장. 문서 번호(D1 등)나 '이 기사' 같은 말은 쓰지 않는다.
- 질문만 읽어도 무엇을 묻는지 알 수 있게 회사·기관·기술 이름을 넣는다.
- '올해', '작년' 같은 상대 표현 대신 연도를 쓰거나 '가장 최근'이라고 쓴다.
- 요소(elements)는 정답에 꼭 들어가야 할 사실을 짧게 하나씩. 각 요소에 근거 문서 번호를 단다.
- 묶음으로 이 유형의 좋은 질문을 만들 수 없으면 possible을 false로 한다. 억지로 만들지 않는다.

[문서 묶음]
{docs}

JSON으로만 출력:
{{"possible": true, "question": "...", "elements": [{{"요소": "...", "문서": "D1"}}], "docs": ["D1", "D3"], "why": "이 질문이 이 유형인 이유 한 문장"}}"""

GUIDE = {
    "비교": "서로 다른 두 대상(회사·기관·지자체·기술)을 같은 측면(방식·조건·규모·성과 등)에서 비교하는 질문.\n"
            "두 대상의 이름을 질문에 모두 넣는다. 서로 다른 문서에서 각 대상의 내용을 찾을 수 있어야 한다.\n"
            "요소는 3~6개, 양쪽 대상에 대해 각각 1개 이상.",
    "시점": "날짜가 다른 문서를 써서 시간에 따른 변화, 또는 특정 시점·가장 최근 정보를 묻는 질문.\n"
            "예) 'OO의 임상 단계는 2023년과 2025년 사이 어떻게 바뀌었나', 'OO에 관한 가장 최근 소식은 언제 무엇인가'.\n"
            "요소 3~5개, 그중 최소 2개는 날짜(연·월)를 포함한다. 날짜는 문서의 보도일 또는 본문에 적힌 사건 날짜를 그대로 쓴다.",
    "종합": "여러 문서를 종합해야 답할 수 있는 쟁점 질문. 예) 'OO 정책의 주요 쟁점은 무엇인가', 'OO에 대한 기대와 우려는'.\n"
            "문서들 사이에 서로 다른 입장이나 엇갈리는 근거(찬성·반대, 기대·우려, 상충하는 수치)가 있어야 한다. 없으면 possible=false.\n"
            "요소 3~6개, 서로 다른 입장을 각각 1개 이상 포함한다.",
    "답변불가": "문서 묶음의 주제와 관련 있어 보이지만, 묶음 어디에도 답이 없는 질문.\n"
               "{variant}\n"
               "elements는 빈 목록 []으로 둔다. docs에는 주제가 가까운 문서 번호를 적는다.",
}
UNANSWERABLE_VARIANT = [
    "실제로 나오는 회사·기술에 대해, 문서에 없는 세부(다른 연도의 수치, 공개되지 않은 계약 조건, 다른 나라의 허가 등)를 묻는다.",
    "실제로 나오는 회사·기술에 대해, 문서에 없는 세부(다른 연도의 수치, 공개되지 않은 계약 조건, 다른 나라의 허가 등)를 묻는다.",
    "그럴듯하지만 실제로 없는 가상의 회사·제품 이름을 만들어 그에 대해 묻는다 (문서에 나오는 이름은 쓰지 않는다).",
]

CHECK_PROMPT = """아래 [검색된 자료]만으로 [질문]에 답할 수 있는지 엄격하게 판단한다.
질문이 묻는 핵심 사실이 자료에 직접 적혀 있으면 answerable=true. 비슷한 주제만 있고 핵심 사실이 없으면 false.

[질문] {question}

[검색된 자료]
{docs}

JSON으로만 출력: {{"answerable": false, "reason": "한 문장"}}"""


# ---------------------------- 문서 ----------------------------
_docs: dict = {}


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def documents() -> dict:
    """문서 키 → {종류, 제목, 날짜, 본문, 출원인, bio}. 뉴스는 청크를 기사 단위로 합침 (키 = 기사ID / 특허 = 출원번호)"""
    if _docs:
        return _docs
    news = rt.get_index("news")
    parts: dict = {}
    for i, m, d in zip(news.ids, news.metas, news.docs):
        k = m.get("기사ID") or i
        body = d.split("\n", 1)[1] if d.startswith("[제목]") and "\n" in d else d
        parts.setdefault(k, {"종류": "뉴스", "제목": m.get("제목", ""), "날짜": str(m.get("발행일", ""))[:10],
                             "출원인": "", "bio": m.get("바이오관련") is not False, "청크": []})
        parts[k]["청크"].append((m.get("청크번호", 0), body))
    for k, v in parts.items():
        v["본문"] = " ".join(b for _, b in sorted(v.pop("청크")))
        _docs[k] = v
    pat = rt.get_index("patents")
    for i, m, d in zip(pat.ids, pat.metas, pat.docs):
        _docs[i] = {"종류": "특허", "제목": m.get("발명의명칭", ""), "날짜": str(m.get("등록일", ""))[:10],
                    "출원인": rt.owner_text(m), "bio": m.get("바이오관련") is not False,
                    "본문": d.split("\n핵심키워드:")[0]}
    return _docs


def doc_block(n: int, k: str, chars: int = DOC_CHARS) -> str:
    d = documents()[k]
    when = f"등록일 {d['날짜']}" if d["종류"] == "특허" else f"보도일 {d['날짜']}"
    who = f" | 출원인: {d['출원인']}" if d["출원인"] else ""
    return f"[D{n}] {d['종류']} | {when} | {d['제목']}{who}\n{_flat(d['본문'])[:chars]}"


def _seeds(kind: str, n: int, rng: random.Random, used: set) -> list[str]:
    want = "뉴스" if kind == "news" else "특허"
    pool = sorted(k for k, d in documents().items()
                  if d["종류"] == want and d["bio"] and len(d["본문"]) >= MIN_CHARS and k not in used)
    return rng.sample(pool, min(n, len(pool)))


def _neighbors(kind: str, key: str) -> list[str]:
    d = documents()[key]
    q = d["제목"]
    target = "news" if kind == "news" else "patent"
    k = {"k_news": NEIGHBORS + 1, "k_patent": 0} if kind == "news" else {"k_news": 0, "k_patent": NEIGHBORS + 1}
    res = rt.retrieve(q, target=target, q_vec=rt._embed(q), **k)
    out = []
    for x in res:
        dk = x["메타"].get("기사ID", x["id"]) if kind == "news" else x["id"]
        if dk != key and dk not in out and dk in documents():
            out.append(dk)
    return out[:NEIGHBORS]


# ---------------------------- 생성 ----------------------------
def _bundles() -> list[dict]:
    """유형별 씨앗 + 이웃 문서 묶음 (같은 SEED면 항상 같은 묶음 → 이어하기 가능)"""
    rng, used, out = random.Random(SEED), set(), []
    for qtype, plan in SEED_PLAN.items():
        n = 0
        for kind, cnt in plan.items():
            for s in _seeds(kind, cnt, rng, used):
                used.add(s)
                out.append({"bid": f"{PREFIX[qtype]}{n:03d}", "유형": qtype, "kind": kind, "seed": s})
                n += 1
    return out


def _gen_prompt(b: dict) -> str:
    keys = [b["seed"]] + b["이웃"]
    guide = GUIDE[b["유형"]]
    if b["유형"] == "답변불가":
        guide = guide.format(variant=UNANSWERABLE_VARIANT[int(b["bid"][1:]) % 3])
    docs = "\n\n".join(doc_block(i, k) for i, k in enumerate(keys, 1))
    return COMMON.format(kind=b["유형"], guide=guide, docs=docs)


def _check_prompt(question: str) -> str:
    res = rt.retrieve(question, q_vec=rt._embed(question), k_patent=8, k_news=8)
    docs = "\n\n".join(f"[{i}] {x['종류']} | {str(x['날짜'])[:10]} | {x['제목']}\n{_flat(x['본문'])[:800]}"
                       for i, x in enumerate(res, 1)) or "(검색 결과 없음)"
    return CHECK_PROMPT.format(question=question, docs=docs)


def generate():
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)
    print("문서 준비 중…")
    documents()
    bundles = _bundles()
    print(f"문서 묶음 {len(bundles)}개 만드는 중 (검색 {len(bundles)}번)…")
    nb_path = OUT_DIR / "qgen2_bundles.json"       # 이웃 문서도 저장 → 다시 실행해도 같은 묶음
    saved = json.loads(nb_path.read_text(encoding="utf-8")) if nb_path.exists() else {}
    for i, b in enumerate(bundles, 1):
        if b["bid"] not in saved:
            saved[b["bid"]] = _neighbors(b["kind"], b["seed"])
            if i % 20 == 0:
                nb_path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        b["이웃"] = saved[b["bid"]]
    nb_path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    gen = ej.run_cached("qgen2_generate", {b["bid"]: _gen_prompt(b) for b in bundles}, ej.chat_json,
                        "질문 생성 (gpt-6-luna)")

    rows = []
    for b in bundles:
        g = gen.get(b["bid"]) or {}
        q = str(g.get("question") or "").strip()
        if not g.get("possible") or not q:
            continue
        keys = [b["seed"]] + b["이웃"]
        dmap = {f"D{i}": k for i, k in enumerate(keys, 1)}
        els = [e for e in g.get("elements") or [] if isinstance(e, dict) and str(e.get("요소", "")).strip()]
        if b["유형"] != "답변불가" and len(els) < 2:
            continue
        doc_keys = [dmap[d] for d in g.get("docs") or [] if d in dmap] or [b["seed"]]
        rows.append({"bid": b["bid"], "유형": b["유형"], "질문": q,
                     "필수요소": [{"요소": str(e["요소"]).strip(), "문서": dmap.get(str(e.get("문서", "")).strip(), "")}
                              for e in els] if b["유형"] != "답변불가" else [],
                     "문서키": doc_keys, "이유": g.get("why", "")})

    # 답변 불가: 실제로 검색해서 답할 수 있는 자료가 있으면 버림
    un = [r for r in rows if r["유형"] == "답변불가"]
    if un:
        print(f"답변 불가 질문 {len(un)}개를 실제로 검색해서 확인해요…")
        path = OUT_DIR / "qgen2_check.jsonl"            # 이미 확인한 질문은 다시 검색하지 않음
        done = {json.loads(x)["key"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()} if path.exists() else set()
        chk = ej.run_cached("qgen2_check", {r["bid"]: "" if r["bid"] in done else _check_prompt(r["질문"]) for r in un}, ej.chat_json,
                            "답변 불가 확인 (gpt-6-luna)")
        bad = {k for k, v in chk.items() if (v or {}).get("answerable") is not False}
        rows = [r for r in rows if r["bid"] not in bad]
        print(f"  → 자료로 답할 수 있어서 뺀 질문 {len(bad & {r['bid'] for r in un})}개")

    seen, out = set(), []
    for r in rows:                                  # 같은 질문 중복 제거
        norm = re.sub(r"\s+", "", r["질문"])
        if norm not in seen:
            seen.add(norm); out.append(r)
    cnt: dict = {}
    table = []
    for r in out:
        cnt[r["유형"]] = cnt.get(r["유형"], 0) + 1
        docs = documents()
        table.append({
            "bid": r["bid"], "유형": r["유형"], "질문": r["질문"],
            "필수요소": " / ".join(f"{e['요소']}" for e in r["필수요소"]),
            "근거문서": " / ".join(f"{docs[k]['종류']} {docs[k]['날짜']} {docs[k]['제목'][:40]}" for k in r["문서키"]),
            "생성이유": r["이유"], "추천": "O" if cnt[r["유형"]] <= N_PER_TYPE else "", "검토": "", "메모": "",
            "데이터(수정 금지)": json.dumps({"필수요소": r["필수요소"], "문서키": r["문서키"]}, ensure_ascii=False)})
    d = pd.DataFrame(table)
    if DRAFT.exists():         # 이미 적어 둔 검토는 지우지 않음
        old = _read_draft()
        keep = {(r["bid"], r["질문"]): (r["검토"], r["메모"]) for _, r in old.iterrows() if r["검토"].strip() or r["메모"].strip()}
        for i, r in d.iterrows():
            if (r["bid"], r["질문"]) in keep:
                d.at[i, "검토"], d.at[i, "메모"] = keep[(r["bid"], r["질문"])]
    d.to_csv(DRAFT, index=False, encoding="utf-8-sig")
    summary = pd.Series(cnt).reindex(list(PREFIX)).fillna(0).astype(int)
    print("\n[만든 질문 수]", summary.to_dict())
    short = [t for t, c in summary.items() if c < N_PER_TYPE]
    if short:
        print(f"⚠️ {', '.join(short)} 유형이 {N_PER_TYPE}개보다 적어요. SEEDS_PER_TYPE을 늘리려면 SEED_PLAN 숫자를 키우고 다시 실행하세요.")
    print(f"\n저장: eval/{DRAFT.name} ({time.time() - t0:.0f}초)")
    print("다음: 엑셀로 열어 '검토' 칸에 O / X 를 적고 저장 → eq.confirm()")
    return d


def _read_draft() -> pd.DataFrame:
    """엑셀에서 저장하면 인코딩이 cp949로 바뀔 수 있어서 둘 다 시도"""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(DRAFT, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{DRAFT.name}을(를) 읽지 못했어요. 엑셀에서 'CSV UTF-8'로 저장해 주세요.")


def confirm():
    """검토 칸이 O인 질문만 확정 → eval/eval_set/eval_questions2.json"""
    d = _read_draft()
    ok = d[d["검토"].str.strip().str.upper() == "O"]
    if ok.empty:
        print("'검토' 칸에 O가 하나도 없어요. 쓸 질문에 O를 적고 저장한 뒤 다시 실행해 주세요.")
        return None
    out, num = [], {t: 0 for t in PREFIX}
    for _, r in ok.iterrows():
        num[r["유형"]] += 1
        data = json.loads(r["데이터(수정 금지)"])
        out.append({"qid": f"{PREFIX[r['유형']]}{num[r['유형']]:02d}", "유형": r["유형"], "질문": r["질문"],
                    "필수요소": data["필수요소"], "문서키": data["문서키"], "bid": r["bid"]})
    FINAL.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[확정한 질문 수]", num)
    off = {t: c for t, c in num.items() if c != N_PER_TYPE}
    if off:
        print(f"ℹ️ 유형별 {N_PER_TYPE}개가 아닌 유형: {off} (그대로 써도 되고, O를 조정해 다시 confirm() 해도 돼요)")
    print(f"저장: eval/{FINAL.name} → 다음: import eval_answers as ea; ea.run()")
    return out


if __name__ == "__main__":
    generate()
