"""
하이브리드 리트리버 (v2) — BM25(키워드 검색) + 벡터 검색을 RRF로 결합

[구성]
  - 벡터 검색 : ChromaDB (text-embedding-3-small)  → 의미가 비슷한 문서
  - BM25      : 같은 DB의 본문으로 만든 키워드 색인 → 인물·회사명·코드(CD3, GLP-1)가 그대로 들어간 문서
  - 결합      : RRF(순위 역수 합산). 점수 = W_VEC/(60+벡터순위) + W_BM25/(60+BM25순위)
  - 필터      : 분야·기간·본문 포함 단어·뉴스 노이즈(바이오관련)를 두 검색에 똑같이 적용
  - 중복 묶기 : 뉴스는 같은 기사, 특허는 같은 제목(패밀리)을 하나로
  - 출원인    : 특허 BM25 색인에 출원인·최종권리자 이름을 넣어 회사명으로도 찾음 (주소·'주식회사' 등은 뺌)
  - 회사명 필터: 질문에 특허 출원인의 '전체 이름'(예: 알테오젠, 삼성바이오로직스)이 있으면 그 출원인의 특허만 결과로 (COMPANY_FILTER)
                v2: 이름 일부('바이오사이언스', '삼성')로는 걸리지 않음. SK·GC·CJ·LG 같은 영문 약칭은 출원인 표기(에스케이…)로 바꿔 비교
                retrieve(companies=[…])로 회사 이름을 직접 주면 그 이름으로 필터 (rag_chain이 질문 분석 결과를 넘김)
                → '알테오젠 기술수출'에서 '핵 수출 조절제' 같은 단어만 같은 특허가 끼지 않음

[사용 - Jupyter, project3 폴더에서]
  from rag.retriever import retrieve, show
  results = retrieve("알테오젠 최근 기술수출")                     # 하이브리드 (기본)
  results = retrieve("CD3 이중항체", target="patent")
  results = retrieve("메디엑스포", date_from="2023", date_to="2023")  # 2023-01-01 ~ 2023-12-31
  results = retrieve("…", mode="vector")   # 비교용: 'vector' | 'bm25' | 'hybrid'
  show(results)

[필요]  pip install kiwipiepy scipy   (kiwipiepy가 없으면 간단한 단어 나누기로 자동 전환)
[처음 실행]  BM25 색인을 만드느라 수 분 걸려요 (store/bm25 폴더에 저장, 이후엔 몇 초)
"""
from __future__ import annotations

import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

# ============================== 설정 ==============================
import config

BASE_DIR = config.ROOT

DB_DIR = config.CHROMA_DIR
INDEX_DIR = config.BM25_DIR
EMBED_MODEL = "text-embedding-3-small"
K_PATENT, K_NEWS = 5, 5
CANDIDATES = 50                 # 각 검색에서 가져올 후보 수
RRF_K = 60
W_VEC, W_BM25 = 1.0, 2.0        # 두 검색의 비중 (벡터 1 : BM25 2 — 부트스트랩 평가로 확정, 2026-11-06)
NOISE_FILTER = True             # 바이오관련=False 뉴스·특허 제외 (표시가 있을 때만. 특허는 indexer/noise_patent.py로 표시)
BM25_K1, BM25_B = 1.2, 0.75
INDEX_VERSION = 2               # 색인 구조가 바뀌면 올림 → 예전 색인은 자동으로 다시 만듦 (v2: 특허 출원인 추가, 단어 희귀도 저장)
COMPANY_FILTER = True           # 특허: 질문 속 출원인 이름이 없는 BM25 결과는 버림
GENERIC_OWNER = {"대한민국", "한국", "미국", "일본", "중국", "독일", "스위스", "영국", "프랑스", "대학교", "대학",
                 "연구원", "연구소", "재단", "진흥원", "센터", "병원", "의료원", "컴퍼니", "산학", "협력단", "공사"}
                                # 출원인 이름에 자주 나오지만 회사명으로 보면 안 되는 흔한 말
STOPWORDS = {"것", "수", "등", "및", "때", "곳", "중", "위해", "통해", "대한", "관련", "이번", "올해", "지난해",
             "기자", "뉴스", "제공", "있다", "한다", "했다", "본", "발명", "방법", "조성물"}
# ==================================================================

_clients: dict = {}
_index: dict = {}
_tok = {"name": None, "kiwi": None}


# ---------------------------- 단어 나누기 ----------------------------
CODE_RE = re.compile(r"[A-Za-z]+-?\d+[A-Za-z0-9-]*|\d+[A-Za-z]+[A-Za-z0-9-]*")   # CD3, GLP-1, FL118, 4D …


def _init_tokenizer():
    if _tok["name"]:
        return
    try:
        from kiwipiepy import Kiwi
        try:
            _tok["kiwi"] = Kiwi(num_workers=-1)
        except TypeError:
            _tok["kiwi"] = Kiwi()
        _tok["name"] = "kiwi"
    except ImportError:
        _tok["name"] = "simple"
        print("ℹ️ kiwipiepy가 없어 간단한 단어 나누기를 써요 (pip install kiwipiepy 권장)")


KEEP_TAGS = ("NNG", "NNP", "NR", "SL", "SH", "SN", "XR")


def _post(tokens: list[str], text: str) -> list[str]:
    out = [t.lower() for t in tokens if t and t not in STOPWORDS and not (len(t) == 1 and not t.isalnum())]
    out += [c.lower() for c in CODE_RE.findall(text)]
    return out


def tokenize(text: str) -> list[str]:
    _init_tokenizer()
    if _tok["name"] == "kiwi":
        toks = [t.form for t in _tok["kiwi"].tokenize(text) if t.tag in KEEP_TAGS]
    else:
        words = re.findall(r"[가-힣]+|[A-Za-z]+|\d+", text)
        toks = []
        for w in words:
            toks.append(w)
            if re.fullmatch(r"[가-힣]{3,}", w):           # 한글 긴 단어는 두 글자씩도 (조사 붙은 형태 대응)
                toks += [w[i:i + 2] for i in range(len(w) - 1)]
    return _post(toks, text)


def tokenize_many(texts: list[str], label: str = "") -> list[list[str]]:
    _init_tokenizer()
    out, t0 = [], time.time()
    if _tok["name"] == "kiwi":
        for i, toks in enumerate(_tok["kiwi"].tokenize(texts), 1):
            out.append(_post([t.form for t in toks if t.tag in KEEP_TAGS], texts[i - 1]))
            if i % 5000 == 0:
                print(f"    {label} 단어 나누기 {i:,}/{len(texts):,} ({time.time() - t0:.0f}초)")
    else:
        out = [tokenize(t) for t in texts]
    return out


# ---------------------------- BM25 색인 ----------------------------
class BM25Index:
    """희소 행렬로 만든 BM25. 문서×단어 가중치를 미리 계산해 두고, 질문 단어 열만 더함"""

    def __init__(self, ids, docs, metas, tokens=None, tokenizer_name=None, state=None):
        if state is not None:   # 저장본에서 복원
            self.__dict__.update(state)
            return
        from scipy.sparse import csc_matrix
        self.ids, self.docs, self.metas, self.tokenizer = ids, docs, metas, tokenizer_name
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        dl = np.zeros(len(ids), dtype=np.float32)
        for i, toks in enumerate(tokens):
            dl[i] = len(toks)
            tf: dict[int, int] = {}
            for t in toks:
                j = vocab.setdefault(t, len(vocab))
                tf[j] = tf.get(j, 0) + 1
            rows += [i] * len(tf)
            cols += list(tf)
            vals += list(tf.values())
        rows, cols, vals = np.array(rows), np.array(cols), np.array(vals, dtype=np.float32)
        n, avgdl = len(ids), max(dl.mean(), 1)
        df = np.bincount(cols, minlength=len(vocab))
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        w = idf[cols] * vals * (BM25_K1 + 1) / (vals + BM25_K1 * (1 - BM25_B + BM25_B * dl[rows] / avgdl))
        self.matrix = csc_matrix((w, (rows, cols)), shape=(n, len(vocab)), dtype=np.float32)
        self.vocab = vocab
        self.id_pos = {x: i for i, x in enumerate(ids)}
        self.version = INDEX_VERSION

    def search(self, query_tokens: list[str], mask: np.ndarray | None, n: int) -> list[str]:
        """BM25 상위 n개 id"""
        cols = [self.vocab[t] for t in set(query_tokens) if t in self.vocab]
        if not cols:
            return []
        scores = np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()
        if mask is not None:
            scores = np.where(mask, scores, 0)
        top = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        top = top[np.argsort(-scores[top])]
        return [self.ids[i] for i in top if scores[i] > 0]


def _all_docs(col):
    ids, docs, metas, off = [], [], [], 0
    while True:
        r = col.get(limit=5000, offset=off, include=["documents", "metadatas"])
        if not r["ids"]:
            break
        ids += r["ids"]; docs += r["documents"]; metas += r["metadatas"]
        off += len(r["ids"])
    return ids, docs, metas


CORP_WORDS = re.compile(r"\((주|유|재|사|학|의)\)|주식회사|유한회사|유한책임회사|재단법인|사단법인|학교법인|의료법인|"
                        r"농업회사법인|가부시키가이샤|가부시끼가이샤|인코포레이티드|코포레이션|엘티디\.?|게엠베하|리미티드")


def owner_text(meta: dict, as_list: bool = False):
    """특허 출원인·최종권리자 → 검색용 이름만. '(주)알테오젠(대전광역시 유성구...)' → '알테오젠'
    여러 명은 '|'로 구분되어 있고, 각 이름 끝의 괄호가 주소라서 끝 괄호 하나만 지움"""
    names = []
    for key in ("출원인", "최종권리자"):
        for part in str(meta.get(key) or "").split("|"):
            part = re.sub(r"\([^()]*\)\s*$", "", part.strip())   # 끝의 주소 괄호
            part = re.sub(r"\s+", " ", CORP_WORDS.sub(" ", part)).strip()
            if part and part not in names:
                names.append(part)
    return names if as_list else " ".join(names)


LATIN_ALIAS = {"sk": "에스케이", "gc": "지씨", "cj": "씨제이", "lg": "엘지", "hk": "에이치케이", "jw": "제이더블유",
               "hlb": "에이치엘비", "ck": "씨케이", "dx": "디엑스"}
_ALIAS_RE = re.compile(r"(?<![a-z])(" + "|".join(sorted(LATIN_ALIAS, key=len, reverse=True)) + r")(?=[가-힣])")


def owner_names(meta: dict) -> list[str]:
    """출원인·최종권리자 이름 목록 (owner_text를 나누지 않은 형태)"""
    return [n for n in owner_text(meta, as_list=True) if n]


def norm_name(s: str) -> str:
    """회사 이름 비교용: 소문자, 띄어쓰기·기호 제거, 'SK바이오' → '에스케이바이오'"""
    s = re.sub(r"[^0-9a-z가-힣]", "", str(s).lower())
    return _ALIAS_RE.sub(lambda m: LATIN_ALIAS[m.group(1)], s)


def owner_index(idx) -> dict[str, list[int]]:
    """특허 색인의 출원인 이름(정규화) → 특허 위치 목록. 처음 한 번 만들어 메모리에 둠"""
    if getattr(idx, "_owner_pos", None) is None:
        pos: dict[str, list[int]] = {}
        for i, m in enumerate(idx.metas):
            for n in owner_names(m):
                key = norm_name(n)
                if len(key) >= 2 and key not in GENERIC_OWNER and not key.startswith("대한민국"):
                    pos.setdefault(key, []).append(i)
        idx._owner_pos = pos
    return idx._owner_pos


TOPIC_MIN_DOCS = 5        # 특허 '본문'에 이만큼 이상 나오는 말은 회사명이 아니라 기술 용어로 봄 (예: 마이크로바이옴)
_topic_cache: dict[str, bool] = {}


def is_topic_word(name: str) -> bool:
    """출원인 이름이면서 흔한 기술 용어인 경우를 걸러냄. 실제 회사명(알테오젠 등)은 특허 본문에 거의 안 나옴(0회),
    '마이크로바이옴'처럼 같은 이름의 작은 회사가 있는 기술 용어는 본문에 여러 번 나옴(22회)"""
    key = norm_name(name)
    if key not in _topic_cache:
        raw = str(name).lower().replace(" ", "")
        idx = get_index("patents")
        n = 0
        for d in idx.docs:
            if raw in d.lower().replace(" ", ""):
                n += 1
                if n >= TOPIC_MIN_DOCS:
                    break
        _topic_cache[key] = n >= TOPIC_MIN_DOCS
    return _topic_cache[key]


def _drop_contained(names: list[str]) -> list[str]:
    """'녹십자'와 '녹십자웰빙'이 둘 다 걸리면 긴 쪽만 (질문에 '녹십자웰빙'이 있을 때)"""
    return [n for n in names if not any(n != o and n in o for o in names)]


def find_companies(question: str) -> list[str]:
    """질문 안에 출원인 '전체 이름'이 단어의 맨 앞부터 들어 있으면 그 이름들 (자동 감지용)
    예) '삼성바이오로직스 CDMO' → 삼성바이오로직스 / '리가켐바이오' 속 '켐바이오'처럼 단어 중간은 제외
        'GC녹십자'처럼 그룹 약칭이 붙은 단어는 약칭을 뗀 위치도 단어 시작으로 봄"""
    words = [norm_name(w) for w in str(question).split()]
    q, starts, cur = "", set(), 0
    for w in words:
        starts.add(cur)
        starts |= {cur + len(p) for p in GROUP_PREFIX if w.startswith(p)}
        q += w
        cur += len(w)
    found = []
    for n in owner_index(get_index("patents")):
        if len(n) == 2:
            ok = n in words                                   # 2글자(예: 보령)는 띄어 쓴 단어와 똑같을 때만
        else:
            ok = any(q.startswith(n, i) for i in starts)
        if ok and not is_topic_word(n):
            found.append(n)
    return _drop_contained(found)


GROUP_PREFIX = ("에스케이", "지씨", "씨제이", "엘지")      # 'GC녹십자' → '녹십자'처럼 그룹 약칭을 뗀 이름도 찾아봄


def resolve_companies(names: list[str]) -> list[str]:
    """질문 분석이 뽑은 회사 이름 → 특허 DB의 출원인 이름들. 없으면 [] (= 특허 DB에 없는 회사)
    ① 이름이 똑같은 출원인 → 그것만  ② 없으면 그 이름으로 시작하는 출원인 ('삼성바이오' → 삼성바이오로직스·에피스)
    ③ 그래도 없으면 그룹 약칭을 떼고 다시 ('지씨녹십자' → 녹십자). 이름 중간에 들어 있는 경우('리가켐바이오' 속 '켐바이오')는 안 씀"""
    pos = owner_index(get_index("patents"))
    out = []
    for raw in names:
        key = norm_name(raw)
        if len(key) < 2 or key in GENERIC_OWNER:
            continue
        tries = [key] + [key[len(p):] for p in GROUP_PREFIX if key.startswith(p) and len(key) - len(p) >= 2]
        for k in tries:
            hits = [k] if k in pos else [n for n in pos if len(k) >= 3 and n.startswith(k)]
            if hits:
                out += [h for h in hits if h not in out]
                break
    return out


def _bm25_texts(name: str, docs: list[str], metas: list[dict]) -> list[str]:
    """BM25 단어 나누기에 쓰는 글. 특허는 본문 + 출원인 이름 (화면·답변에 쓰는 본문은 그대로)"""
    if name != "patents":
        return docs
    return [f"{d}\n{owner_text(m)}" for d, m in zip(docs, metas)]


def get_index(name: str) -> BM25Index:
    """컬렉션의 BM25 색인. 저장본이 있고 개수·버전이 같으면 불러오고, 아니면 새로 만듦"""
    col = _get_clients()[name]
    if name in _index and len(_index[name].ids) == col.count() and getattr(_index[name], "version", 1) == INDEX_VERSION:
        return _index[name]
    _init_tokenizer()
    path = INDEX_DIR / f"{name}.pkl"
    if path.exists():
        with path.open("rb") as f:
            idx = BM25Index(None, None, None, state=pickle.load(f))   # 일반 dict로 저장 → 셀·파일 어디서 만들어도 불러오기 가능
        if len(idx.ids) == col.count() and idx.tokenizer == _tok["name"] and getattr(idx, "version", 1) == INDEX_VERSION:
            # 메타데이터가 바뀌었으면(예: 노이즈 표시 추가) 메타데이터만 새로 읽음 — 단어 나누기는 다시 안 함
            sample = col.get(limit=1, include=["metadatas"])
            if sample["ids"] and set(sample["metadatas"][0]) != set(idx.metas[idx.id_pos[sample["ids"][0]]]):
                print(f"ℹ️ '{name}' 메타데이터가 바뀌어 색인의 메타데이터를 새로 읽어요")
                ids, _, metas = _all_docs(col)
                new = dict(zip(ids, metas))
                idx.metas = [new[i] for i in idx.ids]
                with path.open("wb") as f:
                    pickle.dump(dict(idx.__dict__), f)
            _index[name] = idx
            return idx
        print(f"ℹ️ '{name}' 문서 수·단어 나누기 방식·색인 버전이 바뀌어 BM25 색인을 다시 만들어요")
    t0 = time.time()
    print(f"[{name}] BM25 색인 만드는 중… (처음 한 번만, 수 분 걸릴 수 있어요)")
    ids, docs, metas = _all_docs(col)
    tokens = tokenize_many(_bm25_texts(name, docs, metas), name)
    idx = BM25Index(ids, docs, metas, tokens, _tok["name"])
    INDEX_DIR.mkdir(exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dict(idx.__dict__), f)
    print(f"  완료: 문서 {len(ids):,}개, 단어 {len(idx.vocab):,}개 ({time.time() - t0:.0f}초) → {path.name}")
    _index[name] = idx
    return idx


# ---------------------------- 연결·필터 ----------------------------
def _get_clients():
    if not _clients:
        import chromadb
        try:
            from dotenv import load_dotenv
            load_dotenv(config.ENV_FILE)
        except ImportError:
            pass
        chroma = chromadb.PersistentClient(path=str(DB_DIR))
        _clients.update(patents=chroma.get_collection("patents"), news=chroma.get_collection("news"))
    if "openai" not in _clients and os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        _clients["openai"] = OpenAI()
    return _clients


def _date_num(s, end: bool = False) -> int | None:
    """'2025' → 20250101 (끝이면 20251231), '2025-03' → 20250301 (끝이면 20250331), '2025-3-5' → 20250305"""
    if not s:
        return None
    parts = [int(p) for p in re.findall(r"\d+", str(s))]
    if not parts or parts[0] < 1900:
        return None
    y = parts[0]
    if len(parts) == 1:
        return y * 10000 + (1231 if end else 101)
    m = min(max(parts[1], 1), 12)
    if len(parts) == 2:
        last = [31, 29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
        return y * 10000 + m * 100 + (last if end else 1)
    return y * 10000 + m * 100 + min(max(parts[2], 1), 31)


def _conditions(kind, field, date_from, date_to, noise_filter, idx: BM25Index):
    """Chroma where 조건 + BM25용 마스크를 같은 기준으로 만듦"""
    conds = []
    date_key = "등록일_num" if kind == "patents" else "발행일_num"
    lo, hi = _date_num(date_from), _date_num(date_to, end=True)
    if kind == "patents" and field:
        conds.append({"분야": field})
    if lo:
        conds.append({date_key: {"$gte": lo}})
    if hi:
        conds.append({date_key: {"$lte": hi}})
    # 노이즈 필터 — 뉴스: 바이오관련=True만 / 특허: 바이오관련=False만 제외 (표시가 없는 새 특허는 보이게)
    use_noise = noise_filter and any("바이오관련" in m for m in (idx.metas[:50] if kind == "news" else idx.metas))
    if use_noise:
        conds.append({"바이오관련": True} if kind == "news" else {"바이오관련": {"$ne": False}})
    where = None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}

    mask = None
    if conds:
        mask = np.ones(len(idx.ids), dtype=bool)
        for i, m in enumerate(idx.metas):
            if kind == "patents" and field and m.get("분야") != field:
                mask[i] = False
            elif lo and m.get(date_key, 0) < lo:
                mask[i] = False
            elif hi and m.get(date_key, 0) > hi:
                mask[i] = False
            elif use_noise and kind == "news" and m.get("바이오관련") is not True:
                mask[i] = False
            elif use_noise and kind == "patents" and m.get("바이오관련") is False:
                mask[i] = False
    return where, mask


# ---------------------------- 검색 ----------------------------
def _embed(text: str) -> list[float]:
    c = _get_clients()
    if "openai" not in c:
        raise RuntimeError(f"OPENAI_API_KEY가 없습니다. project3 폴더의 .env 파일에 넣어주세요: {config.ENV_FILE}")
    return c["openai"].embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


def _search_one(kind, question, q_vec, q_tokens, k, mode, field, date_from, date_to, contains, noise_filter,
                companies=None):
    col = _get_clients()[kind]
    idx = get_index(kind)
    where, mask = _conditions(kind, field, date_from, date_to, noise_filter, idx)
    if contains:
        cm = np.array([contains in d for d in idx.docs])
        mask = cm if mask is None else (mask & cm)

    vec_ids, vec_sim = [], {}
    if mode in ("hybrid", "vector"):
        r = col.query(query_embeddings=[q_vec], n_results=CANDIDATES, where=where,
                      where_document={"$contains": contains} if contains else None, include=["distances"])
        vec_ids = r["ids"][0]
        vec_sim = {i: 1 - d for i, d in zip(r["ids"][0], r["distances"][0])}
    # 회사명 필터: 그 출원인의 특허만 결과로 인정 (BM25·벡터 모두)
    #   → 회사 특허가 2건뿐이면 2건만 보여 줌 (빈자리를 다른 회사 특허로 채우지 않음)
    #   companies가 주어지면(질문 분석 결과) 그 이름으로, 없으면 질문 속 출원인 전체 이름을 자동 감지
    if kind == "patents" and COMPANY_FILTER:
        asked = [c for c in companies if not is_topic_word(c)] if companies is not None else None   # 기술 용어는 회사명에서 제외
        companies = resolve_companies(asked) if asked is not None else find_companies(question)
        if asked and not companies:
            return []        # 회사 이름을 줬는데 특허 DB에 그 회사가 없음 → 다른 회사 특허로 채우지 않음
    else:
        companies = []
    if companies:
        pos = owner_index(idx)
        allowed = np.zeros(len(idx.ids), dtype=bool)
        allowed[[p for n in companies for p in pos.get(n, [])]] = True
        mask = allowed if mask is None else (mask & allowed)
        vec_ids = [i for i in vec_ids if allowed[idx.id_pos[i]]]
    bm_ids = idx.search(q_tokens, mask, CANDIDATES) if mode in ("hybrid", "bm25") else []

    score, vrank, brank = {}, {}, {}
    for r_, i in enumerate(vec_ids, 1):
        vrank[i] = r_
        score[i] = score.get(i, 0) + W_VEC / (RRF_K + r_)
    for r_, i in enumerate(bm_ids, 1):
        brank[i] = r_
        score[i] = score.get(i, 0) + W_BM25 / (RRF_K + r_)

    results, seen = [], set()
    for i in sorted(score, key=lambda x: -score[x]):
        m = idx.metas[idx.id_pos[i]]
        group = m.get("기사ID") if kind == "news" else m.get("발명의명칭", i)
        if group in seen:
            continue
        seen.add(group)
        doc = idx.docs[idx.id_pos[i]]
        how = "둘 다" if i in vrank and i in brank else ("벡터" if i in vrank else "BM25")
        results.append({
            "종류": "특허" if kind == "patents" else "뉴스", "id": i,
            "제목": m.get("발명의명칭", "") if kind == "patents" else m.get("제목", ""),
            "날짜": m.get("등록일", "") if kind == "patents" else m.get("발행일", ""),
            "점수": round(score[i], 5), "벡터유사도": round(vec_sim[i], 3) if i in vec_sim else None,
            "벡터순위": vrank.get(i), "BM25순위": brank.get(i), "검색경로": how,
            "회사명필터": companies,
            "본문": doc, "출처": f"출원번호 {m.get('출원번호', i)}" if kind == "patents" else m.get("원문URL", ""),
            "메타": m})
        if len(results) == k:
            break
    return results


def retrieve(question: str, target: str = "both", field: str | None = None, date_from: str | None = None,
             date_to: str | None = None, contains: str | None = None, k_patent: int = K_PATENT,
             k_news: int = K_NEWS, mode: str = "hybrid", noise_filter: bool | None = None,
             q_vec: list[float] | None = None, companies: list[str] | None = None) -> list[dict]:
    """
    companies: 특허 회사명 필터에 쓸 회사 이름 (None이면 질문에서 자동 감지, []면 필터 없음,
               이름을 줬는데 특허 DB에 없는 회사면 특허 결과 없음)
    target: 'both' | 'patent' | 'news'   mode: 'hybrid' | 'vector' | 'bm25'
    field: 특허 분야 ('바이오의약품' | '바이오헬스케어')
    date_from/date_to: '2025', '2025-03', '2025-03-05' (종료일은 그 기간의 끝까지 포함)
    q_vec: 질문 벡터를 이미 갖고 있으면 넘겨서 재사용 (평가용)
    """
    if mode not in ("hybrid", "vector", "bm25"):
        raise ValueError("mode는 'hybrid', 'vector', 'bm25' 중 하나예요")
    noise_filter = NOISE_FILTER if noise_filter is None else noise_filter     # 지정하지 않으면 현재 설정값
    q_vec = q_vec if q_vec is not None or mode == "bm25" else _embed(question)
    q_tokens = tokenize(question)
    out = []
    if target in ("both", "patent") and k_patent > 0:
        out += _search_one("patents", question, q_vec, q_tokens, k_patent, mode, field, date_from, date_to, contains,
                           noise_filter, companies)
    if target in ("both", "news") and k_news > 0:
        out += _search_one("news", question, q_vec, q_tokens, k_news, mode, None, date_from, date_to, contains, noise_filter)
    return out


def show(results: list[dict], body_chars: int = 100):
    if not results:
        print("관련 자료를 찾지 못했어요.")
        return
    comp = next((x["회사명필터"] for x in results if x.get("회사명필터")), None)
    if comp:
        print(f"ℹ️ 회사명 필터: {', '.join(comp)} — 특허 키워드 검색은 이 이름이 출원인/본문에 있는 특허만")
    for x in results:
        body = re.sub(r"\s+", " ", x["본문"].split("\n", 1)[-1])[:body_chars]
        sim = f"유사도 {x['벡터유사도']:.3f}" if x["벡터유사도"] is not None else "유사도 -"
        print(f"[{x['종류']}] {x['날짜'][:10]} | {x['제목'][:55]}")
        print(f"    경로: {x['검색경로']} (벡터 {x['벡터순위'] or '-'}위 / BM25 {x['BM25순위'] or '-'}위, {sim})")
        if x["종류"] == "특허" and owner_text(x["메타"]):
            print(f"    출원인: {owner_text(x['메타'])[:60]}")
        print(f"    {body}…")
        print(f"    출처: {x['출처']}")


if __name__ == "__main__":
    show(retrieve("알테오젠 기술수출"))
