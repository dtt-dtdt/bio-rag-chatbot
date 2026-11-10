"""
뉴스 노이즈 분류 — 방법 A(규칙 + LLM) 실행 → 그 결과로 분류 모델 학습 → 다음 임베딩 때 재사용

[단계]
  run()          ① 규칙으로 구간 나누기 (바이오·의료 용어 0회 / 1~5회 / 6회 이상)
                 ② 애매한 구간(1~5회) 전부 + 규칙 구간 표본(각 300건)을 GPT로 판정
                    → 규칙 구간 표본으로 '규칙이 얼마나 맞는지'도 측정
                 ③ 최종 라벨 = GPT 판정이 있으면 GPT, 없으면 규칙
                 ④ GPT가 판정한 기사로 분류 모델 학습 (기사 벡터 = 청크 임베딩 평균, 재임베딩 없음)
                    → 학습에 안 쓴 20%로 정확도 측정 → store/noise/news_bio_classifier.joblib 저장
  apply_to_db()  ⑤ (결과 확인 후 따로 실행) 본 DB 뉴스 청크에 메타데이터 추가
                    바이오관련(True/False), 바이오확률(0~1), 노이즈판정(규칙/LLM)   ※ 임베딩은 그대로
  tag_new_chunks(col, ids)  ⑥ indexer/embed.py가 새 뉴스를 넣은 뒤 자동 호출 → 저장된 모델로 분류·표시

[필요]  pip install scikit-learn joblib   (+ openai, chromadb, pandas, python-dotenv)
[실행]  from indexer import noise_news; noise_news.run() → 결과 확인 → noise_news.apply_to_db()
        (또는 python -m indexer noise → python -m indexer apply)
        GPT 판정 약 8,800회 (약 1~2시간, 약 $2~3). 중간에 멈춰도 이어서 진행.
        오래 걸리니 indexer/background.py의 start("indexer.noise_news")로 백그라운드 실행 권장
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

import numpy as np
import pandas as pd

# ============================== 설정 ==============================
import config

BASE_DIR = config.ROOT

RAW_NEWS_FILE = config.NEWS_RAW
MAIN_DB_DIR = config.CHROMA_DIR
OUT_DIR = config.NOISE_DIR
MODEL_FILE = OUT_DIR / "news_bio_classifier.joblib"
JUDGE_MODEL = "gpt-6-luna"
EMBED_MODEL = "text-embedding-3-small"   # 분류 모델은 이 임베딩으로만 쓸 수 있음
WORKERS = 8
MAX_RETRY = 30
JUDGE_CHARS = 800          # 판정에 쓸 본문 앞부분 길이
RULE_CHECK_N = 300         # 규칙 구간(0회, 6회 이상) 검사 표본 수
LOW_MAX, HIGH_MIN = 0, 6   # 용어 0회 = 노이즈 후보 / 6회 이상 = 바이오 후보 / 그 사이 = 애매(GPT 판정)
THRESHOLD = 0.3            # 분류 모델: 바이오확률이 이 값 미만이면 노이즈 (진짜 기사를 버리지 않게 낮게)
SEED = 42

TERMS = ["바이오", "제약", "의약", "신약", "임상", "치료", "의료", "병원", "헬스케어", "진단", "질환", "환자",
         "백신", "항체", "세포", "유전자", "식약처", "FDA", "건강", "디지털치료", "CDMO", "바이오시밀러", "의사",
         "약사", "웨어러블", "원격의료", "원격진료", "비대면 진료", "약물", "감염", "항암", "당뇨", "혈당",
         "유전체", "바이오마커", "의료기기", "보건"]
BOILERPLATE_RE = re.compile(r"\s*(기사를 더 보고 싶다면\??|기사 섹션 분류 안내)\s*")
# ==================================================================

TERM_RE = re.compile("|".join(map(re.escape, TERMS)))
_clients: dict = {}
_unsupported: set[str] = set()
_lock = threading.Lock()


def get_clients(need_openai=True):
    if need_openai and "openai" not in _clients:
        from openai import OpenAI
        try:
            from dotenv import load_dotenv
            load_dotenv(config.ENV_FILE)
        except ImportError:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(f"OPENAI_API_KEY가 없습니다. project3 폴더의 .env 파일에 넣어주세요: {config.ENV_FILE}")
        _clients["openai"] = OpenAI()
    if "main" not in _clients:
        import chromadb
        _clients["main"] = chromadb.PersistentClient(path=str(MAIN_DB_DIR))
    return _clients


def article_id(url, title, date):  # indexer/chunk.py와 동일
    return hashlib.sha1(f"{url}|{title}|{date}".encode("utf-8")).hexdigest()[:12]


# ---------------------------- GPT 판정 ----------------------------
JUDGE_PROMPT = """너는 뉴스 기사를 분류하는 사람이다.
[제목] {title}
[본문 앞부분] {text}

이 기사의 '주된 내용'이 바이오의약품·제약·바이오헬스케어·디지털헬스·의료(병원·의료기기·진단·보건 정책 포함) 산업에 관한 것인가?
- 헬스케어가 여러 사업 분야 중 하나로 잠깐 언급되거나, 행사 참여 기업 목록에만 있으면 false.
- 일반 경제·IT·금융·정치·지자체 행정·인사 기사는, 바이오·의료가 중심이 아니면 false.
JSON으로 출력: {{"bio_related": true/false, "reason": "한 문장"}}"""


def chat_json(prompt: str) -> dict:
    oa = get_clients()["openai"]
    attempt = 0
    while attempt < MAX_RETRY:
        params = dict(model=JUDGE_MODEL, timeout=120,
                      messages=[{"role": "system", "content": "Respond only with a valid JSON object."},
                                {"role": "user", "content": prompt}])
        if "temperature" not in _unsupported:
            params["temperature"] = 0
        if "response_format" not in _unsupported:
            params["response_format"] = {"type": "json_object"}
        if "reasoning_effort" not in _unsupported:
            params["reasoning_effort"] = "low"     # 단순 분류라 빠르고 저렴하게
        try:
            r = oa.chat.completions.create(**params)
            text = r.choices[0].message.content or ""
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", text, re.S)
                return json.loads(m.group(0)) if m else {}
        except Exception as e:
            status, msg = getattr(e, "status_code", None), str(e)
            if status == 400:
                bad = next((p for p in ("temperature", "response_format", "reasoning_effort")
                            if p in msg and p not in _unsupported), None)
                if bad:
                    with _lock:
                        if bad not in _unsupported:
                            _unsupported.add(bad)
                            print(f"    ℹ️ {JUDGE_MODEL}은(는) '{bad}' 설정을 지원하지 않아 빼고 진행해요")
                    continue
            if (status is not None and status != 429 and status < 500) or attempt == MAX_RETRY - 1:
                print(f"    ❌ GPT 호출 실패: {type(e).__name__}: {msg[:200]}")
                raise
            m = re.search(r"try again in ([\d.]+)\s*(ms|s)", msg)
            wait = max((float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 1) if m else 0,
                       min(60, 3 * 2 ** attempt))
            print(f"    재시도 {attempt + 1} ({type(e).__name__}) {wait:.0f}초 대기")
            time.sleep(wait)
            attempt += 1
    return {}


def judge_all(tasks: dict) -> dict:
    """tasks = {기사ID: (제목, 본문)} → {기사ID: {"bio_related":…, "reason":…}}  (하나씩 저장, 이어서 진행)"""
    path = OUT_DIR / "judge_bio.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["key"]] = rec["out"]
    todo = [k for k in tasks if k not in done]
    print(f"\n[GPT 판정] 전체 {len(tasks):,} / 이미 판정 {len(tasks) - len(todo):,} / 새로 판정 {len(todo):,}")
    if not todo:
        return done

    def work(k):
        title, text = tasks[k]
        return k, chat_json(JUDGE_PROMPT.format(title=title, text=text))

    with path.open("a", encoding="utf-8") as f:
        print("  연결 확인 중 (첫 판정 1개)…")
        k, out = work(todo[0])
        f.write(json.dumps({"key": k, "out": out}, ensure_ascii=False) + "\n")
        done[k] = out
        print(f"  ✅ 연결 정상 ({JUDGE_MODEL})")
        with ThreadPoolExecutor(WORKERS) as ex:
            for n, (k, out) in enumerate(ex.map(work, todo[1:]), 2):
                with _lock:
                    f.write(json.dumps({"key": k, "out": out}, ensure_ascii=False) + "\n")
                    f.flush()
                    done[k] = out
                if n % 200 == 0 or n == len(todo):
                    print(f"  진행 {n:,}/{len(todo):,}")
    return done


# ---------------------------- 기사 벡터 (청크 임베딩 평균) ----------------------------
def article_vectors(col, article_ids: set | None = None) -> dict[str, np.ndarray]:
    """본 DB 뉴스 청크 임베딩을 기사별로 평균. article_ids가 있으면 그 기사만"""
    sums, counts, off = {}, {}, 0
    while True:
        r = col.get(limit=2000, offset=off, include=["embeddings", "metadatas"])
        if not len(r["ids"]):
            break
        for e, m in zip(r["embeddings"], r["metadatas"]):
            a = m["기사ID"]
            if article_ids is not None and a not in article_ids:
                continue
            v = np.asarray(e, dtype=np.float32)
            sums[a] = sums.get(a, 0) + v
            counts[a] = counts.get(a, 0) + 1
        off += len(r["ids"])
    out = {}
    for a, s in sums.items():
        v = s / counts[a]
        out[a] = v / (np.linalg.norm(v) or 1)
    return out


# ---------------------------- ①~④ 실행 ----------------------------
def run():
    t0 = time.time()
    try:  # 오래 걸리는 GPT 판정 전에 필요한 라이브러리부터 확인
        import sklearn  # noqa: F401
        import joblib  # noqa: F401
    except ImportError:
        print("❌ scikit-learn 또는 joblib이 없어요. 새 셀에서 %pip install scikit-learn joblib 실행 후 "
              "커널을 재시작하고 다시 실행해 주세요.")
        return None
    OUT_DIR.mkdir(exist_ok=True)

    # ① 규칙 구간
    df = pd.read_csv(RAW_NEWS_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["기사ID"] = [article_id(u, t, d) for u, t, d in zip(df["원문URL"], df["제목"], df["발행일"])]
    df = df.drop_duplicates("기사ID").reset_index(drop=True)
    df["본문"] = df["본문전문"].map(lambda b: re.sub(r"\s+", " ", BOILERPLATE_RE.sub(" ", b)).strip())
    df["용어수"] = (df["제목"] + " " + df["본문"].str[:600]).map(lambda s: len(TERM_RE.findall(s)))
    df["구간"] = np.where(df["용어수"] <= LOW_MAX, "노이즈 후보",
                        np.where(df["용어수"] >= HIGH_MIN, "바이오 후보", "애매"))
    print("기사", f"{len(df):,}건 →", df["구간"].value_counts().to_dict())

    # ② GPT 판정 대상: 애매 전부 + 규칙 구간 표본
    check = pd.concat([df[df["구간"] == "애매"],
                       df[df["구간"] == "노이즈 후보"].sample(min(RULE_CHECK_N, (df["구간"] == "노이즈 후보").sum()), random_state=SEED),
                       df[df["구간"] == "바이오 후보"].sample(min(RULE_CHECK_N, (df["구간"] == "바이오 후보").sum()), random_state=SEED)])
    judged = judge_all({r.기사ID: (r.제목, r.본문[:JUDGE_CHARS]) for r in check.itertuples()})

    # ③ 최종 라벨
    df["GPT판정"] = df["기사ID"].map(lambda a: judged.get(a, {}).get("bio_related"))
    df["GPT근거"] = df["기사ID"].map(lambda a: judged.get(a, {}).get("reason", ""))
    df["바이오관련"] = df["GPT판정"].where(df["GPT판정"].notna(), df["구간"] == "바이오 후보").astype(bool)
    df["판정출처"] = np.where(df["GPT판정"].notna(), "LLM", "규칙")

    rule_rows = []
    for g, expect in (("노이즈 후보", False), ("바이오 후보", True)):
        s = df[(df["구간"] == g) & df["GPT판정"].notna()]
        if len(s):
            rule_rows.append(f"- 규칙 '{g}' 표본 {len(s)}건 중 GPT도 같은 판단: {(s['GPT판정'] == expect).mean():.1%}")
    amb = df[df["구간"] == "애매"]
    amb_bio = amb["바이오관련"].mean() if len(amb) else 0

    # ④ 분류 모델 학습 (GPT 판정 기사만 사용)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
    from sklearn.model_selection import train_test_split
    import joblib

    col = get_clients(need_openai=False)["main"].get_collection("news")
    labeled = df[df["판정출처"] == "LLM"]
    print(f"\n기사 벡터 계산 중… (청크 임베딩 평균, 학습용 {len(labeled):,}건)")
    vecs = article_vectors(col, set(labeled["기사ID"]))
    labeled = labeled[labeled["기사ID"].isin(vecs)]
    X = np.stack([vecs[a] for a in labeled["기사ID"]])
    y = labeled["바이오관련"].astype(int).values
    if len(set(y)) < 2 or min(np.bincount(y)) < 10:
        print(f"❌ 학습 불가: GPT 판정 결과가 한쪽으로 치우쳤어요 (바이오 {int(y.sum())}건 / 노이즈 {int(len(y) - y.sum())}건). "
              "noise/judge_bio.jsonl의 판정 내용을 확인해 주세요.")
        return df
    aid_arr = np.asarray(labeled["기사ID"].tolist(), dtype=object)   # 최신 pandas(pyarrow 문자열) 대응
    Xtr, Xte, ytr, yte, _, ids_te = train_test_split(X, y, aid_arr, test_size=0.2,
                                                     random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p >= THRESHOLD).astype(int)          # 1 = 바이오
    noise_prec = precision_score(yte, pred, pos_label=0, zero_division=0)   # '노이즈'라고 한 것 중 진짜 노이즈
    noise_rec = recall_score(yte, pred, pos_label=0, zero_division=0)       # 실제 노이즈 중 찾아낸 비율
    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred, labels=[0, 1])

    clf.fit(X, y)   # 평가 후 전체 데이터로 다시 학습해서 저장
    joblib.dump({"model": clf, "threshold": THRESHOLD, "embed_model": EMBED_MODEL,
                 "trained_at": time.strftime("%Y-%m-%d %H:%M"), "n_train": len(y),
                 "test_metrics": {"accuracy": acc, "noise_precision": noise_prec, "noise_recall": noise_rec}},
                MODEL_FILE)

    df[["기사ID", "제목", "발행일", "매체명", "수집키워드", "용어수", "구간", "판정출처", "GPT판정", "GPT근거", "바이오관련"]].to_csv(
        OUT_DIR / "news_labels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"기사ID": ids_te, "정답(LLM)": yte, "예측": pred, "바이오확률": p.round(3)}).to_csv(
        OUT_DIR / "classifier_test_predictions.csv", index=False, encoding="utf-8-sig")

    report = (
        f"# 뉴스 노이즈 분류 결과\n\n- 실행: {time.strftime('%Y-%m-%d %H:%M')} / 판정 모델: {JUDGE_MODEL}"
        + (f" (뺀 설정: {', '.join(sorted(_unsupported))})" if _unsupported else "") + "\n\n"
        f"## 방법 A (규칙 + LLM)\n- 기사 {len(df):,}건: " + ", ".join(f"{k} {v:,}" for k, v in df['구간'].value_counts().items()) + "\n"
        + "\n".join(rule_rows) + "\n"
        f"- 애매 구간 {len(amb):,}건 중 바이오 관련: {amb_bio:.1%}\n"
        f"- **최종: 바이오 {df['바이오관련'].sum():,}건 / 노이즈 {(~df['바이오관련']).sum():,}건 ({(~df['바이오관련']).mean():.1%})**\n\n"
        f"## 분류 모델 (로지스틱 회귀, 학습 {len(y):,}건 = GPT 판정 기사)\n"
        f"- 시험(학습에 안 쓴 20%, {len(yte):,}건), 기준 바이오확률 < {THRESHOLD} → 노이즈\n"
        f"- 정확도 {acc:.1%} / 노이즈 정밀도 {noise_prec:.1%} (노이즈라고 한 것 중 진짜 노이즈) / "
        f"노이즈 재현율 {noise_rec:.1%} (실제 노이즈 중 찾아낸 비율)\n"
        f"- 혼동행렬 [실제 노이즈→(노이즈, 바이오)={cm[0].tolist()}, 실제 바이오→(노이즈, 바이오)={cm[1].tolist()}]\n"
        f"  · 실제 바이오인데 노이즈로 버려지는 기사: {cm[1][0]}건\n"
        f"- 저장: {MODEL_FILE.name}\n")
    (OUT_DIR / "noise_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"저장: noise/news_labels.csv, classifier_test_predictions.csv, noise_report.md ({(time.time() - t0) / 60:.0f}분)")
    print("→ 결과를 확인한 뒤 apply_to_db()를 실행하면 본 DB에 표시돼요.")
    return df


# ---------------------------- ⑤ 본 DB에 반영 ----------------------------
def _merge_update(col, ids, new_metas):
    """기존 메타데이터를 유지하면서 키만 추가 (임베딩·본문은 그대로)"""
    for i in range(0, len(ids), 500):          # DB 한도에 걸리지 않게 500개씩
        part = ids[i:i + 500]
        cur = col.get(ids=part, include=["metadatas"])
        by_id = dict(zip(cur["ids"], cur["metadatas"]))
        col.update(ids=list(by_id), metadatas=[{**by_id[x], **new_metas[x]} for x in by_id])


def apply_to_db():
    import joblib
    labels = pd.read_csv(OUT_DIR / "news_labels.csv", dtype=str, keep_default_na=False)
    lab = dict(zip(labels["기사ID"], labels["바이오관련"] == "True"))
    src = dict(zip(labels["기사ID"], labels["판정출처"]))
    bundle = joblib.load(MODEL_FILE)
    col = get_clients(need_openai=False)["main"].get_collection("news")

    print("기사 벡터 계산 중… (전체)")
    vecs = article_vectors(col)
    aids = list(vecs)
    prob = dict(zip(aids, bundle["model"].predict_proba(np.stack([vecs[a] for a in aids]))[:, 1].round(4)))

    all_ids, all_metas, off = [], [], 0      # 한 번에 다 꺼내면 DB 한도 초과 → 나눠서
    while True:
        part = col.get(include=["metadatas"], limit=2000, offset=off)
        if not part["ids"]:
            break
        all_ids += part["ids"]; all_metas += part["metadatas"]
        off += len(part["ids"])
    new = {}
    for _id, m in zip(all_ids, all_metas):
        a = m["기사ID"]
        if a in lab:   # 방법 A 라벨 우선
            new[_id] = {"바이오관련": bool(lab[a]), "바이오확률": float(prob.get(a, -1)), "노이즈판정": src[a]}
        else:          # 라벨이 없는 기사는 분류 모델로
            new[_id] = {"바이오관련": bool(prob.get(a, 1) >= bundle["threshold"]),
                        "바이오확률": float(prob.get(a, -1)), "노이즈판정": "분류모델"}
    _merge_update(col, list(new), new)
    n_noise = sum(not v["바이오관련"] for v in new.values())
    agree = np.mean([(prob[a] >= bundle["threshold"]) == lab[a] for a in aids if a in lab])
    print(f"✅ 뉴스 청크 {len(new):,}개에 표시 완료 (노이즈 청크 {n_noise:,}개)")
    print(f"   참고: 방법 A 라벨과 분류 모델 예측 일치율 {agree:.1%}")
    print("   검색에서 제외하려면 where={'바이오관련': True} 조건을 쓰면 돼요.")


# ---------------------------- ⑥ 다음 임베딩 때 (indexer/embed.py가 호출) ----------------------------
def tag_new_chunks(col, new_ids: list[str]):
    """새로 넣은 뉴스 청크를 저장된 분류 모델로 분류해서 표시. 모델이 없으면 건너뜀"""
    if not new_ids:
        return
    if not MODEL_FILE.exists():
        print("  (노이즈 분류 모델이 없어 바이오관련 표시는 건너뛰었어요)")
        return
    import joblib
    bundle = joblib.load(MODEL_FILE)
    if bundle.get("embed_model") != EMBED_MODEL:
        print(f"  ⚠️ 분류 모델은 {bundle.get('embed_model')} 임베딩용이라 건너뛰었어요")
        return
    r = col.get(ids=new_ids, include=["metadatas"])
    aids = {m["기사ID"] for m in r["metadatas"]}
    # 새 청크의 기사 전체(이전에 들어간 청크 포함)로 평균
    full = col.get(where={"기사ID": {"$in": list(aids)}}, include=["embeddings", "metadatas"])
    sums, cnt = {}, {}
    for e, m in zip(full["embeddings"], full["metadatas"]):
        a = m["기사ID"]
        sums[a] = sums.get(a, 0) + np.asarray(e, dtype=np.float32)
        cnt[a] = cnt.get(a, 0) + 1
    aid_list = list(sums)
    X = np.stack([(sums[a] / cnt[a]) / (np.linalg.norm(sums[a] / cnt[a]) or 1) for a in aid_list])
    prob = dict(zip(aid_list, bundle["model"].predict_proba(X)[:, 1]))
    new = {_id: {"바이오관련": bool(prob[m["기사ID"]] >= bundle["threshold"]),
                 "바이오확률": round(float(prob[m["기사ID"]]), 4), "노이즈판정": "분류모델"}
           for _id, m in zip(full["ids"], full["metadatas"]) if m.get("노이즈판정") != "LLM"}
    _merge_update(col, list(new), new)
    print(f"  노이즈 분류: 새 기사 {len(aid_list):,}건 중 노이즈 {sum(p < bundle['threshold'] for p in prob.values()):,}건 표시")


if __name__ == "__main__":
    run()
