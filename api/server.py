"""
챗봇 API 서버 — Lovable 같은 외부 웹 화면이 RAG 챗봇을 부를 수 있게 HTTP로 열어 줌 (조회 전용)

[실행]  (project3 폴더에서)
  python -m uvicorn api.server:app --port 8000
  → http://localhost:8000/api/health 가 열리면 정상. 처음 시작할 때 검색 색인을 불러오느라 수십 초 걸려요.
[외부(Lovable)에서 부르기]  내 PC를 터널로 열기 (예: ngrok)
  ngrok http 8000        → https://xxxx.ngrok-free.app 같은 주소를 Lovable의 API_BASE_URL에 넣기
[보안]
  - OpenAI 키는 이 서버의 .env에만 있고 화면(브라우저)으로 나가지 않아요.
  - 터널 주소를 아는 사람은 누구나 질문할 수 있고 비용은 내 키로 나가요.
    .env에 API_TOKEN=아무문자열 을 넣으면, 요청 헤더 X-API-Key가 같은 값일 때만 응답해요.
  - CORS: .env의 ALLOWED_ORIGINS(쉼표로 구분)로 허용할 화면 주소를 정할 수 있어요. 없으면 모두 허용.
[엔드포인트]
  GET  /api/health          서버·DB 상태
  POST /api/ask             질문 → 답변 + 출처 (멀티턴: 이전 대화를 history로)
  GET  /api/eval/summary    평가 결과 (버전·실험별 지표, 최종 버전의 실패 유형·원인 계층·거절 정확도)
"""
from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
import threading
import time

import pandas as pd
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import config

FINAL = "v5"
HISTORY_TURNS = 3                 # 질문 분석에 넘길 이전 대화 수 (질문+답변 한 쌍 = 1) — Streamlit 화면과 같음
MAX_QUESTION = 500                # 질문 최대 길이
DATE_RE = re.compile(r"^(19|20)\d{2}(-(0[1-9]|1[0-2]))?$")
_lock = threading.Lock()          # 검색 색인은 동시에 여러 질문을 처리하도록 만들지 않아서 한 번에 하나씩
_state: dict = {}


def _env():
    from dotenv import load_dotenv
    load_dotenv(config.ENV_FILE)
    return os.getenv("API_TOKEN", "").strip(), [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]


API_TOKEN, ALLOWED_ORIGINS = _env()


def _chain():
    """처음 요청 때 체인·검색 색인을 한 번만 불러옴"""
    if "rc" not in _state:
        from rag import chain as rc
        rc.rt.get_index("patents")
        rc.rt.get_index("news")
        _state["rc"] = rc
    return _state["rc"]


def _auth(request: Request):
    if API_TOKEN and request.headers.get("x-api-key", "") != API_TOKEN:
        return JSONResponse({"error": "unauthorized", "message": "X-API-Key가 맞지 않아요."}, status_code=401)
    return None


def _source(s: dict, cited: set) -> dict:
    """출처 1건 → 화면용 (영문 키)"""
    return {"number": s["번호"], "kind": "patent" if s["종류"] == "특허" else "news", "title": s["제목"],
            "date": s["날짜"], "url": s.get("링크") or None, "reference": s["출처"],
            "applicant": s.get("출원인") or None, "media": s.get("매체") or None,
            "body": s["본문"], "cited": s["번호"] in cited}


# ---------------------------- /api/health ----------------------------
async def health(request: Request):
    rc = _state.get("rc")
    return JSONResponse({"ok": True, "ready": rc is not None, "final_version": FINAL,
                         "model": rc.CHAT_MODEL if rc else None, "auth_required": bool(API_TOKEN)})


# ---------------------------- /api/ask ----------------------------
def _ask_sync(question: str, history: list, filters: dict) -> dict:
    rc = _chain()
    with _lock:
        out = rc.ask(question, history=history, **filters)
    cited = set(out.get("인용번호", []))
    a = out.get("analysis") or {}
    cond = out.get("적용조건", {}) or {}
    return {
        "type": {"답변": "answer", "조언 제한": "advice_limited", "거절": "refused", "안내": "info"}.get(out["type"], out["type"]),
        "refusal_reason": out.get("거절사유"),
        "answer": out["answer"],
        "sources": [_source(s, cited) for s in out.get("sources", [])],
        "analysis": {"category": a.get("category"), "search_query": a.get("search_query"),
                     "target": cond.get("target", "both"), "date_from": cond.get("date_from"),
                     "date_to": cond.get("date_to"), "date_relaxed": bool(out.get("기간완화"))} if a else None,
        "timing": {"analyze_s": round(out.get("분석초", 0), 2), "search_s": round(out.get("검색초", 0), 2),
                   "answer_s": round(out.get("답변초", 0), 2), "total_s": round(out.get("전체초", 0), 2)},
    }


async def ask(request: Request):
    if (denied := _auth(request)):
        return denied
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "bad_request", "message": "JSON 본문이 필요해요."}, status_code=400)
    question = str(body.get("question", "")).strip()
    if not question or len(question) > MAX_QUESTION:
        return JSONResponse({"error": "bad_request", "message": f"질문은 1~{MAX_QUESTION}자여야 해요."}, status_code=400)
    history = [(m.get("role"), str(m.get("content", ""))) for m in (body.get("history") or [])
               if isinstance(m, dict) and m.get("role") in ("user", "assistant")][-2 * HISTORY_TURNS:]
    f = body.get("filters") or {}
    filters = {}
    if f.get("target") in ("both", "patent", "news"):
        filters["target"] = f["target"]
    if f.get("field") in ("바이오의약품", "바이오헬스케어"):
        filters["field"] = f["field"]
    for k in ("date_from", "date_to"):
        if f.get(k) and DATE_RE.match(str(f[k])):
            filters[k] = str(f[k])
    from starlette.concurrency import run_in_threadpool
    try:
        result = await run_in_threadpool(_ask_sync, question, history, filters)
    except Exception as e:     # 외부 오류 문구에는 키 일부가 섞일 수 있어 종류만 돌려줌 (RuntimeError는 우리 코드의 안내문)
        detail = str(e)[:200] if isinstance(e, RuntimeError) else type(e).__name__
        return JSONResponse({"error": "server_error", "message": f"답변을 만들지 못했어요: {detail}"}, status_code=500)
    return JSONResponse(result)


# ---------------------------- /api/eval/summary ----------------------------
def _read(path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.DataFrame()


def _summary() -> dict:
    root = config.EVAL_RESULTS_DIR
    notes, decisions = {}, {}
    reg = root / "experiments.csv"
    if reg.exists():
        for r in _read(reg).itertuples(index=False):
            notes[r.태그] = r.변수 if r.변수.startswith("(") else f"{r.변수}: {r.기준값} → {r.변경값}"
            decisions[r.태그] = r.결정
    versions = []
    for folder in root.iterdir():
        p = folder / f"answer_eval_details_{folder.name}.csv"
        if not (folder.is_dir() and p.exists()):
            continue
        d = _read(p)
        ev, base = d[d["그룹"].str.startswith("평가")], d[~d["그룹"].str.startswith("설계")]
        ref = folder / "refusal_eval.csv"
        refusal = (_read(ref)["올바른 거절"] == "True").mean() if ref.exists() else None
        versions.append({"tag": folder.name, "note": notes.get(folder.name, ""), "decision": decisions.get(folder.name, ""),
                         "correct": round((ev["정답성"] == "correct").mean(), 4),
                         "grounded": round((ev["근거성"] == "yes").mean(), 4),
                         "gold_retrieved": round((ev["정답검색됨"] == "True").mean(), 4),
                         "refusal_ok": round((base["거절적절"] == "True").mean(), 4),
                         "failures": int((base["실패"] == "True").sum()), "questions": len(base),
                         "refusal_accuracy": None if refusal is None else round(refusal, 4)})
    order = lambda t: (0, int(t[1:]), "") if t[:1] == "v" and t[1:].isdigit() else (1, 0, t)   # noqa: E731
    versions.sort(key=lambda v: order(v["tag"]))
    final = {}
    fd = root / FINAL
    if (fd / f"answer_eval_details_{FINAL}.csv").exists():
        d = _read(fd / f"answer_eval_details_{FINAL}.csv")
        final["failure_types"] = d[d["실패"] == "True"]["실패유형"].value_counts().to_dict()
    if (fd / "failure_layers.csv").exists():
        final["failure_layers"] = _read(fd / "failure_layers.csv")["계층"].value_counts().to_dict()
    return {"final_version": FINAL, "versions": versions, "final": final,
            "generated_at": time.strftime("%Y-%m-%d %H:%M")}


async def eval_summary(request: Request):
    if (denied := _auth(request)):
        return denied
    return JSONResponse(_summary())


@asynccontextmanager
async def lifespan(app):
    print("ℹ️ 검색 색인을 불러오는 중… (처음 한 번, 수십 초)")
    from starlette.concurrency import run_in_threadpool
    await run_in_threadpool(_chain)
    print("✅ 준비 완료 — /api/health")
    yield


app = Starlette(
    routes=[Route("/api/health", health), Route("/api/ask", ask, methods=["POST"]),
            Route("/api/eval/summary", eval_summary)],
    middleware=[Middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS or ["*"], allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-API-Key", "ngrok-skip-browser-warning"])],
    lifespan=lifespan,
)
