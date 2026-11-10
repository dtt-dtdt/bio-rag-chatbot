"""
임베딩을 백그라운드로 실행 (노트북 셀·창을 닫아도 계속 실행)

[준비]  project3 폴더에서 실행해요 (from indexer import background).

[사용 - Jupyter, project3 폴더에서]
  from indexer.background import start, status, stop
  start()      # 백그라운드 시작 → logs 폴더에 로그 파일 생성
  status()     # 실행 중인지 + 로그 마지막 15줄 보기 (언제든 다시 실행해서 확인)
  stop()       # 중지 (중지해도 저장된 건 남고, 다시 start() 하면 이어서 처리)

[주의]
  - 컴퓨터가 꺼지거나 절전 모드에 들어가면 멈춰요. 다시 start() 하면 이어서 처리해요.
  - 실행 중에는 노트북에서 embed_all 셀을 동시에 돌리지 마세요 (같은 DB에 동시에 쓰면 꼬일 수 있어요).
    검색·조회는 같이 써도 괜찮아요.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import config

BASE_DIR = config.ROOT

MODULE = "indexer.embed"
LOG_DIR = BASE_DIR / "logs"
PID_FILE = BASE_DIR / "embed_bg.pid"
IS_WINDOWS = os.name == "nt"


def _is_running(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:  # 이 노트북이 띄운 프로세스가 끝났으면 정리 (끝났는데 '실행 중'으로 보이는 문제 방지)
        done_pid, _ = os.waitpid(pid, os.WNOHANG)
        if done_pid == pid:
            return False
    except ChildProcessError:
        pass  # 다른 세션에서 띄운 프로세스
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _current_pid() -> int | None:
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip() or 0)
        if pid and _is_running(pid):
            return pid
        PID_FILE.unlink()
    return None


def start(module: str = MODULE, sample: str = "all"):
    """module: 실행할 모듈 (예: 'indexer.noise_patent').  sample='all' 이면 전체 임베딩, '200' 이면 시험용"""
    import importlib.util
    if importlib.util.find_spec(module) is None:
        print(f"❌ {module} 모듈을 찾지 못했어요. project3 폴더에서 실행해 주세요.")
        return
    if (pid := _current_pid()):
        print(f"이미 실행 중이에요 (PID {pid}). status()로 진행 상황을 확인하세요.")
        return

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{module.split('.')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    env = {**os.environ, "EMBED_SAMPLE_N": sample, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    log = open(log_path, "w", encoding="utf-8")
    kwargs = dict(cwd=str(BASE_DIR), stdout=log, stderr=subprocess.STDOUT, env=env)
    if IS_WINDOWS:  # 노트북·창을 닫아도 계속 실행되도록 분리
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    p = subprocess.Popen([sys.executable, "-u", "-m", module], **kwargs)
    PID_FILE.write_text(str(p.pid))
    print(f"✅ 백그라운드 시작 (PID {p.pid})")
    print(f"   로그: {log_path}")
    print("   status()로 진행 상황을 확인하세요.")


def status(lines: int = 15):
    pid = _current_pid()
    print("상태:", f"🔄 실행 중 (PID {pid})" if pid else "⏹ 실행 중 아님")
    logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime) if LOG_DIR.exists() else []
    if not logs:
        print("로그가 아직 없어요.")
        return
    text = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"--- {logs[-1].name} (마지막 {lines}줄) ---")
    print("\n".join(text[-lines:]))
    if not pid and any("Traceback" in t for t in text):
        print("\n⚠ 오류로 멈춘 것 같아요. 위 로그를 보내주세요.")


def stop():
    pid = _current_pid()
    if not pid:
        print("실행 중인 작업이 없어요.")
        return
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(os.getpgid(pid), 15)
    PID_FILE.unlink(missing_ok=True)
    print(f"⏹ 중지했어요 (PID {pid}). 저장된 건 남아 있고, start()로 이어서 처리할 수 있어요.")
