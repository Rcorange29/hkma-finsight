"""
start.py — Boot the Financial Report Q&A Engine from scratch.

Steps:
  1. Report .env / chat-model key status (a key is only needed for answers).
  2. Ingest PDFs (skipped when chroma_db already exists; use --reingest to force).
  3. Start the FastAPI server on port 8000.
  4. Start the Streamlit UI on port 8501.
  5. Wait until both ports are accepting connections, then open the browser.

Ingestion and retrieval are fully local (BGE embeddings + cross-encoder).
Only the answer step needs ARK_API_KEY (Doubao) or OPENAI_API_KEY.

Usage:
    python start.py            # skip ingestion if chroma_db exists
    python start.py --reingest # always wipe chroma_db and re-ingest
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
ENV_FILE = ROOT / ".env"

API_HOST = "127.0.0.1"
API_PORT = 8000
UI_PORT = 8501

PYTHON = sys.executable          # same interpreter / venv that launched this script
READY_TIMEOUT = 60               # seconds to wait for each server to become ready
POLL_INTERVAL = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(msg: str) -> None:
    print(f"[start] {msg}", flush=True)


def _abort(msg: str) -> None:
    print(f"[start] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, label: str) -> None:
    _print(f"Waiting for {label} on {host}:{port} …")
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        if _port_open(host, port):
            _print(f"{label} is ready.")
            return
        time.sleep(POLL_INTERVAL)
    _abort(f"{label} did not become ready within {READY_TIMEOUT}s.")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_env() -> None:
    """
    Check configuration without blocking local use.

    Ingestion and retrieval run fully on the machine (local embeddings and the
    local cross-encoder). Only the answer step calls a chat model, which needs
    ARK_API_KEY (Doubao/Volcano Ark) or OPENAI_API_KEY. So a missing key is a
    warning — the app still starts and the question box explains what to set.
    """
    has_key = bool(os.getenv("ARK_API_KEY", "").strip()) or bool(
        os.getenv("OPENAI_API_KEY", "").strip()
    )
    if not ENV_FILE.exists():
        _print(
            f"No .env at {ENV_FILE}. Ingestion/retrieval run locally; to get "
            "answers, copy .env.example → .env and set ARK_API_KEY (Doubao) "
            "or OPENAI_API_KEY."
        )
        return
    # start.py doesn't load .env itself (the child processes do), so read the
    # key lines directly to report whether answering is configured.
    if not has_key:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in ("ARK_API_KEY", "OPENAI_API_KEY") and value.strip().strip(
                "\"'") not in ("", "your-key-here"):
                has_key = True
                break
    if not has_key:
        _print(
            "No ARK_API_KEY / OPENAI_API_KEY found in .env: the app starts, but "
            "answering questions is disabled until a key is set. Ingestion and "
            "retrieval work without it."
        )
        return
    _print(".env OK (a chat-model API key is set).")


def check_data() -> bool:
    pdfs = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    if not pdfs:
        _print(f"No PDF files in {DATA_DIR} — upload filings from the UI sidebar.")
        return False
    _print(f"Found {len(pdfs)} PDF(s): {', '.join(p.name for p in pdfs)}")
    return True


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_ingest() -> None:
    _print("Running financial_rag.ingest …")
    result = subprocess.run([PYTHON, "-m", "financial_rag.ingest"], cwd=ROOT)
    if result.returncode != 0:
        _abort("Ingestion failed. Check the output above.")
    _print("Ingestion complete.")


def api_env() -> dict[str, str]:
    """The API checks the same login the UI uses, or trusts the local user."""
    env = os.environ.copy()
    auth = login_settings()
    if auth is None:
        env["AUTH_DISABLED"] = "1"
        return env
    # Cognito's metadata URL is <issuer>/.well-known/openid-configuration
    metadata = auth.get("server_metadata_url", "")
    env.setdefault("COGNITO_ISSUER", metadata.split("/.well-known/")[0])
    env.setdefault("COGNITO_CLIENT_ID", auth.get("client_id", ""))
    return env


def start_api() -> subprocess.Popen:
    _print(f"Starting FastAPI server on port {API_PORT} …")
    proc = subprocess.Popen(
        [
            PYTHON, "-m", "uvicorn",
            "financial_rag.api:app",
            "--host", API_HOST,
            "--port", str(API_PORT),
        ],
        cwd=ROOT,
        env=api_env(),
    )
    return proc


def login_settings() -> dict[str, str] | None:
    """The [auth] section of .streamlit/secrets.toml, or None without one.

    Read line by line rather than with tomllib, which needs Python 3.11+.
    """
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if not secrets.exists():
        return None
    section = None
    settings: dict[str, str] | None = None
    for line in secrets.read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
            if section == "[auth]":
                settings = {}
        elif section == "[auth]" and "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            settings[key.strip()] = value.strip().strip('"').strip("'")
    return settings


def login_configured() -> bool:
    return login_settings() is not None


def start_streamlit() -> subprocess.Popen:
    _print(f"Starting Streamlit on port {UI_PORT} …")
    env = os.environ.copy()
    if not login_configured():
        _print("No [auth] in .streamlit/secrets.toml — running without login (local mode).")
        env["AUTH_DISABLED"] = "1"
    proc = subprocess.Popen(
        [
            PYTHON, "-m", "streamlit", "run",
            str(ROOT / "financial_rag" / "app.py"),
            "--server.port", str(UI_PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    return proc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Boot the Financial RAG Q&A engine.")
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Wipe chroma_db and re-ingest all PDFs even if the store already exists.",
    )
    args = parser.parse_args()

    _print("=== Financial Report Q&A Engine — startup ===")

    check_env()
    has_pdfs = check_data()

    # Ingestion
    if not has_pdfs:
        _print("Skipping ingestion.")
    elif args.reingest:
        if CHROMA_DIR.exists():
            _print("--reingest: removing existing chroma_db …")
            shutil.rmtree(CHROMA_DIR)
        run_ingest()
    elif not CHROMA_DIR.exists():
        _print("chroma_db not found — running ingestion …")
        run_ingest()
    else:
        _print("chroma_db exists — skipping ingestion (pass --reingest to rebuild).")

    # Start servers
    api_proc = start_api()
    ui_proc = start_streamlit()

    # Register cleanup so Ctrl-C shuts both down cleanly
    procs = [api_proc, ui_proc]

    def _shutdown(sig, frame):
        _print("Shutting down servers …")
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Wait until ready, then open browser
    _wait_for_port(API_HOST, API_PORT, "FastAPI")
    _wait_for_port("localhost", UI_PORT, "Streamlit")

    url = f"http://localhost:{UI_PORT}"
    _print(f"Opening browser at {url}")
    webbrowser.open(url)

    _print("All services running. Press Ctrl-C to stop.")
    try:
        # Keep the process alive so Ctrl-C works
        while True:
            # Restart a crashed server automatically
            for i, proc in enumerate(procs):
                if proc.poll() is not None:
                    label = "FastAPI" if i == 0 else "Streamlit"
                    _print(f"{label} exited unexpectedly (code {proc.returncode}) — restarting …")
                    procs[i] = start_api() if i == 0 else start_streamlit()
            time.sleep(2)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
