#!/usr/bin/env python3
"""
Launch PaperHelper with one command.

Usage:
    python run_app.py
    ./run_app.py
    python run_app.py --port 8502
    python run_app.py --open-browser
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_FILE = ROOT / "app.py"
SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PaperHelper Streamlit app.")
    parser.add_argument("--host", default="localhost", help="Server host. Default: localhost")
    parser.add_argument("--port", type=int, default=8501, help="Server port. Default: 8501")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Ask Streamlit to open the browser automatically.",
    )
    return parser.parse_args()


def ensure_ready() -> None:
    if not APP_FILE.exists():
        raise SystemExit(f"找不到主程式：{APP_FILE}")

    if not SECRETS_FILE.exists():
        print(
            "提醒：找不到 .streamlit/secrets.toml。若使用 Google/Groq/OpenAI，請先設定 API Key。",
            file=sys.stderr,
        )

    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit(
            "找不到 streamlit 套件。請先執行：\n"
            "  pip install -r requirements.txt"
        )


def main() -> int:
    args = parse_args()
    ensure_ready()

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "false" if args.open_browser else "true",
    ]

    print(f"啟動 PaperHelper：http://{args.host}:{args.port}")
    print("若瀏覽器沒有自動開啟，請手動打開上方網址。按 Ctrl+C 可停止服務。")
    try:
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        print("\n已停止 PaperHelper。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
