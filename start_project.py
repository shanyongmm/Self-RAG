from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start the Self-RAG FastAPI project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-reload", action="store_true", help="Disable uvicorn reload.")
    args = parser.parse_args()

    python_executable = _project_python()
    command = [
        str(python_executable),
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if not args.no_reload:
        command.append("--reload")

    _print_start_message(args.host, args.port, python_executable)
    return subprocess.call(command, cwd=PROJECT_ROOT)


def _project_python() -> Path:
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def _print_start_message(host: str, port: int, python_executable: Path) -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print("WARNING: .env was not found. Copy .env.example to .env and fill it first.")

    print(f"Python: {python_executable}")
    print(f"Starting Self-RAG API: http://{host}:{port}/")
    print(f"Swagger docs: http://{host}:{port}/docs")
    print("Press Ctrl+C to stop.")


if __name__ == "__main__":
    raise SystemExit(main())
