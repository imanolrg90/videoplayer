from __future__ import annotations

import os
from pathlib import Path


def resolve_log_path() -> Path:
    default = Path(__file__).resolve().parents[3] / "LOG" / "abrearch_premium.log"
    configured = os.getenv("APP_LOG_PATH", "").strip()
    return Path(configured) if configured else default


def read_log_tail(max_lines: int = 300) -> str:
    max_lines = max(20, min(int(max_lines), 2000))
    log_path = resolve_log_path()
    if not log_path.exists():
        return f"No se encontro el log en: {log_path}"

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"No se pudo leer el log: {exc}"

    lines = text.splitlines()
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    return "\n".join(tail)
