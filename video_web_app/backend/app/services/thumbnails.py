from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[2] / "thumb_cache"


def _resolve_ffmpeg() -> str | None:
    env_value = os.getenv("FFMPEG_PATH", "").strip()
    if env_value and Path(env_value).exists():
        return env_value

    hardcoded = Path("D:/projects/FINANZAS/tools/ffmpeg.exe")
    if hardcoded.exists():
        return str(hardcoded)

    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def ensure_thumbnail(video_path: Path) -> Path | None:
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg or not video_path.exists() or not video_path.is_file():
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key_raw = f"{video_path}|{video_path.stat().st_mtime_ns}|{video_path.stat().st_size}"
    key = hashlib.sha1(key_raw.encode("utf-8")).hexdigest()
    out_path = CACHE_DIR / f"{key}.jpg"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        "00:00:02",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale=320:-1",
        "-y",
        str(out_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=20)
        if result.returncode != 0:
            return None
    except Exception:
        return None

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    return None
