from __future__ import annotations

import hashlib
import mimetypes
import os
import sqlite3
import subprocess
from datetime import datetime
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, session, url_for

load_dotenv()

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv"}

MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", "/mnt/media_share")).resolve()
STATE_DB_PATH = Path(os.getenv("STATE_DB_PATH", "./video_state.db")).resolve()
APP_LOG_PATH = Path(os.getenv("APP_LOG_PATH", "./app.log")).resolve()
THUMB_CACHE_DIR = (Path(__file__).resolve().parent / "thumb_cache").resolve()
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "/usr/bin/ffmpeg").strip()
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "change-this-password")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret")


def init_db() -> None:
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(STATE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_state (
                relative_path TEXT PRIMARY KEY,
                favorite INTEGER NOT NULL DEFAULT 0,
                watched INTEGER NOT NULL DEFAULT 0,
                last_viewed TEXT,
                views INTEGER NOT NULL DEFAULT 0,
                watched_seconds INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()


def auth_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return jsonify({"detail": "Unauthorized"}), 401
        return view_func(*args, **kwargs)

    return wrapped


def is_inside_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def normalize_rel(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def resolve_folder_path(relative_folder: str) -> Path | None:
    normalized = normalize_rel(relative_folder or "")
    if not normalized:
        return MEDIA_ROOT
    candidate = (MEDIA_ROOT / normalized).resolve()
    if not is_inside_root(MEDIA_ROOT, candidate):
        return None
    if not candidate.exists() or not candidate.is_dir():
        return None
    return candidate


def resolve_video_path(relative_path: str) -> Path | None:
    normalized = normalize_rel(relative_path)
    candidate = (MEDIA_ROOT / normalized).resolve()
    if not is_inside_root(MEDIA_ROOT, candidate):
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        return None
    return candidate


def list_videos(relative_folder: str = "", search: str = "", recursive: bool = True) -> list[Path]:
    folder = resolve_folder_path(relative_folder)
    if not folder:
        return []
    search_lower = search.strip().lower()
    iterator = folder.rglob("*") if recursive else folder.glob("*")

    items: list[Path] = []
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if search_lower and search_lower not in p.name.lower():
            continue
        items.append(p)

    items.sort(key=lambda x: x.name.lower())
    return items


def list_folders() -> list[dict]:
    if not MEDIA_ROOT.exists() or not MEDIA_ROOT.is_dir():
        return []

    root_name = MEDIA_ROOT.name or "Root"
    output = [{"name": root_name, "relative_path": "", "depth": 0, "video_count": count_videos_in_folder(MEDIA_ROOT)}]

    dirs = [p for p in MEDIA_ROOT.rglob("*") if p.is_dir()]
    dirs.sort(key=lambda p: str(p).lower())
    for d in dirs:
        rel = str(d.relative_to(MEDIA_ROOT)).replace("\\", "/")
        output.append(
            {
                "name": d.name,
                "relative_path": rel,
                "depth": len(Path(rel).parts),
                "video_count": count_videos_in_folder(d),
            }
        )
    return output


def count_videos_in_folder(folder: Path) -> int:
    try:
        return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    except OSError:
        return 0


def get_states(rel_paths: list[str]) -> dict[str, dict]:
    if not rel_paths:
        return {}
    placeholders = ",".join("?" for _ in rel_paths)
    with sqlite3.connect(STATE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT relative_path, favorite, watched, last_viewed, views, watched_seconds FROM video_state WHERE relative_path IN ({placeholders})",
            rel_paths,
        ).fetchall()

    return {
        row["relative_path"]: {
            "favorite": bool(row["favorite"]),
            "watched": bool(row["watched"]),
            "last_viewed": row["last_viewed"],
            "views": int(row["views"] or 0),
            "watched_seconds": int(row["watched_seconds"] or 0),
        }
        for row in rows
    }


def upsert_state(relative_path: str, favorite: bool | None = None, watched: bool | None = None, mark_viewed: bool = False) -> dict:
    with sqlite3.connect(STATE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT favorite, watched, last_viewed, views, watched_seconds FROM video_state WHERE relative_path = ?",
            (relative_path,),
        ).fetchone()

        current_fav = bool(row["favorite"]) if row else False
        current_watched = bool(row["watched"]) if row else False
        current_last = row["last_viewed"] if row else None
        current_views = int(row["views"] or 0) if row else 0
        current_secs = int(row["watched_seconds"] or 0) if row else 0

        new_fav = current_fav if favorite is None else bool(favorite)
        new_watched = current_watched if watched is None else bool(watched)
        if mark_viewed:
            new_watched = True
            current_views += 1
            current_last = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        conn.execute(
            """
            INSERT INTO video_state (relative_path, favorite, watched, last_viewed, views, watched_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(relative_path)
            DO UPDATE SET favorite = excluded.favorite,
                          watched = excluded.watched,
                          last_viewed = excluded.last_viewed,
                          views = excluded.views,
                          watched_seconds = excluded.watched_seconds
            """,
            (relative_path, int(new_fav), int(new_watched), current_last, current_views, current_secs),
        )
        conn.commit()

    return {
        "relative_path": relative_path,
        "favorite": new_fav,
        "watched": new_watched,
        "last_viewed": current_last,
        "views": current_views,
        "watched_seconds": current_secs,
    }


def ensure_thumbnail(video_path: Path) -> Path | None:
    if not Path(FFMPEG_PATH).exists():
        return None

    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key_raw = f"{video_path}|{video_path.stat().st_mtime_ns}|{video_path.stat().st_size}"
    key = hashlib.sha1(key_raw.encode("utf-8")).hexdigest()
    out = THUMB_CACHE_DIR / f"{key}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return out

    cmd = [
        FFMPEG_PATH,
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
        str(out),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=20)
        if result.returncode != 0:
            return None
    except Exception:
        return None

    if out.exists() and out.stat().st_size > 0:
        return out
    return None


def parse_range_header(range_header: str, file_size: int) -> tuple[int, int] | None:
    if not range_header or not range_header.startswith("bytes="):
        return None
    value = range_header.replace("bytes=", "", 1)
    if "-" not in value:
        return None
    start_s, end_s = value.split("-", 1)
    try:
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        return None
    if start > end or start < 0 or end >= file_size:
        return None
    return start, end


@app.get("/")
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login_page"))
    return render_template("index.html")


@app.get("/login")
def login_page():
    return render_template("login.html")


@app.post("/api/login")
def login_api():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    if username == APP_USERNAME and password == APP_PASSWORD:
        session["logged_in"] = True
        return jsonify({"ok": True})

    return jsonify({"detail": "Credenciales invalidas"}), 401


@app.post("/api/logout")
@auth_required
def logout_api():
    session.pop("logged_in", None)
    return jsonify({"ok": True})


@app.get("/api/health")
@auth_required
def health():
    return jsonify(
        {
            "status": "ok",
            "media_root": str(MEDIA_ROOT),
            "media_root_exists": MEDIA_ROOT.exists(),
        }
    )


@app.get("/api/folders")
@auth_required
def folders_api():
    if not MEDIA_ROOT.exists() or not MEDIA_ROOT.is_dir():
        return jsonify({"detail": "MEDIA_ROOT no existe o no es carpeta"}), 400

    items = list_folders()
    for item in items:
        item["thumbnail_url"] = f"/api/thumb/folder?path={item['relative_path']}"
    return jsonify({"count": len(items), "items": items})


@app.get("/api/videos")
@auth_required
def videos_api():
    if not MEDIA_ROOT.exists() or not MEDIA_ROOT.is_dir():
        return jsonify({"detail": "MEDIA_ROOT no existe o no es carpeta"}), 400

    search = request.args.get("search", "")
    folder = request.args.get("folder", "")
    recursive = request.args.get("recursive", "true").lower() == "true"

    found = list_videos(folder, search, recursive)
    rel_paths = [str(p.relative_to(MEDIA_ROOT)).replace("\\", "/") for p in found]
    state_map = get_states(rel_paths)

    items = []
    for p in found:
        rel = str(p.relative_to(MEDIA_ROOT)).replace("\\", "/")
        state = state_map.get(rel, {})
        inferred_fav = p.name.lower().startswith("top ")
        size_bytes = int(p.stat().st_size)
        items.append(
            {
                "name": p.name,
                "relative_path": rel,
                "folder": str(p.parent.relative_to(MEDIA_ROOT)).replace("\\", "/"),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "favorite": bool(state.get("favorite", False)) or inferred_fav,
                "watched": bool(state.get("watched", False)),
                "last_viewed": state.get("last_viewed"),
                "views": int(state.get("views", 0)),
                "watched_seconds": int(state.get("watched_seconds", 0)),
                "has_hash": False,
                "thumbnail_url": f"/api/thumb/video?path={rel}",
            }
        )

    return jsonify({"count": len(items), "folder": folder, "recursive": recursive, "items": items})


@app.get("/api/thumb/video")
@auth_required
def thumb_video_api():
    rel = request.args.get("path", "")
    video_path = resolve_video_path(rel)
    if not video_path:
        return jsonify({"detail": "Video no encontrado"}), 404

    thumb = ensure_thumbnail(video_path)
    if not thumb:
        return jsonify({"detail": "Miniatura no disponible"}), 404
    return send_file(thumb, mimetype="image/jpeg")


@app.get("/api/thumb/folder")
@auth_required
def thumb_folder_api():
    rel = request.args.get("path", "")
    folder_path = resolve_folder_path(rel)
    if not folder_path:
        return jsonify({"detail": "Carpeta no encontrada"}), 404

    candidates = list_videos(rel, "", True)
    if not candidates:
        return jsonify({"detail": "No hay videos para miniatura"}), 404

    thumb = ensure_thumbnail(candidates[0])
    if not thumb:
        return jsonify({"detail": "Miniatura no disponible"}), 404
    return send_file(thumb, mimetype="image/jpeg")


@app.get("/api/stream")
@auth_required
def stream_api():
    rel = request.args.get("path", "")
    video_path = resolve_video_path(rel)
    if not video_path:
        return jsonify({"detail": "Video no encontrado"}), 404

    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range", "")
    media_type, _ = mimetypes.guess_type(video_path.name)
    media_type = media_type or "video/mp4"

    parsed = parse_range_header(range_header, file_size)
    if not parsed:
        return send_file(video_path, mimetype=media_type, as_attachment=False)

    start, end = parsed
    length = end - start + 1

    with open(video_path, "rb") as f:
        f.seek(start)
        data = f.read(length)

    response = Response(data, 206, mimetype=media_type, direct_passthrough=True)
    response.headers.add("Content-Range", f"bytes {start}-{end}/{file_size}")
    response.headers.add("Accept-Ranges", "bytes")
    response.headers.add("Content-Length", str(length))
    return response


@app.post("/api/video/state")
@auth_required
def set_state_api():
    rel = request.args.get("path", "")
    video_path = resolve_video_path(rel)
    if not video_path:
        return jsonify({"detail": "Video no encontrado"}), 404

    favorite_raw = request.args.get("favorite")
    watched_raw = request.args.get("watched")

    favorite = None if favorite_raw is None else favorite_raw.lower() == "true"
    watched = None if watched_raw is None else watched_raw.lower() == "true"

    rel_norm = str(video_path.relative_to(MEDIA_ROOT)).replace("\\", "/")
    state = upsert_state(rel_norm, favorite=favorite, watched=watched)
    return jsonify({"ok": True, "state": state})


@app.post("/api/video/viewed")
@auth_required
def viewed_api():
    rel = request.args.get("path", "")
    video_path = resolve_video_path(rel)
    if not video_path:
        return jsonify({"detail": "Video no encontrado"}), 404

    rel_norm = str(video_path.relative_to(MEDIA_ROOT)).replace("\\", "/")
    state = upsert_state(rel_norm, mark_viewed=True)
    return jsonify({"ok": True, "state": state})


@app.get("/api/log")
@auth_required
def log_api():
    lines = int(request.args.get("lines", "250"))
    lines = max(20, min(lines, 2000))

    if not APP_LOG_PATH.exists():
        return Response(f"No se encontro el log en: {APP_LOG_PATH}", mimetype="text/plain")

    try:
        text = APP_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return Response(f"No se pudo leer el log: {exc}", mimetype="text/plain")

    all_lines = text.splitlines()
    tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
    return Response("\n".join(tail), mimetype="text/plain")


if __name__ == "__main__":
    init_db()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    app.run(host=host, port=port, debug=False)
