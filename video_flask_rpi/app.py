from __future__ import annotations

import hashlib
import json
import logging
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
DESKTOP_DB_PATH = Path(os.getenv("DESKTOP_DB_PATH", "./videos.db")).resolve()
APP_LOG_PATH = Path(os.getenv("APP_LOG_PATH", "./LOG/abrearch_premium.log")).resolve()
FOLDER_VIEWS_LOG_DIR = Path(os.getenv("FOLDER_VIEWS_LOG_DIR", str(APP_LOG_PATH.parent))).resolve()
FOLDER_VIEWS_LOG_BASENAME = os.getenv("FOLDER_VIEWS_LOG_BASENAME", "folder_views").strip() or "folder_views"
THUMB_CACHE_DIR = (Path(__file__).resolve().parent / "thumb_cache").resolve()
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "/usr/bin/ffmpeg").strip()
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "change-this-password")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret")


def init_db() -> None:
    DESKTOP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DESKTOP_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ruta TEXT UNIQUE NOT NULL,
                nombre_archivo TEXT,
                reproducciones INTEGER DEFAULT 0,
                tiempo_visto_seg INTEGER DEFAULT 0,
                ultima_reproduccion TEXT,
                es_favorito BOOLEAN DEFAULT 0,
                fue_visto BOOLEAN DEFAULT 0,
                miniatura BLOB,
                fecha_created TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ruta TEXT UNIQUE NOT NULL,
                tamaño_bytes INTEGER,
                hash_visual BLOB,
                fecha_calculado TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                clave TEXT PRIMARY KEY,
                valor TEXT,
                fecha_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_renames (
                ruta_origen TEXT PRIMARY KEY,
                fecha_added TEXT DEFAULT CURRENT_TIMESTAMP,
                intentos INTEGER DEFAULT 0,
                ultimo_error TEXT
            )
            """
        )
        conn.commit()


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("abrearch_premium")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    try:
        APP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
    except OSError:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(threadName)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = setup_logging()
init_db()


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DESKTOP_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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


def norm_abs(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def is_marked_rwd(name: str) -> bool:
    lower = name.lower()
    if lower.startswith("rwd ") or lower.startswith("top rwd "):
        return True
    stem = Path(name).stem.lower()
    return stem.endswith("_rwd")


def target_top_name(name: str, make_favorite: bool) -> str:
    if make_favorite:
        if name.lower().startswith("top "):
            return name
        return f"top {name}"
    if name.lower().startswith("top "):
        return name[4:]
    return name


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
    abs_paths = [norm_abs((MEDIA_ROOT / rel).resolve()) for rel in rel_paths]
    placeholders = ",".join("?" for _ in abs_paths)
    out = {
        rel: {
            "favorite": False,
            "watched": False,
            "last_viewed": None,
            "views": 0,
            "watched_seconds": 0,
            "has_hash": False,
            "thumb_blob": None,
        }
        for rel in rel_paths
    }

    abs_to_rel = dict(zip(abs_paths, rel_paths))

    with db_connect() as conn:
        rows = conn.execute(
            f"""
            SELECT ruta, reproducciones, tiempo_visto_seg, ultima_reproduccion,
                   es_favorito, fue_visto, miniatura
            FROM video_stats
            WHERE ruta IN ({placeholders})
            """,
            abs_paths,
        ).fetchall()
        for row in rows:
            rel = abs_to_rel.get(norm_abs(row["ruta"]))
            if not rel:
                continue
            out[rel].update(
                {
                    "favorite": bool(row["es_favorito"]),
                    "watched": bool(row["fue_visto"]),
                    "last_viewed": row["ultima_reproduccion"],
                    "views": int(row["reproducciones"] or 0),
                    "watched_seconds": int(row["tiempo_visto_seg"] or 0),
                    "thumb_blob": bytes(row["miniatura"]) if row["miniatura"] else None,
                }
            )

        hash_rows = conn.execute(
            f"SELECT ruta FROM video_hashes WHERE hash_visual IS NOT NULL AND ruta IN ({placeholders})",
            abs_paths,
        ).fetchall()
        for row in hash_rows:
            rel = abs_to_rel.get(norm_abs(row["ruta"]))
            if rel:
                out[rel]["has_hash"] = True

    return out


def upsert_state(relative_path: str, favorite: bool | None = None, watched: bool | None = None, mark_viewed: bool = False) -> dict:
    abs_path = norm_abs((MEDIA_ROOT / relative_path).resolve())
    nombre = Path(abs_path).name

    with db_connect() as conn:
        row = conn.execute(
            "SELECT reproducciones, tiempo_visto_seg, ultima_reproduccion, es_favorito, fue_visto FROM video_stats WHERE ruta = ?",
            (abs_path,),
        ).fetchone()

        current_fav = bool(row["es_favorito"]) if row else False
        current_watched = bool(row["fue_visto"]) if row else False
        current_last = row["ultima_reproduccion"] if row else None
        current_views = int(row["reproducciones"] or 0) if row else 0
        current_secs = int(row["tiempo_visto_seg"] or 0) if row else 0

        new_fav = current_fav if favorite is None else bool(favorite)
        new_watched = current_watched if watched is None else bool(watched)
        if mark_viewed:
            new_watched = True
            current_views += 1
            current_last = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        conn.execute(
            """
            INSERT INTO video_stats (ruta, nombre_archivo, reproducciones, tiempo_visto_seg, ultima_reproduccion, es_favorito, fue_visto)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ruta) DO UPDATE SET
                nombre_archivo = excluded.nombre_archivo,
                reproducciones = excluded.reproducciones,
                tiempo_visto_seg = excluded.tiempo_visto_seg,
                ultima_reproduccion = excluded.ultima_reproduccion,
                es_favorito = excluded.es_favorito,
                fue_visto = excluded.fue_visto
            """,
            (abs_path, nombre, current_views, current_secs, current_last, int(new_fav), int(new_watched)),
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


def add_pending_rename(abs_path: str, error_msg: str | None = None) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO pending_renames (ruta_origen, intentos, ultimo_error)
            VALUES (?, 1, ?)
            ON CONFLICT(ruta_origen) DO UPDATE SET
                intentos = intentos + 1,
                ultimo_error = excluded.ultimo_error,
                fecha_added = CURRENT_TIMESTAMP
            """,
            (norm_abs(abs_path), error_msg),
        )
        conn.commit()


def get_setting(clave: str, default: str = "") -> str:
    with db_connect() as conn:
        row = conn.execute("SELECT valor FROM app_settings WHERE clave = ?", (clave,)).fetchone()
    if row is None:
        return default
    return row["valor"] if row["valor"] is not None else default


def set_setting(clave: str, valor: str) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO app_settings (clave, valor, fecha_updated)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(clave) DO UPDATE SET
                valor = excluded.valor,
                fecha_updated = CURRENT_TIMESTAMP
            """,
            (clave, valor),
        )
        conn.commit()


def queue_pending_delete(abs_path: str) -> None:
    raw = get_setting("pending_delete_paths", "[]")
    try:
        data = json.loads(raw) if raw else []
    except Exception:
        data = []
    s = {str(p).replace("\\", "/") for p in data if p}
    s.add(norm_abs(abs_path))
    set_setting("pending_delete_paths", json.dumps(sorted(s), ensure_ascii=False))


def queue_pending_fav_rename(old_abs: str, new_abs: str) -> None:
    raw = get_setting("pending_fav_renames", "[]")
    try:
        data = json.loads(raw) if raw else []
    except Exception:
        data = []
    mapping = {}
    for row in data:
        if isinstance(row, dict) and row.get("old") and row.get("new"):
            mapping[str(row["old"]).replace("\\", "/")] = str(row["new"]).replace("\\", "/")
    mapping[norm_abs(old_abs)] = norm_abs(new_abs)
    rows = [{"old": k, "new": v} for k, v in sorted(mapping.items())]
    set_setting("pending_fav_renames", json.dumps(rows, ensure_ascii=False))


def queue_pending_rwd(abs_path: str) -> None:
    add_pending_rename(abs_path)


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


def folder_views_today_path() -> Path:
    fecha = datetime.now().strftime("%Y-%m-%d")
    filename = f"{FOLDER_VIEWS_LOG_BASENAME}_{fecha}.log"
    return FOLDER_VIEWS_LOG_DIR / filename


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
        inferred_watched = is_marked_rwd(p.name)
        size_bytes = int(p.stat().st_size)
        items.append(
            {
                "name": p.name,
                "relative_path": rel,
                "folder": str(p.parent.relative_to(MEDIA_ROOT)).replace("\\", "/"),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "favorite": bool(state.get("favorite", False)) or inferred_fav,
                "watched": bool(state.get("watched", False)) or inferred_watched,
                "last_viewed": state.get("last_viewed"),
                "views": int(state.get("views", 0)),
                "watched_seconds": int(state.get("watched_seconds", 0)),
                "has_hash": bool(state.get("has_hash", False)),
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

    rel_norm = str(video_path.relative_to(MEDIA_ROOT)).replace("\\", "/")
    state = get_states([rel_norm]).get(rel_norm, {})
    blob = state.get("thumb_blob")
    if blob:
        return Response(blob, mimetype="image/jpeg")

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
    abs_norm = norm_abs(video_path)

    if watched is True:
        queue_pending_rwd(abs_norm)

    if favorite is not None:
        new_name = target_top_name(video_path.name, favorite)
        new_abs = norm_abs(video_path.with_name(new_name))
        if new_abs != abs_norm:
            queue_pending_fav_rename(abs_norm, new_abs)

    state = upsert_state(rel_norm, favorite=favorite, watched=watched)
    LOGGER.info(
        "API /video/state path=%s favorite=%s watched=%s",
        rel_norm,
        favorite,
        watched,
    )
    return jsonify({"ok": True, "state": state})


@app.post("/api/video/viewed")
@auth_required
def viewed_api():
    rel = request.args.get("path", "")
    video_path = resolve_video_path(rel)
    if not video_path:
        return jsonify({"detail": "Video no encontrado"}), 404

    rel_norm = str(video_path.relative_to(MEDIA_ROOT)).replace("\\", "/")
    abs_norm = norm_abs(video_path)
    queue_pending_rwd(abs_norm)
    state = upsert_state(rel_norm, mark_viewed=True)
    LOGGER.info("API /video/viewed path=%s queued_rwd=1", rel_norm)
    return jsonify({"ok": True, "state": state})


@app.post("/api/video/delete")
@auth_required
def defer_delete_api():
    rel = request.args.get("path", "")
    video_path = resolve_video_path(rel)
    if not video_path:
        return jsonify({"detail": "Video no encontrado"}), 404

    rel_norm = str(video_path.relative_to(MEDIA_ROOT)).replace("\\", "/")
    queue_pending_delete(norm_abs(video_path))
    LOGGER.info("API /video/delete path=%s queued_delete=1", rel_norm)
    return jsonify({"ok": True, "queued": True, "path": rel_norm})


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


@app.get("/api/log/folder-views/today")
@auth_required
def folder_views_today_api():
    path = folder_views_today_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return jsonify({"detail": f"No se pudo leer el log diario: {exc}"}), 500

    return jsonify(
        {
            "ok": True,
            "file_name": path.name,
            "file_path": str(path),
            "content": text,
        }
    )


@app.post("/api/log/folder-views/today")
@auth_required
def save_folder_views_today_api():
    path = folder_views_today_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = request.get_json(silent=True) or {}
    content = data.get("content", "")
    if not isinstance(content, str):
        return jsonify({"detail": "content debe ser texto"}), 400

    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return jsonify({"detail": f"No se pudo guardar el log diario: {exc}"}), 500

    LOGGER.info("API /log/folder-views/today saved file=%s bytes=%d", path.name, len(content.encode("utf-8")))
    return jsonify({"ok": True, "file_name": path.name, "file_path": str(path)})


if __name__ == "__main__":
    init_db()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    app.run(host=host, port=port, debug=False)
