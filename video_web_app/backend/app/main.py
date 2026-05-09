from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.services.auth import ensure_access, get_configured_token
from app.services.legacy_stats import get_legacy_stats_for_paths
from app.services.log_reader import read_log_tail
from app.services.media import (
    get_media_root,
    iter_videos_in_folder,
    list_folders,
    resolve_folder_path,
    resolve_video_path,
)
from app.services.state_store import get_states, init_db, upsert_state
from app.services.thumbnails import ensure_thumbnail

app = FastAPI(title="Video Web App", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict:
    root = get_media_root()
    return {
        "status": "ok",
        "media_root": str(root),
        "media_root_exists": root.exists(),
        "token_enabled": bool(get_configured_token()),
    }


@app.get("/api/videos")
def list_videos(
    search: str = Query(default="", max_length=120),
    folder: str = Query(default="", max_length=500),
    recursive: bool = Query(default=True),
    _auth: None = Depends(ensure_access),
) -> dict:
    media_root = get_media_root()
    if not media_root.exists() or not media_root.is_dir():
        raise HTTPException(
            status_code=400,
            detail=(
                "MEDIA_ROOT no existe o no es carpeta. "
                "Configura la variable de entorno MEDIA_ROOT."
            ),
        )

    found = list(iter_videos_in_folder(media_root, relative_folder=folder, search=search, recursive=recursive))
    rel_paths = [str(video_path.relative_to(media_root)).replace("\\", "/") for video_path in found]
    states = get_states(media_root, rel_paths)
    legacy_stats = get_legacy_stats_for_paths(media_root, rel_paths)

    videos = []
    for video_path in found:
        rel_path = str(video_path.relative_to(media_root)).replace("\\", "/")
        size_mb = round(video_path.stat().st_size / (1024 * 1024), 2)
        state = states.get(rel_path, {})
        legacy = legacy_stats.get(rel_path, {})
        inferred_favorite = video_path.name.lower().startswith("top ")
        videos.append(
            {
                "name": video_path.name,
                "relative_path": rel_path,
                "size_mb": size_mb,
                "size_bytes": int(video_path.stat().st_size),
                "favorite": bool(state.get("favorite", False)) or inferred_favorite,
                "watched": bool(state.get("watched", False)),
                "last_viewed": state.get("last_viewed"),
                "views": int(legacy.get("views", 0)),
                "watched_seconds": int(legacy.get("watched_seconds", 0)),
                "has_hash": bool(legacy.get("has_hash", False)),
                "folder": str(video_path.parent.relative_to(media_root)).replace("\\", "/"),
                "thumbnail_url": f"/api/thumb/video?path={rel_path}",
            }
        )

    return {
        "folder": folder,
        "recursive": recursive,
        "count": len(videos),
        "items": videos,
    }


@app.get("/api/folders")
def folders(_auth: None = Depends(ensure_access)) -> dict:
    media_root = get_media_root()
    if not media_root.exists() or not media_root.is_dir():
        raise HTTPException(
            status_code=400,
            detail=(
                "MEDIA_ROOT no existe o no es carpeta. "
                "Configura la variable de entorno MEDIA_ROOT."
            ),
        )

    items = list_folders(media_root)
    for folder_info in items:
        folder_info["thumbnail_url"] = f"/api/thumb/folder?path={folder_info['relative_path']}"
    return {
        "count": len(items),
        "items": items,
    }


@app.get("/api/thumb/video")
def video_thumb(
    path: str = Query(..., min_length=1, max_length=500),
    _auth: None = Depends(ensure_access),
):
    media_root = get_media_root()
    video_path = resolve_video_path(media_root, path)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    thumb_path = ensure_thumbnail(video_path)
    if not thumb_path:
        raise HTTPException(status_code=404, detail="Miniatura no disponible")
    return FileResponse(path=thumb_path, media_type="image/jpeg")


@app.get("/api/thumb/folder")
def folder_thumb(
    path: str = Query(default="", max_length=500),
    _auth: None = Depends(ensure_access),
):
    media_root = get_media_root()
    folder_path = resolve_folder_path(media_root, path)
    if not folder_path:
        raise HTTPException(status_code=404, detail="Carpeta no encontrada")

    candidates = list(iter_videos_in_folder(media_root, relative_folder=path, search="", recursive=True))
    if not candidates:
        raise HTTPException(status_code=404, detail="No hay videos para miniatura")

    thumb_path = ensure_thumbnail(candidates[0])
    if not thumb_path:
        raise HTTPException(status_code=404, detail="Miniatura no disponible")
    return FileResponse(path=thumb_path, media_type="image/jpeg")


@app.get("/api/log", response_class=PlainTextResponse)
def get_log(
    lines: int = Query(default=250, ge=20, le=2000),
    _auth: None = Depends(ensure_access),
):
    return read_log_tail(lines)


@app.get("/api/stream")
def stream_video(
    path: str = Query(..., min_length=1, max_length=500),
    _auth: None = Depends(ensure_access),
):
    media_root = get_media_root()
    video_path = resolve_video_path(media_root, path)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    media_type, _ = mimetypes.guess_type(video_path.name)
    return FileResponse(path=video_path, media_type=media_type or "video/mp4", filename=video_path.name)


@app.post("/api/video/state")
def set_video_state(
    path: str = Query(..., min_length=1, max_length=500),
    favorite: Optional[bool] = Query(default=None),
    watched: Optional[bool] = Query(default=None),
    _auth: None = Depends(ensure_access),
) -> dict:
    media_root = get_media_root()
    video_path = resolve_video_path(media_root, path)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    rel_path = str(video_path.relative_to(media_root)).replace("\\", "/")
    state = upsert_state(rel_path, favorite=favorite, watched=watched)
    return {"ok": True, "state": state}


@app.post("/api/video/viewed")
def mark_video_viewed(
    path: str = Query(..., min_length=1, max_length=500),
    _auth: None = Depends(ensure_access),
) -> dict:
    media_root = get_media_root()
    video_path = resolve_video_path(media_root, path)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video no encontrado")

    rel_path = str(video_path.relative_to(media_root)).replace("\\", "/")
    state = upsert_state(rel_path, mark_viewed=True)
    return {"ok": True, "state": state}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
