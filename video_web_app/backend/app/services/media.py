from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv"}
DEFAULT_MEDIA_ROOT = "D:/videos"


def get_media_root() -> Path:
    media_root = Path(os.getenv("MEDIA_ROOT", DEFAULT_MEDIA_ROOT)).resolve()
    return media_root


def is_inside_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def iter_videos(root: Path, search: str = "") -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []

    search_lower = search.strip().lower()
    items: list[Path] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if search_lower and search_lower not in file_path.name.lower():
            continue
        items.append(file_path)

    items.sort(key=lambda p: p.name.lower())
    return items


def resolve_folder_path(root: Path, relative_folder: str) -> Path | None:
    normalized = (relative_folder or "").replace("\\", "/").strip("/")
    if not normalized:
        return root
    candidate = (root / normalized).resolve()
    if not is_inside_root(root, candidate):
        return None
    if not candidate.exists() or not candidate.is_dir():
        return None
    return candidate


def iter_videos_in_folder(
    root: Path,
    relative_folder: str = "",
    search: str = "",
    recursive: bool = False,
) -> Iterable[Path]:
    folder_path = resolve_folder_path(root, relative_folder)
    if not folder_path:
        return []

    search_lower = search.strip().lower()
    globber = folder_path.rglob("*") if recursive else folder_path.glob("*")

    items: list[Path] = []
    for file_path in globber:
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if search_lower and search_lower not in file_path.name.lower():
            continue
        items.append(file_path)

    items.sort(key=lambda p: p.name.lower())
    return items


def list_folders(root: Path) -> list[dict]:
    if not root.exists() or not root.is_dir():
        return []

    folders: list[dict] = [
        {
            "name": root.name or "Root",
            "relative_path": "",
            "depth": 0,
            "video_count": _count_videos_in_folder(root),
        }
    ]

    for dir_path in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: str(p).lower()):
        rel = str(dir_path.relative_to(root)).replace("\\", "/")
        folders.append(
            {
                "name": dir_path.name,
                "relative_path": rel,
                "depth": len(Path(rel).parts),
                "video_count": _count_videos_in_folder(dir_path),
            }
        )

    return folders


def _count_videos_in_folder(folder: Path) -> int:
    count = 0
    try:
        for entry in folder.iterdir():
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
                count += 1
    except OSError:
        return 0
    return count


def resolve_video_path(root: Path, relative_path: str) -> Path | None:
    normalized = relative_path.replace("\\", "/").strip("/")
    candidate = (root / normalized).resolve()
    if not is_inside_root(root, candidate):
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    if candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        return None
    return candidate
