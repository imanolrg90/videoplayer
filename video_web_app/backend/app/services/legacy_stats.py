from __future__ import annotations

import os
import sqlite3
from pathlib import Path

LEGACY_DB_PATH = Path(os.getenv("LEGACY_DB_PATH", str(Path(__file__).resolve().parents[3] / "videos.db")))
_CHUNK_SIZE = 800


def get_legacy_stats_for_paths(media_root: Path, relative_paths: list[str]) -> dict[str, dict]:
    if not LEGACY_DB_PATH.exists() or not relative_paths:
        return {}

    abs_to_rel = {}
    for rel in relative_paths:
        abs_norm = _norm_path(str((media_root / rel).resolve()))
        abs_to_rel[abs_norm] = rel

    stats_map = {
        rel: {
            "views": 0,
            "watched_seconds": 0,
            "has_hash": False,
        }
        for rel in relative_paths
    }

    try:
        with sqlite3.connect(LEGACY_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            _fill_video_stats(conn, abs_to_rel, stats_map)
            _fill_hash_stats(conn, abs_to_rel, stats_map)
    except sqlite3.Error:
        return stats_map

    return stats_map


def _fill_video_stats(conn: sqlite3.Connection, abs_to_rel: dict[str, str], out: dict[str, dict]) -> None:
    paths = list(abs_to_rel.keys())
    for i in range(0, len(paths), _CHUNK_SIZE):
        chunk = paths[i : i + _CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT ruta, reproducciones, tiempo_visto_seg FROM video_stats WHERE lower(replace(ruta, '\\\\', '/')) IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            rel = abs_to_rel.get(_norm_path(row["ruta"]))
            if not rel:
                continue
            out[rel]["views"] = int(row["reproducciones"] or 0)
            out[rel]["watched_seconds"] = int(row["tiempo_visto_seg"] or 0)


def _fill_hash_stats(conn: sqlite3.Connection, abs_to_rel: dict[str, str], out: dict[str, dict]) -> None:
    paths = list(abs_to_rel.keys())
    for i in range(0, len(paths), _CHUNK_SIZE):
        chunk = paths[i : i + _CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT ruta, hash_visual FROM video_hashes WHERE lower(replace(ruta, '\\\\', '/')) IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            rel = abs_to_rel.get(_norm_path(row["ruta"]))
            if not rel:
                continue
            out[rel]["has_hash"] = bool(row["hash_visual"])


def _norm_path(path: str) -> str:
    return str(path).replace("\\", "/").lower()
