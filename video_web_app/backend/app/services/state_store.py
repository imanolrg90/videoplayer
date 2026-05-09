from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

DB_PATH = Path(__file__).resolve().parents[2] / "video_web.db"
LEGACY_DB_PATH = Path(os.getenv("LEGACY_DB_PATH", str(Path(__file__).resolve().parents[3] / "videos.db")))


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_state (
                relative_path TEXT PRIMARY KEY,
                favorite INTEGER NOT NULL DEFAULT 0,
                watched INTEGER NOT NULL DEFAULT 0,
                last_viewed TEXT
            )
            """
        )
        conn.commit()


def get_states(media_root: Path, relative_paths: Iterable[str]) -> dict[str, dict]:
    paths = [p for p in relative_paths if p]
    if not paths:
        return {}

    placeholders = ",".join("?" for _ in paths)
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT relative_path, favorite, watched, last_viewed FROM video_state WHERE relative_path IN ({placeholders})",
            paths,
        ).fetchall()

    current = {
        row["relative_path"]: {
            "favorite": bool(row["favorite"]),
            "watched": bool(row["watched"]),
            "last_viewed": row["last_viewed"],
        }
        for row in rows
    }

    legacy = _get_legacy_states(media_root, set(paths))
    for rel_path, legacy_state in legacy.items():
        if rel_path not in current:
            current[rel_path] = legacy_state

    return current


def _get_legacy_states(media_root: Path, requested_paths: set[str]) -> dict[str, dict]:
    if not LEGACY_DB_PATH.exists() or not requested_paths:
        return {}

    try:
        with sqlite3.connect(LEGACY_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ruta, es_favorito, fue_visto, ultima_reproduccion
                FROM video_stats
                WHERE es_favorito = 1 OR fue_visto = 1
                """
            ).fetchall()
    except sqlite3.Error:
        return {}

    results: dict[str, dict] = {}
    for row in rows:
        abs_path = row["ruta"]
        if not abs_path:
            continue
        try:
            rel = str(Path(abs_path).resolve().relative_to(media_root)).replace("\\", "/")
        except Exception:
            continue
        if rel not in requested_paths:
            continue
        results[rel] = {
            "favorite": bool(row["es_favorito"]),
            "watched": bool(row["fue_visto"]),
            "last_viewed": row["ultima_reproduccion"],
        }
    return results


def upsert_state(relative_path: str, favorite: bool | None = None, watched: bool | None = None, mark_viewed: bool = False) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT favorite, watched, last_viewed FROM video_state WHERE relative_path = ?",
            (relative_path,),
        ).fetchone()

        current_fav = bool(row["favorite"]) if row else False
        current_watched = bool(row["watched"]) if row else False
        current_last_viewed = row["last_viewed"] if row else None

        new_fav = current_fav if favorite is None else bool(favorite)
        new_watched = current_watched if watched is None else bool(watched)
        if mark_viewed:
            new_watched = True
            current_last_viewed = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        conn.execute(
            """
            INSERT INTO video_state (relative_path, favorite, watched, last_viewed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(relative_path)
            DO UPDATE SET favorite = excluded.favorite,
                          watched = excluded.watched,
                          last_viewed = excluded.last_viewed
            """,
            (relative_path, int(new_fav), int(new_watched), current_last_viewed),
        )
        conn.commit()

    return {
        "relative_path": relative_path,
        "favorite": new_fav,
        "watched": new_watched,
        "last_viewed": current_last_viewed,
    }
