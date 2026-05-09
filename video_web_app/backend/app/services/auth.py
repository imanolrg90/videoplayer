from __future__ import annotations

import os
from typing import Optional

from fastapi import Header, HTTPException, Query


def get_configured_token() -> str:
    return os.getenv("ACCESS_TOKEN", "").strip()


def ensure_access(x_access_token: Optional[str] = Header(default=None), token: Optional[str] = Query(default=None)) -> None:
    configured = get_configured_token()
    if not configured:
        return

    provided = (x_access_token or token or "").strip()
    if provided != configured:
        raise HTTPException(status_code=401, detail="Token invalido")
