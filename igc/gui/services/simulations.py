from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

# psycopg3 (binary or source)
try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore

def _db_url() -> Optional[str]:
    # Prefer IGC_DATABASE_URL, then DATABASE_URL
    return os.getenv("IGC_DATABASE_URL") or os.getenv("DATABASE_URL")

def list_simulations(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Return rows from table public.simulations as:
      [{"id": ..., "name": ..., "description": ...}, ...]
    On any failure (no psycopg / no URL / query error) -> [].
    """
    url = _db_url()
    if not url or not psycopg:
        return []

    sql = """
        SELECT id, name, description
        FROM public.simulations
        ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST, id DESC
        LIMIT %s
    """
    try:
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall() or []
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({"id": r[0], "name": r[1], "description": r[2]})
    return out
