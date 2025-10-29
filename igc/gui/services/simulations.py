from __future__ import annotations
import os
from typing import Any, Dict, List

# Try to use psycopg (binary or source)
try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None  # type: ignore

# Optional: integrate with an existing pool if your project defines one later.
# Here we keep it self-contained and simple.

def _db_url() -> str | None:
    return os.getenv("IGC_DATABASE_URL") or os.getenv("DATABASE_URL")

def list_simulations(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Return rows from table `simulations` as a list of dicts:
      [{id: ..., name: ..., description: ...}, ...]
    Falls back to [] if no DB is configured or reachable.
    """
    url = _db_url()
    if not url or not psycopg:
        return []

    sql = """
        SELECT id, name, description
        FROM simulations
        ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST, id DESC
        LIMIT %s
    """
    try:
        # psycopg 3: connect() accepts a URL
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall() or []
    except Exception:
        return []

    # Build list of dicts
    out: List[Dict[str, Any]] = []
    for r in rows:
        # r is a tuple (id, name, description)
        out.append({"id": r[0], "name": r[1], "description": r[2]})
    return out
