from __future__ import annotations
import os, sys, traceback
from typing import Any, Dict, List, Optional

try:
    import psycopg  # psycopg3
except Exception:
    psycopg = None  # type: ignore

def _db_url() -> Optional[str]:
    return os.getenv("IGC_DATABASE_URL") or os.getenv("DATABASE_URL")

def list_simulations(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Return rows from public.simulations as:
      [{id, name, description}, ...]
    Uses EXACT columns: id, name, description.
    On error, logs to stderr and returns [].
    """
    url = _db_url()
    if not url:
        print("[simulations] No DATABASE_URL/IGC_DATABASE_URL set", file=sys.stderr)
        return []
    if not psycopg:
        print("[simulations] psycopg not available in env", file=sys.stderr)
        return []

    sql = """
        SELECT id, name, description
        FROM public.simulations
        ORDER BY name ASC, id DESC
        LIMIT %s
    """
    try:
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall() or []
                return [{"id": r[0], "name": r[1], "description": r[2]} for r in rows]
    except Exception:
        print("[simulations] query failed:", file=sys.stderr)
        traceback.print_exc()
        return []
