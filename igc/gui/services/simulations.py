from __future__ import annotations
import os, sys, traceback
from typing import Any, Dict, List, Optional

try:
    import psycopg  # psycopg3
except Exception:
    psycopg = None  # type: ignore

def _db_url() -> Optional[str]:
    # Prefer explicit URL; otherwise fall back to Unix socket (peer auth)
    return os.getenv("IGC_DATABASE_URL") or os.getenv("DATABASE_URL") or "postgresql:///igc"

# ---------- list for /sims/select ----------
def list_simulations(limit: int = 500) -> List[Dict[str, Any]]:
    """
    [{id, label, name, description}, ...] sorted by label, name.
    """
    url = _db_url()
    if not psycopg:
        return []
    sql = """
        SELECT id, label, name, description
        FROM public.simulations
        ORDER BY label ASC, name ASC, id DESC
        LIMIT %s
    """
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall() or []
            return [{"id": r[0], "label": r[1], "name": r[2], "description": r[3]} for r in rows]
    except Exception:
        traceback.print_exc()
        return []

# ---------- single-row (used by /sims/edit) ----------
def get_simulation(sim_id: int) -> Optional[Dict[str, Any]]:
    """
    One row by id with the common fields (id,label,name,description).
    """
    url = _db_url()
    if not psycopg:
        return None
    sql = """
        SELECT id, label, name, description
        FROM public.simulations
        WHERE id = %s
    """
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.execute(sql, (sim_id,))
            r = cur.fetchone()
            if not r:
                return None
            return {"id": r[0], "label": r[1], "name": r[2], "description": r[3]}
    except Exception:
        traceback.print_exc()
        return None

def get_simulation_full(sim_id: int) -> Optional[Dict[str, Any]]:
    """
    One row by id as a full dict (SELECT *).
    """
    url = _db_url()
    if not psycopg:
        return None
    sql = "SELECT * FROM public.simulations WHERE id = %s"
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.execute(sql, (sim_id,))
            row = cur.fetchone()
            if row is None:
                return None
            names = [d[0] for d in cur.description]
            return dict(zip(names, row))
    except Exception:
        traceback.print_exc()
        return None

# ---------- columns metadata for template rendering ----------
def get_simulation_columns() -> List[Dict[str, Any]]:
    """
    Columns for public.simulations:
      [{name, data_type, is_nullable, default, description}]
    Description comes from pg_description if present (may be empty).
    """
    url = _db_url()
    if not psycopg:
        return []
    cols_sql = """
        SELECT c.column_name, c.data_type, c.is_nullable, c.column_default
        FROM information_schema.columns c
        WHERE c.table_schema = 'public' AND c.table_name = 'simulations'
        ORDER BY c.ordinal_position
    """
    desc_sql = """
        SELECT a.attname AS column_name, d.description
        FROM pg_catalog.pg_attribute a
        LEFT JOIN pg_catalog.pg_description d
          ON d.objoid = a.attrelid AND d.objsubid = a.attnum
        WHERE a.attrelid = 'public.simulations'::regclass
          AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY a.attnum
    """
    try:
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.execute(cols_sql)
            cols = cur.fetchall() or []
            cur.execute(desc_sql)
            descs = {r[0]: r[1] for r in (cur.fetchall() or [])}
    except Exception:
        traceback.print_exc()
        return []

    out: List[Dict[str, Any]] = []
    for name, dtype, nullable, default in cols:
        out.append({
            "name": name,
            "data_type": dtype,
            "is_nullable": (nullable == "YES"),
            "default": default,
            "description": descs.get(name) or "",
        })
    return out
