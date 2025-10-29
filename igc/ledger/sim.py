from __future__ import annotations
from typing import Any, Dict, List, Optional
from igc.ledger.core import _connect  # use the core connection helper

def list_simulations(limit: int = 500) -> List[Dict[str, Any]]:
    sql = """
      SELECT id, label, name, description
      FROM public.simulations
      ORDER BY label ASC, name ASC, id DESC
      LIMIT %s
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (limit,))
        rows = cur.fetchall() or []
    return [{"id": r[0], "label": r[1], "name": r[2], "description": r[3]} for r in rows]

def get_simulation(sim_id: int) -> Optional[Dict[str, Any]]:
    sql = "SELECT id, label, name, description FROM public.simulations WHERE id = %s"
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (sim_id,))
        r = cur.fetchone()
    if not r:
        return None
    return {"id": r[0], "label": r[1], "name": r[2], "description": r[3]}

def get_simulation_full(sim_id: int) -> Optional[Dict[str, Any]]:
    sql = "SELECT * FROM public.simulations WHERE id = %s"
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (sim_id,))
        row = cur.fetchone()
        if row is None:
            return None
        names = [d[0] for d in cur.description]
        return dict(zip(names, row))

def get_simulation_columns() -> List[Dict[str, Any]]:
    cols_sql = """
      SELECT c.column_name, c.data_type, c.is_nullable, c.column_default
      FROM information_schema.columns c
      WHERE c.table_schema='public' AND c.table_name='simulations'
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
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(cols_sql); cols = cur.fetchall() or []
        cur.execute(desc_sql); descs = cur.fetchall() or []
    dmap = {r[0]: (r[1] or "") for r in descs}
    out: List[Dict[str, Any]] = []
    for name, dtype, nullable, default in cols:
        out.append({
            "name": name,
            "data_type": dtype,
            "is_nullable": (nullable == "YES"),
            "default": default,
            "description": dmap.get(name, ""),
        })
    return out

def _simulation_columns_set():
    # cache the set of updatable columns from information_schema (lowercase names)
    cols_sql = """
      SELECT lower(c.column_name)
      FROM information_schema.columns c
      WHERE c.table_schema='public' AND c.table_name='simulations'
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(cols_sql)
        return {r[0] for r in (cur.fetchall() or [])}

_SIM_COLS = None

def _filter_values(values: dict) -> dict:
    global _SIM_COLS
    if _SIM_COLS is None:
        _SIM_COLS = _simulation_columns_set()
    return {k: v for k, v in values.items() if k.lower() in _SIM_COLS and k.lower() != 'id'}

def update_simulation(sim_id: int, values: dict) -> int:
    """UPDATE public.simulations SET ... WHERE id=%s; returns affected row count."""
    vals = _filter_values(values)
    if not vals:
        return 0
    cols = list(vals.keys())
    sets = ", ".join(f'"{c}" = %s' for c in cols)
    sql = f'UPDATE public.simulations SET {sets} WHERE id = %s'
    params = [vals[c] for c in cols] + [sim_id]
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params); conn.commit()
        return cur.rowcount or 0

def create_simulation(values: dict) -> int:
    """INSERT into public.simulations (...) values (...); returns new id."""
    vals = _filter_values(values)
    if not vals:
        # minimal insert if nothing provided
        sql = 'INSERT INTO public.simulations DEFAULT VALUES RETURNING id'
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(sql); new_id = cur.fetchone()[0]; conn.commit(); return int(new_id)
    cols = list(vals.keys())
    placeholders = ", ".join(["%s"] * len(cols))
    cols_sql = ", ".join(f'"{c}"' for c in cols)
    sql = f'INSERT INTO public.simulations ({cols_sql}) VALUES ({placeholders}) RETURNING id'
    params = [vals[c] for c in cols]
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params); new_id = cur.fetchone()[0]; conn.commit(); return int(new_id)
