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
