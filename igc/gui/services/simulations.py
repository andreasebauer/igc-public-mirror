from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from igc.db.pg import cx, fetchone_dict, fetchall_dict, execute

# --- Read operations ---------------------------------------------------------

def list_simulations(q: Optional[str]=None, limit: int=100, offset: int=0) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM public.simulations"
    args: Tuple[Any, ...] = ()
    if q:
        sql += " WHERE name ILIKE %s OR label ILIKE %s"
        args = (f"%{q}%", f"%{q}%")
    sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
    args += (limit, offset)
    with cx() as conn:
        return fetchall_dict(conn, sql, args)

def get_simulation(sim_id: int) -> Optional[Dict[str, Any]]:
    with cx() as conn:
        return fetchone_dict(conn, "SELECT * FROM public.simulations WHERE id=%s", (sim_id,))

# --- Write operations --------------------------------------------------------

def insert_simulation(payload: Dict[str, Any]) -> int:
    """
    Insert a new simulation row. Expects keys matching column names.
    Returns new simulation id.
    """
    # Build dynamic insert (columns from payload)
    cols = list(payload.keys())
    vals = [payload[c] for c in cols]
    placeholders = ", ".join(["%s"] * len(cols))
    collist = ", ".join(cols)
    sql = f"INSERT INTO public.simulations ({collist}) VALUES ({placeholders}) RETURNING id"
    with cx() as conn:
        row = fetchone_dict(conn, sql, tuple(vals))
        return int(row["id"])  # type: ignore

def update_simulation(sim_id: int, payload: Dict[str, Any]) -> int:
    """
    Update an existing simulation by id. Only provided keys are updated.
    Returns number of rows affected.
    """
    if not payload:
        return 0
    sets = ", ".join([f"{k}=%s" for k in payload.keys()])
    vals = list(payload.values()) + [sim_id]
    sql = f"UPDATE public.simulations SET {sets} WHERE id=%s"
    with cx() as conn:
        return execute(conn, sql, tuple(vals))
