from __future__ import annotations
from igc.db.pg import cx, fetchall_dict

def list_actions() -> list[dict]:
    return [
        {"code":"create_sim","label":"Create Simulation"},
        {"code":"select_data","label":"Select / Inspect Data"},
        {"code":"select_metrics","label":"Select & Seed Metrics"},
        {"code":"monitor_jobs","label":"Monitor Jobs"}
    ]

def db_health() -> dict:
    with cx() as conn:
        row = fetchall_dict(conn, "select now() as now, current_database() as db, current_user as user")[0]
        return row

def recent_sims(limit:int=50) -> list[dict]:
    with cx() as conn:
        return fetchall_dict(conn, """
            select id, simid, label, name, gridx, gridy, gridz, status, createdate
            from simulations
            order by id desc
            limit %s
        """, (limit,))
