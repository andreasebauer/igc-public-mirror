from __future__ import annotations
from typing import Optional
from igc.db.pg import cx, fetchall_dict, fetchone_dict, execute

def list_active_jobs(sim_id: Optional[int]=None, limit:int=200) -> list[dict]:
    sql = "select * from big_view"
    params=[]
    if sim_id:
        sql += " where smj_simid = %s"
        params.append(sim_id)
    sql += " order by smj_jobid desc limit %s"
    params.append(limit)
    with cx() as conn:
        return fetchall_dict(conn, sql, tuple(params))

def job_detail(job_id:int) -> dict:
    with cx() as conn:
        row = fetchone_dict(conn, "select * from big_view where smj_jobid=%s", (job_id,))
        return row or {}

def requeue_job(job_id:int) -> int:
    with cx() as conn:
        execute(conn, "update simmetricjobs set status='queued' where jobid=%s", (job_id,))
        return 1

def cancel_job(job_id:int) -> int:
    with cx() as conn:
        execute(conn, "update simmetricjobs set status='canceled' where jobid=%s", (job_id,))
        return 1
