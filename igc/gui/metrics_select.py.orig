from __future__ import annotations
from typing import Sequence
from datetime import datetime
from igc.db.pg import cx, fetchall_dict, execute

def list_metrics_for_sim(sim_id:int) -> list[dict]:
    with cx() as conn:
        return fetchall_dict(conn, """
          select distinct m.id as metricid, m.name, m.description, m.outputtypes
          from simmetricgroupmatcher smgm
          join metrics m on m.id = smgm.metric_id
          where smgm.sim_id = %s
          order by m.name
        """, (sim_id,))

def validate_selection(metric_ids:Sequence[int]) -> list[str]:
    if not metric_ids: return ["no metrics selected"]
    with cx() as conn:
        rows = fetchall_dict(conn, "select id from metrics where id = any(%s)", (list(metric_ids),))
    found = {r["id"] for r in rows}
    missing = [str(mid) for mid in metric_ids if mid not in found]
    return [f"unknown metric id: {', '.join(missing)}"] if missing else []

def seed_jobs_for_frames(sim_id:int, metric_ids:Sequence[int], frames:Sequence[int]) -> int:
    if not frames or not metric_ids: return 0
    with cx() as conn:
        execute(conn, "create unique index if not exists simmetricjobs_unique on simmetricjobs(simid,metricid,frame)")
        count=0
        for mid in metric_ids:
            for f in frames:
                execute(conn, """
                  insert into simmetricjobs (simid,metricid,groupid,frame,status,jobtype,jobsubtype,priority,createdate)
                  select %s,%s,smgm.group_id,%s,'queued','metric','final',0, now()::text
                  from simmetricgroupmatcher smgm
                  where smgm.sim_id=%s and smgm.metric_id=%s
                  on conflict (simid,metricid,frame) do nothing
                """, (sim_id, mid, f, sim_id, mid))
                count += 1
        return count
