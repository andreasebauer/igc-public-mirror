from __future__ import annotations
from typing import Sequence, Dict, List
from datetime import datetime
from igc.db.pg import cx, fetchall_dict, execute

def list_metrics_for_sim(sim_id: int) -> list[dict]:
    """
    Legacy flat loader: metrics bound to a sim via simmetricgroupmatcher.
    Kept for backward compatibility / fallback rendering.
    """
    with cx() as conn:
        return fetchall_dict(conn, """
          select distinct m.id as metricid, m.name, m.description, m.outputtypes
          from simmetricgroupmatcher smgm
          join metrics m on m.id = smgm.metric_id
          where smgm.sim_id = %s
          order by m.name
        """, (sim_id,))

def list_metrics_grouped() -> List[Dict]:
    """
    Load metrics grouped (6 groups) using the DB view `metrics_group`.
    Expected columns:
      - group_label
      - id            (metric id)
      - name
      - description
      - outputtypes
    Returns:
      [
        { "label": <group>,
          "metrics": [
            {"id":..,"name":..,"desc":..,"out":..}, ...
          ]
        }, ...
      ]
    """
    with cx() as conn:
        rows = fetchall_dict(conn, """
          select group_label, id, name, description, outputtypes
          from metrics_group
          order by group_label, name
        """)
    groups: Dict[str, Dict] = {}
    for r in rows:
        glabel = r["group_label"]
        g = groups.get(glabel)
        if not g:
            g = {"label": glabel, "metrics": []}
            groups[glabel] = g
        g["metrics"].append({
            "id":   r["id"],
            "name": r["name"],
            "desc": r.get("description") or "",
            "out":  r.get("outputtypes") or ""
        })
    return [groups[k] for k in sorted(groups.keys())]

def validate_selection(metric_ids: Sequence[int]) -> list[str]:
    """
    Validate that at least one metric is selected.
    """
    if not metric_ids:
        return ["No metrics selected."]
    return []

def discover_frames_in_root(run_root: str) -> list[int]:
    """
    Discover existing frames for a given run_root directory on disk.

    Expects a folder structure like:
      run_root/
        Frame_0000/
        Frame_0100/
        Frame_0200/
        ...

    Returns a sorted list of integer frame indices.
    """
    import os
    import re

    frames: set[int] = set()
    if not run_root:
        return []
    try:
        for entry in os.scandir(run_root):
            if not entry.is_dir():
                continue
            name = entry.name
            # Examples: Frame_0000, frame10, 0000
            m = re.search(r"(\d+)$", name)
            if m:
                frames.add(int(m.group(1)))
    except Exception:
        # On any error, just return whatever we collected so far (possibly empty)
        return sorted(frames)
    return sorted(frames)

def discover_sweep_members(sweep_root: str) -> list[str]:
    """
    Discover sweep member run_root directories under a given sweep_root.

    Expects a folder structure like:
      sweep_root/
        <run_folder_1>/
        <run_folder_2>/
        ...

    Returns a sorted list of full paths to child directories.
    """
    import os

    members: list[str] = []
    if not sweep_root:
        return members
    try:
        for entry in os.scandir(sweep_root):
            if entry.is_dir():
                members.append(entry.path)
    except Exception:
        return sorted(members)
    return sorted(members)

def seed_jobs_for_frames(sim_id: int, metric_ids: Sequence[int], frames: Sequence[int]) -> int:
    """
    Create metric jobs (one per metric × frame).
    Uses group_id from simmetricgroupmatcher for (sim, metric).
    Keeps original insert and unique index semantics.
    """
    if not frames or not metric_ids:
        return 0
    with cx() as conn:
        execute(conn, "create unique index if not exists simmetjobs_unique on simmetjobs(simid,metricid,frame)")
        count = 0
        for mid in metric_ids:
            for f in frames:
                execute(conn, """
                  insert into simmetjobs (simid,metricid,frame,status,priority,createdate)
                  select %s,%s,%s,'queued',0, now()
                  from simmetricgroupmatcher smgm
                  where smgm.sim_id=%s and smgm.metric_id=%s
                  on conflict (simid,metricid,frame) do nothing
                """, (sim_id, mid, f, sim_id, mid))
                count += 1
        return count
