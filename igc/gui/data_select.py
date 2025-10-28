from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import os, re
from igc.db.pg import cx, fetchall_dict, fetchone_dict

STORE = Path(os.environ.get("IGC_STORE", "/data/igc"))
FRAME_RE = re.compile(r'^Frame_(\d{4})$')

def list_runs(limit:int=100) -> list[dict]:
    with cx() as conn:
        return fetchall_dict(conn, """
          select id, simid, label, name, gridx, gridy, gridz, status, createdate
          from simulations
          order by id desc
          limit %s
        """, (limit,))

def list_frames_by_fs(sim_label:str) -> list[int]:
    base = STORE / f"Sim_{sim_label}"
    if not base.exists(): return []
    frames=[]
    for p in base.iterdir():
        if not p.is_dir(): continue
        m = FRAME_RE.match(p.name)
        if m: frames.append(int(m.group(1)))
    return sorted(frames)

def describe_run(sim_id:int) -> dict:
    with cx() as conn:
        sim = fetchone_dict(conn, "select id, simid, label, name, gridx, gridy, gridz, createdate, status from simulations where id=%s", (sim_id,))
    if not sim: return {"error":"simulation not found"}
    frames = list_frames_by_fs(sim["label"])
    return {"sim":sim,"frames":frames,"frames_count":len(frames)}

def select_frame_range(frames:Sequence[int], start:int, end:int, stride:int) -> list[int]:
    if not frames: return []
    s = max(start, min(frames))
    e = min(end,   max(frames))
    chosen=[f for f in frames if s <= f <= e]
    if stride>1:
        chosen = [f for i,f in enumerate(chosen) if i % stride == 0]
    return chosen
