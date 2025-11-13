# igc/gui/services/runtime_sampler.py
from __future__ import annotations
import os, time, shutil
from pathlib import Path

try:
    import psutil  # optional
except Exception:
    psutil = None  # type: ignore

_STORE = Path(os.environ.get("IGC_STORE", "/data/simulations"))

_last_t = 0.0
_cache = {
    "cpu_pct_proc": 0.0,
    "cpu_pct_sys":  0.0,
    "ram_used_mb":  0,
    "disk_free_mb": 0,
    "disk_total_mb": 0,
}

def _sample() -> None:
    global _last_t, _cache
    now = time.time()
    # sample at most once per second
    if now - _last_t < 1.0:
        return
    cpu_proc = cpu_sys = 0.0
    rss_mb = 0
    if psutil:
        try:
            p = psutil.Process(os.getpid())
            cpu_proc = float(p.cpu_percent(interval=0.0))
            cpu_sys  = float(psutil.cpu_percent(interval=0.0))
            rss_mb   = int(getattr(p.memory_info(), "rss", 0) // (1024*1024))
        except Exception:
            pass
    try:
        du = shutil.disk_usage(str(_STORE))
        df_mb  = int(du.free  // (1024*1024))
        dt_mb  = int(du.total // (1024*1024))
    except Exception:
        df_mb = dt_mb = 0
    _cache.update({
        "cpu_pct_proc": cpu_proc,
        "cpu_pct_sys":  cpu_sys,
        "ram_used_mb":  rss_mb,
        "disk_free_mb": df_mb,
        "disk_total_mb": dt_mb,
    })
    _last_t = now

def get() -> dict:
    _sample()
    return dict(_cache)