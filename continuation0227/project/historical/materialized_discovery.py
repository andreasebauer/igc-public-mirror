from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Mapping
import atexit
import gc
import json
import os
import shutil
import sys
import tempfile
import threading
import uuid
import zipfile
from infinity_grid.canon import canonical_sha256
from . import maturation_parallel as mp_exec
from . import regime_scanner as rs

DISCOVERY_SEED_SHA256 = "cb6f48641eb99d374d19d5dc8d5bfada13e12cf1ecb20ed0e66958629ec08f1b"

_O7_RUNTIME_MODULE_NAME = "ig_materialized_discovery_o7_runtime"

_PROCESS_O7_RUNTIME: dict[str, Any] | None = None

_PROCESS_O7_RUNTIME_ATEXIT_REGISTERED = False

def _close_process_o7_runtime() -> None:
    """Release the one warm O7 replay kernel retained for this Python process."""
    global _PROCESS_O7_RUNTIME
    runtime = _PROCESS_O7_RUNTIME
    _PROCESS_O7_RUNTIME = None
    if not runtime:
        return
    try:
        base_states = runtime.get("base_states")
        if isinstance(base_states, list):
            base_states.clear()
        engine = runtime.get("engine")
        if engine is not None:
            try:
                engine.O6 = None
            except Exception:
                pass
        sys.modules.pop(_O7_RUNTIME_MODULE_NAME, None)
        prereg = sys.modules.get("ig_oscout_prereg_core")
        prereg_file = getattr(prereg, "__file__", None) if prereg is not None else None
        tmp = runtime.get("tmp")
        if prereg_file and tmp is not None:
            try:
                if Path(prereg_file).resolve().is_relative_to(Path(tmp).resolve()):
                    sys.modules.pop("ig_oscout_prereg_core", None)
            except Exception:
                pass
        ctx = runtime.get("seed_context")
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        if tmp is not None:
            shutil.rmtree(Path(tmp), ignore_errors=True)
    finally:
        gc.collect()

def _get_process_o7_runtime(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable warm O7 replay kernel for the current interpreter.

    Caller holds _DISCOVERY_SESSION_LOCK. Scientific O8+ carrier state is never shared.
    """
    global _PROCESS_O7_RUNTIME, _PROCESS_O7_RUNTIME_ATEXIT_REGISTERED
    if _PROCESS_O7_RUNTIME is not None:
        return _PROCESS_O7_RUNTIME
    tmp = Path(tempfile.mkdtemp(prefix="ig_o_materialized_discovery_runtime_"))
    seed_context = None
    old_data_root = os.environ.get("OSCOUT_DATA_ROOT")
    try:
        seed_context = as_file(files("infinity_grid").joinpath("resources", "decoder", "O7_MATERIAL_ROOT_cb6f48641eb9.zip"))
        seed_path = Path(seed_context.__enter__())
        seed_root, o7root = _extract_compact_seed(seed_path, tmp)
        os.environ["OSCOUT_DATA_ROOT"] = str(o7root.resolve())
        sys.modules.pop(_O7_RUNTIME_MODULE_NAME, None)
        engine = rs._load_module(_O7_RUNTIME_MODULE_NAME, o7root / "02_CODE" / "o7_live_engine.py")
        engine.O6 = engine.import_o6()
        parent_map = engine.load_parent_records()
        _, bpairs = engine.O6.load_rules()
        bridge_pairs = sorted(tuple(map(int, x)) for x in bpairs)
        records = json.loads(
            (seed_root / "graduation_compact" / "07_INPUT_SNAPSHOTS" / "O7_IMMUTABLE_SURVIVORS.json").read_text(encoding="utf-8")
        )["records"]
        base = []
        for r in records:
            ctx = engine._profile_row_context(r, parent_map)
            edges = tuple(tuple(x) for x in r["edges"])
            base.append(rs.O7State(engine, ctx, edges, (0, 0, 0, 0, 0, 0, 0), r["state_digest"], r["lane"]))
        selected = rs._farthest_select(base, int(spec["panel"]["beam"]))
        _PROCESS_O7_RUNTIME = {
            "tmp": tmp, "seed_context": seed_context, "seed_path": seed_path,
            "seed_root": seed_root, "o7root": o7root, "engine": engine,
            "bridge_pairs": bridge_pairs, "base_states": selected, "base_candidate_count": len(base),
        }
        if not _PROCESS_O7_RUNTIME_ATEXIT_REGISTERED:
            atexit.register(_close_process_o7_runtime)
            _PROCESS_O7_RUNTIME_ATEXIT_REGISTERED = True
        return _PROCESS_O7_RUNTIME
    except Exception:
        sys.modules.pop(_O7_RUNTIME_MODULE_NAME, None)
        if seed_context is not None:
            try: seed_context.__exit__(None, None, None)
            except Exception: pass
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        if old_data_root is None:
            os.environ.pop("OSCOUT_DATA_ROOT", None)
        else:
            os.environ["OSCOUT_DATA_ROOT"] = old_data_root

class MaterializedDiscoveryError(RuntimeError):
    pass

def _extract_compact_seed(seed_path: Path, work: Path) -> tuple[Path, Path]:
    seed_path = Path(seed_path)
    got = rs._sha_file(seed_path)
    if got != DISCOVERY_SEED_SHA256:
        raise MaterializedDiscoveryError(f"O7 compact discovery seed SHA mismatch: {got}")
    seed_out = work / "seed"
    with zipfile.ZipFile(seed_path) as zf:
        bad = zf.testzip()
        if bad:
            raise MaterializedDiscoveryError(f"O7 compact discovery seed CRC failure at {bad}")
        zf.extractall(seed_out)
    roots = [p for p in seed_out.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise MaterializedDiscoveryError("ambiguous O7 compact discovery seed root")
    seed_root = roots[0]
    replay_zip = seed_root / "replay" / "Infinity_Grid_O7_COMPACT_REPLAY_ROOT_v1_2026-08-29.zip"
    if not replay_zip.is_file():
        raise MaterializedDiscoveryError("compact discovery seed is missing the certified O7 replay root")
    replay_out = work / "o7"
    with zipfile.ZipFile(replay_zip) as zf:
        bad = zf.testzip()
        if bad:
            raise MaterializedDiscoveryError(f"O7 compact replay CRC failure at {bad}")
        zf.extractall(replay_out)
    roots = [p for p in replay_out.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise MaterializedDiscoveryError("ambiguous O7 compact replay root")
    return seed_root, roots[0]
