from __future__ import annotations

import json
from pathlib import Path

from .canon import canonical_sha256, write_json_atomic
from .records import utc_now


PACKAGING_SCHEMA = "IG_DECODER_V05_PACKAGING_STATE_V1_3"
_ALLOWED = {
    "NOT_STARTED": {"PACKAGING"},
    "PACKAGING": {"COMPLETE", "FAILED"},
    "FAILED": {"PACKAGING"},
    "COMPLETE": set(),
}


def _path(run_dir: Path) -> Path:
    return Path(run_dir) / "v05_packaging.json"


def initialize_packaging_state(run_dir: Path) -> dict:
    p = _path(run_dir)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    base = {
        "schema_id": PACKAGING_SCHEMA,
        "run_id": Path(run_dir).name,
        "state": "NOT_STARTED",
        "attempt": 0,
        "updated_utc": utc_now(),
        "detail": None,
    }
    rec = dict(base, state_sha256=canonical_sha256(base))
    write_json_atomic(p, rec)
    return rec


def load_packaging_state(run_dir: Path) -> dict:
    rec = initialize_packaging_state(run_dir)
    observed = canonical_sha256({k: v for k, v in rec.items() if k != "state_sha256"})
    if rec.get("state_sha256") != observed:
        raise RuntimeError("packaging state hash mismatch")
    return rec


def set_packaging_state(run_dir: Path, state: str, *, detail=None) -> dict:
    old = load_packaging_state(run_dir)
    if state not in _ALLOWED.get(old["state"], set()):
        raise RuntimeError(f"invalid packaging transition {old['state']} -> {state}")
    attempt = old["attempt"] + (1 if state == "PACKAGING" else 0)
    base = {
        "schema_id": PACKAGING_SCHEMA,
        "run_id": old["run_id"],
        "state": state,
        "attempt": attempt,
        "updated_utc": utc_now(),
        "detail": detail,
    }
    rec = dict(base, state_sha256=canonical_sha256(base))
    write_json_atomic(_path(run_dir), rec)
    return rec
