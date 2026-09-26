from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .canon import canonical_sha256, canonical_text, write_json_atomic
from .records import utc_now


from .core.boundary import destination_tuple as _dtup, canonical_boundary_record as _canon, record_sha256 as _sha_record, bridge_ports as _bridge, compose_boundary_records as _compose

class CorruptChunkCheckpoint(RuntimeError):
    pass


class StaleChunkCheckpoint(RuntimeError):
    pass


def chunk_parent_ranges(sources: list[dict[str, Any]], chunk_size: int) -> list[dict[str, Any]]:
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    ordered = sorted(sources, key=lambda x: x["sha256"])
    out = []
    for ordinal, start in enumerate(range(0, len(ordered), chunk_size)):
        part = ordered[start:start + chunk_size]
        if not part:
            continue
        out.append({
            "ordinal": ordinal,
            "start": start,
            "stop": start + len(part),
            "first_parent_sha256": part[0]["sha256"],
            "last_parent_sha256": part[-1]["sha256"],
            "chunk_id": f"{ordinal:06d}-{part[0]['sha256'][:12]}-{part[-1]['sha256'][:12]}",
            "sources": part,
        })
    return out


def compute_generation_chunk(sources: list[dict[str, Any]], primitive_records: list[Any]) -> dict[str, Any]:
    children: dict[str, Any] = {}
    pars: dict[str, set[str]] = defaultdict(set)
    wit: Counter[str] = Counter()
    src_child: dict[str, set[str]] = defaultdict(set)
    attempts = lawful = 0
    for s in sorted(sources, key=lambda x: x["sha256"]):
        a = s["record"]
        for b in primitive_records:
            for sa in range(len(a[1])):
                for sp in range(len(b[1])):
                    attempts += 1
                    ch = _compose(a, sa, b, sp)
                    if ch is None:
                        continue
                    lawful += 1
                    h = _sha_record(ch)
                    children[h] = ch
                    pars[h].add(s["sha256"])
                    wit[h] += 1
                    src_child[s["sha256"]].add(h)
    return {
        "attempts": attempts,
        "lawful": lawful,
        "children": {h: repr(children[h]) for h in sorted(children)},
        "parents": {h: sorted(pars[h]) for h in sorted(pars)},
        "witness_count": {h: int(wit[h]) for h in sorted(wit)},
        "source_children": {s: sorted(src_child[s]) for s in sorted(src_child)},
    }


def merge_generation_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    children: dict[str, str] = {}
    pars: dict[str, set[str]] = defaultdict(set)
    wit: Counter[str] = Counter()
    src_child: dict[str, set[str]] = defaultdict(set)
    attempts = lawful = 0
    for chunk in chunks:
        attempts += int(chunk["attempts"])
        lawful += int(chunk["lawful"])
        for h, rec in chunk["children"].items():
            prev = children.get(h)
            if prev is not None and prev != rec:
                raise RuntimeError(f"child SHA collision with unequal record: {h}")
            children[h] = rec
        for h, ps in chunk["parents"].items():
            pars[h].update(ps)
        for h, n in chunk["witness_count"].items():
            wit[h] += int(n)
        for s, hs in chunk["source_children"].items():
            src_child[s].update(hs)
    return {
        "attempts": attempts,
        "lawful": lawful,
        "children": {h: children[h] for h in sorted(children)},
        "parents": {h: sorted(pars[h]) for h in sorted(pars)},
        "witness_count": {h: int(wit[h]) for h in sorted(wit)},
        "source_children": {s: sorted(src_child[s]) for s in sorted(src_child)},
    }


def generation_projection_sha256(merged: dict[str, Any]) -> str:
    return canonical_sha256(merged)


class ChunkCheckpointStore:
    """Immutable chunk checkpoint store bound to exact generation dependencies.

    Existing non-matching or corrupt checkpoints fail closed.  Missing checkpoints
    may be computed.  This is intentionally stricter than ordinary stage resume.
    """

    def __init__(self, root: Path, binding: dict[str, Any]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = json.loads(canonical_text(binding))
        self.binding_sha256 = canonical_sha256(self.binding)

    def _paths(self, chunk_id: str) -> tuple[Path, Path]:
        if not chunk_id or "/" in chunk_id or "\\" in chunk_id or ".." in chunk_id:
            raise ValueError("unsafe chunk_id")
        return self.root / f"{chunk_id}.json", self.root / f"{chunk_id}.sha256"

    def _verify_existing(self, chunk_id: str) -> dict[str, Any] | None:
        p, sp = self._paths(chunk_id)
        if not p.exists() and not sp.exists():
            return None
        if not p.is_file() or not sp.is_file():
            raise CorruptChunkCheckpoint(f"incomplete chunk checkpoint {chunk_id}")
        raw = p.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        declared = sp.read_text(encoding="utf-8").strip().split()[0]
        if observed != declared:
            raise CorruptChunkCheckpoint(f"chunk checkpoint byte hash mismatch {chunk_id}")
        obj = json.loads(raw)
        if obj.get("schema_id") != "IG_SCOUT_CHUNK_CHECKPOINT_V0_1":
            raise CorruptChunkCheckpoint(f"chunk checkpoint schema mismatch {chunk_id}")
        content = {k: v for k, v in obj.items() if k != "checkpoint_content_sha256"}
        if canonical_sha256(content) != obj.get("checkpoint_content_sha256"):
            raise CorruptChunkCheckpoint(f"chunk checkpoint content hash mismatch {chunk_id}")
        if obj.get("binding_sha256") != self.binding_sha256 or obj.get("binding") != self.binding:
            raise StaleChunkCheckpoint(f"chunk checkpoint dependency binding mismatch {chunk_id}")
        if obj.get("chunk_id") != chunk_id or obj.get("status") != "COMPLETE_VALID":
            raise CorruptChunkCheckpoint(f"chunk checkpoint identity/status mismatch {chunk_id}")
        return obj

    def get_or_compute(self, *, chunk_id: str, chunk_spec: dict[str, Any], compute: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        existing = self._verify_existing(chunk_id)
        spec_sha = canonical_sha256(chunk_spec)
        if existing is not None:
            if existing.get("chunk_spec_sha256") != spec_sha or existing.get("chunk_spec") != chunk_spec:
                raise StaleChunkCheckpoint(f"chunk specification mismatch {chunk_id}")
            return existing["result"], True
        result = compute()
        record = {
            "schema_id": "IG_SCOUT_CHUNK_CHECKPOINT_V0_1",
            "chunk_id": chunk_id,
            "status": "COMPLETE_VALID",
            "binding": self.binding,
            "binding_sha256": self.binding_sha256,
            "chunk_spec": chunk_spec,
            "chunk_spec_sha256": spec_sha,
            "result": result,
            "result_sha256": canonical_sha256(result),
            "created_utc": utc_now(),
        }
        record["checkpoint_content_sha256"] = canonical_sha256(record)
        p, sp = self._paths(chunk_id)
        if p.exists() or sp.exists():
            raise RuntimeError(f"chunk checkpoint race: {chunk_id}")
        write_json_atomic(p, record)
        raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        tmp = sp.with_suffix(sp.suffix + f".tmp-{os.getpid()}")
        tmp.write_text(sha + "\n", encoding="utf-8")
        with tmp.open("rb") as h:
            try:
                os.fsync(h.fileno())
            except OSError:
                pass
        os.replace(tmp, sp)
        return result, False

    def verify_all(self) -> dict[str, Any]:
        checked = 0
        for p in sorted(self.root.glob("*.json")):
            self._verify_existing(p.stem)
            checked += 1
        return {"status": "PASS", "checked_chunks": checked, "binding_sha256": self.binding_sha256}


def load_fixture_sources(fixture: Path, selected_name: str = "PARENT_SELECTED_RELATION.json") -> list[dict[str, Any]]:
    raw = json.loads((Path(fixture) / selected_name).read_text(encoding="utf-8"))
    sources = []
    for x in raw.get("selected", []):
        rec = _canon(ast.literal_eval(x["record"]))
        if _sha_record(rec) != x["sha256"]:
            raise RuntimeError("parent selected relation record SHA mismatch")
        sources.append({"sha256": x["sha256"], "record": rec})
    sources.sort(key=lambda x: x["sha256"])
    return sources


def load_fixture_primitive(fixture: Path) -> list[Any]:
    with (Path(fixture) / "primitive.pkl").open("rb") as h:
        prim = pickle.load(h)
    return [_canon(p["record"]) for p in prim]
