from __future__ import annotations

"""Frozen G-uplift architecture and nomenclature contract (Decoder v0.29.0 Phase 0).

This module contains no G2 generator and performs no G2 science.  It only exposes and
verifies the machine-readable contract that future structural-uplift experiments must obey.
"""

from dataclasses import dataclass
from importlib.resources import files
import hashlib
import json
import re
from typing import Any, Mapping


class UpliftArchitectureError(RuntimeError):
    pass


_STAGE_RE = re.compile(r"^G([1-9][0-9]*):([CSR])([0-9]+)$")
_RESOURCE_ROOT = "resources/uplift"


def _load(name: str) -> dict[str, Any]:
    return json.loads(files("infinity_grid").joinpath(_RESOURCE_ROOT, name).read_text(encoding="utf-8"))


def _self_hash(obj: Mapping[str, Any], field: str) -> str:
    payload = {k: v for k, v in obj.items() if k != field}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def uplift_contract() -> dict[str, Any]:
    obj = _load("G_UPLIFT_ARCHITECTURE_CONTRACT_V1.json")
    expected = obj.get("contract_sha256")
    observed = _self_hash(obj, "contract_sha256")
    if expected != observed:
        raise UpliftArchitectureError(f"uplift contract identity mismatch: expected {expected}, observed {observed}")
    if obj.get("status") != "FROZEN_PHASE0_ARCHITECTURE_ONLY":
        raise UpliftArchitectureError("unexpected uplift contract status")
    return obj


def nomenclature_contract() -> dict[str, Any]:
    obj = _load("G_UPLIFT_NOMENCLATURE_V1.json")
    expected = obj.get("science_sha256")
    observed = _self_hash(obj, "science_sha256")
    if expected != observed:
        raise UpliftArchitectureError("uplift nomenclature identity mismatch")
    return obj


@dataclass(frozen=True, order=True)
class StageRef:
    layer: int
    axis: str
    index: int

    @classmethod
    def parse(cls, label: str) -> "StageRef":
        m = _STAGE_RE.fullmatch(label or "")
        if m is None:
            raise UpliftArchitectureError(f"invalid active G-stage ref: {label!r}")
        return cls(int(m.group(1)), m.group(2), int(m.group(3)))

    def __str__(self) -> str:
        return f"G{self.layer}:{self.axis}{self.index}"

    @property
    def is_candidate(self) -> bool:
        return self.axis == "C"

    @property
    def is_structural(self) -> bool:
        return self.axis == "S"

    @property
    def is_recursive(self) -> bool:
        return self.axis == "R"


def legacy_o_to_active(level: int) -> str:
    """Map historical post-graduation O-depth labels to active G1 recursive depth labels.

    O0..O7 remain historical discovery/graduation labels and are intentionally not silently
    rewritten. O8..O100 are the frozen legacy range re-described as G1 recursive depths.
    """
    if not isinstance(level, int):
        raise UpliftArchitectureError("legacy O level must be int")
    if 8 <= level <= 100:
        return f"G1:R{level}"
    if 0 <= level <= 7:
        raise UpliftArchitectureError(f"O{level} is historical discovery/graduation provenance, not an active recursive-depth alias")
    raise UpliftArchitectureError(f"legacy mapping not frozen for O{level}")


def assert_stage_authorized(label: str, *, graduated_layers: set[int] | frozenset[int]) -> StageRef:
    """Fail closed if a post-graduation R stage is referenced without layer graduation."""
    ref = StageRef.parse(label)
    if ref.axis == "R" and ref.layer not in graduated_layers:
        raise UpliftArchitectureError(f"{ref}: recursive axis requires a graduation certificate for G{ref.layer}")
    return ref


def structural_program() -> tuple[dict[str, Any], ...]:
    contract = uplift_contract()
    return tuple(dict(x) for x in contract["canonical_structural_program"])


def phase0_verification_result() -> dict[str, Any]:
    c = uplift_contract()
    n = nomenclature_contract()
    stages = structural_program()
    expected = [f"S{i}" for i in range(7)]
    observed = [x["stage"] for x in stages]
    failures: list[str] = []
    if observed != expected:
        failures.append("STRUCTURAL_STAGE_SEQUENCE")
    if c["graduation_firewall"].get("candidate_stage_can_graduate") is not False:
        failures.append("CANDIDATE_PROMOTION_FIREWALL")
    if c["graduation_firewall"].get("Gk_R_axis_requires_graduation_certificate") is not True:
        failures.append("R_AXIS_GRADUATION_FIREWALL")
    if n["current_status"].get("G2") != "NOT_EARNED":
        failures.append("G2_STATUS")
    payload = {
        "schema_id":"IG_G_UPLIFT_PHASE0_VERIFICATION_V1",
        "status":"PASS" if not failures else "FAIL",
        "contract_sha256":c["contract_sha256"],
        "nomenclature_sha256":n["science_sha256"],
        "structural_stages":observed,
        "legacy_mapping_checks":{str(i):legacy_o_to_active(i) for i in (8,14,20,100)},
        "failures":failures,
        "new_g_science":False,
        "g2_graduated":False,
    }
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    payload["science_sha256"]=hashlib.sha256(raw).hexdigest()
    return payload
