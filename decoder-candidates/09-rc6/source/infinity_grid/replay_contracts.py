from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, write_json_atomic


class ReplayContractError(RuntimeError):
    pass


CONTRACT_SCHEMA = "IG_REPLAY_CONTRACT_V1"
REGISTRY_SCHEMA = "IG_REPLAY_CONTRACT_REGISTRY_V1"
REQUIRED_MARKER_SCHEMA = "IG_REPLAY_CONTRACTS_REQUIRED_V1"

REPLAY_CLASSES = {
    "VERIFY_AUTHORITY",
    "RERUN_GRADUATION_AUDIT",
    "REPLAY_BOUNDED_EXPERIMENT",
    "SCIENTIFIC_RECONSTRUCTION",
    "CHECKPOINT_PREFLIGHT",
    "FULL_HISTORICAL_CAMPAIGN",
}

AUTHORITY_CLASSES = {
    "CANONICAL_SCIENCE_AUTHORITY",
    "CERTIFIED_AUTHORITY",
    "RECOVERED_CERTIFIED_AUTHORITY",
    "RECONSTRUCTED_SCIENCE",
    "BOUNDED_SCIENTIFIC_EVIDENCE",
    "CHECKPOINT_STATE_EVIDENCE",
    "ORCHESTRATION_CONTROL_EVIDENCE",
    "HISTORICAL_CAMPAIGN_EVIDENCE",
}

HISTORICAL_IDENTITY_CLASSES = {
    "BYTE_IDENTICAL_WITHIN_SCOPE",
    "SCIENCE_EQUIVALENT_NOT_BYTE_IDENTICAL",
    "CERTIFICATE_ONLY",
    "CHECKPOINT_NATIVE",
    "NOT_APPLICABLE",
    "UNAVAILABLE",
}


def _contract_identity(contract: dict[str, Any]) -> str:
    return canonical_sha256({k: v for k, v in contract.items() if k != "contract_sha256"})


def validate_replay_contract(contract: dict[str, Any]) -> None:
    required = {
        "schema_id", "profile_id", "profile_sha256", "target_node_id", "target_alias",
        "replay_class", "authority_class", "historical_identity", "recomputes",
        "does_not_recompute", "limitations", "legacy_resume_semantics", "contract_sha256",
    }
    missing = required - set(contract)
    if missing:
        raise ReplayContractError(f"replay contract missing fields: {sorted(missing)}")
    if contract.get("schema_id") != CONTRACT_SCHEMA:
        raise ReplayContractError("bad replay contract schema")
    if contract.get("replay_class") not in REPLAY_CLASSES:
        raise ReplayContractError(f"unsupported replay_class {contract.get('replay_class')}")
    if contract.get("authority_class") not in AUTHORITY_CLASSES:
        raise ReplayContractError(f"unsupported authority_class {contract.get('authority_class')}")
    if contract.get("historical_identity") not in HISTORICAL_IDENTITY_CLASSES:
        raise ReplayContractError(f"unsupported historical_identity {contract.get('historical_identity')}")
    for key in ("profile_id", "profile_sha256", "target_node_id", "recomputes", "does_not_recompute"):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise ReplayContractError(f"{key} must be a nonempty string")
    if not isinstance(contract.get("limitations"), list) or not contract["limitations"] or any(not isinstance(x, str) or not x.strip() for x in contract["limitations"]):
        raise ReplayContractError("limitations must be a nonempty string list")
    if _contract_identity(contract) != contract.get("contract_sha256"):
        raise ReplayContractError("replay contract hash mismatch")


def build_replay_contract(**kwargs: Any) -> dict[str, Any]:
    c = {
        "schema_id": CONTRACT_SCHEMA,
        "profile_id": kwargs["profile_id"],
        "profile_sha256": kwargs["profile_sha256"],
        "target_node_id": kwargs["target_node_id"],
        "target_alias": kwargs.get("target_alias"),
        "replay_class": kwargs["replay_class"],
        "authority_class": kwargs["authority_class"],
        "historical_identity": kwargs["historical_identity"],
        "recomputes": kwargs["recomputes"],
        "does_not_recompute": kwargs["does_not_recompute"],
        "limitations": list(kwargs["limitations"]),
        "legacy_resume_semantics": kwargs.get("legacy_resume_semantics", ""),
    }
    c["contract_sha256"] = _contract_identity(c)
    validate_replay_contract(c)
    return c


class ReplayContractStore:
    def __init__(self, paths):
        self.paths = paths
        self.root = paths.store / "graph" / "replay_contracts"
        self.marker = paths.store / "graph" / "REPLAY_CONTRACTS_REQUIRED.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def required(self) -> bool:
        if not self.marker.is_file():
            return False
        obj = json.loads(self.marker.read_text(encoding="utf-8"))
        return obj.get("schema_id") == REQUIRED_MARKER_SCHEMA and bool(obj.get("required"))

    def install_required_marker(self) -> dict[str, Any]:
        obj = {"schema_id": REQUIRED_MARKER_SCHEMA, "required": True, "contract_schema": CONTRACT_SCHEMA}
        write_json_atomic(self.marker, obj)
        return obj

    def register(self, contract: dict[str, Any]) -> dict[str, Any]:
        validate_replay_contract(contract)
        p = self.root / f"{contract['contract_sha256']}.json"
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            validate_replay_contract(old)
            if old != contract:
                raise ReplayContractError("immutable replay contract conflict")
            return old
        write_json_atomic(p, contract)
        return contract

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            o = json.loads(p.read_text(encoding="utf-8"))
            validate_replay_contract(o)
            out.append(o)
        return out

    def resolve_for_target(self, target_node_id: str) -> dict[str, Any]:
        matches = [c for c in self.list() if c["target_node_id"] == target_node_id]
        if len(matches) != 1:
            if not matches:
                raise ReplayContractError(f"no replay contract registered for {target_node_id}")
            raise ReplayContractError(f"ambiguous replay contracts for {target_node_id}: {len(matches)}")
        return matches[0]

    def validate_profile_coverage(self, profiles: list[dict[str, Any]]) -> dict[str, Any]:
        contracts = self.list()
        by_target: dict[str, list[dict[str, Any]]] = {}
        for c in contracts:
            by_target.setdefault(c["target_node_id"], []).append(c)
        failures = []
        rows = []
        for p in profiles:
            ms = by_target.get(p["target_node_id"], [])
            if len(ms) != 1:
                failures.append({"profile_id": p["profile_id"], "target_node_id": p["target_node_id"], "reason": "MISSING_OR_AMBIGUOUS_CONTRACT", "count": len(ms)})
                continue
            c = ms[0]
            if c["profile_id"] != p["profile_id"] or c["profile_sha256"] != p["profile_sha256"]:
                failures.append({"profile_id": p["profile_id"], "target_node_id": p["target_node_id"], "reason": "PROFILE_IDENTITY_MISMATCH", "contract_profile_id": c["profile_id"], "contract_profile_sha256": c["profile_sha256"], "profile_sha256": p["profile_sha256"]})
            rows.append({"profile_id": p["profile_id"], "target_node_id": p["target_node_id"], "replay_class": c["replay_class"], "authority_class": c["authority_class"]})
        extra = sorted(set(by_target) - {p["target_node_id"] for p in profiles})
        if extra:
            failures.append({"reason": "EXTRA_CONTRACT_TARGETS", "target_node_ids": extra})
        return {
            "schema_id": REGISTRY_SCHEMA,
            "status": "PASS" if not failures else "FAIL",
            "profiles": len(profiles),
            "contracts": len(contracts),
            "rows": sorted(rows, key=lambda r: r["profile_id"]),
            "failures": failures,
        }
