from __future__ import annotations

"""Authoritative scientific ResearchFrontier, deliberately separate from software CURRENT_STATE.

The ResearchFrontier is the programme's scientific position at the active zoom.  It is
never inferred from process liveness or Decoder release metadata.  This module only
validates, loads and verifies a frozen frontier artifact; it does not mutate science.
"""

from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import json

from .canon import canonical_sha256
from .scientific_architecture import (
    ScientificArchitectureError,
    ScientificProtocolRegistry,
    make_ref,
    validate_scientific_artifact,
)


CURRENT_FRONTIER_RESOURCE = "resources/science_architecture/frontier/RESEARCH_FRONTIER_CURRENT_V1.json"


class ResearchFrontierError(ScientificArchitectureError):
    pass


def _load_json_path(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResearchFrontierError(f"cannot parse ResearchFrontier {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ResearchFrontierError("ResearchFrontier must be a JSON object")
    return obj


def load_research_frontier(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        obj = json.loads(files("infinity_grid").joinpath(CURRENT_FRONTIER_RESOURCE).read_text(encoding="utf-8"))
    else:
        obj = _load_json_path(Path(path))
    verify_research_frontier(obj, raise_on_error=True)
    return obj


def _depth_as_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ResearchFrontierError(f"{label} cannot be boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.upper().startswith("O") and raw[1:].isdigit():
            return int(raw[1:])
        if raw.isdigit():
            return int(raw)
    raise ResearchFrontierError(f"{label} must be an integer depth, O<number>, or null")


def verify_research_frontier(
    frontier: Mapping[str, Any],
    *,
    registry: ScientificProtocolRegistry | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        validate_scientific_artifact(frontier)
        registry = registry or ScientificProtocolRegistry()
        entity_ref = str(frontier["current_entity_class_ref"])
        regime_ref = str(frontier["current_regime_ref"])
        entity = registry.entity(entity_ref)
        regime = registry.regime(regime_ref)
        if regime["input_entity_class_ref"] != entity_ref:
            errors.append(
                f"Regime {regime_ref} input_entity_class_ref={regime['input_entity_class_ref']} does not match frontier entity {entity_ref}"
            )

        active = list(frontier.get("active_test_pack_refs", []))
        available = list(frontier.get("available_test_pack_refs", []))
        if len(active) != len(set(active)):
            errors.append("active_test_pack_refs contains duplicates")
        if len(available) != len(set(available)):
            errors.append("available_test_pack_refs contains duplicates")
        if not set(active).issubset(set(available)):
            errors.append("active_test_pack_refs must be a subset of available_test_pack_refs")
        for pref in available:
            try:
                registry.pack(pref)
            except ScientificArchitectureError as exc:
                errors.append(f"frontier references unknown TestPack {pref}: {exc}")

        depth = _depth_as_int(frontier.get("latest_verified_regime_depth"), label="latest_verified_regime_depth")
        origin = _depth_as_int(regime.get("depth_coordinate", {}).get("origin"), label="regime depth origin")
        if depth is not None and origin is not None and depth < origin:
            errors.append(f"latest verified depth {depth} precedes regime origin {origin}")

        # Keep plateau and lift semantically distinct.  This pass does not certify either.
        plateau = str(frontier.get("plateau_state", ""))
        lift = str(frontier.get("lift_state", ""))
        if "CERTIFIED" in lift and "NOT_CERTIFIED" not in lift and "CERTIFIED" not in plateau:
            errors.append("a certified lift cannot be represented without a plateau/maturity authority in the frontier")

        if frontier.get("rule") != "SCIENTIFIC_FRONTIER_IS_SEPARATE_FROM_SOFTWARE_RELEASE_STATE":
            errors.append("ResearchFrontier separation rule is missing")
        swref = str(frontier.get("software_release_state_ref", ""))
        if not swref:
            errors.append("software_release_state_ref is required as a pointer, never as embedded software state")
        if "CURRENT_STATE" not in swref:
            errors.append("software_release_state_ref must point to a Decoder CURRENT_STATE authority")

        # Entity/ref identities are versioned scientific identities.
        make_ref(entity["entity_class_id"], entity["version"])
        make_ref(regime["regime_id"], regime["version"])
    except Exception as exc:
        if not errors:
            errors.append(f"{type(exc).__name__}: {exc}")

    result = {
        "schema_id": "IG_RESEARCH_FRONTIER_VERIFICATION_V1",
        "status": "PASS" if not errors else "FAIL",
        "frontier_id": frontier.get("frontier_id") if isinstance(frontier, Mapping) else None,
        "frontier_science_sha256": canonical_sha256(dict(frontier)) if isinstance(frontier, Mapping) else None,
        "errors": errors,
        "science_state_is_separate_from_software_state": True,
        "promotion_performed": False,
    }
    result["science_sha256"] = canonical_sha256(result)
    if errors and raise_on_error:
        raise ResearchFrontierError("ResearchFrontier verification failed: " + "; ".join(errors))
    return result


def current_frontier_status() -> dict[str, Any]:
    frontier = load_research_frontier()
    verification = verify_research_frontier(frontier)
    return {
        "schema_id": "IG_RESEARCH_FRONTIER_STATUS_V1",
        "status": verification["status"],
        "frontier": frontier,
        "frontier_science_sha256": canonical_sha256(frontier),
        "verification_science_sha256": verification["science_sha256"],
        "rule": "SCIENTIFIC_FRONTIER_IS_SEPARATE_FROM_SOFTWARE_RELEASE_STATE",
    }
