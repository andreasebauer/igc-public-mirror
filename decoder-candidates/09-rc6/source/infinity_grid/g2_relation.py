from __future__ import annotations

"""Relation-valued G2 external reservation semantics (v0.30.2).

This module is deliberately *G2-facing*.  It does not modify or replace the frozen
``reserve_external`` operation carried by G1 states.  Instead it lifts that already-earned
single-successor operation into a complete finite relation when a G2 composite may expose
an endpoint through more than one eligible child.

The scientific rule is:

* every eligible immediate child is a lawful owner choice;
* if the child is a frozen G1 object, call its existing deterministic ``reserve_external``;
* if the child is itself a G2 composite, recurse through the same G2 relation;
* retain every distinct exact implementation successor (construction identity is allowed only
  for operational deduplication, never as a public selector or semantic field);
* public observers must quotient/project the returned successor relation without reading the
  operational owner witness.

The S4-R0 audit selected this semantics after exhaustive comparison of all 3,604 nontrivial
S2 realization classes.  That audit is a bounded implementation authority, not an all-depth
G2 congruence theorem.
"""

from importlib.resources import files
from typing import Any
import json

from .canon import canonical_sha256
from . import regime_scanner as rs


class G2RelationError(RuntimeError):
    pass


_SPEC_RESOURCE = "resources/uplift/G2_RELATION_VALUED_RESERVATION_SPEC_V1.json"

# Execution-only exact-kernel caches.  They are process-local, never serialized, and
# never enter scientific payloads.  State identity is the frozen construction digest
# scoped to the concrete engine instance so equal digests from independent materializations
# can never return objects owned by the wrong engine.
_RESERVATION_RELATION_CACHE: dict[tuple[int, str, int], tuple[tuple[Any, Any], ...]] = {}
_PAIR_COMPOSITION_CACHE: dict[tuple[int, int, str, str, int, int, str, str], tuple[Any, ...]] = {}
_CACHE_STATS = {
    "reservation_hits": 0, "reservation_misses": 0,
    "pair_hits": 0, "pair_misses": 0,
}


def clear_exact_relation_caches() -> None:
    _RESERVATION_RELATION_CACHE.clear()
    _PAIR_COMPOSITION_CACHE.clear()
    for k in _CACHE_STATS:
        _CACHE_STATS[k] = 0


def exact_relation_cache_stats() -> dict[str, int]:
    return {**_CACHE_STATS,
            "reservation_entries": len(_RESERVATION_RELATION_CACHE),
            "pair_entries": len(_PAIR_COMPOSITION_CACHE)}


def _engine_scope_key(state: Any) -> int:
    return id(getattr(state, "engine", None))


def relation_spec() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(_SPEC_RESOURCE).read_text(encoding="utf-8"))
    expected = str(obj.get("spec_sha256", ""))
    payload = {k: v for k, v in obj.items() if k != "spec_sha256"}
    observed = canonical_sha256(payload)
    if expected != observed:
        raise G2RelationError(f"G2 relation-valued reservation spec hash mismatch: expected {expected}, observed {observed}")
    return obj


def public_state_payload(state: Any) -> dict[str, Any]:
    """The strictly public state read used by the S4 branch observer."""
    skin = str(state.skin)
    caps = [int(x) for x in state.total_caps]
    if len(caps) != 7 or any(x < 0 for x in caps):
        raise G2RelationError("G2 public state requires seven nonnegative resource counters")
    return {"skin": skin, "caps": caps}


def _exact_successor_key(state: Any) -> str:
    """Operational-only exact deduplication key.

    ``construction_digest`` is never used to *choose* a branch and is not returned in any
    public/scientific payload.  It is used only to avoid returning the same exact implementation
    successor more than once when equivalent recursive paths converge.
    """
    d = getattr(state, "construction_digest", None)
    if isinstance(d, str) and len(d) == 64:
        return d
    # Fallback is intentionally not a selector; it is only an exact-object dedupe surrogate.
    return f"obj:{id(state)}"


def is_g2_composite(state: Any) -> bool:
    return isinstance(state, rs.LiftState) and int(state.level) >= 101


def _reserve_external_relation_with_witness(state: Any, endpoint_type: int) -> tuple[tuple[Any, Any], ...]:
    """Internal complete successor relation retaining operational witnesses.

    Frozen G1 semantics are left untouched.  A non-G2 state therefore contributes exactly the
    successor returned by its existing ``reserve_external`` method.  A G2 ``LiftState`` branches
    over *all* eligible immediate children and recursively applies this same lifting if an
    eligible child is already a G2 composite.

    The result contains exact implementation states so later branch-sensitive observers may
    preserve distinct futures even when two immediate public states happen to coincide.
    """
    t = int(endpoint_type)
    if t < 0 or t >= 7:
        raise G2RelationError(f"endpoint type outside frozen seven-type alphabet: {t}")
    digest = str(getattr(state, "construction_digest", ""))
    cache_key = (_engine_scope_key(state), digest, t) if len(digest) == 64 else None
    if cache_key is not None and cache_key in _RESERVATION_RELATION_CACHE:
        _CACHE_STATS["reservation_hits"] += 1
        return _RESERVATION_RELATION_CACHE[cache_key]
    _CACHE_STATS["reservation_misses"] += 1
    caps = tuple(int(x) for x in state.total_caps)
    if len(caps) != 7:
        raise G2RelationError("state resource vector must contain seven counters")
    if caps[t] <= 0:
        result: tuple[tuple[Any, Any], ...] = tuple()
        if cache_key is not None:
            _RESERVATION_RELATION_CACHE[cache_key] = result
        return result

    if not is_g2_composite(state):
        try:
            succ, _hidden_witness_discarded = state.reserve_external(t)
        except Exception as exc:
            raise G2RelationError(f"frozen lower-G reserve_external({t}) failed despite positive capacity") from exc
        result = ((succ, _hidden_witness_discarded),)
        if cache_key is not None:
            _RESERVATION_RELATION_CACHE[cache_key] = result
        return result

    successors: list[tuple[Any, Any]] = []
    for i, child in enumerate(state.children):
        child_caps = tuple(int(x) for x in child.total_caps)
        if len(child_caps) != 7 or child_caps[t] <= 0:
            continue
        child_successors = _reserve_external_relation_with_witness(child, t)
        for child_succ, child_witness in child_successors:
            ch = list(state.children)
            ch[i] = child_succ
            successors.append((rs.LiftState(state.engine, state.level, tuple(ch), state.top_edges_full, state.lane, state.motif_id), (i, child_witness, t)))

    if not successors:
        raise G2RelationError(f"G2 relation found no eligible owner for positive type-{t} capacity")

    # Set-valued semantics: exact duplicate branches are removed, but no provenance-based
    # preference is ever imposed between distinct successors.
    uniq: dict[str, tuple[Any, Any]] = {}
    for s, witness in successors:
        uniq.setdefault(_exact_successor_key(s), (s, witness))
    result = tuple(uniq[k] for k in sorted(uniq))
    if cache_key is not None:
        _RESERVATION_RELATION_CACHE[cache_key] = result
    return result


def reserve_external_relation(state: Any, endpoint_type: int) -> tuple[Any, ...]:
    """Return the complete finite G2 successor relation without exposing witnesses."""
    return tuple(s for s, _w in _reserve_external_relation_with_witness(state, endpoint_type))


def compose_binary_relation(
    engine: Any, level: int, left: Any, right: Any, endpoint_type_left: int, endpoint_type_right: int,
    *, lane: str = "G2_RELATION", motif_id: str = "G2:RELATION:BINARY"
) -> tuple[Any, ...]:
    """Compose two complete units by one typed bridge under relation-valued G2 routing.

    This is the relation-valued counterpart of the one-edge ``_build_lift`` operation used by
    implementation-level S4.  Every lawful reservation branch on either complete input unit is
    retained.  Lower-G children still use their frozen deterministic reservation operation.
    """
    a, b = int(endpoint_type_left), int(endpoint_type_right)
    if a < 0 or a >= 7 or b < 0 or b >= 7:
        raise G2RelationError("bridge endpoint type outside frozen seven-type alphabet")
    if int(left.total_caps[a]) <= 0 or int(right.total_caps[b]) <= 0:
        return tuple()
    ld, rd = str(getattr(left, "construction_digest", "")), str(getattr(right, "construction_digest", ""))
    pair_key = None
    if len(ld) == 64 and len(rd) == 64:
        pair_key = (id(engine), int(level), ld, rd, a, b, str(lane), str(motif_id))
        if pair_key in _PAIR_COMPOSITION_CACHE:
            _CACHE_STATS["pair_hits"] += 1
            return _PAIR_COMPOSITION_CACHE[pair_key]
    _CACHE_STATS["pair_misses"] += 1
    lrel = _reserve_external_relation_with_witness(left, a)
    rrel = _reserve_external_relation_with_witness(right, b)
    out: dict[str, Any] = {}
    for ls, lw in lrel:
        for rsucc, rw in rrel:
            edge = (0, 1, a, b, lw, rw)
            st = rs.LiftState(engine, int(level), (ls, rsucc), (edge,), str(lane), str(motif_id))
            out.setdefault(_exact_successor_key(st), st)
    result = tuple(out[k] for k in sorted(out))
    if pair_key is not None:
        _PAIR_COMPOSITION_CACHE[pair_key] = result
    return result


def public_successor_relation(state: Any, endpoint_type: int) -> list[dict[str, Any]]:
    """Canonical public projection of one G2 reservation relation."""
    payloads = {json.dumps(public_state_payload(s), sort_keys=True, separators=(",", ":")) for s in reserve_external_relation(state, endpoint_type)}
    return [json.loads(x) for x in sorted(payloads)]


def post_reservation_public_branch_relation_projection(state: Any, first_reserved_type: int) -> dict[str, Any]:
    """Branch-sensitive depth-2 public continuation used by implementation-level S4.

    This is the implemented counterpart of S4-R0 option C.  For each lawful first owner branch,
    it preserves that branch's immediate public successor and its complete one-more-reservation
    public successor relation.  Hidden owner/path witnesses and construction digests are absent
    from the semantic payload.
    """
    t = int(first_reserved_type)
    firsts = reserve_external_relation(state, t)
    if not firsts:
        raise G2RelationError("S4 D4 projection requires an available first G2 reservation")

    branches: set[str] = set()
    for first in firsts:
        next_relation = []
        for u in range(7):
            succs = public_successor_relation(first, u)
            next_relation.append({"endpoint_type": u, "successors": succs})
        branch = {"first_state": public_state_payload(first), "next_relation": next_relation}
        branches.add(json.dumps(branch, sort_keys=True, separators=(",", ":")))

    semantic = {
        "schema_id": "IG_G2_POST_RESERVATION_PUBLIC_BRANCH_RELATION_V1",
        "first_reserved_type": t,
        "branches": [json.loads(x) for x in sorted(branches)],
        "routing_semantics": "ALL_ELIGIBLE_OWNER_RELATION_WITH_FROZEN_LOWER_G_RESERVE",
        "hidden_owner_witnesses_retained": False,
        "construction_identity_used_as_selector": False,
        "authority": relation_spec()["s4_r0_authority"],
    }
    semantic["science_sha256"] = canonical_sha256(semantic)
    return semantic
