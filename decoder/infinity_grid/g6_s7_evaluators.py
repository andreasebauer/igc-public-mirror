from __future__ import annotations

"""Controller-only evaluators for G6:S7 ordinary branch-sensitive quotient work.

S7 is pre-R and marker/reconstruction free. Exact ordinary relations are used only
for successor branching. Depth-1 evaluation uses an exact relation-profile fast
path: the public successor block is fixed by the inherited public write law, so
only the exact multiplicity of distinct alternatives must be computed. This is an
execution optimization; the returned full depth-1 signature is byte-equivalent to
materializing every exact child and PUBLIC_READing it.
"""

from collections import OrderedDict
import json
from typing import Any, Mapping

from .canon import canonical_text
from .v05_kernel_services import current_kernel_view


_BASIS_RECORD_CACHE: dict[str, Mapping[str, Any]] | None = None
_BASIS_PUBLIC_CACHE: dict[str, tuple[Any, ...]] | None = None


def _public_signature(read: Mapping[str, Any]) -> tuple[Any, ...]:
    if read.get("legal") is not True or read.get("descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        return ("ILLEGAL",)
    return (
        "D",
        tuple(int(x) for x in read.get("caps7") or ()),
        tuple(sorted((str(k), int(v)) for k, v in (read.get("H_class_bag") or {}).items())),
    )


def _multiset_signature(values: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    """Exact structural bag encoding; text bytes are equality keys, never hashes."""
    rows: dict[str, list[Any]] = {}
    for value in values:
        key = canonical_text(value, pretty=False)
        row = rows.get(key)
        if row is None:
            rows[key] = [value, 1]
        else:
            if canonical_text(row[0], pretty=False) != key:
                raise ValueError("S7_MULTISET_STRUCTURAL_KEY_MISMATCH")
            row[1] = int(row[1]) + 1
    return tuple((rows[k][0], int(rows[k][1])) for k in sorted(rows))


def _compose_public_signature(left: tuple[Any, ...], right: tuple[Any, ...], op: tuple[int, int]) -> tuple[Any, ...]:
    """Frozen inherited G5 public write law on already-validated public signatures."""
    if not left or not right or left[0] != "D" or right[0] != "D":
        return ("ILLEGAL",)
    lc, rc = tuple(left[1]), tuple(right[1])
    if len(lc) != 7 or len(rc) != 7:
        raise ValueError("S7_PUBLIC_CAPS7_ARITY")
    a, b = int(op[0]), int(op[1])
    caps = [int(lc[i]) + int(rc[i]) for i in range(7)]
    caps[a] -= 1; caps[b] -= 1
    if any(x < 0 for x in caps):
        raise ValueError("S7_PUBLIC_WRITE_NEGATIVE_CAPACITY")
    bag: dict[str, int] = {}
    for k, v in tuple(left[2]) + tuple(right[2]):
        bag[str(k)] = bag.get(str(k), 0) + int(v)
    return ("D", tuple(caps), tuple(sorted((k, int(v)) for k, v in bag.items() if int(v))))


def _basis_records_snapshot(view, basis_refs: tuple[str, ...]):
    """Load/cache only exact frozen basis records; requires no PUBLIC_READ."""
    global _BASIS_RECORD_CACHE
    if _BASIS_RECORD_CACHE is None:
        basis = view.call("AUTHORITY_BASIS")
        if set(basis) != set(basis_refs):
            raise ValueError("S7_AUTHORITY_BASIS_MISMATCH")
        _BASIS_RECORD_CACHE = {str(k): basis[str(k)] for k in basis_refs}
    return _BASIS_RECORD_CACHE


def _basis_snapshot(view, basis_refs: tuple[str, ...]):
    global _BASIS_PUBLIC_CACHE
    basis = _basis_records_snapshot(view, basis_refs)
    if _BASIS_PUBLIC_CACHE is None:
        _BASIS_PUBLIC_CACHE = {
            str(k): _public_signature(view.call("PUBLIC_READ", basis[str(k)])) for k in basis_refs
        }
    return basis, _BASIS_PUBLIC_CACHE


def s7_public_state_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    view = current_kernel_view()
    sig = _public_signature(view.call("PUBLIC_READ", payload["state_tree"]))
    return {"signature": sig, "outcome_count": 1, "metrics": {"public_read_count": 1}}


def s7_ordinary_branch_count_component_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One frozen batch of depth-1 observer components for monotone refinement.

    The inherited public partition is computed separately. Within one inherited
    public class, the successor public block for each frozen action is deterministic,
    so the vector of exact branch multiplicities is an exact component batch of the
    unchanged 248-action observer. Batching is execution-only.
    """
    view = current_kernel_view()
    refs = tuple(str(x) for x in payload.get("basis_refs") or ())
    if tuple(sorted(refs)) != refs or len(refs) != 4:
        raise ValueError("S7_BASIS_REFS")
    basis = _basis_records_snapshot(view, refs)
    contexts = tuple(payload.get("contexts") or ())
    if not 1 <= len(contexts) <= 8:
        raise ValueError("S7_COMPONENT_BATCH_SIZE")
    state = payload["state_tree"]
    counts=[]; metrics={
        "exact_relation_profile_call_count":0, "attempted_owner_pairs":0,
        "legal_owner_pairs":0, "rooted_owner_pair_candidates":0,
        "child_canon_constructions":0,
    }
    for row in contexts:
        ref = str(row["basis_ref"])
        if ref not in basis:
            raise ValueError("S7_COMPONENT_BASIS_REF")
        op = tuple(int(x) for x in row["operator"])
        pos = str(row["position"])
        if pos not in {"LEFT", "RIGHT"}:
            raise ValueError("S7_COMPONENT_POSITION")
        seed=basis[ref]; left,right=(state,seed) if pos=="LEFT" else (seed,state)
        prof=view.call("EXACT_RELATION_PROFILE",left,right,op)
        counts.append(int(prof["exact_outcome_count"]))
        metrics["exact_relation_profile_call_count"] += 1
        m=prof.get("metrics") or {}
        for key in ("attempted_owner_pairs","legal_owner_pairs","rooted_owner_pair_candidates","child_canon_constructions"):
            metrics[key] += int(m.get(key,0))
    return {
        "signature": ("S7_D1_BRANCH_MULTIPLICITY_COMPONENT_BATCH", tuple(counts)),
        "outcome_count": sum(counts),
        "metrics": metrics,
    }


def s7_ordinary_branch_relation_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Complete depth-1/depth-2 branch signature under the frozen ordinary basis.

    depth=1 is byte-equivalent to the original materializing evaluator but uses
    exact relation profiles plus the inherited public write law, avoiding child
    carrier construction and per-child PUBLIC_READ.

    depth=2 materializes exact first-step children because their depth-1 behavior
    blocks are genuinely required, but every child depth-1 signature uses the fast
    profile path.
    """
    view = current_kernel_view()
    state = payload["state_tree"]
    depth = int(payload.get("future_depth", 1))
    if depth not in (1, 2):
        raise ValueError("S7_FUTURE_DEPTH")

    basis_refs = tuple(str(x) for x in payload.get("basis_refs") or ())
    operators = tuple(tuple(int(y) for y in x) for x in (payload.get("operator_basis") or ()))
    if tuple(sorted(basis_refs)) != basis_refs or len(basis_refs) != 4:
        raise ValueError("S7_BASIS_REFS")
    if len(operators) != 31 or len(set(operators)) != 31:
        raise ValueError("S7_OPERATOR_BASIS")
    basis, basis_public = _basis_snapshot(view, basis_refs)

    metrics = {
        "public_read_count": 0,
        "ordinary_relation_call_count": 0,
        "ordinary_relation_profile_call_count": 0,
        "ordinary_total_distinct_child_count": 0,
        "ordinary_zero_context_count": 0,
        "ordinary_max_distinct_child_count": 0,
        "attempted_owner_pairs": 0,
        "legal_owner_pairs": 0,
        "rooted_owner_pair_candidates": 0,
        "child_canon_constructions": 0,
        "recursive_state_cache_hits": 0,
        "recursive_state_cache_misses": 0,
        "depth1_child_materializations_avoided": 0,
    }
    state_cache: dict[tuple[int, str], tuple[Any, ...]] = {}
    pub_cache: dict[str, tuple[Any, ...]] = {}
    full_step_cache: dict[str, tuple[tuple[Any, ...], tuple[tuple[Mapping[str, Any], ...], ...]]] = {}

    def add_rel_metrics(m: Mapping[str, Any]) -> None:
        for key in ("attempted_owner_pairs", "legal_owner_pairs", "rooted_owner_pair_candidates", "child_canon_constructions"):
            metrics[key] += int(m.get(key, 0))

    def public_sig(tree_record: Mapping[str, Any]) -> tuple[Any, ...]:
        skey = canonical_text(tree_record, pretty=False)
        cached = pub_cache.get(skey)
        if cached is not None:
            metrics["recursive_state_cache_hits"] += 1
            return cached
        pub = _public_signature(view.call("PUBLIC_READ", tree_record))
        metrics["public_read_count"] += 1
        if pub == ("ILLEGAL",):
            raise ValueError("S7_PARENT_PUBLIC_READ_ILLEGAL")
        pub_cache[skey] = pub
        return pub

    def fast_depth1(tree_record: Mapping[str, Any]) -> tuple[Any, ...]:
        key = (1, canonical_text(tree_record, pretty=False))
        cached = state_cache.get(key)
        if cached is not None:
            metrics["recursive_state_cache_hits"] += 1
            return cached
        metrics["recursive_state_cache_misses"] += 1
        root_pub = public_sig(tree_record)
        action_blocks = []
        for ref in basis_refs:
            seed = basis[ref]; seed_pub = basis_public[ref]
            for op in operators:
                for pos in ("LEFT", "RIGHT"):
                    left, right = (tree_record, seed) if pos == "LEFT" else (seed, tree_record)
                    lpub, rpub = (root_pub, seed_pub) if pos == "LEFT" else (seed_pub, root_pub)
                    prof = view.call("EXACT_RELATION_PROFILE", left, right, op)
                    n = int(prof["exact_outcome_count"])
                    metrics["ordinary_relation_profile_call_count"] += 1
                    metrics["ordinary_total_distinct_child_count"] += n
                    metrics["ordinary_zero_context_count"] += int(n == 0)
                    metrics["ordinary_max_distinct_child_count"] = max(int(metrics["ordinary_max_distinct_child_count"]), n)
                    metrics["depth1_child_materializations_avoided"] += n
                    add_rel_metrics(prof.get("metrics") or {})
                    if n == 0:
                        action_blocks.append(())
                    else:
                        action_blocks.append(((_compose_public_signature(lpub, rpub, op), n),))
        if len(action_blocks) != 248:
            raise ValueError(f"S7_CONTEXT_COUNT:{len(action_blocks)}")
        out = ("ORDINARY_BRANCH_MULTISET", 1, root_pub, tuple(action_blocks))
        state_cache[key] = out
        return out

    def one_step_full(tree_record: Mapping[str, Any]):
        skey = canonical_text(tree_record, pretty=False)
        cached = full_step_cache.get(skey)
        if cached is not None:
            metrics["recursive_state_cache_hits"] += 1
            return cached
        root_pub = public_sig(tree_record)
        action_children: list[tuple[Mapping[str, Any], ...]] = []
        for ref in basis_refs:
            seed = basis[ref]
            for op in operators:
                for pos in ("LEFT", "RIGHT"):
                    left, right = (tree_record, seed) if pos == "LEFT" else (seed, tree_record)
                    rel = view.call("EXACT_RELATION", left, right, op)
                    children = tuple(rel["children"])
                    action_children.append(children)
                    n = len(children)
                    metrics["ordinary_relation_call_count"] += 1
                    metrics["ordinary_total_distinct_child_count"] += n
                    metrics["ordinary_zero_context_count"] += int(n == 0)
                    metrics["ordinary_max_distinct_child_count"] = max(int(metrics["ordinary_max_distinct_child_count"]), n)
                    add_rel_metrics(rel.get("metrics") or {})
        if len(action_children) != 248:
            raise ValueError(f"S7_CONTEXT_COUNT:{len(action_children)}")
        out = (root_pub, tuple(action_children)); full_step_cache[skey] = out
        return out

    def signature(tree_record: Mapping[str, Any], level: int) -> tuple[Any, ...]:
        if int(level) == 1:
            return fast_depth1(tree_record)
        key = (int(level), canonical_text(tree_record, pretty=False))
        cached = state_cache.get(key)
        if cached is not None:
            metrics["recursive_state_cache_hits"] += 1
            return cached
        metrics["recursive_state_cache_misses"] += 1
        previous = fast_depth1(tree_record)
        _pub, action_children = one_step_full(tree_record)
        action_blocks = []
        for children in action_children:
            vals = [fast_depth1(child) for child in children]
            action_blocks.append(_multiset_signature(vals))
        out = ("ORDINARY_BRANCH_MULTISET", int(level), previous, tuple(action_blocks))
        state_cache[key] = out
        return out

    sig = signature(state, depth)
    return {
        "signature": sig,
        "outcome_count": int(metrics["ordinary_total_distinct_child_count"]),
        "metrics": dict(metrics, future_depth=depth, ordinary_context_count=248),
    }


# G6:S7 depth-2 completion execution-only child-prefix cache.  The key contains
# the complete exact child service record text plus the frozen observer basis.
# Exact text equality, never a digest, decides cache reuse.  Entries are bounded
# and worker-local; eviction changes performance only.
_D2_CHILD_PREFIX_CACHE: "OrderedDict[tuple[Any, ...], tuple[int, ...]]" = OrderedDict()
# Parent-prefix execution has negligible cross-parent exact-child overlap and the
# prior 2048-entry worker cache measured zero hits while retaining tens of MiB.
# Keep the symbol for compatibility/telemetry, but disable storage.
_D2_CHILD_PREFIX_CACHE_MAX = 0


def _s7_frozen_contexts(basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...]):
    return tuple((ref, op, pos) for ref in basis_refs for op in operators for pos in ("LEFT", "RIGHT"))


def _s7_factor_swap_representative_contexts(
    basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...]
):
    """Lossless representatives of the 248 frozen contexts.

    Exact factor swap with endpoint-label reversal gives
      (seed,(a,b),RIGHT) == (seed,(b,a),LEFT).
    The frozen operator basis is transpose-closed, so normalizing the variable
    state to LEFT leaves exactly 4*31 = 124 representatives.
    """
    op_set = set(operators)
    if any((b, a) not in op_set for a, b in operators):
        raise ValueError("S7_FACTOR_SWAP_OPERATOR_TRANSPOSE_CLOSURE")
    contexts = tuple((ref, op, "LEFT") for ref in basis_refs for op in operators)
    if len(contexts) != 124:
        raise ValueError("S7_FACTOR_SWAP_CONTEXT_COUNT")
    return contexts


def _d2_child_count_prefix(view, child: Mapping[str, Any], *, basis: Mapping[str, Mapping[str, Any]],
                           basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...],
                           prefix_count: int, metrics: dict[str, Any],
                           execution_context_normalization: str | None = None) -> tuple[int, ...]:
    if execution_context_normalization is None:
        contexts = _s7_frozen_contexts(basis_refs, operators)
        allowed_prefixes = (1, 8, 32, 128, 248)
        expected_context_count = 248
    elif execution_context_normalization == "FACTOR_SWAP_LEFT_V1":
        contexts = _s7_factor_swap_representative_contexts(basis_refs, operators)
        allowed_prefixes = (1, 8, 32, 64, 124)
        expected_context_count = 124
    else:
        raise ValueError("S7_D2_EXECUTION_CONTEXT_NORMALIZATION")
    if len(contexts) != expected_context_count or prefix_count not in allowed_prefixes:
        raise ValueError("S7_D2_INNER_PREFIX")
    # The historical worker-local child-prefix cache is disabled for the current
    # parent-prefix execution path.  Cross-parent exact-child overlap is negligible
    # and the accepted audit measured zero hits, so retaining canonical child text
    # only increases anonymous memory.  If a future registered execution explicitly
    # re-enables the cache, exact canonical child text remains the equality key.
    cache_enabled = _D2_CHILD_PREFIX_CACHE_MAX > 0
    key = None
    cached = None
    if cache_enabled:
        key = (execution_context_normalization, basis_refs, operators, canonical_text(child, pretty=False))
        cached = _D2_CHILD_PREFIX_CACHE.get(key)
        if cached is not None and len(cached) >= prefix_count:
            _D2_CHILD_PREFIX_CACHE.move_to_end(key)
            metrics["child_prefix_cache_hits"] += 1
            return tuple(cached[:prefix_count])
    counts = list(cached or ())
    start = len(counts)
    metric_keys = ("attempted_owner_pairs", "legal_owner_pairs", "rooted_owner_pair_candidates", "child_canon_constructions",
                   "relation_profile_zero_fast_hits", "relation_profile_bridge_fingerprint_fast_hits",
                   "relation_profile_bridge_fingerprint_fallbacks", "relation_profile_bridge_fingerprint_canon_avoided",
                   "relation_profile_batch_calls", "relation_profile_batch_operator_count",
                   "relation_profile_batch_bridge_index_builds",
                   "relation_profile_family_calls", "relation_profile_family_right_count",
                   "relation_profile_family_operator_count", "relation_profile_family_profile_count",
                   "relation_profile_family_general_fallbacks", "relation_profile_family_old_bridge_index_builds",
                   "relation_profile_family_certified_unique_candidates",
                   "relation_profile_family_ambiguous_candidates",
                   "relation_profile_family_exact_graft_signatures",
                   "relation_profile_family_path_updates", "relation_profile_family_full_canon_avoided")
    # A25's active full-P124 schedule is exactly four contiguous LEFT blocks of
    # 31 operators, one per frozen seed. A30 evaluates that complete registered
    # family in one exact compact transform. Partial/legacy prefixes retain the
    # already-certified scalar path.
    if (start == 0 and execution_context_normalization == "FACTOR_SWAP_LEFT_V1" and prefix_count == 124):
        family = view.call(
            "EXACT_RELATION_PROFILE_FAMILY", child,
            tuple(basis[ref] for ref in basis_refs), operators,
        )
        vals = tuple(int(x) for x in family.get("exact_outcome_counts") or ())
        if len(vals) != 124:
            raise ValueError("S7_D2_FAMILY_PROFILE_LENGTH")
        counts.extend(vals)
        metrics["exact_relation_profile_family_call_count"] = int(metrics.get("exact_relation_profile_family_call_count", 0)) + 1
        # Preserve the historical logical-profile counter: science still has
        # 124 inner profile coordinates although execution uses one family call.
        metrics["exact_relation_profile_call_count"] += len(vals)
        m = family.get("metrics") or {}
        for mk in metric_keys:
            metrics[mk] = int(metrics.get(mk, 0)) + int(m.get(mk, 0))
    else:
        for ref, op, pos in contexts[start:prefix_count]:
            seed = basis[ref]
            left, right = (child, seed) if pos == "LEFT" else (seed, child)
            prof = view.call("EXACT_RELATION_PROFILE", left, right, op)
            counts.append(int(prof["exact_outcome_count"]))
            metrics["exact_relation_profile_call_count"] += 1
            m = prof.get("metrics") or {}
            for mk in metric_keys:
                # Backward-compatible with registered callers that provide the historical
                # four-key metric accumulator. New operational counters are additive only.
                metrics[mk] = int(metrics.get(mk, 0)) + int(m.get(mk, 0))
    if cache_enabled:
        assert key is not None
        _D2_CHILD_PREFIX_CACHE[key] = tuple(counts)
        _D2_CHILD_PREFIX_CACHE.move_to_end(key)
        while len(_D2_CHILD_PREFIX_CACHE) > _D2_CHILD_PREFIX_CACHE_MAX:
            _D2_CHILD_PREFIX_CACHE.popitem(last=False)
            metrics["child_prefix_cache_evictions"] += 1
    return tuple(counts[:prefix_count])


def s7_depth2_outer_prefix_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Exact monotone component of the frozen depth-2 S7 observer.

    The input parent is already inside one certified A6 depth-1 class.  For one
    frozen outer action, all exact children therefore share the same inherited
    public D across candidate parents.  Comparing the bag of child Sig1 blocks is
    consequently equivalent to comparing the bag of their complete 248 exact
    branch-count vectors.  This evaluator returns a cumulative exact prefix of
    those vectors.  Exact child identity is used only as a worker-local cache key
    and never enters the scientific signature.
    """
    view = current_kernel_view()
    refs = tuple(str(x) for x in payload.get("basis_refs") or ())
    operators = tuple(tuple(int(y) for y in x) for x in (payload.get("operator_basis") or ()))
    if tuple(sorted(refs)) != refs or len(refs) != 4:
        raise ValueError("S7_D2_BASIS_REFS")
    if len(operators) != 31 or len(set(operators)) != 31:
        raise ValueError("S7_D2_OPERATOR_BASIS")
    execution_context_normalization = payload.get("execution_context_normalization")
    if execution_context_normalization is None:
        contexts = _s7_frozen_contexts(refs, operators)
        allowed_prefixes = (1, 8, 32, 128, 248)
        expected_context_count = 248
        signature_tag = "S7_D2_OUTER_SUCCESSOR_SIG1_PREFIX_MULTISET"
    elif execution_context_normalization == "FACTOR_SWAP_LEFT_V1":
        contexts = _s7_factor_swap_representative_contexts(refs, operators)
        allowed_prefixes = (1, 8, 32, 64, 124)
        expected_context_count = 124
        signature_tag = "S7_D2_OUTER_SUCCESSOR_SIG1_FACTOR_SWAP_REP_PREFIX_MULTISET"
    else:
        raise ValueError("S7_D2_EXECUTION_CONTEXT_NORMALIZATION")
    if len(contexts) != expected_context_count:
        raise ValueError("S7_D2_CONTEXT_COUNT")
    outer_index = int(payload.get("outer_context_index", -1))
    prefix_count = int(payload.get("inner_prefix_context_count", -1))
    if not 0 <= outer_index < expected_context_count or prefix_count not in allowed_prefixes:
        raise ValueError("S7_D2_COMPONENT_INDEX")
    basis = _basis_records_snapshot(view, refs)
    state = payload["state_tree"]
    ref, op, pos = contexts[outer_index]
    seed = basis[ref]
    left, right = (state, seed) if pos == "LEFT" else (seed, state)
    rel = view.call("EXACT_RELATION", left, right, op)
    children = tuple(rel["children"])
    metrics = {
        "exact_relation_call_count": 1,
        "exact_relation_profile_call_count": 0,
        "exact_relation_profile_batch_call_count": 0,
        "exact_relation_profile_family_call_count": 0,
        "outer_distinct_child_count": len(children),
        "child_prefix_cache_hits": 0,
        "child_prefix_cache_evictions": 0,
        "attempted_owner_pairs": 0,
        "legal_owner_pairs": 0,
        "rooted_owner_pair_candidates": 0,
        "child_canon_constructions": 0,
        "relation_profile_zero_fast_hits": 0,
        "relation_profile_bridge_fingerprint_fast_hits": 0,
        "relation_profile_bridge_fingerprint_fallbacks": 0,
        "relation_profile_bridge_fingerprint_canon_avoided": 0,
    }
    rm = rel.get("metrics") or {}
    for mk in ("attempted_owner_pairs", "legal_owner_pairs", "rooted_owner_pair_candidates", "child_canon_constructions",
               "relation_profile_zero_fast_hits", "relation_profile_bridge_fingerprint_fast_hits",
               "relation_profile_bridge_fingerprint_fallbacks", "relation_profile_bridge_fingerprint_canon_avoided"):
        metrics[mk] += int(rm.get(mk, 0))
    child_prefixes = []
    for child in children:
        counts = _d2_child_count_prefix(
            view, child, basis=basis, basis_refs=refs, operators=operators,
            prefix_count=prefix_count, metrics=metrics,
            execution_context_normalization=execution_context_normalization,
        )
        child_prefixes.append(("S7_CHILD_D1_BRANCH_COUNT_PREFIX", counts))
    return {
        "signature": (
            signature_tag,
            int(prefix_count),
            _multiset_signature(child_prefixes),
        ),
        "outcome_count": len(children),
        "metrics": dict(metrics, outer_context_index=outer_index, inner_prefix_context_count=prefix_count,
                        scientific_context_count=248, execution_context_count=expected_context_count,
                        execution_context_normalization=execution_context_normalization),
    }


def s7_depth2_outer_children_generation_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize one parent's exact children for one frozen outer S7 action.

    This is the generation half of the global-dedup S7D2 execution.  The
    StageScienceRuntime content-indexed generation store performs exact
    structural deduplication across all parent tasks.  Exact child identity is
    execution-only and never enters a scientific observer signature.
    """
    view = current_kernel_view()
    refs = tuple(str(x) for x in payload.get("basis_refs") or ())
    operators = tuple(tuple(int(y) for y in x) for x in (payload.get("operator_basis") or ()))
    if tuple(sorted(refs)) != refs or len(refs) != 4:
        raise ValueError("S7_D2_GEN_BASIS_REFS")
    if len(operators) != 31 or len(set(operators)) != 31:
        raise ValueError("S7_D2_GEN_OPERATOR_BASIS")
    contexts = _s7_frozen_contexts(refs, operators)
    outer_index = int(payload.get("outer_context_index", -1))
    if len(contexts) != 248 or not 0 <= outer_index < 248:
        raise ValueError("S7_D2_GEN_OUTER_INDEX")
    basis = _basis_records_snapshot(view, refs)
    state = payload["state_tree"]
    ref, op, pos = contexts[outer_index]
    seed = basis[ref]
    left, right = (state, seed) if pos == "LEFT" else (seed, state)
    rel = view.call("EXACT_RELATION", left, right, op)
    canons = tuple(rel.get("canons") or ())
    children = tuple(rel.get("children") or ())
    if len(canons) != len(children):
        raise ValueError("S7_D2_GEN_CANON_CHILD_ALIGNMENT")
    metrics = {
        "exact_relation_call_count": 1,
        "outer_distinct_child_count": len(children),
        "attempted_owner_pairs": 0,
        "legal_owner_pairs": 0,
        "rooted_owner_pair_candidates": 0,
        "child_canon_constructions": 0,
    }
    rm = rel.get("metrics") or {}
    for mk in ("attempted_owner_pairs", "legal_owner_pairs", "rooted_owner_pair_candidates", "child_canon_constructions"):
        metrics[mk] += int(rm.get(mk, 0))
    return {
        "states": [
            {"identity": canon, "state": {"tree": child}}
            for canon, child in zip(canons, children)
        ],
        "metrics": dict(metrics, outer_context_index=outer_index),
    }


def s7_depth2_child_profile_generation_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extend one globally deduplicated exact child's frozen depth-1 count profile.

    Compact controller payloads carry canonical JSON text and are decoded only
    inside the executing worker. Legacy expanded fields remain accepted for
    reference-equivalence tests. Exact child identity remains the generation-store
    structural authority; child_token is execution-only join metadata.
    """
    view = current_kernel_view()
    refs = tuple(str(x) for x in payload.get("basis_refs") or ())
    operators = tuple(tuple(int(y) for y in x) for x in (payload.get("operator_basis") or ()))
    if tuple(sorted(refs)) != refs or len(refs) != 4:
        raise ValueError("S7_D2_PROFILE_BASIS_REFS")
    if len(operators) != 31 or len(set(operators)) != 31:
        raise ValueError("S7_D2_PROFILE_OPERATOR_BASIS")
    contexts = _s7_frozen_contexts(refs, operators)
    start = int(payload.get("start_context_index", -1))
    end = int(payload.get("end_context_index", -1))
    previous = tuple(int(x) for x in (payload.get("previous_counts") or ()))
    if len(contexts) != 248 or start not in (0, 8, 32, 128) or end not in (8, 32, 128, 248) or start >= end:
        raise ValueError("S7_D2_PROFILE_RANGE")
    if len(previous) != start:
        raise ValueError("S7_D2_PROFILE_PREVIOUS_PREFIX")

    identity = payload.get("child_identity")
    if identity is None:
        identity_json = payload.get("child_identity_json")
        if type(identity_json) is not str:
            raise ValueError("S7_D2_PROFILE_IDENTITY_JSON")
        identity = json.loads(identity_json)
    if identity is None:
        raise ValueError("S7_D2_PROFILE_IDENTITY")

    child = payload.get("child_tree")
    if child is None:
        state_json = payload.get("child_state_json")
        if type(state_json) is not str:
            raise ValueError("S7_D2_PROFILE_STATE_JSON")
        state = json.loads(state_json)
        child = state.get("tree") if type(state) is dict else None
    if type(child) is not dict:
        raise ValueError("S7_D2_PROFILE_CHILD_TREE")

    child_token_raw = payload.get("child_token")
    child_token = None if child_token_raw is None else str(child_token_raw)
    if child_token is not None and not child_token:
        raise ValueError("S7_D2_PROFILE_CHILD_TOKEN")

    basis = _basis_records_snapshot(view, refs)
    counts = list(previous)
    metrics = {
        "exact_relation_profile_call_count": 0,
        "attempted_owner_pairs": 0,
        "legal_owner_pairs": 0,
        "rooted_owner_pair_candidates": 0,
        "child_canon_constructions": 0,
        "relation_profile_zero_fast_hits": 0,
        "relation_profile_bridge_fingerprint_fast_hits": 0,
        "relation_profile_bridge_fingerprint_fallbacks": 0,
        "relation_profile_bridge_fingerprint_canon_avoided": 0,
        "profile_contexts_reused": int(start),
        "profile_contexts_computed": int(end - start),
        "compact_payload": bool("child_state_json" in payload),
    }
    for ref, op, pos in contexts[start:end]:
        seed = basis[ref]
        left, right = (child, seed) if pos == "LEFT" else (seed, child)
        prof = view.call("EXACT_RELATION_PROFILE", left, right, op)
        counts.append(int(prof["exact_outcome_count"]))
        metrics["exact_relation_profile_call_count"] += 1
        pm = prof.get("metrics") or {}
        for mk in ("attempted_owner_pairs", "legal_owner_pairs", "rooted_owner_pair_candidates", "child_canon_constructions",
                   "relation_profile_zero_fast_hits", "relation_profile_bridge_fingerprint_fast_hits",
                   "relation_profile_bridge_fingerprint_fallbacks", "relation_profile_bridge_fingerprint_canon_avoided"):
            metrics[mk] += int(pm.get(mk, 0))
    if len(counts) != end:
        raise ValueError("S7_D2_PROFILE_LENGTH")
    state_payload = {"profile_counts": tuple(counts)}
    if child_token is not None:
        state_payload["child_token"] = child_token
    return {
        "states": [{
            "identity": identity,
            "state": state_payload,
        }],
        "metrics": dict(metrics, start_context_index=start, end_context_index=end),
    }

def s7_depth2_parent_profile_multiset_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the reference S7D2 parent signature from exact child profiles.

    The compact path receives one deterministic JSON string rather than a large
    nested Python list retained in the controller. Legacy expanded input remains
    accepted. The evaluator is kernel-free and emits the identical exact multiset
    signature.
    """
    prefix_count = int(payload.get("inner_prefix_context_count", -1))
    if prefix_count not in (8, 32, 128, 248):
        raise ValueError("S7_D2_ASSEMBLY_PREFIX")
    raw = payload.get("child_profile_counts")
    if raw is None:
        compact = payload.get("child_profile_counts_json")
        if type(compact) is not str:
            raise ValueError("S7_D2_ASSEMBLY_PROFILE_JSON")
        raw = json.loads(compact)
    raw = tuple(raw or ())
    values = []
    for row in raw:
        counts = tuple(int(x) for x in row)
        if len(counts) != prefix_count:
            raise ValueError("S7_D2_ASSEMBLY_CHILD_PROFILE_LENGTH")
        values.append(("S7_CHILD_D1_BRANCH_COUNT_PREFIX", counts))
    return {
        "signature": (
            "S7_D2_OUTER_SUCCESSOR_SIG1_PREFIX_MULTISET",
            prefix_count,
            _multiset_signature(values),
        ),
        "outcome_count": len(values),
        "metrics": {
            "assembled_child_profile_count": len(values),
            "kernel_calls": 0,
            "compact_payload": bool("child_profile_counts_json" in payload),
        },
    }
