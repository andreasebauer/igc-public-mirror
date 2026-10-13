from __future__ import annotations

"""Controller-only evaluators for G6:S8 deeper ordinary congruence tests.

S8 is identity-free at the scientific boundary. Exact child carriers are used only
transiently by accepted kernel services and never appear in returned signatures.
The scientific observer is the frozen 248-context S7 ordinary multiset observer;
124 factor-swap-normalized LEFT contexts are an execution-only optimization.
"""

from collections import OrderedDict
import resource
from typing import Any, Mapping

from .canon import canonical_text
from .v05_kernel_services import current_kernel_view


class S8ResourceLimit(RuntimeError):
    pass


class S8ScientificBudgetExceeded(S8ResourceLimit):
    """Frozen scientific compute counter exhausted (V3 outcome E1)."""


class S8TransientExecutionPause(S8ResourceLimit):
    """Execution/resource interruption without scientific-counter exhaustion (V3 E2)."""


def _public_signature(read: Mapping[str, Any]) -> tuple[Any, ...]:
    if read.get("legal") is not True or read.get("descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        return ("ILLEGAL",)
    return (
        "D",
        tuple(int(x) for x in read.get("caps7") or ()),
        tuple(sorted((str(k), int(v)) for k, v in (read.get("H_class_bag") or {}).items())),
    )


def _multiset_signature(values: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    rows: dict[str, list[Any]] = {}
    for value in values:
        key = canonical_text(value, pretty=False)
        row = rows.get(key)
        if row is None:
            rows[key] = [value, 1]
        else:
            if canonical_text(row[0], pretty=False) != key:
                raise ValueError("S8_MULTISET_STRUCTURAL_KEY_MISMATCH")
            row[1] = int(row[1]) + 1
    return tuple((rows[k][0], int(rows[k][1])) for k in sorted(rows))


def _compose_public_signature(left: tuple[Any, ...], right: tuple[Any, ...], op: tuple[int, int]) -> tuple[Any, ...]:
    if not left or not right or left[0] != "D" or right[0] != "D":
        return ("ILLEGAL",)
    lc, rc = tuple(left[1]), tuple(right[1])
    if len(lc) != 7 or len(rc) != 7:
        raise ValueError("S8_PUBLIC_CAPS7_ARITY")
    a, b = int(op[0]), int(op[1])
    caps = [int(lc[i]) + int(rc[i]) for i in range(7)]
    caps[a] -= 1; caps[b] -= 1
    if any(x < 0 for x in caps):
        raise ValueError("S8_PUBLIC_WRITE_NEGATIVE_CAPACITY")
    bag: dict[str, int] = {}
    for k, v in tuple(left[2]) + tuple(right[2]):
        bag[str(k)] = bag.get(str(k), 0) + int(v)
    return ("D", tuple(caps), tuple(sorted((k, int(v)) for k, v in bag.items() if int(v))))


def _execution_contexts(basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...]):
    op_set = set(operators)
    if any((b, a) not in op_set for a, b in operators):
        raise ValueError("S8_FACTOR_SWAP_OPERATOR_TRANSPOSE_CLOSURE")
    rows = tuple((ref, op) for ref in basis_refs for op in operators)
    if len(rows) != 124:
        raise ValueError("S8_FACTOR_SWAP_CONTEXT_COUNT")
    return rows


def _scientific_alias_map(basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    execution = _execution_contexts(basis_refs, operators)
    index = {(ref, op): i for i, (ref, op) in enumerate(execution)}
    out = []
    for ref in basis_refs:
        for op in operators:
            out.append(index[(ref, op)])
            out.append(index[(ref, (op[1], op[0]))])
    if len(out) != 248 or len(set(out)) != 124:
        raise ValueError("S8_FACTOR_SWAP_ALIAS_COVERAGE")
    return tuple(out)


class _RecursiveComputer:
    def __init__(self, payload: Mapping[str, Any]):
        self.view = current_kernel_view()
        self.basis_refs = tuple(str(x) for x in payload.get("basis_refs") or ())
        self.operators = tuple(tuple(int(y) for y in x) for x in (payload.get("operator_basis") or ()))
        if tuple(sorted(self.basis_refs)) != self.basis_refs or len(self.basis_refs) != 4:
            raise ValueError("S8_BASIS_REFS")
        if len(self.operators) != 31 or len(set(self.operators)) != 31:
            raise ValueError("S8_OPERATOR_BASIS")
        self.max_states = int(payload.get("max_recursive_states", 100000))
        self.max_relations = int(payload.get("max_exact_relation_calls", 50000))
        self.max_worker_rss_bytes = int(payload.get("max_worker_rss_bytes", 536870912))
        self.v3_budget_semantics = bool(payload.get("v3_budget_semantics", False))
        self.max_profile_coordinate_equivalents = int(payload.get("max_profile_coordinate_equivalents", self.max_relations))
        if self.max_states < 100 or self.max_relations < 100 or self.max_worker_rss_bytes < 134217728:
            raise ValueError("S8_RESOURCE_BUDGET_TOO_SMALL")
        if self.v3_budget_semantics and self.max_profile_coordinate_equivalents < 100:
            raise ValueError("S8_PROFILE_COORDINATE_BUDGET_TOO_SMALL")
        self.basis = self.view.call("AUTHORITY_BASIS")
        if set(self.basis) != set(self.basis_refs):
            raise ValueError("S8_AUTHORITY_BASIS_MISMATCH")
        self.basis_public = {ref: _public_signature(self.view.call("PUBLIC_READ", self.basis[ref])) for ref in self.basis_refs}
        self.execution = _execution_contexts(self.basis_refs, self.operators)
        self.aliases = _scientific_alias_map(self.basis_refs, self.operators)
        self.metrics: dict[str, Any] = {
            "ordinary_context_count": 248, "execution_context_count": 124,
            "public_read_count": 4, "exact_relation_call_count": 0,
            "exact_relation_profile_call_count": 0,
            "exact_relation_profile_family_call_count": 0,
            "logical_profile_coordinate_count": 0,
            "recursive_state_count": 0, "recursive_cache_hits": 0,
            "signature_cache_hits": 0, "signature_cache_evictions": 0,
            "public_cache_hits": 0, "public_cache_evictions": 0,
            "child_cache_hits": 0, "child_cache_evictions": 0,
            "max_recursive_states": self.max_states,
            "max_exact_relation_calls": self.max_relations,
            "max_profile_coordinate_equivalents": self.max_profile_coordinate_equivalents,
            "v3_budget_semantics": self.v3_budget_semantics,
            "max_worker_rss_bytes": self.max_worker_rss_bytes,
            "max_worker_rss_bytes_observed": 0,
            "depth1_encoding": "PUBLIC_ROOT_PLUS_EXECUTION_EXACT_COUNTS_V2",
            "depth2_encoding": "S7_A30_COUNT_VECTOR_BLOCKS_EXECUTION_124_V1",
            "scientific_alias_duplication_materialized": False,
            "resource_pause": False,
        }
        # S7 showed negligible child-prefix reuse and high retention cost.  S8 therefore
        # keeps only tiny exact task-local LRUs and disables exact-child retention.
        self.sig_cache_max_entries = 8
        self.public_cache_max_entries = 32
        self.child_cache_max_entries = 0
        self.sig_cache: "OrderedDict[tuple[int,str], tuple[Any,...]]" = OrderedDict()
        self.public_cache: "OrderedDict[str, tuple[Any, ...]]" = OrderedDict()
        self.child_cache: "OrderedDict[str, tuple[tuple[Mapping[str, Any], ...], ...]]" = OrderedDict()
        # Gen33: bounded cache for class-first recursive projections.  It is
        # execution-only; exact child identity never enters the scientific signature.
        self.projection_cache_max_entries = 16
        self.projection_cache: "OrderedDict[Any, tuple[Any, ...]]" = OrderedDict()
        self.metrics["projection_cache_hits"] = 0
        self.metrics["projection_cache_evictions"] = 0
        # Gen34: carry S7's certified lazy-prefix rule all the way down to
        # depth 1.  Partial P1/P8/P32 projections use only the required scalar
        # exact-profile coordinates; P124 alone uses the A30 family transform.
        # This cache is task-local, execution-only, exact-keyed, and bounded.
        self.profile_prefix_cache_max_entries = 16
        self.profile_prefix_cache: "OrderedDict[str, tuple[int, ...]]" = OrderedDict()
        self.metrics["profile_prefix_cache_hits"] = 0
        self.metrics["profile_prefix_cache_evictions"] = 0

    def _guard(self) -> None:
        if int(self.metrics["recursive_state_count"]) >= self.max_states:
            if self.v3_budget_semantics:
                raise S8ScientificBudgetExceeded("MAX_RECURSIVE_STATES")
            raise S8ResourceLimit("MAX_RECURSIVE_STATES")
        if not self.v3_budget_semantics:
            relation_service_calls = (
                int(self.metrics["exact_relation_call_count"])
                + int(self.metrics["exact_relation_profile_call_count"])
                + int(self.metrics["exact_relation_profile_family_call_count"])
            )
            if relation_service_calls >= self.max_relations:
                raise S8ResourceLimit("MAX_EXACT_RELATION_SERVICE_CALLS")
        rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        if rss_bytes > int(self.metrics.get("max_worker_rss_bytes_observed", 0)):
            self.metrics["max_worker_rss_bytes_observed"] = rss_bytes
        if rss_bytes >= self.max_worker_rss_bytes:
            if self.v3_budget_semantics:
                raise S8TransientExecutionPause("MAX_WORKER_RSS_BYTES")
            raise S8ResourceLimit("MAX_WORKER_RSS_BYTES")

    def _guard_exact_relation(self) -> None:
        self._guard()
        if self.v3_budget_semantics and int(self.metrics["exact_relation_call_count"]) >= self.max_relations:
            raise S8ScientificBudgetExceeded("MAX_EXACT_RELATION_CALLS")

    def _guard_profile_coordinates(self, count: int) -> None:
        self._guard()
        n = int(count)
        if n < 1:
            raise ValueError("S8_PROFILE_COORDINATE_COUNT")
        if self.v3_budget_semantics and int(self.metrics["logical_profile_coordinate_count"]) + n > self.max_profile_coordinate_equivalents:
            raise S8ScientificBudgetExceeded("MAX_PROFILE_COORDINATE_EQUIVALENTS")

    @staticmethod
    def _lru_get(cache: OrderedDict, key: Any, metrics: dict[str, Any], hit_key: str):
        got = cache.get(key)
        if got is not None:
            cache.move_to_end(key)
            metrics["recursive_cache_hits"] = int(metrics["recursive_cache_hits"]) + 1
            metrics[hit_key] = int(metrics.get(hit_key, 0)) + 1
        return got

    @staticmethod
    def _lru_put(cache: OrderedDict, key: Any, value: Any, max_entries: int, metrics: dict[str, Any], eviction_key: str) -> None:
        if max_entries <= 0:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)
            metrics[eviction_key] = int(metrics.get(eviction_key, 0)) + 1

    def public_sig(self, tree: Mapping[str, Any]) -> tuple[Any, ...]:
        key = canonical_text(tree, pretty=False)
        got = self._lru_get(self.public_cache, key, self.metrics, "public_cache_hits")
        if got is not None:
            return got
        value = _public_signature(self.view.call("PUBLIC_READ", tree))
        self.metrics["public_read_count"] = int(self.metrics["public_read_count"]) + 1
        if value == ("ILLEGAL",):
            raise ValueError("S8_PARENT_PUBLIC_READ_ILLEGAL")
        self._lru_put(self.public_cache, key, value, self.public_cache_max_entries, self.metrics, "public_cache_evictions")
        return value

    def profile_counts(self, tree: Mapping[str, Any]) -> tuple[int, ...]:
        """Exact S7/A30 P124 count vector without a redundant child PUBLIC_READ.

        For one fixed outer coordinate, every exact child has the same inherited
        public descriptor, determined by the parent public descriptor, frozen seed,
        and operator.  S7 depth-2 therefore certified comparison of children by the
        124 exact multiplicities alone.  Reuse exactly that accepted reduction here.
        """
        self._guard()
        self._guard_profile_coordinates(len(self.execution))
        family = self.view.call(
            "EXACT_RELATION_PROFILE_FAMILY", tree,
            tuple(self.basis[ref] for ref in self.basis_refs), self.operators,
        )
        counts = tuple(int(x) for x in (family.get("exact_outcome_counts") or ()))
        if len(counts) != len(self.execution):
            raise ValueError("S8_D1_FAMILY_PROFILE_LENGTH")
        self.metrics["exact_relation_profile_family_call_count"] = int(self.metrics["exact_relation_profile_family_call_count"]) + 1
        self.metrics["logical_profile_coordinate_count"] = int(self.metrics["logical_profile_coordinate_count"]) + len(counts)
        return counts

    def profile_counts_prefix(self, tree: Mapping[str, Any], prefix_count: int) -> tuple[int, ...]:
        """Exact monotone prefix of the accepted 124-coordinate count vector.

        This is the S7 lazy-prefix execution rule lifted into S8 recursion.
        P1/P8/P32 call only the scalar exact-profile coordinates that are
        actually required.  P124 delegates to ``profile_counts`` so the full
        equality boundary remains the accepted A30 family transform.
        """
        p = int(prefix_count)
        if p not in (1, 8, 32, 124):
            raise ValueError("S8_D1_PROFILE_PREFIX_COUNT")
        tkey = canonical_text(tree, pretty=False)
        cached = self._lru_get(
            self.profile_prefix_cache, tkey, self.metrics, "profile_prefix_cache_hits"
        )
        if cached is not None and len(cached) >= p:
            return tuple(cached[:p])
        if p == 124:
            counts = self.profile_counts(tree)
            self._lru_put(
                self.profile_prefix_cache, tkey, counts,
                self.profile_prefix_cache_max_entries, self.metrics,
                "profile_prefix_cache_evictions",
            )
            return counts
        counts = list(cached or ())
        start = len(counts)
        if start > p:
            return tuple(counts[:p])
        for ref, op in self.execution[start:p]:
            self._guard_profile_coordinates(1)
            prof = self.view.call("EXACT_RELATION_PROFILE", tree, self.basis[ref], op)
            counts.append(int(prof["exact_outcome_count"]))
            self.metrics["exact_relation_profile_call_count"] = int(self.metrics["exact_relation_profile_call_count"]) + 1
            self.metrics["logical_profile_coordinate_count"] = int(self.metrics["logical_profile_coordinate_count"]) + 1
        value = tuple(counts)
        self._lru_put(
            self.profile_prefix_cache, tkey, value,
            self.profile_prefix_cache_max_entries, self.metrics,
            "profile_prefix_cache_evictions",
        )
        return value

    def depth1(self, tree: Mapping[str, Any]) -> tuple[Any, ...]:
        tkey = canonical_text(tree, pretty=False); ckey = (1, tkey)
        got = self._lru_get(self.sig_cache, ckey, self.metrics, "signature_cache_hits")
        if got is not None:
            return got
        self._guard(); self.metrics["recursive_state_count"] = int(self.metrics["recursive_state_count"]) + 1
        root_pub = self.public_sig(tree)
        counts = self.profile_counts(tree)
        # Factor-swap normalization is lossless.  The 248 scientific coordinates are
        # deterministic duplicates/transposes of these 124 execution coordinates, so
        # materializing the aliases changes size, not equality.
        value = ("ORDINARY_BRANCH_MULTISET", 1, root_pub, counts)
        self._lru_put(self.sig_cache, ckey, value, self.sig_cache_max_entries, self.metrics, "signature_cache_evictions")
        return value

    def depth2_compact(self, tree: Mapping[str, Any]) -> tuple[Any, ...]:
        """Equality-exact depth-2 signature using the accepted S7 A30 reduction.

        The historical recursive implementation called PUBLIC_READ on every
        grandchild and expanded 124 execution coordinates back to 248 aliases.
        S7 already proved both are redundant for equality at a fixed coordinate:
        child public D is inherited deterministically, while factor-swap aliases are
        lossless duplicates.  Stream each outer block and retain only its multiset of
        A30 P124 count vectors, plus the root public descriptor.
        """
        tkey = canonical_text(tree, pretty=False); ckey = (2, tkey)
        got = self._lru_get(self.sig_cache, ckey, self.metrics, "signature_cache_hits")
        if got is not None:
            return got
        self._guard(); self.metrics["recursive_state_count"] = int(self.metrics["recursive_state_count"]) + 1
        root_pub = self.public_sig(tree)
        blocks = []
        for ref, op in self.execution:
            self._guard_exact_relation()
            rel = self.view.call("EXACT_RELATION", tree, self.basis[ref], op)
            self.metrics["exact_relation_call_count"] = int(self.metrics["exact_relation_call_count"]) + 1
            vectors = [self.profile_counts(child) for child in tuple(rel["children"])]
            blocks.append(_multiset_signature(vectors))
        value = ("ORDINARY_BRANCH_MULTISET_COMPACT_D2", 2, root_pub, tuple(blocks))
        self._lru_put(self.sig_cache, ckey, value, self.sig_cache_max_entries, self.metrics, "signature_cache_evictions")
        return value

    def one_step(self, tree: Mapping[str, Any]) -> tuple[tuple[Mapping[str, Any], ...], ...]:
        tkey = canonical_text(tree, pretty=False)
        got = self._lru_get(self.child_cache, tkey, self.metrics, "child_cache_hits")
        if got is not None:
            return got
        blocks = []
        for ref, op in self.execution:
            self._guard_exact_relation()
            rel = self.view.call("EXACT_RELATION", tree, self.basis[ref], op)
            self.metrics["exact_relation_call_count"] = int(self.metrics["exact_relation_call_count"]) + 1
            blocks.append(tuple(rel["children"]))
        value = tuple(blocks)
        self._lru_put(self.child_cache, tkey, value, self.child_cache_max_entries, self.metrics, "child_cache_evictions")
        return value

    def signature(self, tree: Mapping[str, Any], level: int) -> tuple[Any, ...]:
        if level == 1:
            return self.depth1(tree)
        if level == 2:
            return self.depth2_compact(tree)
        tkey = canonical_text(tree, pretty=False); ckey = (int(level), tkey)
        got = self._lru_get(self.sig_cache, ckey, self.metrics, "signature_cache_hits")
        if got is not None:
            return got
        self._guard(); self.metrics["recursive_state_count"] = int(self.metrics["recursive_state_count"]) + 1
        previous = self.signature(tree, level - 1)
        # Stream one execution block at a time.  Do not retain the full exact child
        # family and do not materialize factor-swap duplicate aliases.
        execution_blocks = []
        for ref, op in self.execution:
            self._guard_exact_relation()
            rel = self.view.call("EXACT_RELATION", tree, self.basis[ref], op)
            self.metrics["exact_relation_call_count"] = int(self.metrics["exact_relation_call_count"]) + 1
            execution_blocks.append(_multiset_signature([self.signature(child, level - 1) for child in tuple(rel["children"])]))
        value = ("ORDINARY_BRANCH_MULTISET", int(level), previous, tuple(execution_blocks))
        self._lru_put(self.sig_cache, ckey, value, self.sig_cache_max_entries, self.metrics, "signature_cache_evictions")
        return value

    def projected_signature(self, tree: Mapping[str, Any], level: int, *, prefix_count: int) -> tuple[Any, ...]:
        """Monotone exact projection of the accepted Gen32 recursive signature.

        P124 is equality-equivalent to ``signature`` under Gen32's certified
        factor-swap normalization/count-vector reduction.  P1/P8/P32 retain a
        literal prefix of those normalized execution blocks.  Therefore unequal
        projected signatures are valid full-signature separators; equality at a
        short prefix makes no scientific claim and only authorizes extension.
        """
        p = int(prefix_count)
        if p not in (1, 8, 32, 124):
            raise ValueError("S8_RECURSIVE_PREFIX_COUNT")
        if level < 1:
            raise ValueError("S8_PROJECTED_LEVEL")
        if level == 1:
            # Gen34 true nested prefix: partial projections retain the root
            # public descriptor plus only the required exact count prefix.
            # P124 returns the accepted full depth-1 representation exactly.
            if p == 124:
                return self.depth1(tree)
            tkey = canonical_text(tree, pretty=False)
            ckey = ("P", 1, p, tkey)
            got = self._lru_get(self.projection_cache, ckey, self.metrics, "projection_cache_hits")
            if got is not None:
                return got
            self._guard(); self.metrics["recursive_state_count"] = int(self.metrics["recursive_state_count"]) + 1
            root_pub = self.public_sig(tree)
            counts = self.profile_counts_prefix(tree, p)
            value = ("ORDINARY_BRANCH_MULTISET_D1_PROJECTION", 1, p, root_pub, counts)
            self._lru_put(self.projection_cache, ckey, value, self.projection_cache_max_entries, self.metrics, "projection_cache_evictions")
            return value
        tkey = canonical_text(tree, pretty=False)
        ckey = ("P", int(level), p, tkey)
        got = self._lru_get(self.projection_cache, ckey, self.metrics, "projection_cache_hits")
        if got is not None:
            return got
        self._guard(); self.metrics["recursive_state_count"] = int(self.metrics["recursive_state_count"]) + 1
        if level == 2:
            # Literal prefix of accepted Gen32 depth2_compact: same root public
            # descriptor and first p normalized outer blocks; each grandchild is
            # represented by its complete A30 P124 exact-count vector.
            root_pub = self.public_sig(tree)
            blocks = []
            for ref, op in self.execution[:p]:
                self._guard()
                rel = self.view.call("EXACT_RELATION", tree, self.basis[ref], op)
                self.metrics["exact_relation_call_count"] = int(self.metrics["exact_relation_call_count"]) + 1
                vectors = [self.profile_counts_prefix(child, p) for child in tuple(rel["children"])]
                blocks.append(_multiset_signature(vectors))
            value = ("ORDINARY_BRANCH_MULTISET_COMPACT_D2_PROJECTION", 2, p, root_pub, tuple(blocks))
        else:
            previous = self.projected_signature(tree, level - 1, prefix_count=p)
            blocks = []
            for ref, op in self.execution[:p]:
                self._guard()
                rel = self.view.call("EXACT_RELATION", tree, self.basis[ref], op)
                self.metrics["exact_relation_call_count"] = int(self.metrics["exact_relation_call_count"]) + 1
                blocks.append(_multiset_signature([
                    self.projected_signature(child, level - 1, prefix_count=p)
                    for child in tuple(rel["children"])
                ]))
            value = ("ORDINARY_BRANCH_MULTISET_EXECUTION_PROJECTION", int(level), p, previous, tuple(blocks))
        self._lru_put(self.projection_cache, ckey, value, self.projection_cache_max_entries, self.metrics, "projection_cache_evictions")
        return value

    def outer_component_projection(self, tree: Mapping[str, Any], *, level: int,
                                   execution_context_index: int, prefix_count: int) -> tuple[Any, ...]:
        if level < 2:
            raise ValueError("S8_COMPONENT_DEPTH")
        if not 0 <= int(execution_context_index) < len(self.execution):
            raise ValueError("S8_EXECUTION_CONTEXT_INDEX")
        p = int(prefix_count)
        if p not in (1, 8, 32, 124):
            raise ValueError("S8_RECURSIVE_PREFIX_COUNT")
        ref, op = self.execution[int(execution_context_index)]
        self._guard_exact_relation()
        rel = self.view.call("EXACT_RELATION", tree, self.basis[ref], op)
        self.metrics["exact_relation_call_count"] = int(self.metrics["exact_relation_call_count"]) + 1
        values = [self.projected_signature(child, level - 1, prefix_count=p) for child in tuple(rel["children"])]
        return ("S8_RECURSIVE_OUTER_COMPONENT_EXECUTION_PROJECTION", int(level), int(execution_context_index), p, _multiset_signature(values))

    def outer_component(self, tree: Mapping[str, Any], *, level: int, execution_context_index: int) -> tuple[Any, ...]:
        if level < 2:
            raise ValueError("S8_COMPONENT_DEPTH")
        if not 0 <= int(execution_context_index) < len(self.execution):
            raise ValueError("S8_EXECUTION_CONTEXT_INDEX")
        ref, op = self.execution[int(execution_context_index)]
        self._guard_exact_relation()
        rel = self.view.call("EXACT_RELATION", tree, self.basis[ref], op)
        self.metrics["exact_relation_call_count"] = int(self.metrics["exact_relation_call_count"]) + 1
        values = [self.signature(child, level - 1) for child in tuple(rel["children"])]
        return ("S8_RECURSIVE_OUTER_COMPONENT", int(level), int(execution_context_index), _multiset_signature(values))


def _v3_pause_result(depth: int, outcome: str, reason: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    if outcome not in ("E1_FROZEN_TASK_BUDGET_EXCEEDED", "E2_TRANSIENT_EXECUTION_PAUSE"):
        raise ValueError("S8_V3_PAUSE_OUTCOME")
    m = dict(metrics); m["v3_outcome"] = outcome; m["pause_reason"] = reason; m["future_depth"] = int(depth)
    return {"signature": ("S8_V3_PAUSE", outcome, int(depth), reason), "outcome_count": int(m.get("exact_relation_call_count", 0)), "metrics": m, "v3_outcome": outcome}


def _resource_result(depth: int, reason: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    m = dict(metrics); m["resource_pause"] = True; m["resource_pause_reason"] = reason; m["future_depth"] = int(depth)
    return {"signature": ("S8_RESOURCE_PAUSE", int(depth), reason), "outcome_count": int(m.get("exact_relation_call_count", 0)), "metrics": m}


def s8_recursive_outer_component_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    depth = int(payload.get("future_depth", 0))
    if depth < 3 or depth > 6:
        raise ValueError("S8_FUTURE_DEPTH")
    computer = _RecursiveComputer(payload)
    try:
        sig = computer.outer_component(payload["state_tree"], level=depth, execution_context_index=int(payload["outer_execution_context_index"]))
    except S8ResourceLimit as exc:
        return _resource_result(depth, str(exc), computer.metrics)
    m = dict(computer.metrics, future_depth=depth, outer_execution_context_index=int(payload["outer_execution_context_index"]))
    return {"signature": sig, "outcome_count": int(m["exact_relation_call_count"]), "metrics": m}


def s8_s7_class_recursive_prefix_comparator_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Compare one complete surviving S7 class and stop at its first separator.

    State tokens/class ids are execution provenance only and never participate in
    the compared signatures.  Exact scientific equality is decided on structural
    recursive projection tuples.  P124 is the full normalized observer; earlier
    prefixes are monotone separator probes only.
    """
    depth = int(payload.get("future_depth", 0))
    if depth < 3 or depth > 6:
        raise ValueError("S8_FUTURE_DEPTH")
    outer = int(payload.get("outer_execution_context_index", -1))
    schedule = tuple(int(x) for x in (payload.get("recursive_prefix_schedule") or ()))
    if schedule != (1, 8, 32, 124):
        raise ValueError("S8_RECURSIVE_PREFIX_SCHEDULE")
    members = list(payload.get("class_members") or ())
    if not 2 <= len(members) <= 12:
        raise ValueError("S8_CLASS_MEMBER_COUNT")
    members.sort(key=lambda r: str(r.get("state_token", "")))
    tokens = [str(r.get("state_token", "")) for r in members]
    if any(not t for t in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("S8_CLASS_MEMBER_TOKENS")
    computer = _RecursiveComputer(payload)
    rep = members[0]
    try:
        for p in schedule:
            rep_sig = computer.outer_component_projection(rep["state_tree"], level=depth, execution_context_index=outer, prefix_count=p)
            for row in members[1:]:
                sig = computer.outer_component_projection(row["state_tree"], level=depth, execution_context_index=outer, prefix_count=p)
                if sig != rep_sig:
                    m = dict(computer.metrics, future_depth=depth, outer_execution_context_index=outer,
                             recursive_prefix_count=p, split_found=True,
                             s7_class_id=str(payload.get("s7_class_id", "")),
                             a_state_token=tokens[0], b_state_token=str(row["state_token"]),
                             separator_a_signature=rep_sig, separator_b_signature=sig,
                             p124_full_equality_equivalent=True)
                    return {"signature": ("S8_S7_CLASS_PREFIX_SPLIT", depth, outer, p, rep_sig, sig),
                            "outcome_count": int(computer.metrics["exact_relation_call_count"]), "metrics": m}
    except S8ScientificBudgetExceeded as exc:
        return _v3_pause_result(depth, "E1_FROZEN_TASK_BUDGET_EXCEEDED", str(exc), dict(computer.metrics, outer_execution_context_index=outer, s7_class_id=str(payload.get("s7_class_id", "")), split_found=False))
    except S8TransientExecutionPause as exc:
        return _v3_pause_result(depth, "E2_TRANSIENT_EXECUTION_PAUSE", str(exc), dict(computer.metrics, outer_execution_context_index=outer, s7_class_id=str(payload.get("s7_class_id", "")), split_found=False))
    except S8ResourceLimit as exc:
        m = dict(computer.metrics, future_depth=depth, outer_execution_context_index=outer,
                 s7_class_id=str(payload.get("s7_class_id", "")), split_found=False,
                 resource_pause=True, resource_pause_reason=str(exc))
        return {"signature": ("S8_RESOURCE_PAUSE", depth, str(exc)),
                "outcome_count": int(computer.metrics.get("exact_relation_call_count", 0)), "metrics": m}
    m = dict(computer.metrics, future_depth=depth, outer_execution_context_index=outer,
             recursive_prefix_count=124, split_found=False,
             s7_class_id=str(payload.get("s7_class_id", "")),
             class_member_count=len(members), p124_full_equality_equivalent=True)
    return {"signature": ("S8_S7_CLASS_NO_SPLIT_P124", depth, outer),
            "outcome_count": int(computer.metrics["exact_relation_call_count"]), "metrics": m}

def s8_recursive_ordinary_signature_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reference/full recursive signature evaluator used for tests and proof replay."""
    depth = int(payload.get("future_depth", 0))
    if depth < 1 or depth > 6:
        raise ValueError("S8_FUTURE_DEPTH")
    computer = _RecursiveComputer(payload)
    try:
        sig = computer.signature(payload["state_tree"], depth)
    except S8ResourceLimit as exc:
        return _resource_result(depth, str(exc), computer.metrics)
    m = dict(computer.metrics, future_depth=depth)
    return {"signature": sig, "outcome_count": int(m["exact_relation_call_count"]), "metrics": m}



def _s8_v3_worker_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def s8_v3_relation_generation_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Generate one exact normalized ordinary-relation block for staged S8 V3.

    This is execution-only decomposition of the existing recursive projection.
    Exact child identity is retained only by the StageRuntime generation store and
    never enters the scientific signature.
    """
    view = current_kernel_view()
    refs = tuple(str(x) for x in payload.get("basis_refs") or ())
    operators = tuple(tuple(int(y) for y in x) for x in (payload.get("operator_basis") or ()))
    if tuple(sorted(refs)) != refs or len(refs) != 4:
        raise ValueError("S8V3_STAGE_BASIS_REFS")
    if len(operators) != 31 or len(set(operators)) != 31:
        raise ValueError("S8V3_STAGE_OPERATOR_BASIS")
    execution = _execution_contexts(refs, operators)
    idx = int(payload.get("execution_context_index", -1))
    if not 0 <= idx < len(execution):
        raise ValueError("S8V3_STAGE_EXECUTION_CONTEXT_INDEX")
    max_rss = int(payload.get("max_worker_rss_bytes", 536870912))
    if max_rss < 134217728:
        raise ValueError("S8V3_STAGE_WORKER_RSS_BUDGET")
    basis = view.call("AUTHORITY_BASIS")
    if set(basis) != set(refs):
        raise ValueError("S8V3_STAGE_AUTHORITY_BASIS")
    ref, op = execution[idx]
    rel = view.call("EXACT_RELATION", payload["state_tree"], basis[ref], op)
    rss = _s8_v3_worker_rss_bytes()
    if rss >= max_rss:
        raise S8TransientExecutionPause("RESOURCE:MAX_WORKER_RSS_BYTES")
    canons = tuple(rel.get("canons") or ())
    children = tuple(rel.get("children") or ())
    if len(canons) != len(children):
        raise ValueError("S8V3_STAGE_CANON_CHILD_ALIGNMENT")
    rm = rel.get("metrics") or {}
    metrics = {
        "exact_relation_call_count": 1,
        "execution_context_index": idx,
        "outer_distinct_child_count": len(children),
        "max_worker_rss_bytes_observed": rss,
    }
    for key in ("attempted_owner_pairs","legal_owner_pairs","rooted_owner_pair_candidates","child_canon_constructions"):
        metrics[key] = int(rm.get(key, 0))
    return {
        "states": [{"identity": canon, "state": {"tree": child}} for canon, child in zip(canons, children)],
        "metrics": metrics,
    }


def s8_v3_profile_extension_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extend one deduplicated grandchild's normalized profile prefix exactly.

    Coordinates are basis-major over the same 124 factor-swap-normalized LEFT
    contexts as the frozen V3 observer.  Each newly requested contiguous seed
    segment is evaluated by EXACT_RELATION_PROFILE_BATCH, so already-computed
    coordinates are never repaid.
    """
    import json as _json
    view = current_kernel_view()
    refs = tuple(str(x) for x in payload.get("basis_refs") or ())
    operators = tuple(tuple(int(y) for y in x) for x in (payload.get("operator_basis") or ()))
    if tuple(sorted(refs)) != refs or len(refs) != 4:
        raise ValueError("S8V3_PROFILE_BASIS_REFS")
    if len(operators) != 31 or len(set(operators)) != 31:
        raise ValueError("S8V3_PROFILE_OPERATOR_BASIS")
    execution = _execution_contexts(refs, operators)
    start = int(payload.get("start_context_index", -1)); end = int(payload.get("end_context_index", -1))
    previous = tuple(int(x) for x in (payload.get("previous_counts") or ()))
    if end not in (1, 8, 32, 124) or start not in (0, 1, 8, 32) or not (0 <= start < end <= 124):
        raise ValueError("S8V3_PROFILE_RANGE")
    if len(previous) != start:
        raise ValueError("S8V3_PROFILE_PREVIOUS_PREFIX")
    identity_json = payload.get("grandchild_identity_json")
    state_json = payload.get("grandchild_state_json")
    if type(identity_json) is not str or type(state_json) is not str:
        raise ValueError("S8V3_PROFILE_COMPACT_INPUT")
    identity = _json.loads(identity_json)
    state = _json.loads(state_json)
    child = state.get("tree") if type(state) is dict else None
    if type(child) is not dict:
        raise ValueError("S8V3_PROFILE_CHILD_TREE")
    max_rss = int(payload.get("max_worker_rss_bytes", 536870912))
    if max_rss < 134217728:
        raise ValueError("S8V3_PROFILE_WORKER_RSS_BUDGET")
    basis = view.call("AUTHORITY_BASIS")
    if set(basis) != set(refs):
        raise ValueError("S8V3_PROFILE_AUTHORITY_BASIS")
    counts = list(previous)
    metrics = {
        "exact_relation_profile_batch_call_count": 0,
        "logical_profile_coordinate_count": 0,
        "profile_contexts_reused": start,
        "profile_contexts_computed": end-start,
        "max_worker_rss_bytes_observed": 0,
    }
    i = start
    while i < end:
        ref = execution[i][0]
        ops = []
        j = i
        while j < end and execution[j][0] == ref:
            ops.append(execution[j][1]); j += 1
        batch = view.call("EXACT_RELATION_PROFILE_BATCH", child, basis[ref], tuple(ops))
        got = tuple(int(x) for x in (batch.get("exact_outcome_counts") or ()))
        if len(got) != len(ops):
            raise ValueError("S8V3_PROFILE_BATCH_LENGTH")
        counts.extend(got)
        metrics["exact_relation_profile_batch_call_count"] += 1
        metrics["logical_profile_coordinate_count"] += len(got)
        bm = batch.get("metrics") or {}
        for key in ("attempted_owner_pairs","legal_owner_pairs","rooted_owner_pair_candidates","child_canon_constructions",
                    "relation_profile_zero_fast_hits","relation_profile_bridge_fingerprint_fast_hits",
                    "relation_profile_bridge_fingerprint_fallbacks","relation_profile_bridge_fingerprint_canon_avoided",
                    "relation_profile_batch_calls","relation_profile_batch_operator_count","relation_profile_batch_bridge_index_builds"):
            metrics[key] = int(metrics.get(key, 0)) + int(bm.get(key, 0))
        rss = _s8_v3_worker_rss_bytes()
        metrics["max_worker_rss_bytes_observed"] = max(int(metrics["max_worker_rss_bytes_observed"]), rss)
        if rss >= max_rss:
            raise S8TransientExecutionPause("RESOURCE:MAX_WORKER_RSS_BYTES")
        i = j
    if len(counts) != end:
        raise ValueError("S8V3_PROFILE_LENGTH")
    return {
        "states": [{"identity": identity, "state": {"profile_counts": tuple(counts)}}],
        "metrics": metrics,
    }
