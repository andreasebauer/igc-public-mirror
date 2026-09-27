from __future__ import annotations

import json
import statistics
import tempfile
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable

from . import regime_scanner as rs


class InsufficientExactCarrierData(RuntimeError):
    pass


class InsufficientExactL2Mapping(RuntimeError):
    pass


def load_exact_carrier_unblinding_spec() -> dict:
    p = files("infinity_grid").joinpath(
        "resources/decoder/O_EXACT_CARRIER_UNBLINDING_SPEC_v1.json"
    )
    return json.loads(p.read_text(encoding="utf-8"))


def _replace_nested(node: tuple, indices: tuple[int, ...], value: Any) -> tuple:
    if not indices:
        return value
    i = int(indices[0])
    out = list(node)
    out[i] = _replace_nested(out[i], indices[1:], value)
    return tuple(out)


def _decrement_site(h6: tuple, path: tuple[int, ...], endpoint_type: int, count: int = 1) -> tuple:
    a, j, k, v, i = map(int, path)
    p, f = h6[a][j][k][v][i]
    nf = list(f)
    nf[int(endpoint_type)] -= int(count)
    if nf[int(endpoint_type)] < 0:
        raise InsufficientExactCarrierData(
            f"explicit H6 reservation oversubscribed path={path} type={endpoint_type}"
        )
    return _replace_nested(h6, (a, j, k, v, i), (p, tuple(nf)))


def _canonical_colored_edge_structure_with_map(colors: list[str], edges: list[tuple]) -> tuple[tuple, tuple[int, ...]]:
    """Canonicalize a small anonymous-owner structure and return old->canonical labels.

    The second return value is execution support for deterministic construction sampling;
    it is not an additional scientific observable.  When several automorphisms produce
    the same canonical encoding, the lexicographically smallest permutation is chosen.
    Those tied choices lie in the same automorphism orbit, so choosing one cannot create
    a new isomorphism class.
    """
    import itertools

    n = len(colors)
    if n > 8:
        raise InsufficientExactCarrierData(f"exact anonymous-owner canonicalization is bounded at 8 owners; got {n}; reopen with a scalable canonicalizer")
    if n == 0:
        return (tuple(), tuple()), tuple()
    best = None
    best_perm = None
    for perm in itertools.permutations(range(n)):
        nc = [None] * n
        for old, new in enumerate(perm):
            nc[new] = colors[old]
        ee = []
        for e in edges:
            u, v, *tail = e
            nu, nv = perm[int(u)], perm[int(v)]
            tail = list(tail)
            if nu <= nv:
                ee.append((nu, nv, *tail))
            else:
                if len(tail) >= 2:
                    tail[0], tail[1] = tail[1], tail[0]
                if len(tail) >= 4:
                    tail[2], tail[3] = tail[3], tail[2]
                ee.append((nv, nu, *tail))
        cand = (tuple(nc), tuple(sorted(ee, key=repr)))
        key = (repr(cand), tuple(perm))
        if best is None or key < (repr(best), tuple(best_perm)):
            best = cand
            best_perm = tuple(int(x) for x in perm)
    return best, best_perm


def _canonical_colored_edge_structure(colors: list[str], edges: list[tuple]) -> tuple:
    """Compatibility wrapper returning only the canonical anonymous-owner encoding."""
    return _canonical_colored_edge_structure_with_map(colors, edges)[0]


@dataclass(frozen=True)
class ReservationOption:
    state: Any
    endpoint_type: int
    witness_token: str
    multiplicity: int
    representative_key: tuple


@dataclass(frozen=True)
class ExactO7State:
    """O7 exact-carrier wrapper over the compact O6 resource carrier.

    It preserves the exact retained E7 incidence and every O7+ external reservation
    explicitly by site path. It does *not* claim recovery of hidden pre-O7 topology
    that the historical compact O6 resource carrier had already quotiented away.
    """

    engine: Any
    ctx: Any
    edges: tuple
    source_id: str
    lane: str = "O7_AUTHORITY"
    overlay: tuple[tuple[int, tuple[int, ...], int, int], ...] = tuple()
    level: int = 7

    @cached_property
    def exactness_scope(self) -> str:
        return "EXACT_O7_PLUS_OVER_COMPACT_O6_RESOURCE_CARRIER"

    @cached_property
    def owner_h6s(self) -> tuple[tuple, ...]:
        base = [self.engine.materialize_owner(p.h6, self.edges, i) for i, p in enumerate(self.ctx.parents)]
        for owner, path, t, count in self.overlay:
            base[int(owner)] = _decrement_site(base[int(owner)], tuple(path), int(t), int(count))
        return tuple(base)

    @cached_property
    def owner_orbit_data(self) -> tuple[tuple[Any, dict], ...]:
        """Cache each materialized O6 owner's root and current site-orbit table once."""
        out = []
        for h in self.owner_h6s:
            root, _entries, _groups, byp = self.engine.site_orbits_current(h, None)
            out.append((root, byp))
        return tuple(out)

    @cached_property
    def owner_roots(self) -> tuple[str, ...]:
        return tuple(root.base_sig for root, _byp in self.owner_orbit_data)

    @cached_property
    def skin(self) -> str:
        return self.engine.O6._digest_container("R7", list(self.owner_roots))

    @cached_property
    def construction_digest(self) -> str:
        # Match the v0.26 identity at the untouched O7 authority anchor so panel
        # selection remains calibrated; explicit overlays extend the presentation.
        if not self.overlay:
            return rs._sha({
                "level": 7,
                "parents": list(self.ctx.source_order),
                "edges": [list(x) for x in self.edges],
                "reserve_counts": [0, 0, 0, 0, 0, 0, 0],
                "source": self.source_id,
            })
        return rs._sha({
            "level": 7,
            "parents": list(self.ctx.source_order),
            "edges": [list(x) for x in self.edges],
            "explicit_overlay": [
                [o, list(path), t, count] for o, path, t, count in self.overlay
            ],
            "source": self.source_id,
        })

    @cached_property
    def h_struct_canonicalization(self) -> tuple[tuple, tuple[int, ...]]:
        owner_data = [(root.base_sig, byp) for root, byp in self.owner_orbit_data]
        ee = []
        for e in self.edges:
            c, p, a, d, q, b = self.engine.eparts(e)
            p = tuple(p)
            q = tuple(q)
            ps = owner_data[c][1][p][3]
            qs = owner_data[d][1][q][3]
            ee.append((int(c), int(d), int(a), int(b), str(ps), str(qs)))
        return _canonical_colored_edge_structure_with_map([x[0] for x in owner_data], ee)

    @cached_property
    def h_struct_encoding(self) -> tuple:
        return self.h_struct_canonicalization[0]

    @cached_property
    def canonical_owner_labels(self) -> tuple[int, ...]:
        return self.h_struct_canonicalization[1]

    @cached_property
    def h_struct_canon(self) -> str:
        return rs._sha({"tag": "H7_STRUCT", "encoding": self.h_struct_encoding})

    @cached_property
    def top_pairs(self) -> list[tuple[int, int]]:
        return [(int(e[0]), int(e[7])) for e in self.edges]

    @cached_property
    def typed_edges(self) -> list[tuple[int, int, int, int]]:
        return [(int(e[0]), int(e[7]), int(e[6]), int(e[13])) for e in self.edges]

    @cached_property
    def owner_caps(self) -> list[tuple[int, ...]]:
        out = []
        for _root, byp in self.owner_orbit_data:
            cap = [0] * 7
            for _path, (_p, f, _leaf, _ps) in byp.items():
                for t, x in enumerate(f):
                    cap[t] += int(x)
            out.append(tuple(cap))
        return out

    @cached_property
    def total_caps(self) -> tuple[int, ...]:
        return tuple(sum(c[t] for c in self.owner_caps) for t in range(7))

    @cached_property
    def owner_colors(self) -> list[str]:
        return list(self.owner_roots)

    @cached_property
    def leaf_count(self) -> int:
        return sum(len(byp) for _root, byp in self.owner_orbit_data)

    @cached_property
    def relation_count_total(self) -> int:
        return len(self.edges)

    def _with_decrement(self, owner: int, path: tuple[int, ...], t: int) -> "ExactO7State":
        counts = Counter((int(o), tuple(p), int(tt)) for o, p, tt, count in self.overlay for _ in range(int(count)))
        counts[(int(owner), tuple(path), int(t))] += 1
        overlay = tuple((o, p, tt, count) for (o, p, tt), count in sorted(counts.items(), key=repr))
        return ExactO7State(self.engine, self.ctx, self.edges, self.source_id, self.lane, overlay)

    def reservation_options(self, endpoint_type: int) -> list[ReservationOption]:
        t = int(endpoint_type)
        if self.total_caps[t] <= 0:
            return []
        raw = []
        pre = self.h_struct_canon
        for owner, h in enumerate(self.owner_h6s):
            _root, _entries, _groups, byp = self.engine.site_orbits_current(h, None)
            groups = defaultdict(list)
            for path, (_p, f, _leaf, ps) in byp.items():
                if int(f[t]) > 0:
                    groups[str(ps)].append(tuple(path))
            for ps, paths in sorted(groups.items()):
                rep = min(paths)
                child = self._with_decrement(owner, rep, t)
                post = child.h_struct_canon
                token = rs._sha({"pre": pre, "post": post, "type": t})
                raw.append(ReservationOption(child, t, token, len(paths), (owner, rep, ps)))
        # Merge exact-isomorphic pointed choices; multiplicity remains observable.
        merged: dict[tuple, ReservationOption] = {}
        for x in raw:
            key = (x.state.h_struct_canon, x.state.skin, x.witness_token)
            if key in merged:
                old = merged[key]
                merged[key] = ReservationOption(
                    old.state, t, old.witness_token, old.multiplicity + x.multiplicity,
                    min(old.representative_key, x.representative_key),
                )
            else:
                merged[key] = x
        return sorted(merged.values(), key=lambda x: (x.state.h_struct_canon, x.state.skin, x.witness_token, repr(x.representative_key)))

    def reserve_canonical(self, endpoint_type: int) -> ReservationOption:
        """Choose one representation-invariant construction reservation without census.

        This path is construction sampling only.  It chooses an endpoint orbit from the
        *pre-state* canonical owner labelling and materializes exactly one post-state.
        Scientific action audits still use the complete projected Counter kernel.
        """
        t = int(endpoint_type)
        if self.total_caps[t] <= 0:
            raise InsufficientExactCarrierData(f"O7 no exact endpoint type {endpoint_type}")
        candidates = []
        canon = self.canonical_owner_labels
        for owner, (_root, byp) in enumerate(self.owner_orbit_data):
            groups = defaultdict(list)
            for path, (_p, f, _leaf, ps) in byp.items():
                if int(f[t]) > 0:
                    groups[str(ps)].append(tuple(path))
            for ps, paths in groups.items():
                rep = min(paths)
                candidates.append(((int(canon[owner]), str(ps), rep), owner, rep, str(ps), len(paths)))
        if not candidates:
            raise InsufficientExactCarrierData(f"O7 no exact endpoint type {endpoint_type}")
        _key, owner, rep, ps, multiplicity = min(candidates, key=lambda x: x[0])
        child = self._with_decrement(owner, rep, t)
        token = rs._sha({"pre": self.h_struct_canon, "post": child.h_struct_canon, "type": t})
        return ReservationOption(child, t, token, int(multiplicity), (int(canon[owner]), rep, ps))


@dataclass(frozen=True)
class ExactLiftState:
    engine: Any
    level: int
    children: tuple[Any, ...]
    top_edges_full: tuple[tuple, ...]
    lane: str
    motif_id: str

    @cached_property
    def exactness_scope(self) -> str:
        return "EXACT_O7_PLUS_OVER_COMPACT_O6_RESOURCE_CARRIER"

    @cached_property
    def skin(self) -> str:
        return self.engine.O6._digest_container(f"R{self.level}", [c.skin for c in self.children])

    @cached_property
    def h_struct_canonicalization(self) -> tuple[tuple, tuple[int, ...]]:
        colors = [c.h_struct_canon for c in self.children]
        edges = []
        for u, v, a, b, wu, wv in self.top_edges_full:
            edges.append((int(u), int(v), int(a), int(b), str(wu), str(wv)))
        return _canonical_colored_edge_structure_with_map(colors, edges)

    @cached_property
    def h_struct_encoding(self) -> tuple:
        return self.h_struct_canonicalization[0]

    @cached_property
    def canonical_owner_labels(self) -> tuple[int, ...]:
        return self.h_struct_canonicalization[1]

    @cached_property
    def h_struct_canon(self) -> str:
        return rs._sha({"tag": f"H{self.level}_STRUCT", "encoding": self.h_struct_encoding})

    @cached_property
    def construction_digest(self) -> str:
        return rs._sha({
            "level": self.level,
            "children": [c.construction_digest for c in self.children],
            "edges": [list(e[:4]) + [e[4], e[5]] for e in self.top_edges_full],
            "lane": self.lane,
            "motif": self.motif_id,
            "exact_carrier": True,
        })

    @cached_property
    def top_pairs(self) -> list[tuple[int, int]]:
        return [(int(e[0]), int(e[1])) for e in self.top_edges_full]

    @cached_property
    def typed_edges(self) -> list[tuple[int, int, int, int]]:
        return [(int(e[0]), int(e[1]), int(e[2]), int(e[3])) for e in self.top_edges_full]

    @cached_property
    def owner_caps(self) -> list[tuple[int, ...]]:
        return [tuple(c.total_caps) for c in self.children]

    @cached_property
    def total_caps(self) -> tuple[int, ...]:
        return tuple(sum(c.total_caps[t] for c in self.children) for t in range(7))

    @cached_property
    def owner_colors(self) -> list[str]:
        return [c.skin for c in self.children]

    @cached_property
    def leaf_count(self) -> int:
        return sum(c.leaf_count for c in self.children)

    @cached_property
    def relation_count_total(self) -> int:
        return sum(c.relation_count_total for c in self.children) + len(self.top_edges_full)

    def reservation_options(self, endpoint_type: int) -> list[ReservationOption]:
        t = int(endpoint_type)
        if self.total_caps[t] <= 0:
            return []
        pre = self.h_struct_canon
        raw = []
        for i, child in enumerate(self.children):
            if child.total_caps[t] <= 0:
                continue
            for opt in child.reservation_options(t):
                ch = list(self.children)
                ch[i] = opt.state
                post_state = ExactLiftState(self.engine, self.level, tuple(ch), self.top_edges_full, self.lane, self.motif_id)
                post = post_state.h_struct_canon
                token = rs._sha({"pre": pre, "post": post, "type": t})
                raw.append(ReservationOption(
                    post_state, t, token, int(opt.multiplicity), (i, opt.representative_key)
                ))
        merged: dict[tuple, ReservationOption] = {}
        for x in raw:
            key = (x.state.h_struct_canon, x.state.skin, x.witness_token)
            if key in merged:
                old = merged[key]
                merged[key] = ReservationOption(
                    old.state, t, old.witness_token, old.multiplicity + x.multiplicity,
                    min(old.representative_key, x.representative_key, key=repr),
                )
            else:
                merged[key] = x
        return sorted(merged.values(), key=lambda x: (x.state.h_struct_canon, x.state.skin, x.witness_token, repr(x.representative_key)))

    def reserve_canonical(self, endpoint_type: int) -> ReservationOption:
        """Choose one canonical nested reservation without enumerating all post-H states."""
        t = int(endpoint_type)
        if self.total_caps[t] <= 0:
            raise InsufficientExactCarrierData(f"O{self.level} no exact endpoint type {endpoint_type}")
        canon = self.canonical_owner_labels
        available = [i for i, child in enumerate(self.children) if int(child.total_caps[t]) > 0]
        if not available:
            raise InsufficientExactCarrierData(f"O{self.level} no exact endpoint type {endpoint_type}")
        i = min(available, key=lambda j: (int(canon[j]), self.children[j].h_struct_canon, self.children[j].skin, tuple(self.children[j].total_caps)))
        opt = self.children[i].reserve_canonical(t)
        ch = list(self.children)
        ch[i] = opt.state
        post_state = ExactLiftState(self.engine, self.level, tuple(ch), self.top_edges_full, self.lane, self.motif_id)
        token = rs._sha({"pre": self.h_struct_canon, "post": post_state.h_struct_canon, "type": t})
        return ReservationOption(post_state, t, token, int(opt.multiplicity), (int(canon[i]), opt.representative_key))


class ExactProjectedActionOracle:
    """Exact-before-projection relation-add oracle with a direct projected kernel.

    The scientific question is H -> admitted action -> R.  Routine profile computation
    therefore aggregates exact endpoint-orbit multiplicity directly into projected R
    successors and does *not* materialize every post-H carrier.  A slow materialized
    reference implementation remains available only for regression/targeted debugging.
    """

    def __init__(self, bridge_pairs: Iterable[tuple[int, int]]):
        self.bridge_pairs = tuple(sorted(tuple(map(int, x)) for x in bridge_pairs))
        self._reservation_cache: dict[tuple[str, int], list[ReservationOption]] = {}
        self._projected_reservation_cache: dict[tuple[str, int], Counter] = {}
        self._o7_owner_projection_cache: dict[tuple[str, int, int], Counter] = {}
        self._profile_cache: dict[str, Counter] = {}

    def reservation_options(self, state: Any, endpoint_type: int) -> list[ReservationOption]:
        """Materialized exact pointed options; reserved for reopen audits/separator debugging."""
        key = (state.h_struct_canon, int(endpoint_type))
        if key not in self._reservation_cache:
            self._reservation_cache[key] = state.reservation_options(int(endpoint_type))
        return self._reservation_cache[key]

    def _o7_owner_projected_counter(self, state: ExactO7State, owner: int, endpoint_type: int) -> Counter:
        """Projected O6-root successors for one O7 owner, preserving exact orbit multiplicity."""
        key = (state.h_struct_canon, int(owner), int(endpoint_type))
        if key in self._o7_owner_projection_cache:
            return self._o7_owner_projection_cache[key].copy()
        t = int(endpoint_type)
        root, byp = state.owner_orbit_data[int(owner)]
        groups = defaultdict(list)
        for path, (p, f, leaf, ps) in byp.items():
            if int(f[t]) > 0:
                groups[str(ps)].append((tuple(path), p, f, leaf))
        out = Counter()
        for _ps, rows in groups.items():
            _path, p, f, leaf = min(rows, key=lambda z: z[0])
            nf = list(f)
            nf[t] -= 1
            succ = state.engine.O6._successor_merkle_sig(root, {leaf: (p, tuple(nf))})
            out[succ] += len(rows)
        self._o7_owner_projection_cache[key] = out.copy()
        return out

    def projected_reservation_counter(self, state: Any, endpoint_type: int) -> Counter:
        """Return Counter(projected successor skin -> exact pointed-choice multiplicity).

        This recursively projects at the earliest safe point.  It is algebraically equal
        to enumerating ReservationOption post-H states and then forgetting H, but avoids
        constructing/canonicalizing those post-H states.
        """
        key = (state.h_struct_canon, int(endpoint_type))
        if key in self._projected_reservation_cache:
            return self._projected_reservation_cache[key].copy()
        t = int(endpoint_type)
        out = Counter()
        if int(state.total_caps[t]) <= 0:
            self._projected_reservation_cache[key] = out
            return out.copy()
        if isinstance(state, ExactO7State):
            roots = list(state.owner_roots)
            for owner in range(len(roots)):
                local = self._o7_owner_projected_counter(state, owner, t)
                for succ_root, mul in local.items():
                    rr = list(roots)
                    rr[owner] = succ_root
                    out[state.engine.O6._digest_container("R7", rr)] += int(mul)
        elif hasattr(state, "children"):
            skins = [c.skin for c in state.children]
            tag = f"R{state.level}"
            for i, child in enumerate(state.children):
                if int(child.total_caps[t]) <= 0:
                    continue
                for succ_skin, mul in self.projected_reservation_counter(child, t).items():
                    rr = list(skins)
                    rr[i] = succ_skin
                    out[state.engine.O6._digest_container(tag, rr)] += int(mul)
        else:
            # Generic exact leaf fixture / future adapter: materialize only this leaf's
            # local pointed options, then project immediately. Real O7+ runtime states
            # take the optimized branches above.
            for opt in self.reservation_options(state, t):
                out[opt.state.skin] += int(opt.multiplicity)
        self._projected_reservation_cache[key] = out.copy()
        return out

    def current_level_profile(self, state: Any) -> Counter:
        """Fast exact multiplicity profile using direct projected reservation Counters."""
        key = state.h_struct_canon
        if key in self._profile_cache:
            return self._profile_cache[key].copy()
        prof = Counter()
        n = len(state.owner_caps)
        tag = f"R{state.level}"
        if isinstance(state, ExactO7State):
            roots = list(state.owner_roots)
            # R7 relation-add requires two distinct O6 owners, so use owner-local kernels.
            for u in range(n):
                for v in range(u + 1, n):
                    for a, b in self.bridge_pairs:
                        left = self._o7_owner_projected_counter(state, u, a)
                        right = self._o7_owner_projected_counter(state, v, b)
                        if not left or not right:
                            continue
                        for su, ml in left.items():
                            for sv, mr in right.items():
                                rr = list(roots)
                                rr[u] = su
                                rr[v] = sv
                                r_succ = state.engine.O6._digest_container(tag, rr)
                                lab = (tag, min(a, b), max(a, b))
                                prof[(lab, r_succ)] += int(ml) * int(mr)
        else:
            skins = [c.skin for c in state.children]
            for u in range(n):
                for v in range(u + 1, n):
                    for a, b in self.bridge_pairs:
                        left = self.projected_reservation_counter(state.children[u], a)
                        right = self.projected_reservation_counter(state.children[v], b)
                        if not left or not right:
                            continue
                        for su, ml in left.items():
                            for sv, mr in right.items():
                                rr = list(skins)
                                rr[u] = su
                                rr[v] = sv
                                r_succ = state.engine.O6._digest_container(tag, rr)
                                lab = (tag, min(a, b), max(a, b))
                                prof[(lab, r_succ)] += int(ml) * int(mr)
        self._profile_cache[key] = prof.copy()
        return prof

    def current_level_profile_materialized_reference(self, state: Any) -> Counter:
        """Slow reference: materialize exact pointed post-H choices before projecting.

        Never use this for routine fixed-grammar scanning.  It exists to prove the fast
        kernel has the same scientific semantics on regression fixtures and small sentinels.
        """
        prof = Counter()
        n = len(state.owner_caps)
        tag = f"R{state.level}"
        if isinstance(state, ExactO7State):
            roots = list(state.owner_roots)
            for u in range(n):
                for v in range(u + 1, n):
                    for a, b in self.bridge_pairs:
                        left = self._o7_owner_projected_counter(state, u, a)
                        right = self._o7_owner_projected_counter(state, v, b)
                        for su, ml in left.items():
                            for sv, mr in right.items():
                                rr = list(roots)
                                rr[u] = su
                                rr[v] = sv
                                r_succ = state.engine.O6._digest_container(tag, rr)
                                lab = (tag, min(a, b), max(a, b))
                                prof[(lab, r_succ)] += int(ml) * int(mr)
        else:
            for u in range(n):
                for v in range(u + 1, n):
                    for a, b in self.bridge_pairs:
                        left = self.reservation_options(state.children[u], a)
                        right = self.reservation_options(state.children[v], b)
                        for lo in left:
                            for ro in right:
                                skins = [c.skin for c in state.children]
                                skins[u] = lo.state.skin
                                skins[v] = ro.state.skin
                                r_succ = state.engine.O6._digest_container(tag, skins)
                                lab = (tag, min(a, b), max(a, b))
                                prof[(lab, r_succ)] += int(lo.multiplicity) * int(ro.multiplicity)
        return prof

def _profile_digest(profile: Counter) -> str:
    rows = [(repr(k), int(v)) for k, v in sorted(profile.items(), key=lambda kv: repr(kv[0]))]
    return rs._sha(rows)


def _profile_summary(profile: Counter) -> dict:
    labels = Counter()
    successors = set()
    for (label, succ), count in profile.items():
        labels[repr(label)] += int(count)
        successors.add(succ)
    return {
        "profile_sha256": _profile_digest(profile),
        "action_successor_pairs": len(profile),
        "total_action_copies": int(sum(profile.values())),
        "enabled_labels": sorted(labels),
        "label_copy_counts": dict(sorted(labels.items())),
        "distinct_R_successors": len(successors),
    }


def compare_same_r_different_h(states: list[Any], oracle: ExactProjectedActionOracle) -> dict:
    by_r: dict[str, list[Any]] = defaultdict(list)
    for s in states:
        by_r[s.skin].append(s)
    collisions = []
    read_witnesses = []
    profiles: dict[str, Counter] = {}
    for r_skin, group in sorted(by_r.items()):
        by_h: dict[str, Any] = {}
        for s in group:
            by_h.setdefault(s.h_struct_canon, s)
        if len(by_h) < 2:
            continue
        reps = [by_h[k] for k in sorted(by_h)]
        rows = []
        for s in reps:
            p = oracle.current_level_profile(s)
            profiles[s.h_struct_canon] = p
            rows.append({
                "H_struct_sha256": s.h_struct_canon,
                "presentation_sha256": s.construction_digest,
                "motif_id": getattr(s, "motif_id", None),
                "lane": getattr(s, "lane", None),
                "profile": _profile_summary(p),
            })
        digests = {row["profile"]["profile_sha256"] for row in rows}
        collision = {
            "R_skin": r_skin,
            "distinct_H_structures": len(rows),
            "profile_classes": len(digests),
            "rows": rows,
        }
        collisions.append(collision)
        if len(digests) > 1:
            # Small exact separator: first action-successor Counter key whose
            # multiplicity differs between two exact H representatives.
            left, right = reps[0], None
            lp = profiles[left.h_struct_canon]
            for cand in reps[1:]:
                if profiles[cand.h_struct_canon] != lp:
                    right = cand; break
            rp = profiles[right.h_struct_canon]
            sep = None
            for k in sorted(set(lp) | set(rp), key=repr):
                if int(lp.get(k, 0)) != int(rp.get(k, 0)):
                    sep = {
                        "action_label": repr(k[0]),
                        "R_successor": k[1],
                        "left_multiplicity": int(lp.get(k, 0)),
                        "right_multiplicity": int(rp.get(k, 0)),
                    }
                    break
            read_witnesses.append({
                "R_skin": r_skin,
                "left_H": left.h_struct_canon,
                "right_H": right.h_struct_canon,
                "left_profile_sha256": _profile_digest(lp),
                "right_profile_sha256": _profile_digest(rp),
                "separator": sep,
            })
    if read_witnesses:
        classification = "R3_NATURAL_READ_FOUND"
    elif collisions:
        classification = "R_FACTORISATION_HOLDS_ON_PANEL"
    else:
        classification = "NO_SAME_R_DIFFERENT_H_WITNESS_IN_PANEL"
    return {
        "classification": classification,
        "same_R_different_H_groups": len(collisions),
        "natural_read_witnesses": read_witnesses,
        "collision_groups": collisions,
    }


def legacy_blind_read_trace() -> dict:
    return {
        "schema": "IG_LEGACY_BLIND_READ_TRACE_V1",
        "legacy_class": "regime_scanner.LiftState",
        "legality_reads": ["child.total_caps", "bridge/type support"],
        "witness_selection_reads": ["child.construction_digest"],
        "write_target": "recursively selected child then O7 aggregate reserve_counts overlay",
        "scientific_disposition": (
            "construction_digest is a hidden presentation/provenance read in the v0.26 sampling implementation; "
            "it is not by itself an earned operational topology read. The exact-carrier lane removes it from the "
            "scientific transition relation by enumerating exact endpoint orbits before R projection."
        ),
    }


class L2BoundaryAdapter:
    """Fail-closed hook for future reconstruction of the original L2 macro boundary T_X."""

    def __init__(self, mapper=None):
        self.mapper = mapper

    def capability(self) -> dict:
        if self.mapper is None:
            return {
                "status": "INSUFFICIENT_EXACT_L2_MAPPING",
                "reason": (
                    "The v0.26 O7+ compact carrier does not include an authoritative map from every O-carrier "
                    "occurrence back to the complete original L2 macro relation T_X. No mapping is synthesized."
                ),
            }
        return {"status": "AVAILABLE_BY_EXPLICIT_INJECTED_MAPPING"}

    def reconstruct(self, state: Any):
        if self.mapper is None:
            raise InsufficientExactL2Mapping(self.capability()["reason"])
        return self.mapper(state)


def _build_exact_lift(
    engine: Any,
    level: int,
    owners: list[Any],
    motif_edges: list[tuple[int, int]],
    pairs: list[tuple[int, int]],
    lane: str,
    motif_id: str,
    schedule_seed: int = 0,
    force_pair: tuple[int, int] | None = None,
) -> ExactLiftState | None:
    children = list(owners)
    full = []
    for idx, (u, v) in enumerate(sorted(tuple(sorted(map(int, e))) for e in motif_edges)):
        candidates = [force_pair] if force_pair is not None else [pairs[(idx + schedule_seed + j) % len(pairs)] for j in range(len(pairs))]
        chosen = None
        for ab in candidates:
            if ab is None:
                continue
            a, b = map(int, ab)
            if children[u].total_caps[a] > 0 and children[v].total_caps[b] > 0:
                chosen = (a, b); break
        if chosen is None:
            return None
        a, b = chosen
        try:
            lu = children[u].reserve_canonical(a)
            rv = children[v].reserve_canonical(b)
        except InsufficientExactCarrierData:
            return None
        children[u] = lu.state
        children[v] = rv.state
        full.append((u, v, a, b, lu.witness_token, rv.witness_token))
    return ExactLiftState(engine, level, tuple(children), tuple(full), lane, motif_id)


def _scientific_state_key(s: Any) -> tuple:
    return (str(s.h_struct_canon), str(s.skin), tuple(int(x) for x in s.total_caps), int(s.leaf_count), int(s.relation_count_total))

def _structural_farthest_select(states: list[Any], k: int, must_include: list[Any] | None = None) -> list[Any]:
    uniq: dict[tuple, Any] = {}
    for st in states: uniq.setdefault(_scientific_state_key(st), st)
    pool=[uniq[x] for x in sorted(uniq)]
    if len(pool)<=k: return pool
    vec=[rs._state_pre_features(st) for st in pool]; dims=len(vec[0])
    mins=[min(v[j] for v in vec) for j in range(dims)]; maxs=[max(v[j] for v in vec) for j in range(dims)]
    norm=[[(v[j]-mins[j])/(maxs[j]-mins[j]) if maxs[j]>mins[j] else 0.0 for j in range(dims)] for v in vec]
    idx={_scientific_state_key(st):i for i,st in enumerate(pool)}; chosen=[]
    for st in must_include or []:
        i=idx.get(_scientific_state_key(st))
        if i is not None and i not in chosen: chosen.append(i)
    if not chosen: chosen=[0]
    while len(chosen)<k:
        best=None
        for i,st in enumerate(pool):
            if i in chosen: continue
            dmin=min(sum((norm[i][j]-norm[c][j])**2 for j in range(dims))**0.5 for c in chosen)
            key=_scientific_state_key(st)
            if best is None or dmin>best[0]+1e-15 or (abs(dmin-best[0])<=1e-15 and key<best[2]): best=(dmin,i,key)
        chosen.append(best[1])
    return [pool[i] for i in chosen[:k]]

def _build_exact_next_panel(engine, prev: list[Any], level: int, pairs: list[tuple[int, int]], motifs: list[dict], spec: dict):
    prev = sorted(prev, key=_scientific_state_key)
    totals = [sum(s.total_caps) for s in prev]
    med = statistics.median(totals)
    center = min(prev, key=lambda s: (abs(sum(s.total_caps) - med), _scientific_state_key(s)))
    diverse = prev
    cand = []
    failures = 0
    twins = []
    for mi, m in enumerate(motifs):
        n = int(m["n"]); edges = [tuple(e) for e in m["edges"]]
        s = _build_exact_lift(engine, level, [center] * n, edges, pairs, "HOM", f"HOM:{n}:{mi}", schedule_seed=mi % len(pairs))
        if s: cand.append(s)
        else: failures += 1
        if n in (4, 5) or (n == 6 and mi % 6 == 0):
            owners = [diverse[j % len(diverse)] for j in range(n)]
            s = _build_exact_lift(engine, level, owners, edges, pairs, "HET", f"HET:{n}:{mi}", schedule_seed=(mi * 3 + 1) % len(pairs))
            if s: cand.append(s)
            else: failures += 1
        if n == 4 and len(diverse) >= 2:
            owners = [diverse[0], diverse[1], diverse[0], diverse[1]]
            s = _build_exact_lift(engine, level, owners, edges, pairs, "MIX", f"MIX:4:{mi}", schedule_seed=(mi * 5 + 2) % len(pairs))
            if s: cand.append(s)
            else: failures += 1
    if center.total_caps[0] >= 6:
        A = _build_exact_lift(engine, level, [center] * 6, list(rs.GA), pairs, "TWIN", "TWIN:A", force_pair=(0, 0))
        B = _build_exact_lift(engine, level, [center] * 6, list(rs.GB), pairs, "TWIN", "TWIN:B", force_pair=(0, 0))
        if A and B:
            cand.extend([A, B]); twins = [A, B]
    cap = int(spec["panel"]["candidate_cap"])
    if len(cand) > cap:
        cand = sorted(cand, key=_scientific_state_key)[:cap]
        for t in twins:
            if all(_scientific_state_key(x) != _scientific_state_key(t) for x in cand):
                cand[-1] = t
    selected = _structural_farthest_select(cand, int(spec["panel"]["beam"]), must_include=twins)
    return selected, {
        "candidates": len(cand), "build_failures": failures,
        "twin_constructed": len(twins) == 2, "selected": len(selected),
        "construction_policy": "CANONICAL_EXACT_POINTED_ORBIT_SAMPLE; ACTION_AUDIT_ENUMERATES_ALL_ORBITS",
    }


def run_exact_carrier_targeted_audit(
    phase8_seed: Path,
    output: Path,
    start_level: int = 7,
    through: int = 9,
    stop_on_read: bool = True,
    keep_work: bool = False,
    reopen_reason: str | None = None,
    allow_full_panel: bool = False,
) -> dict:
    """Expensive exact-carrier census reserved for an explicit theorem reopen event.

    Routine fixed-grammar factorisation must use theorem acceleration.  This path is
    deliberately fail-closed so an ordinary calibration cannot accidentally launch
    the former 193-candidate per-level census again.
    """
    if not reopen_reason:
        raise RuntimeError("targeted exact-carrier audit requires explicit reopen_reason")
    if not allow_full_panel:
        raise RuntimeError("full-panel exact-carrier audit disabled by default; set allow_full_panel=True after targeted audit is judged insufficient")
    spec = load_exact_carrier_unblinding_spec()
    scanner_spec = rs.load_regime_scanner_spec()
    motifs = rs.load_motif_library()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    start_level = int(start_level); through = int(through)
    if start_level < 7 or through < start_level:
        raise ValueError("require 7 <= start_level <= through")
    work = Path(tempfile.mkdtemp(prefix="ig_exact_h_", dir=str(output)))
    try:
        phase8, o7root = rs._extract_seed(Path(phase8_seed), work)
        import os
        os.environ["OSCOUT_DATA_ROOT"] = str(o7root.resolve())
        engine = rs._load_module("ig_exact_carrier_o7_engine", o7root / "02_CODE" / "o7_live_engine.py")
        engine.O6 = engine.import_o6()
        parent_map = engine.load_parent_records()
        _ports, bpairs = engine.O6.load_rules()
        bridge_pairs = sorted(tuple(map(int, x)) for x in bpairs)
        o8auth = json.loads((phase8 / "authority" / "O8_BP0_RESULT.json").read_text())
        o9auth = json.loads((phase8 / "authority" / "O9_BP0_RESULT.json").read_text())
        if o8auth.get("normalized_grammar", {}).get("candidate_sha256") != rs.GRAMMAR_EXPECTED or o9auth.get("normalized_grammar", {}).get("candidate_sha256") != rs.GRAMMAR_EXPECTED:
            raise RuntimeError("authority grammar hash mismatch")
        records = json.loads((phase8 / "graduation_compact" / "07_INPUT_SNAPSHOTS" / "O7_IMMUTABLE_SURVIVORS.json").read_text())["records"]
        base = []
        for r in records:
            ctx = engine._profile_row_context(r, parent_map)
            edges = tuple(tuple(x) for x in r["edges"])
            base.append(ExactO7State(engine, ctx, edges, r["state_digest"], r["lane"]))
        selected = _structural_farthest_select(base, int(scanner_spec["panel"]["beam"]))
        levels: dict[int, list[Any]] = {7: selected}
        build_meta = {7: {"source": "IMMUTABLE_O7_AUTHORITY", "selected": len(selected)}}
        level_results = {}
        prior_pass = True  # O6-and-below resource factorisation is an earned parent theorem.
        stop = None
        for level in range(7, through + 1):
            if level > 7:
                selected, bm = _build_exact_next_panel(engine, levels[level - 1], level, bridge_pairs, motifs, scanner_spec)
                levels[level] = selected; build_meta[level] = bm
            if level < start_level:
                continue
            oracle = ExactProjectedActionOracle(bridge_pairs)
            audit = compare_same_r_different_h(levels[level], oracle)
            audit.update({
                "level": level,
                "states": len(levels[level]),
                "distinct_R_skins": len({s.skin for s in levels[level]}),
                "distinct_H_structures": len({s.h_struct_canon for s in levels[level]}),
                "exactness_scope": "EXACT_O7_PLUS_OVER_COMPACT_O6_RESOURCE_CARRIER",
                "new_action_audited": f"R{level}_RELATION_ADD",
                "inherited_action_lane": (
                    "DISCHARGED_BY_PRIOR_FACTORISATION" if prior_pass else "NOT_DISCHARGED_AFTER_PRIOR_FAILURE"
                ),
                "build_meta": build_meta[level],
            })
            level_results[str(level)] = audit
            if audit["classification"] == "R3_NATURAL_READ_FOUND":
                prior_pass = False
                if stop_on_read:
                    stop = {"classification": "R3_NATURAL_READ_FOUND", "level": level, "reason": "same inherited R, different exact H, different exact-before-projection relation-add profile"}
                    break
            elif audit["classification"] == "R_FACTORISATION_HOLDS_ON_PANEL":
                prior_pass = prior_pass and True
            # No witness does not prove factorisation; it also does not invalidate an earned prior level.
        if stop is None:
            stop = {
                "classification": "UNRESOLVED_NO_NATURAL_READ_WITHIN_BOUNDED_PANEL",
                "through": max(map(int, level_results)) if level_results else through,
                "reason": "no exact same-R/different-H separator was found in the bounded exact-carrier panels; this is not a universal no-read theorem",
            }
        result = {
            "schema": "IG_O_EXACT_CARRIER_UNBLINDING_RESULT_V1",
            "date": "2026-08-30",
            "status": "PASS",
            "spec_sha256": rs._sha(spec),
            "phase8_seed_sha256": rs._sha_file(Path(phase8_seed)),
            "start_level": start_level,
            "requested_through": through,
            "levels": level_results,
            "stop": stop,
            "legacy_blind_read_trace": legacy_blind_read_trace(),
            "l2_boundary_adapter": L2BoundaryAdapter().capability(),
            "scientific_rule": "COMPUTE_LEGAL_ACTION_RELATION_ON_EXACT_H_FIRST_THEN_PROJECT_SUCCESSORS_TO_INHERITED_R",
            "nonclaims": spec["nonclaims"],
        }
        result["science_sha256"] = rs._sha({k: v for k, v in result.items() if k != "science_sha256"})
        (output / "O_EXACT_CARRIER_UNBLINDING_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    finally:
        if keep_work:
            (output / "WORKDIR.txt").write_text(str(work) + "\n", encoding="utf-8")
        else:
            shutil.rmtree(work, ignore_errors=True)

def run_exact_carrier_unblinding(
    phase8_seed: Path,
    output: Path,
    start_level: int = 7,
    through: int = 9,
    stop_on_read: bool = True,
    keep_work: bool = False,
    *,
    mode: str = "THEOREM_ACCELERATED",
    reopen_reason: str | None = None,
    allow_full_panel: bool = False,
) -> dict:
    """Public exact-carrier entry point; fast theorem transport is the default.

    THEOREM_ACCELERATED performs the O7 fixed exact sentinel plus O8/O9 theorem replay
    without constructing O8/O9 census panels. TARGETED_EXACT is available only after
    an explicit reopen event and requires allow_full_panel=True for the old broad path.
    """
    m = str(mode).upper()
    if m in {"THEOREM_ACCELERATED", "FAST", "AUTO"}:
        if int(start_level) != 7 or int(through) != 9:
            raise RuntimeError(
                "theorem-accelerated calibration currently covers the authoritative O7/O8/O9 anchor; "
                "future material O-levels must call theorem_transport_certificate with their material-instance hash"
            )
        from .exact_carrier_acceleration import run_theorem_accelerated_calibration
        return run_theorem_accelerated_calibration(
            Path(phase8_seed), Path(output), keep_work=keep_work
        )
    if m in {"TARGETED_EXACT", "FULL_PANEL_EXACT"}:
        return run_exact_carrier_targeted_audit(
            Path(phase8_seed), Path(output), start_level=start_level, through=through,
            stop_on_read=stop_on_read, keep_work=keep_work, reopen_reason=reopen_reason,
            allow_full_panel=allow_full_panel,
        )
    raise ValueError(f"unknown exact-carrier mode: {mode}")

