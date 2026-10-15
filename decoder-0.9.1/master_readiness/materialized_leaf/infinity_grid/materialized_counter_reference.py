"""Independent site-occurrence census and scanner-to-exact carrier mapping.

Qualification support only. One choice is one available site/type occurrence,
not one capacity slot and not one orbit. No theorem or production authority.
"""
from collections import Counter

from .exact_carrier_unblinding import (
    ExactO7State, ExactLiftState, InsufficientExactCarrierData, _decrement_site,
)
from . import regime_scanner as rs


def o7_owner_reference(state, owner, endpoint_type):
    """Materialize every site decrement and rebuild its complete O6 root.

    Does not use the direct Merkle successor kernel, orbit representatives,
    reservation_options, or the oracle caches. Equal projections accumulate.
    """
    t = int(endpoint_type)
    if not 0 <= t < 7:
        raise ValueError("endpoint type outside 0..6")
    h = state.owner_h6s[int(owner)]
    _root, _entries, _groups, byp = state.engine.site_orbits_current(h, None)
    result = Counter()
    for path, (_p, f, _leaf, _pointed) in byp.items():
        if int(f[t]) > 0:
            post_h = _decrement_site(h, tuple(path), t)
            post_root, _e, _g, _b = state.engine.site_orbits_current(post_h, None)
            result[post_root.base_sig] += 1
    return result


def reservation_reference(state, endpoint_type):
    t = int(endpoint_type)
    if not 0 <= t < 7:
        raise ValueError("endpoint type outside 0..6")
    result = Counter()
    if isinstance(state, ExactO7State):
        roots = list(state.owner_roots)
        choices = ((i, o7_owner_reference(state, i, t)) for i in range(len(roots)))
    elif isinstance(state, ExactLiftState):
        roots = [c.skin for c in state.children]
        choices = ((i, reservation_reference(c, t)) for i, c in enumerate(state.children))
    else:
        raise InsufficientExactCarrierData("reference requires mapped exact O7/LIFT carrier")
    for i, counter in choices:
        for successor, multiplicity in counter.items():
            post = list(roots)
            post[i] = successor
            result[state.engine.O6._digest_container(f"R{state.level}", post)] += multiplicity
    return result


def profile_reference(state, bridge_pairs):
    """Independent full labelled Counter, including distinct owner occurrences.

    Bridge pairs retain their registered orientation; labels use the existing
    unordered endpoint-type pair. Repeated supplied Bridge entries count twice,
    matching the existing oracle; callers must bind their Bridge source.
    """
    if isinstance(state, ExactO7State):
        roots = list(state.owner_roots)
        local = o7_owner_reference
    elif isinstance(state, ExactLiftState):
        roots = [c.skin for c in state.children]
        local = lambda s, i, t: reservation_reference(s.children[i], t)
    else:
        raise InsufficientExactCarrierData("reference requires mapped exact O7/LIFT carrier")
    prof = Counter()
    # This cache is confined to this exact presentation and this reference call.
    counters = {(i, int(t)): local(state, i, t)
                for i in range(len(roots)) for t in {x for p in bridge_pairs for x in p}}
    for u in range(len(roots)):
        for v in range(u + 1, len(roots)):
            for a, b in bridge_pairs:
                for su, ml in counters[u, int(a)].items():
                    for sv, mr in counters[v, int(b)].items():
                        post = list(roots)
                        post[u], post[v] = su, sv
                        key = ((f"R{state.level}", min(a, b), max(a, b)),
                               state.engine.O6._digest_container(f"R{state.level}", post))
                        prof[key] += ml * mr
    return prof


class SavedScannerActionMapper:
    """Preserve saved coordinates/overlays while exposing the complete action domain.

    This is a new qualification view, not an assertion that reserve_external
    enumerates the action Counter. Historical construction digests stay external
    and unchanged. Scanner top-edge witnesses are retained verbatim as provenance;
    they are not promoted to exact-action witness tokens or transport authority.
    """
    def __init__(self):
        self._memo = {}
        self._sources = {}  # strong references prevent object-id reuse
        self.checked = 0
        self.overlay_sites = 0

    def map(self, state):
        key = id(state)
        if key in self._memo:
            return self._memo[key]
        self._sources[key] = state
        if isinstance(state, rs.O7State):
            counts = tuple(state.reserve_counts)
            if len(counts) != 7 or any(type(c) is not int or c < 0 for c in counts):
                raise InsufficientExactCarrierData("invalid saved O7 reserve counts")
            # Reconstruct the scanner's exact type-major, owner/path-major overlay.
            base = [state.engine.materialize_owner(p.h6, state.edges, i)
                    for i, p in enumerate(state.ctx.parents)]
            tables = [state.engine.site_orbits_current(h, None)[3] for h in base]
            overlay = []
            for t, count in enumerate(counts):
                remaining = count
                for owner, byp in enumerate(tables):
                    for path, (_p, f, _leaf, _ps) in sorted(byp.items(), key=lambda x: tuple(x[0])):
                        take = min(remaining, int(f[t]))
                        if take:
                            overlay.append((owner, tuple(path), t, take))
                            remaining -= take
                if remaining:
                    raise InsufficientExactCarrierData("saved O7 overlay exceeds capacity")
            mapped = ExactO7State(state.engine, state.ctx, state.edges, state.source_id,
                                  state.lane, tuple(overlay))
            self.overlay_sites += len(overlay)
        elif isinstance(state, rs.LiftState):
            if state.level < 8 or any(c.level != state.level - 1 for c in state.children):
                raise InsufficientExactCarrierData("non-adjacent saved LIFT levels")
            mapped = ExactLiftState(state.engine, state.level,
                                    tuple(self.map(c) for c in state.children),
                                    state.top_edges_full, state.lane, state.motif_id)
        else:
            raise InsufficientExactCarrierData("unsupported saved scanner state")
        if (mapped.skin != state.skin or mapped.owner_caps != state.owner_caps
                or mapped.total_caps != state.total_caps
                or mapped.typed_edges != state.typed_edges):
            raise InsufficientExactCarrierData("scanner-to-exact projection mismatch")
        self._memo[key] = mapped
        self.checked += 1
        return mapped
