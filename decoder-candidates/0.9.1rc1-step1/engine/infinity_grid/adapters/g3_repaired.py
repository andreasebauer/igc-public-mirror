from __future__ import annotations

"""P5 adapter for the accepted corrected G3 public/hidden boundary."""

from typing import Any, Sequence

from ..science_adapter_contract import ScienceAdapterBase, ScienceAdapterDescriptor
from ..g4_term_state import G3TermState
from ..uplift_g3_r3 import (
    r3_spec,
    rebase_authority,
    hidden_state,
    rooted_full_incidence_read,
    binary_graft_state_from_parent_reads,
)
from ..canon import canonical_sha256


class G3RepairedAdapter(ScienceAdapterBase):
    DESCRIPTOR = ScienceAdapterDescriptor(
        adapter_id="v05.science.g3-repaired-r3",
        family="G3_CORRECTED",
        version="1.0.0",
        carrier_kind="G3_TERM_STATE_WITH_G2_UNITS_ATOMIC",
        public_descriptor="CAPS7",
        relation_semantics="FINITE_SET_OF_EXACT_RESERVATION_SUCCESSORS",
        capabilities=(
            "PUBLIC_READ",
            "RESTRICTED_HIDDEN_DIAGNOSTIC",
            "RESERVATION_RELATION",
            "CANONICALIZATION",
            "BINARY_GRAFT_HIDDEN_UPDATE",
            "COUNTEREXAMPLE_WITNESS",
            "SCOPE_AND_PROOF_OBLIGATIONS",
        ),
    )

    @staticmethod
    def public_read(state: G3TermState) -> dict[str, Any]:
        return {
            "schema_id": "IG_DECODER_V05_G3_PUBLIC_READ_V1",
            "descriptor": "CAPS7",
            "caps7": list(state.total_caps),
        }

    @staticmethod
    def hidden_diagnostic(n: int, edges: Sequence[Sequence[int]]) -> dict[str, Any]:
        return hidden_state(int(n), edges)

    @staticmethod
    def canonicalize(state: G3TermState) -> dict[str, Any]:
        return {
            "schema_id": "IG_DECODER_V05_G3_CANONICAL_STATE_V1",
            "term_ref": state.construction_digest,
            "topology_canon_sha256": state.topology_canon_sha256,
            "wire": state.to_wire(),
        }

    @staticmethod
    def reserve_relation(state: G3TermState, endpoint_type: int) -> tuple[dict[str, Any], ...]:
        # Return the complete exact successor set.  Empty is a lawful failed reservation relation.
        return tuple(s.to_wire() for s in state.reserve_external_relation(int(endpoint_type)))

    @staticmethod
    def binary_graft_hidden_update(
        left_n: int,
        left_edges: Sequence[Sequence[int]],
        left_root: int,
        right_n: int,
        right_edges: Sequence[Sequence[int]],
        right_root: int,
    ) -> dict[str, Any]:
        lr = rooted_full_incidence_read(left_n, left_edges, left_root)
        rr = rooted_full_incidence_read(right_n, right_edges, right_root)
        return binary_graft_state_from_parent_reads(lr, rr)

    @staticmethod
    def same_public_distinct_hidden_witness() -> dict[str, Any]:
        seed = (4, 1, 0, 0, 0, 0, 0)
        path_typed = ((0, 1, 0, 0), (1, 2, 0, 0), (2, 3, 0, 0))
        star_typed = ((0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0))
        path = G3TermState.from_homogeneous_seed(seed, path_typed, unit_count=4)
        star = G3TermState.from_homogeneous_seed(seed, star_typed, unit_count=4)
        hp = hidden_state(4, [(0, 1), (1, 2), (2, 3)])
        hs = hidden_state(4, [(0, 1), (0, 2), (0, 3)])
        out = {
            "schema_id": "IG_DECODER_V05_G3_SAME_PUBLIC_DISTINCT_H_WITNESS_V1",
            "same_caps7": path.total_caps == star.total_caps,
            "caps7": list(path.total_caps),
            "path_H_science_sha256": hp["science_sha256"],
            "star_H_science_sha256": hs["science_sha256"],
            "hidden_H_distinct": hp["science_sha256"] != hs["science_sha256"],
            "public_descriptor_promoted_hidden_H": False,
        }
        out["science_sha256"] = canonical_sha256(out)
        return out

    def scope_contract(self) -> dict[str, Any]:
        sp = r3_spec(); auth = rebase_authority()
        return {
            "schema_id": "IG_DECODER_V05_G3_REPAIRED_ADAPTER_SCOPE_V1",
            "public_descriptor": "CAPS7",
            "hidden_diagnostic": "SHELL_PROFILE_MULTISET_PLUS_PAIRED_SUPPORT_LOAD_DISTANCE_SIGNATURE_MULTISET",
            "r3_spec_science_sha256": sp["science_sha256"],
            "authority_science_sha256": auth["science_sha256"],
            "hidden_state_promoted": False,
            "construction_identity_public": False,
            "binary_graft_read_minimality_claim": False,
            "promotion": False,
        }
