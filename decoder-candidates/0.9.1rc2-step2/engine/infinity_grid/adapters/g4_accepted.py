from __future__ import annotations

"""P5 adapter for the accepted G4 CAPS7 + repaired-H-class-bag branch."""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from ..science_adapter_contract import ScienceAdapterBase, ScienceAdapterDescriptor, UnsupportedScienceCapability
from ..decorated_tree_messages import graft_fixed_leaf
from ..typed_tree_messages import (
    endpoint_rooted_truncated_cavity_message,
    endpoint_rooted_tree_canon,
    endpoint_unrooted_tree_canon,
)
from ..uplift_g4_r7_rebase import spec as r7_spec, _candidate_rows, _resource_realizability_certificate, _fresh_n7_observer_challenge
from ..canon import canonical_sha256


@dataclass(frozen=True)
class DecoratedG4Tree:
    n: int
    edges: tuple[tuple[int, int], ...]
    H_classes: tuple[str, ...]
    edge_operators: tuple[tuple[int, int], ...]


class G4AcceptedAdapter(ScienceAdapterBase):
    DESCRIPTOR = ScienceAdapterDescriptor(
        adapter_id="v05.science.g4-accepted-r7",
        family="G4_ACCEPTED",
        version="1.1.0",
        carrier_kind="FINITE_DECORATED_G3_UNIT_TREE",
        public_descriptor="CAPS7_PLUS_H_CLASS_BAG",
        relation_semantics="EXACT_FINITE_HIDDEN_TREE_OUTCOMES_WITH_PUBLIC_QUOTIENT",
        capabilities=(
            "PUBLIC_READ",
            "EXACT_HIDDEN_CANONICALIZATION",
            "TRUNCATED_CAVITY_ACTION_READ",
            "FIXED_LEAF_GRAFT_RELATION",
            "REALIZABILITY_WITNESS",
            "FROZEN_OPERATOR_BASIS",
            "COMPRESSION_CORRECTION",
            "SCOPE_AND_PROOF_OBLIGATIONS",
        ),
    )

    def __init__(self) -> None:
        self._rows = _candidate_rows()
        s6 = __import__("infinity_grid.uplift_g4_r7_rebase", fromlist=["_load_embedded_authority"])._load_embedded_authority()[1]
        self._ops = {tuple(map(int, x)) for x in s6["frozen_grammar"]["operator_basis"]}

    def _vertex_key(self, klass: str) -> str:
        if klass not in self._rows:
            raise UnsupportedScienceCapability(f"unsupported repaired-H class {klass}")
        return str(self._rows[klass]["candidate_key_sha256"])

    def vertex_key(self, klass: str) -> str:
        """Stable exact vertex-decoration key for execution kernels."""
        return self._vertex_key(klass)

    def class_caps7(self, klass: str) -> tuple[int, ...]:
        """Frozen CAPS7 row for one repaired-H class; read-only authority surface."""
        if klass not in self._rows:
            raise UnsupportedScienceCapability(f"unsupported repaired-H class {klass}")
        return tuple(map(int, self._rows[klass]["caps7"]))

    def relation_class_table(self) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
        """Immutable frozen class values for Decoder execution preparation.

        No scientific read or canonicalization rule changes. The engine snapshots
        these values once instead of reconstructing the adapter per relation.
        """
        return tuple((k, self.vertex_key(k), self.class_caps7(k)) for k in sorted(self._rows))

    def _validate(self, tree: DecoratedG4Tree) -> None:
        if int(tree.n) != len(tree.H_classes):
            raise ValueError("G4 tree H-class arity mismatch")
        if len(tree.edges) != len(tree.edge_operators):
            raise ValueError("G4 tree edge/operator arity mismatch")
        # The canonicalizer performs the exact finite-tree validation.
        endpoint_unrooted_tree_canon(
            tree.n,
            tree.edges,
            [self._vertex_key(x) for x in tree.H_classes],
            [list(x) for x in tree.edge_operators],
        )
        for op in tree.edge_operators:
            if tuple(op) not in self._ops:
                raise UnsupportedScienceCapability(f"operator {tuple(op)} is outside frozen G4 basis")

    def operator_basis(self) -> tuple[tuple[int, int], ...]:
        """Exact frozen directed G4 operator basis, sorted; read-only authority surface for G5 pair census."""
        return tuple(sorted(self._ops))

    def public_read(self, tree: DecoratedG4Tree) -> dict[str, Any]:
        self._validate(tree)
        local = [list(map(int, self._rows[k]["caps7"])) for k in tree.H_classes]
        for (u, v), (a, b) in zip(tree.edges, tree.edge_operators):
            if local[u][a] <= 0 or local[v][b] <= 0:
                return {
                    "schema_id": "IG_DECODER_V05_G4_PUBLIC_READ_V1",
                    "legal": False,
                    "caps7": None,
                    "H_class_bag": dict(sorted(Counter(tree.H_classes).items())),
                    "failure": "RESOURCE_UNAVAILABLE",
                }
            local[u][a] -= 1; local[v][b] -= 1
        total = [sum(row[t] for row in local) for t in range(7)]
        return {
            "schema_id": "IG_DECODER_V05_G4_PUBLIC_READ_V1",
            "legal": True,
            "descriptor": "CAPS7_PLUS_H_CLASS_BAG",
            "caps7": total,
            "H_class_bag": dict(sorted(Counter(tree.H_classes).items())),
        }

    def unrooted_canon(self, tree: DecoratedG4Tree) -> tuple[Any, ...]:
        self._validate(tree)
        return endpoint_unrooted_tree_canon(tree.n, tree.edges, [self._vertex_key(x) for x in tree.H_classes], [list(x) for x in tree.edge_operators])

    def rooted_canon(self, tree: DecoratedG4Tree, root: int) -> tuple[Any, ...]:
        self._validate(tree)
        return endpoint_rooted_tree_canon(tree.n, tree.edges, [self._vertex_key(x) for x in tree.H_classes], [list(x) for x in tree.edge_operators], int(root))

    def action_read(self, tree: DecoratedG4Tree, root: int, depth: int) -> tuple[Any, ...]:
        self._validate(tree)
        return endpoint_rooted_truncated_cavity_message(tree.n, tree.edges, [self._vertex_key(x) for x in tree.H_classes], [list(x) for x in tree.edge_operators], int(root), int(depth))

    def graft_relation(self, tree: DecoratedG4Tree, root: int, *, new_H_class: str, operator: Sequence[int]) -> tuple[DecoratedG4Tree, ...]:
        self._validate(tree)
        op = tuple(map(int, operator))
        if op not in self._ops:
            raise UnsupportedScienceCapability(f"operator {op} is outside frozen G4 basis")
        # Resource check is exact at the explicit attachment owner and new vertex.
        local = [list(map(int, self._rows[k]["caps7"])) for k in tree.H_classes]
        for (u, v), (a, b) in zip(tree.edges, tree.edge_operators):
            local[u][a] -= 1; local[v][b] -= 1
        new_caps = list(map(int, self._rows[new_H_class]["caps7"])) if new_H_class in self._rows else None
        if new_caps is None:
            raise UnsupportedScienceCapability(f"unsupported repaired-H class {new_H_class}")
        if local[int(root)][op[0]] <= 0 or new_caps[op[1]] <= 0:
            return tuple()
        n, edges, _labels, _grades = graft_fixed_leaf(
            tree.n, tree.edges, tree.H_classes, tree.edge_operators, int(root),
            new_vertex_label=new_H_class, new_edge_label=op,
        )
        return (DecoratedG4Tree(n, tuple(edges), tuple(tree.H_classes) + (new_H_class,), tuple(tree.edge_operators) + (op,)),)

    @staticmethod
    def realizability_witness(depth: int = 2) -> dict[str, Any]:
        cert = _resource_realizability_certificate(int(depth))
        return cert["support_rows"][int(depth)]

    @staticmethod
    def compression_correction() -> dict[str, Any]:
        fresh = _fresh_n7_observer_challenge()
        d5 = next(r for r in fresh["depth_rows"] if int(r["depth"]) == 5)
        return {
            "schema_id": "IG_DECODER_V05_G4_COMPRESSION_CORRECTION_V1",
            "r4_r5_p6_depth5_signature_classes": 6144,
            "r4_r5_p6_exact_rooted_cases": 6144,
            "r4_r5_nontrivial_compression_earned": False,
            "fresh_n7_raw_action_instances": fresh["raw_action_instances"],
            "fresh_n7_exact_rooted_action_cases": fresh["exact_rooted_action_cases_after_dedup"],
            "fresh_n7_exact_child_canons": fresh["exact_child_canon_count"],
            "fresh_n7_first_predictive_depth": 5,
            "fresh_n7_depth5_classes": d5["action_read_class_count"],
            "fresh_n7_depth5_nonpredictive_classes": d5["nonpredictive_class_count"],
            "fresh_n7_nontrivial_compression_earned": False,
            "global_minimality_claim": False,
        }

    def scope_contract(self) -> dict[str, Any]:
        sp = r7_spec()
        return {
            "schema_id": "IG_DECODER_V05_G4_ACCEPTED_ADAPTER_SCOPE_V1",
            "r7_spec_science_sha256": sp["science_sha256"],
            "Q": "PUBLIC_G4_DESCRIPTOR_CAPS7_PLUS_H_CLASS_BAG",
            "A": "R6_NONBACKTRACKING_CAVITY_MESSAGE_AT_DECLARED_DEPTH",
            "O": "EXACT_UNROOTED_DECORATED_CHILD_CANON",
            "predicate": "PARENT_CONDITIONED_ONE_STEP_PREDICTION",
            "actual_g4_realizability_bridge_required": True,
            "abstract_tree_transfer_without_bridge_forbidden": True,
            "topology_promoted": False,
            "physical_geometry_claim": False,
            "recursive_future_claim": False,
            "global_minimality_claim": False,
            "promotion": False,
        }
