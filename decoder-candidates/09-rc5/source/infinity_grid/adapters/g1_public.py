from __future__ import annotations

"""P5 adapter for the frozen G1:R100 public one-endpoint interface family."""

from typing import Any, Iterable, Sequence

from ..science_adapter_contract import ScienceAdapterBase, ScienceAdapterDescriptor
from ..uplift_structural import (
    extract_carrier_interface,
    extract_interface_population,
    pair_connection_record,
    implementation_spec,
)
from ..canon import canonical_sha256


class G1PublicAdapter(ScienceAdapterBase):
    DESCRIPTOR = ScienceAdapterDescriptor(
        adapter_id="v05.science.g1-public-r100",
        family="G1_PUBLIC_INTERFACE",
        version="1.0.0",
        carrier_kind="COMPLETE_G1_R100_CARRIER",
        public_descriptor="G1_PUBLIC_ONE_ENDPOINT_INTERFACE_SEMANTICS_V1",
        relation_semantics="COMPLETE_TYPED_PAIR_ATTEMPT_RELATION_WITH_FAILED_CONNECTIONS_RETAINED",
        capabilities=(
            "PUBLIC_READ",
            "ONE_ENDPOINT_RESERVATION",
            "PAIR_CONNECTION_RELATION",
            "CANONICALIZATION",
            "FAILURE_RECORDS",
            "SCOPE_AND_PROOF_OBLIGATIONS",
        ),
    )

    def extract(self, carrier: Any, *, source_authority_sha256: str) -> dict[str, Any]:
        return extract_carrier_interface(
            carrier,
            source_stage="G1:R100",
            source_authority_sha256=source_authority_sha256,
        )

    def population(self, carriers: Iterable[Any], *, source_authority_sha256: str) -> dict[str, Any]:
        return extract_interface_population(
            carriers,
            source_stage="G1:R100",
            source_authority_sha256=source_authority_sha256,
        )

    def pair_relation(
        self,
        left_interface: dict[str, Any],
        right_interface: dict[str, Any],
        *,
        operators: Sequence[Sequence[int]],
        bridge_pairs: Sequence[Sequence[int]],
        source_authority_sha256: str,
    ) -> tuple[dict[str, Any], ...]:
        rows = []
        for op in operators:
            a, b = map(int, op)
            rows.append(
                pair_connection_record(
                    left_interface,
                    right_interface,
                    a,
                    b,
                    bridge_pairs=bridge_pairs,
                    source_authority_sha256=source_authority_sha256,
                )
            )
        # Preserve every attempt, including illegal/failed attempts.  Do not choose a representative.
        return tuple(rows)

    def canonical_relation_sha256(self, rows: Sequence[dict[str, Any]]) -> str:
        return canonical_sha256(list(rows))

    def scope_contract(self) -> dict[str, Any]:
        spec = implementation_spec()
        return {
            "schema_id": "IG_DECODER_V05_G1_PUBLIC_ADAPTER_SCOPE_V1",
            "source_stage": "G1:R100",
            "implementation_spec_sha256": spec["spec_sha256"],
            "hidden_reads": False,
            "failed_connections_retained": True,
            "representative_selection": False,
            "contraction_order_statement": (
                "Any inherited observer-level contraction/rebracketing equality is kept at its declared "
                "quotient scope; this adapter does not assert associative binary connected composition."
            ),
            "promotion": False,
        }
