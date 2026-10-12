from __future__ import annotations

"""v0.5 P5 science-adapter contract.

Adapters expose existing mathematical semantics through a uniform capability boundary.
They do not alter the underlying scientific implementation, select representatives from
relation-valued outcomes, or acquire graduation authority.
"""

from dataclasses import dataclass
from typing import Any, Mapping


class ScienceAdapterError(RuntimeError):
    pass


class UnsupportedScienceCapability(ScienceAdapterError):
    pass


@dataclass(frozen=True)
class ScienceAdapterDescriptor:
    adapter_id: str
    family: str
    version: str
    carrier_kind: str
    public_descriptor: str
    relation_semantics: str
    capabilities: tuple[str, ...]
    authority_effect: str = "NONE_P5"

    def to_wire(self) -> dict[str, Any]:
        return {
            "schema_id": "IG_DECODER_V05_SCIENCE_ADAPTER_DESCRIPTOR_V1",
            "adapter_id": self.adapter_id,
            "family": self.family,
            "version": self.version,
            "carrier_kind": self.carrier_kind,
            "public_descriptor": self.public_descriptor,
            "relation_semantics": self.relation_semantics,
            "capabilities": list(self.capabilities),
            "authority_effect": self.authority_effect,
        }


class ScienceAdapterBase:
    DESCRIPTOR: ScienceAdapterDescriptor

    def descriptor(self) -> dict[str, Any]:
        return self.DESCRIPTOR.to_wire()

    def require_capability(self, capability: str) -> None:
        if capability not in self.DESCRIPTOR.capabilities:
            raise UnsupportedScienceCapability(
                f"{self.DESCRIPTOR.adapter_id} does not support capability {capability}"
            )

    def scope_contract(self) -> Mapping[str, Any]:
        raise NotImplementedError
