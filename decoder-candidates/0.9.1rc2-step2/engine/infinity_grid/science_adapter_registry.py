from __future__ import annotations

"""v0.5 P5 bounded science-adapter registry.

This is deliberately small and static for P5.  It provides one supported lookup
point so migrated mathematical families are not selected by ad-hoc imports.
Unknown adapters fail early.  The registry has no experiment or graduation authority.
"""

from importlib.resources import files
import json
from typing import Any

from .canon import canonical_sha256
from .science_adapter_contract import ScienceAdapterBase, UnsupportedScienceCapability
from .adapters.g1_public import G1PublicAdapter
from .adapters.g3_repaired import G3RepairedAdapter
from .adapters.g4_accepted import G4AcceptedAdapter

_CONTRACT_RESOURCE = "resources/v05/P5_SCIENCE_ADAPTER_COMPATIBILITY_CONTRACT_V1.json"

_ADAPTERS = {
    G1PublicAdapter.DESCRIPTOR.adapter_id: G1PublicAdapter,
    G3RepairedAdapter.DESCRIPTOR.adapter_id: G3RepairedAdapter,
    G4AcceptedAdapter.DESCRIPTOR.adapter_id: G4AcceptedAdapter,
}

P5_SEQUENTIAL_ADAPTER_IDS = (
    G1PublicAdapter.DESCRIPTOR.adapter_id,
    G3RepairedAdapter.DESCRIPTOR.adapter_id,
    G4AcceptedAdapter.DESCRIPTOR.adapter_id,
)


def load_p5_adapter_contract() -> dict[str, Any]:
    obj = json.loads(files("infinity_grid").joinpath(_CONTRACT_RESOURCE).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_DECODER_V05_P5_SCIENCE_ADAPTER_COMPATIBILITY_CONTRACT_V1":
        raise RuntimeError("bad P5 adapter compatibility contract schema")
    expected = str(obj.get("science_sha256", ""))
    observed = canonical_sha256({k: v for k, v in obj.items() if k != "science_sha256"})
    if expected != observed:
        raise RuntimeError("P5 adapter compatibility contract hash mismatch")
    if tuple(obj.get("sequential_adapter_ids", [])) != P5_SEQUENTIAL_ADAPTER_IDS:
        raise RuntimeError("P5 adapter compatibility contract order mismatch")
    return obj


def list_science_adapters() -> tuple[dict[str, Any], ...]:
    load_p5_adapter_contract()
    return tuple(_ADAPTERS[k].DESCRIPTOR.to_wire() for k in P5_SEQUENTIAL_ADAPTER_IDS)


def get_science_adapter(adapter_id: str) -> ScienceAdapterBase:
    load_p5_adapter_contract()
    cls = _ADAPTERS.get(str(adapter_id))
    if cls is None:
        raise UnsupportedScienceCapability(f"unregistered v0.5 science adapter {adapter_id}")
    return cls()
