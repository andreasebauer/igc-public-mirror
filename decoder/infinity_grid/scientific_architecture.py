from __future__ import annotations

"""Generic scientific protocol layer for Decoder Architecture Spec v1.

This module is deliberately thin.  It does not generate algebraic states and it does
not alter legacy Scout/OScout/O-regime science.  It provides the typed contracts that
bind a meaningful EntityClass + Regime to read-only TestSpecs/TestPacks and typed
ObservationViews.
"""

from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping
import json

from .canon import canonical_sha256
from .schema import validate


class ScientificArchitectureError(RuntimeError):
    pass


class UndeclaredCapabilityRead(ScientificArchitectureError):
    pass


_RESOURCE_ROOT = "resources/science_architecture"
_SCHEMA_FILES = {
    "IG_ENTITY_CLASS_V1": "IG_ENTITY_CLASS_V1.schema.json",
    "IG_REGIME_V1": "IG_REGIME_V1.schema.json",
    "IG_CAPABILITY_PROVIDER_V1": "IG_CAPABILITY_PROVIDER_V1.schema.json",
    "IG_TEST_SPEC_V1": "IG_TEST_SPEC_V1.schema.json",
    "IG_TEST_PACK_V1": "IG_TEST_PACK_V1.schema.json",
    "IG_OBSERVATION_VIEW_V1": "IG_OBSERVATION_VIEW_V1.schema.json",
    "IG_FINDING_V1": "IG_FINDING_V1.schema.json",
    "IG_MATURATION_RECORD_V1": "IG_MATURATION_RECORD_V1.schema.json",
    "IG_PLATEAU_CERTIFICATE_V1": "IG_PLATEAU_CERTIFICATE_V1.schema.json",
    "IG_LIFT_CANDIDATE_V1": "IG_LIFT_CANDIDATE_V1.schema.json",
    "IG_LIFT_CERTIFICATE_V1": "IG_LIFT_CERTIFICATE_V1.schema.json",
    "IG_RESEARCH_FRONTIER_V1": "IG_RESEARCH_FRONTIER_V1.schema.json",
    "IG_RUN_PLAN_V2": "IG_RUN_PLAN_V2.schema.json",
    "IG_ARCHITECTURE_CONTRACT_V2": "IG_ARCHITECTURE_CONTRACT_V2.schema.json",
    "IG_SCIENTIFIC_AUDIT_EVIDENCE_V1": "IG_SCIENTIFIC_AUDIT_EVIDENCE_V1.schema.json",
    "IG_LIFT_CERTIFICATION_INPUT_V1": "IG_LIFT_CERTIFICATION_INPUT_V1.schema.json",
    "IG_PLATEAU_AUDIT_RESULT_V1": "IG_PLATEAU_AUDIT_RESULT_V1.schema.json",
    "IG_LIFT_AUDIT_RESULT_V1": "IG_LIFT_AUDIT_RESULT_V1.schema.json",
}


def _root():
    return files("infinity_grid").joinpath(_RESOURCE_ROOT)


def _load_json(rel: str) -> dict[str, Any]:
    return json.loads(_root().joinpath(rel).read_text(encoding="utf-8"))


def _schema(schema_id: str) -> dict[str, Any]:
    try:
        name = _SCHEMA_FILES[schema_id]
    except KeyError as exc:
        raise ScientificArchitectureError(f"unknown scientific architecture schema {schema_id}") from exc
    return _load_json(f"schemas/{name}")


def validate_scientific_artifact(obj: Mapping[str, Any]) -> str:
    sid = obj.get("schema_id")
    if not isinstance(sid, str):
        raise ScientificArchitectureError("scientific artifact missing schema_id")
    if sid not in _SCHEMA_FILES:
        raise ScientificArchitectureError(f"unsupported scientific architecture schema {sid}")
    validate(_schema(sid), dict(obj))
    return sid


def make_ref(identity: str, version: str) -> str:
    return f"{identity}@{version}"


def split_ref(ref: str) -> tuple[str, str]:
    if not isinstance(ref, str) or "@" not in ref:
        raise ScientificArchitectureError(f"versioned ref required: {ref!r}")
    identity, version = ref.rsplit("@", 1)
    if not identity or not version:
        raise ScientificArchitectureError(f"bad versioned ref: {ref!r}")
    return identity, version


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(v) for v in value)
    return value


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: thaw_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(v) for v in value]
    return value


class ScientificProtocolRegistry:
    """Frozen registry for EntityClass/Regime/Test/TestPack compatibility specs.

    The initial implementation loads only the O7 compatibility example from Architecture
    Spec v1.  The registry itself is generic and versioned; future EntityClasses/Regimes
    can be added as resources without changing the runner contract.
    """

    def __init__(self) -> None:
        self.capability_vocabulary = _load_json("registries/CAPABILITY_VOCABULARY_V1.json")
        self.invariants = _load_json("registries/SCIENTIFIC_ARCHITECTURE_INVARIANTS_V1.json")
        self.finding_status_vocabulary = _load_json("registries/FINDING_STATUS_VOCABULARY_V1.json")
        self._capabilities = {x["id"]: x for x in self.capability_vocabulary["capabilities"]}

        entity = _load_json("examples/O7_ENTITY_CLASS_COMPAT.json")
        regime = _load_json("examples/O7_REGIME_COMPAT.json")
        pack = _load_json("examples/O7_MATURATION_TEST_PACK_COMPAT.json")
        tests = []
        eroot = _root().joinpath("examples")
        for p in sorted(eroot.iterdir(), key=lambda x: x.name):
            if p.name.startswith("test_") and p.name.endswith(".json"):
                tests.append(json.loads(p.read_text(encoding="utf-8")))

        for obj in [entity, regime, pack, *tests]:
            validate_scientific_artifact(obj)

        self._entities = {make_ref(entity["entity_class_id"], entity["version"]): entity}
        self._regimes = {make_ref(regime["regime_id"], regime["version"]): regime}
        self._tests = {make_ref(t["test_id"], t["version"]): t for t in tests}
        self._packs = {make_ref(pack["pack_id"], pack["version"]): pack}
        self._verify_cross_contracts()

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(self._capabilities)

    def entity(self, ref: str) -> dict[str, Any]:
        try:
            return self._entities[ref]
        except KeyError as exc:
            raise ScientificArchitectureError(f"unknown EntityClass {ref}") from exc

    def regime(self, ref: str) -> dict[str, Any]:
        try:
            return self._regimes[ref]
        except KeyError as exc:
            raise ScientificArchitectureError(f"unknown Regime {ref}") from exc

    def test(self, ref: str) -> dict[str, Any]:
        try:
            return self._tests[ref]
        except KeyError as exc:
            raise ScientificArchitectureError(f"unknown Test {ref}") from exc

    def pack(self, ref: str) -> dict[str, Any]:
        try:
            return self._packs[ref]
        except KeyError as exc:
            raise ScientificArchitectureError(f"unknown TestPack {ref}") from exc

    def tests(self) -> list[dict[str, Any]]:
        return [self._tests[k] for k in sorted(self._tests)]

    def packs(self) -> list[dict[str, Any]]:
        return [self._packs[k] for k in sorted(self._packs)]

    def _verify_cross_contracts(self) -> None:
        if len(self._tests) != 12:
            raise ScientificArchitectureError(f"O7 compatibility pack expects 12 Tests, found {len(self._tests)}")
        if len(self._capabilities) != 15:
            raise ScientificArchitectureError(f"Architecture Spec v1 capability vocabulary expected 15 entries, found {len(self._capabilities)}")

        for eref, ent in self._entities.items():
            for cref in ent.get("capability_provider_refs", []):
                if not isinstance(cref, str) or not cref:
                    raise ScientificArchitectureError(f"{eref}: bad capability provider ref")

        for rref, reg in self._regimes.items():
            ent_ref = reg["input_entity_class_ref"]
            if ent_ref not in self._entities:
                raise ScientificArchitectureError(f"{rref}: missing input EntityClass {ent_ref}")
            unknown = set(reg["exposed_capabilities"]) - set(self._capabilities)
            if unknown:
                raise ScientificArchitectureError(f"{rref}: unknown exposed capabilities {sorted(unknown)}")

        for tref, test in self._tests.items():
            if test.get("read_only") is not True:
                raise ScientificArchitectureError(f"{tref}: Test must be read_only")
            unknown = set(test["required_capabilities"]) - set(self._capabilities)
            if unknown:
                raise ScientificArchitectureError(f"{tref}: unknown capabilities {sorted(unknown)}")
            for eref in test["admissibility"]["entity_class_refs"]:
                if eref not in self._entities:
                    raise ScientificArchitectureError(f"{tref}: unknown admissible EntityClass {eref}")
            for rref in test["admissibility"]["regime_refs"]:
                reg = self.regime(rref)
                missing = set(test["required_capabilities"]) - set(reg["exposed_capabilities"])
                if missing:
                    raise ScientificArchitectureError(f"{tref}: Regime {rref} does not expose {sorted(missing)}")
            for dep in test.get("dependency_test_refs", []):
                if dep not in self._tests:
                    raise ScientificArchitectureError(f"{tref}: unknown dependency Test {dep}")
            if test["input_mode"] == "FINDING_DEPENDENCIES" and not test.get("dependency_test_refs"):
                raise ScientificArchitectureError(f"{tref}: finding dependency Test has no dependencies")

        for pref, pack in self._packs.items():
            members = [m["test_ref"] for m in pack["members"]]
            if len(members) != len(set(members)):
                raise ScientificArchitectureError(f"{pref}: duplicate TestPack member")
            if set(members) != set(self._tests):
                raise ScientificArchitectureError(f"{pref}: O7 compatibility pack membership does not equal the frozen 12-Test set")
            for a, b in pack["dependency_edges"]:
                if a not in members or b not in members:
                    raise ScientificArchitectureError(f"{pref}: dependency edge references nonmember")
            self._assert_acyclic(members, pack["dependency_edges"], pref)
            if pack.get("feedback_mode") != "RECOGNITION_ONLY":
                raise ScientificArchitectureError(f"{pref}: TestPack feedback must be recognition-only")

    @staticmethod
    def _assert_acyclic(nodes: Iterable[str], edges: Iterable[Iterable[str]], label: str) -> None:
        nodes = list(nodes)
        outgoing = {n: [] for n in nodes}
        indegree = {n: 0 for n in nodes}
        for raw in edges:
            edge = list(raw)
            if len(edge) != 2:
                raise ScientificArchitectureError(f"{label}: dependency edge must have 2 endpoints")
            a, b = edge
            outgoing[a].append(b)
            indegree[b] += 1
        ready = sorted(n for n, d in indegree.items() if d == 0)
        seen = 0
        while ready:
            n = ready.pop(0)
            seen += 1
            for m in sorted(outgoing[n]):
                indegree[m] -= 1
                if indegree[m] == 0:
                    ready.append(m)
                    ready.sort()
        if seen != len(nodes):
            raise ScientificArchitectureError(f"{label}: TestPack dependency graph contains a cycle")

    def verification_result(self) -> dict[str, Any]:
        payload = {
            "schema_id": "IG_SCIENTIFIC_PROTOCOL_REGISTRY_VERIFICATION_V1",
            "schema_version": "1.0.0",
            "status": "PASS",
            "entity_classes": len(self._entities),
            "regimes": len(self._regimes),
            "tests": len(self._tests),
            "test_packs": len(self._packs),
            "capabilities": len(self._capabilities),
            "all_tests_read_only": all(t["read_only"] is True for t in self._tests.values()),
            "legacy_science_routing_changed": False,
            "architecture_spec_bundle_sha256": _load_json("provenance/ARCHITECTURE_SPEC_BUNDLE_PIN.json")["architecture_spec_bundle_sha256"],
        }
        payload["science_sha256"] = canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class CapabilityBinding:
    capability_id: str
    provider_ref: str
    artifact_ref: str
    science_sha256: str


class ObservationView:
    """Read-only Test view that enforces declared capability access.

    Only the capability IDs listed by the bound TestSpec are placed in the runtime
    view.  Attempting to read anything else fails closed.
    """

    def __init__(
        self,
        *,
        entity_instance_ref: str,
        entity_class_ref: str,
        regime_ref: str,
        regime_depth: int | str,
        test_ref: str,
        required_capabilities: Iterable[str],
        capability_payloads: Mapping[str, Any],
        source_science_sha256: str,
        provider_ref: str,
    ) -> None:
        required = tuple(sorted(set(required_capabilities)))
        missing = set(required) - set(capability_payloads)
        if missing:
            raise ScientificArchitectureError(f"ObservationView missing required capabilities {sorted(missing)}")
        self._allowed = frozenset(required)
        self._data = MappingProxyType({k: _deep_freeze(capability_payloads[k]) for k in required})
        self._bindings = {
            k: CapabilityBinding(
                capability_id=k,
                provider_ref=provider_ref,
                artifact_ref=f"memory:{entity_instance_ref}:{k}",
                science_sha256=canonical_sha256(thaw_json(self._data[k])),
            )
            for k in required
        }
        self.entity_instance_ref = entity_instance_ref
        self.entity_class_ref = entity_class_ref
        self.regime_ref = regime_ref
        self.regime_depth = regime_depth
        self.test_ref = test_ref
        self.source_science_sha256 = source_science_sha256

    @property
    def allowed_capabilities(self) -> frozenset[str]:
        return self._allowed

    def read(self, capability_id: str) -> Any:
        if capability_id not in self._allowed:
            raise UndeclaredCapabilityRead(
                f"Test {self.test_ref} attempted undeclared capability read {capability_id}"
            )
        return self._data[capability_id]

    def artifact(self) -> dict[str, Any]:
        binding_map = {
            k: {
                "provider_ref": b.provider_ref,
                "artifact_ref": b.artifact_ref,
                "science_sha256": b.science_sha256,
            }
            for k, b in sorted(self._bindings.items())
        }
        # view_id is the semantic identity of the immutable observation, not merely its
        # declared scope. Provider and per-capability science bindings are therefore part
        # of the preimage. Two views with different evidence can no longer collide.
        view_preimage = {
            "entity_instance_ref": self.entity_instance_ref,
            "entity_class_ref": self.entity_class_ref,
            "regime_ref": self.regime_ref,
            "regime_depth": self.regime_depth,
            "test_ref": self.test_ref,
            "capability_bindings": binding_map,
            "source_science_sha256": self.source_science_sha256,
            "hidden_reads_forbidden": True,
        }
        obj = {
            "schema_id": "IG_OBSERVATION_VIEW_V1",
            "schema_version": "1.0.0",
            "view_id": canonical_sha256(view_preimage),
            "entity_instance_ref": self.entity_instance_ref,
            "entity_class_ref": self.entity_class_ref,
            "regime_ref": self.regime_ref,
            "regime_depth": self.regime_depth,
            "test_ref": self.test_ref,
            "capability_bindings": binding_map,
            "source_science_sha256": self.source_science_sha256,
            "hidden_reads_forbidden": True,
        }
        validate_scientific_artifact(obj)
        return obj


@dataclass(frozen=True)
class TestExecution:
    test_ref: str
    evidence_payload: Any
    evidence_sha256: str
    finding: Mapping[str, Any]
    observation_view: Mapping[str, Any] | None


def build_finding(
    *,
    test_spec: Mapping[str, Any],
    entity_instance_ref: str,
    regime_ref: str,
    regime_depth: int | str,
    source_science_sha256: str,
    evidence_payload: Any,
    scientific_status: str,
    summary: str,
    historical_labels: Iterable[str] = (),
    operational_status: str = "PASS",
    reopen_triggered: bool = False,
) -> dict[str, Any]:
    tref = make_ref(str(test_spec["test_id"]), str(test_spec["version"]))
    ev_sha = canonical_sha256(evidence_payload)
    obj = {
        "schema_id": "IG_FINDING_V1",
        "schema_version": "1.0.0",
        "finding_id": canonical_sha256({
            "test_ref": tref,
            "entity_instance_ref": entity_instance_ref,
            "regime_ref": regime_ref,
            "regime_depth": regime_depth,
            "source_science_sha256": source_science_sha256,
            "evidence_sha256": ev_sha,
        }),
        "test_ref": tref,
        "entity_instance_ref": entity_instance_ref,
        "regime_ref": regime_ref,
        "regime_depth": regime_depth,
        "operational_status": operational_status,
        "scientific_status": scientific_status,
        "evidence_class": test_spec["scope_class"],
        "claim_ceiling": test_spec["claim_ceiling"],
        "summary": summary,
        "evidence_refs": [{
            "id": f"memory-evidence:{ev_sha}",
            "version": "1",
            "role": "test evidence",
            "scope": f"{tref} at regime depth {regime_depth}",
            "sha256": ev_sha,
        }],
        "source_science_sha256": source_science_sha256,
        "historical_labels": list(historical_labels),
        "reopen_triggered": bool(reopen_triggered),
    }
    validate_scientific_artifact(obj)
    # Fail closed if a Finding attempts to overstate its Test ceiling.
    rank = {"RECONNAISSANCE": 0, "BOUNDED_EXACT": 1, "CONSTRUCTION_AUDIT": 2, "THEOREM_TRANSPORT": 3, "FORMAL_THEOREM": 4}
    if rank[obj["evidence_class"]] > rank[obj["claim_ceiling"]]:
        raise ScientificArchitectureError("Finding evidence class exceeds Test claim ceiling")
    return obj


TestAdapter = Callable[[ObservationView], Any]
