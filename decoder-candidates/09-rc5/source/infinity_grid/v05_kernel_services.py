from __future__ import annotations

"""O3A semantic kernel-service boundary.

This module is intentionally policy/infrastructure only.  O3A does not move any
scientific semantics.  It makes the existing coverage machine-readable, binds an
execution-local restricted view, and prevents new evaluator code from adding a
second implementation of a kernel-covered operation.

O3B migrates all registered production evaluators to this service surface.
No production evaluator retains a semantic grandfathering exemption.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import ast
import contextvars
import hashlib

KERNEL_SERVICE_CATALOG_SCHEMA = "IG_DECODER_KERNEL_SERVICE_CATALOG_V1"
EVALUATOR_SPEC_SCHEMA = "IG_DECODER_EVALUATOR_SPEC_V1"
SEMANTIC_GATE_SCHEMA = "IG_DECODER_SEMANTIC_DUPLICATION_GATE_V1"
KERNEL_VIEW_SCHEMA = "IG_DECODER_RESTRICTED_KERNEL_VIEW_V1"

PRIMARY = "PRIMARY"
REFERENCE_ORACLE = "REFERENCE_ORACLE"
KERNEL_TEST = "KERNEL_TEST"
VALID_ROLES = frozenset({PRIMARY, REFERENCE_ORACLE, KERNEL_TEST})

PUBLIC = "PUBLIC"
KERNEL_INTERNAL = "KERNEL_INTERNAL"
REFERENCE_ONLY = "REFERENCE_ONLY"

# Public names are deliberately small.  Tests ask for scientific operations,
# never implementation machinery.
KERNEL_SERVICE_CATALOG: dict[str, dict[str, Any]] = {
    "AUTHORITY_BASIS": {"visibility": PUBLIC, "meaning": "Frozen exact G6 basis carrier snapshot."},
    "OPERATOR_BASIS": {"visibility": PUBLIC, "meaning": "Frozen G6 operator basis."},
    "PUBLIC_READ": {"visibility": PUBLIC, "meaning": "Frozen inherited G5 public resource read D=(f,m)."},
    "EXACT_IDENTITY": {"visibility": PUBLIC, "meaning": "Exact structural identity under the bound authority."},
    "EXACT_RELATION": {"visibility": PUBLIC, "meaning": "Exact relation-valued composition with set semantics."},
    "EXACT_RELATION_PROFILE": {"visibility": PUBLIC, "meaning": "Exact relation outcome count and operational profile without child-carrier materialization."},
    "EXACT_RELATION_PROFILE_BATCH": {"visibility": PUBLIC, "meaning": "Ordered exact relation profiles for many operators on one prepared left/right pair; execution-only vectorization of EXACT_RELATION_PROFILE."},
    "EXACT_RELATION_PROFILE_FAMILY": {"visibility": PUBLIC, "meaning": "Ordered exact relation profiles for many right carriers and operators sharing one left carrier; exact compact structural transform with conservative fallback."},
    "RELATION_ENABLED": {"visibility": PUBLIC, "meaning": "Exact enabledness without retaining children."},
    "FIRST_EXACT_CHILD": {"visibility": PUBLIC, "meaning": "Deterministic first exact child under canonical ordering."},
    "OBSERVER_Q": {"visibility": PUBLIC, "meaning": "Exact frozen observer state q(X)."},
    "OBSERVER_DECODE": {"visibility": PUBLIC, "meaning": "Exact q-state decoding with verified fast path and exact fallback."},
    "OBSERVER_WRITE": {"visibility": PUBLIC, "meaning": "Exact relation-valued observer-state write law."},
    "MARKER_Q": {"visibility": PUBLIC, "meaning": "Exact fresh-D-marker relation state q_D(X)."},
    "MARKER_DECODE": {"visibility": PUBLIC, "meaning": "Exact q_D decoding by unique-D deletion."},
    "MARKER_WRITE": {"visibility": PUBLIC, "meaning": "Exact relation-valued q_D-state write law."},
    "ATTACHMENT_RESPONSE_DESCRIPTOR": {"visibility": PUBLIC, "meaning": "Exact anonymous typed attachment-response descriptor."},
    "PREPARE_TREE": {"visibility": KERNEL_INTERNAL, "meaning": "Prepared tree/cavity/root data."},
    "RECONSTRUCT_FROM_ROOTED_CANON": {"visibility": KERNEL_INTERNAL, "meaning": "Exact representative reconstruction from rooted canon."},
    "EDGE_CUT_PROFILE": {"visibility": KERNEL_INTERNAL, "meaning": "Exact edge split/cut metadata."},
    "COMPONENT_CANONS": {"visibility": KERNEL_INTERNAL, "meaning": "Exact component canons after an edge cut."},
    "GRAFT_CANON": {"visibility": KERNEL_INTERNAL, "meaning": "Exact graft canonicalization primitive."},
    "OWNER_ORBIT_REDUCTION": {"visibility": KERNEL_INTERNAL, "meaning": "Exact owner orbit reduction."},
    "LEGACY_EXACT_RELATION_ORACLE": {"visibility": REFERENCE_ONLY, "meaning": "Independent historical exact-relation oracle."},
    "LEGACY_OBSERVER_DECODE_ORACLE": {"visibility": REFERENCE_ONLY, "meaning": "Independent historical observer-decode oracle."},
    "LEGACY_MARKER_DECODE_ORACLE": {"visibility": REFERENCE_ONLY, "meaning": "Independent fresh-marker unique-D deletion oracle."},
    "INDEPENDENT_CANON_ORACLE": {"visibility": REFERENCE_ONLY, "meaning": "Independent canonicalization oracle."},
}

PUBLIC_KERNEL_SERVICES = frozenset(k for k,v in KERNEL_SERVICE_CATALOG.items() if v["visibility"] == PUBLIC)
REFERENCE_KERNEL_SERVICES = frozenset(k for k,v in KERNEL_SERVICE_CATALOG.items() if v["visibility"] == REFERENCE_ONLY)


class KernelServiceViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluatorSpec:
    ref: str
    role: str
    allowed_kernel_services: tuple[str, ...]
    reference_oracle_for: tuple[str, ...] = ()
    legacy_semantic_source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or ":" not in self.ref:
            raise ValueError("EVALUATOR_SPEC_REF")
        if self.role not in VALID_ROLES:
            raise ValueError("EVALUATOR_SPEC_ROLE")
        if len(self.allowed_kernel_services) != len(set(self.allowed_kernel_services)):
            raise ValueError("EVALUATOR_SPEC_DUPLICATE_SERVICE")
        for service in self.allowed_kernel_services:
            if service not in KERNEL_SERVICE_CATALOG:
                raise ValueError("EVALUATOR_SPEC_UNKNOWN_SERVICE:" + service)
            visibility = KERNEL_SERVICE_CATALOG[service]["visibility"]
            if self.role == PRIMARY and visibility != PUBLIC:
                raise ValueError("EVALUATOR_SPEC_PRIMARY_NONPUBLIC:" + service)
            if self.role == REFERENCE_ORACLE and visibility == KERNEL_INTERNAL:
                raise ValueError("EVALUATOR_SPEC_REFERENCE_INTERNAL:" + service)
        for oracle in self.reference_oracle_for:
            if oracle not in REFERENCE_KERNEL_SERVICES:
                raise ValueError("EVALUATOR_SPEC_REFERENCE_TARGET:" + oracle)
        if self.role == REFERENCE_ORACLE:
            ref_allowed = set(self.allowed_kernel_services) & set(REFERENCE_KERNEL_SERVICES)
            if not ref_allowed <= set(self.reference_oracle_for):
                raise ValueError("EVALUATOR_SPEC_REFERENCE_SERVICE_NOT_DECLARED")
        sha = self.legacy_semantic_source_sha256
        if sha is not None and (len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("EVALUATOR_SPEC_LEGACY_SHA")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_id": EVALUATOR_SPEC_SCHEMA,
            "ref": self.ref,
            "role": self.role,
            "allowed_kernel_services": list(self.allowed_kernel_services),
            "reference_oracle_for": list(self.reference_oracle_for),
            "legacy_semantic_source_sha256": self.legacy_semantic_source_sha256,
        }


class KernelView:
    """Execution-local authorization view for one registered evaluator.

    O3B binds only declared providers into this execution-local view.
    Undeclared or unbound services always fail closed.
    """
    __slots__ = ("evaluator_ref", "role", "_allowed", "_providers")

    def __init__(self, spec: EvaluatorSpec, providers: Mapping[str, Callable[..., Any]] | None = None) -> None:
        self.evaluator_ref = spec.ref
        self.role = spec.role
        self._allowed = frozenset(spec.allowed_kernel_services)
        self._providers = dict(providers or {})
        unknown = set(self._providers) - self._allowed
        if unknown:
            raise KernelServiceViolation("KERNEL_VIEW_PROVIDER_UNDECLARED:" + sorted(unknown)[0])

    @property
    def allowed_services(self) -> tuple[str, ...]:
        return tuple(sorted(self._allowed))

    def allows(self, service_id: str) -> bool:
        return service_id in self._allowed

    def require(self, service_id: str) -> None:
        if service_id not in KERNEL_SERVICE_CATALOG:
            raise KernelServiceViolation("KERNEL_SERVICE_UNKNOWN:" + str(service_id))
        if service_id not in self._allowed:
            raise KernelServiceViolation("KERNEL_SERVICE_NOT_DECLARED:" + str(service_id))

    def call(self, service_id: str, *args: Any, **kwargs: Any) -> Any:
        self.require(service_id)
        provider = self._providers.get(service_id)
        if provider is None:
            raise KernelServiceViolation("KERNEL_SERVICE_PROVIDER_NOT_BOUND:" + service_id)
        return provider(*args, **kwargs)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_id": KERNEL_VIEW_SCHEMA,
            "evaluator_ref": self.evaluator_ref,
            "role": self.role,
            "allowed_kernel_services": list(self.allowed_services),
            "bound_provider_services": sorted(self._providers),
        }


_CURRENT_KERNEL_VIEW: contextvars.ContextVar[KernelView | None] = contextvars.ContextVar(
    "IG_DECODER_CURRENT_KERNEL_VIEW", default=None
)


def bind_kernel_view(spec: EvaluatorSpec, providers: Mapping[str, Callable[..., Any]] | None = None) -> KernelView:
    view = KernelView(spec, providers)
    _CURRENT_KERNEL_VIEW.set(view)
    return view


def clear_kernel_view() -> None:
    _CURRENT_KERNEL_VIEW.set(None)


def current_kernel_view() -> KernelView:
    view = _CURRENT_KERNEL_VIEW.get()
    if view is None:
        raise KernelServiceViolation("KERNEL_VIEW_NOT_BOUND")
    return view


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _normalized_import(node: ast.ImportFrom) -> str:
    module = node.module or ""
    if node.level:
        return "infinity_grid" + (("." + module) if module else "")
    return module


# (module, imported symbol) -> already-covered kernel service / guidance.
# This is intentionally explicit rather than broad substring matching, so the
# rejection tells a test author exactly what Decoder already owns.
_COVERED_SYMBOLS: dict[tuple[str, str], tuple[str, str]] = {
    ("infinity_grid.exact_tree_relation_kernel", "get_relation_kernel"): ("EXACT_RELATION", "use the restricted KernelView instead of obtaining the kernel object"),
    ("infinity_grid.exact_tree_relation_kernel", "configure_relation_kernel"): ("PREPARE_TREE", "kernel lifecycle is Decoder-runtime owned"),
    ("infinity_grid.exact_tree_relation_kernel", "tree_from_record"): ("PREPARE_TREE", "exact carrier decode/validation is kernel covered"),
    ("infinity_grid.exact_tree_relation_kernel", "ExactTreeRelationKernel"): ("EXACT_RELATION", "kernel construction is Decoder-runtime owned"),
    ("infinity_grid.adapters.g4_accepted", "G4AcceptedAdapter"): ("EXACT_IDENTITY", "exact identity is kernel covered"),
    ("infinity_grid.g6_s1_repair", "_basis"): ("AUTHORITY_BASIS", "use AUTHORITY_BASIS"),
    ("infinity_grid.g6_stage_executors", "_basis"): ("AUTHORITY_BASIS", "use AUTHORITY_BASIS"),
    ("infinity_grid.g6_s5r_crw_kernel", "exact_observer_state"): ("OBSERVER_Q", "use OBSERVER_Q"),
    ("infinity_grid.g6_s5r_crw_kernel", "decode_exact_observer_state"): ("OBSERVER_DECODE", "use OBSERVER_DECODE"),
    ("infinity_grid.g6_s5r_crw_kernel", "write_exact_observer_states"): ("OBSERVER_WRITE", "use OBSERVER_WRITE"),
    ("infinity_grid.g6_s5r_crw_kernel", "attachment_response_descriptor"): ("ATTACHMENT_RESPONSE_DESCRIPTOR", "use ATTACHMENT_RESPONSE_DESCRIPTOR"),
    ("infinity_grid.g6_s5r_crw_kernel", "tree_from_rooted_canon"): ("RECONSTRUCT_FROM_ROOTED_CANON", "rooted-canon reconstruction is kernel-internal"),
    ("infinity_grid.g6_s5r_crw_kernel", "_legacy_parent_candidates_from_observer_canons"): ("LEGACY_OBSERVER_DECODE_ORACLE", "legacy inversion is reference-oracle only"),
}

_COVERED_MODULES = frozenset(m for m,_s in _COVERED_SYMBOLS)

_FORBIDDEN_DIRECT_MODULES: dict[str, tuple[str,str]] = {
    "infinity_grid.v05_kernel_service_providers": ("MULTIPLE", "providers are runtime/stage infrastructure; evaluators use current_kernel_view only"),
    "infinity_grid.g6_reference_oracles": ("LEGACY_OBSERVER_DECODE_ORACLE", "reference algorithms are available only through a REFERENCE_ONLY KernelView service"),
}


def audit_evaluator_semantics(module_path: str | Path, spec: EvaluatorSpec) -> dict[str, Any]:
    path = Path(module_path).resolve(strict=True)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _COVERED_MODULES or any(alias.name.startswith(m + '.') for m in _COVERED_MODULES) or alias.name in _FORBIDDEN_DIRECT_MODULES:
                    violations.append({
                        "line": int(node.lineno), "kind": "DIRECT_KERNEL_SEMANTIC_MODULE_IMPORT",
                        "module": alias.name, "symbol": "*", "covered_service": _FORBIDDEN_DIRECT_MODULES.get(alias.name,("MULTIPLE", ""))[0],
                        "guidance": _FORBIDDEN_DIRECT_MODULES.get(alias.name,("MULTIPLE", "use only services declared in the evaluator KernelView"))[1] or "use only services declared in the evaluator KernelView",
                    })
        elif isinstance(node, ast.ImportFrom):
            module = _normalized_import(node)
            for alias in node.names:
                if module in _FORBIDDEN_DIRECT_MODULES:
                    service,guidance=_FORBIDDEN_DIRECT_MODULES[module]
                    violations.append({
                        "line": int(node.lineno), "kind": "DIRECT_KERNEL_PROVIDER_OR_REFERENCE_IMPORT",
                        "module": module, "symbol": alias.name, "covered_service": service, "guidance": guidance,
                    })
                    continue
                row = _COVERED_SYMBOLS.get((module, alias.name))
                if row is not None:
                    service, guidance = row
                    violations.append({
                        "line": int(node.lineno), "kind": "KERNEL_COVERED_SEMANTIC_IMPORT",
                        "module": module, "symbol": alias.name, "covered_service": service,
                        "guidance": guidance,
                    })
                elif module in _COVERED_MODULES and alias.name == "*":
                    violations.append({
                        "line": int(node.lineno), "kind": "DIRECT_KERNEL_SEMANTIC_MODULE_IMPORT",
                        "module": module, "symbol": "*", "covered_service": "MULTIPLE",
                        "guidance": "use only services declared in the evaluator KernelView",
                    })
    violations.sort(key=lambda x: (x["line"], x["module"], x["symbol"]))
    source_sha = _sha_file(path)
    if not violations:
        status = "PASS"
        migration_required = False
    elif spec.legacy_semantic_source_sha256 == source_sha:
        # Exact-byte grandfathering is the O3A bridge only.  It cannot authorize
        # edits or new duplicate implementations, and O3B must remove it.
        status = "PASS_LEGACY_FROZEN"
        migration_required = True
    else:
        status = "FAIL"
        migration_required = True
    return {
        "schema_id": SEMANTIC_GATE_SCHEMA,
        "status": status,
        "evaluator_ref": spec.ref,
        "role": spec.role,
        "module_path": str(path),
        "module_source_sha256": source_sha,
        "legacy_semantic_source_sha256": spec.legacy_semantic_source_sha256,
        "allowed_kernel_services": list(spec.allowed_kernel_services),
        "reference_oracle_for": list(spec.reference_oracle_for),
        "migration_required": migration_required,
        "violation_count": len(violations),
        "violations": violations,
    }


def require_evaluator_semantics(module_path: str | Path, spec: EvaluatorSpec) -> dict[str, Any]:
    result = audit_evaluator_semantics(module_path, spec)
    if result["status"] not in {"PASS", "PASS_LEGACY_FROZEN"}:
        first = result["violations"][:5]
        raise KernelServiceViolation(
            "DUPLICATE_KERNEL_SEMANTICS:" + spec.ref + ":" + repr(first)
        )
    return result


def service_catalog_snapshot() -> dict[str, Any]:
    return {
        "schema_id": KERNEL_SERVICE_CATALOG_SCHEMA,
        "services": [
            {"id": k, **KERNEL_SERVICE_CATALOG[k]}
            for k in sorted(KERNEL_SERVICE_CATALOG)
        ],
    }
