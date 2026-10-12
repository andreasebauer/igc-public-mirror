from __future__ import annotations
import ast, importlib, json
from pathlib import Path
from typing import Any

class P7SupportError(RuntimeError):
    pass

ROOT = Path(__file__).resolve().parent
RES = ROOT / "resources"
V05 = RES / "v05"


def _load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def approved_skip_ledger() -> dict[str, Any]:
    obj = _load(V05 / "P7_APPROVED_SKIP_LEDGER_V1.json")
    if obj.get("schema_id") != "IG_DECODER_V05_P7_APPROVED_SKIP_LEDGER_V1":
        raise P7SupportError("bad P7 skip-ledger schema")
    ids = [x.get("test_id") for x in obj.get("entries", [])]
    if len(ids) != len(set(ids)) or any(not x for x in ids):
        raise P7SupportError("P7 skip ledger IDs must be unique nonempty strings")
    return obj


def load_uplift_registry() -> dict[str, Any]:
    from .uplift_campaign import load_uplift_experiment_registry
    return load_uplift_experiment_registry()


def load_legacy_scientific_registry() -> dict[str, Any]:
    return _load(RES / "decoder" / "SCIENTIFIC_TEST_REGISTRY_v1.json")


def resolve_handler_ref(ref: str) -> bool:
    mod, sep, name = ref.partition(":")
    if not sep or not mod or not name:
        return False
    try:
        obj = getattr(importlib.import_module(mod), name)
    except Exception:
        return False
    return callable(obj)


def scan_pytest_source(source_root: Path) -> dict[str, Any]:
    tests_root = Path(source_root) / "tests"
    rows=[]; bare=[]; self_equal=[]; skip_calls=[]
    for p in sorted(tests_root.glob("test_*.py")):
        try:
            tree=ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except SyntaxError as e:
            raise P7SupportError(f"syntax error in {p}: {e}") from e
        rel=p.relative_to(source_root).as_posix()
        for node in tree.body:
            if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
                continue
            tid=f"{rel}::{node.name}"
            has_bare=False; explicit=[]; same=[]
            for n in ast.walk(node):
                if isinstance(n,ast.Return) and n.value is None:
                    has_bare=True
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=="pytest" and n.func.attr=="skip":
                    val=None
                    if n.args and isinstance(n.args[0],ast.Constant): val=n.args[0].value
                    explicit.append(val)
                if isinstance(n,ast.Assert) and isinstance(n.test,ast.Compare) and len(n.test.ops)==1 and isinstance(n.test.ops[0],ast.Eq) and len(n.test.comparators)==1:
                    if ast.dump(n.test.left,include_attributes=False)==ast.dump(n.test.comparators[0],include_attributes=False):
                        same.append(getattr(n,"lineno",None))
            row={"test_id":tid,"file":rel,"line":node.lineno,"bare_return":has_bare,"explicit_skip_calls":explicit,"self_equality_assert_lines":same}
            rows.append(row)
            if has_bare: bare.append(tid)
            if explicit: skip_calls.append({"test_id":tid,"reasons":explicit})
            if same: self_equal.append({"test_id":tid,"lines":same})
    return {"schema_id":"IG_DECODER_V05_P7_PYTEST_SOURCE_SCAN_V1","tests":rows,"test_count":len(rows),"bare_return_tests":bare,"explicit_skip_tests":skip_calls,"self_equality_asserts":self_equal}


def classify_uplift_experiments() -> list[dict[str, Any]]:
    reg=load_uplift_registry(); rows=[]
    migrated_s={f"G3:S{i}" for i in range(7)} | {f"G4:S{i}.REBASE" for i in range(7)}
    for e in reg.get("experiments",[]):
        eid=e["experiment_id"]; mode=e.get("execution_mode")
        ref=e.get("handler_ref")
        resolved=bool(ref and resolve_handler_ref(ref)) if mode=="NATIVE_HANDLER" else None
        if mode=="IMPORTED_EVIDENCE":
            support="IMPORTED_AND_VERIFIED_HISTORY"
            current=False
        elif eid in migrated_s:
            support="V05_MIGRATED_HISTORY_VIA_GENERIC_S_WORKFLOW"
            current=True
        elif eid=="G4:R7.REBASE":
            support="V05_MIGRATED_BOUNDED_R_REGRESSION"
            current=True
        else:
            support="HISTORICAL_NATIVE_ARCHIVED_NOT_CURRENT_V05_SUPPORT"
            current=False
        rows.append({"experiment_id":eid,"historical_execution_mode":mode,"handler_ref":ref,"handler_resolves":resolved,"v05_support_status":support,"claimed_current_support":current})
    return rows


def current_v05_runner_classification() -> list[dict[str,Any]]:
    from . import adapters  # noqa: F401
    from .controller import RUNNERS, RUNNER_META
    accepted={"adapter.base_l2_step10","adapter.scout_l15","adapter.oscout_phase0","adapter.v05_workflow"}
    rows=[]
    for name in sorted(RUNNERS):
        meta=RUNNER_META.get(name,{})
        if name in accepted:
            status="V05_PRODUCTION_GUARDED"
        elif meta.get("v05_operation") is None:
            status="LEGACY_COMPATIBILITY_NOT_V05_PRODUCTION"
        else:
            status="UNCLASSIFIED_V05_ROUTE"
        rows.append({"runner":name,"operation":meta.get("v05_operation"),"call_style":meta.get("call_style"),"classification":status})
    return rows
