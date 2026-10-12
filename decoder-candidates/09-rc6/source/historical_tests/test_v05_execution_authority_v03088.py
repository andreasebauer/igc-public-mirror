from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from infinity_grid.canon import canonical_sha256
from infinity_grid.controller_only_fixture import stage_handler
from infinity_grid.v05_chain import ScientificChainController, ChainExecutionResult, CHAIN_SCHEMA_V2, seal_chain_registration
from infinity_grid.v05_execution_authority import (
    EngineeringAuthority, ExecutionAuthorityError, ENGINEERING_ROLE, RuntimePermit,
    digest, science_digest, source_tree_digest, verify_execution_receipt,
)
from infinity_grid.v05_stage_runtime import StageScienceRuntime, StageRuntimeError, _resolve_ref
from infinity_grid.v05_stage_architecture import audit_module_source, StageArchitectureViolation


def registration(*, chain_id="LD-ENG", workers=1, hierarchy="ENGINEERING", mirror=False):
    return seal_chain_registration({
        "schema_id": CHAIN_SCHEMA_V2, "chain_id": chain_id,
        "subject": {"hierarchy": hierarchy, "level": 0, "parent_ref": "ENGINEERING_FIXTURE"},
        "release_line": "v0.5 / 0.50", "mode": "CONTROLLED_S_THEN_ADAPTIVE_R",
        "parent_authority": {"authority_ref": "fixture", "science_sha256": "0"*64,
            "verification_sha256": "1"*64, "status": "CERTIFIED_PASS"},
        "stages": [{
            "stage_id": "ENG:S0", "series": "S", "stage_kind": "DECODER_STAGE",
            "question_ref": "fixture-only", "question_sha256": "2"*64, "depends_on": [],
            "execution": {"handler_key": "fixture", "parameters": {"values": list(range(12)),
                "modulus": 3, "workers": workers, "reducer_memory_budget_bytes": 536870912,
                "workspace_budget_bytes": 16777216}},
            "result_contract": {"artifact_logical_name": "result.json", "outcome_pointer": "/outcome",
                "allowed_outcomes": ["PASS"]},
            "transitions": {"PASS": {"action": "END", "next_stage": None, "reason": "ENGINEERING_END"}},
            "auto_run": True, "promotion_effect": "NONE",
        }],
        "budgets": {"max_stage_executions": 4, "max_chain_wall_seconds": 60, "default_workers": workers},
        "durability": {"fsync_each_transition": True, "stage_commits": "APPEND_ONLY", "external_mirror_required": mirror},
        "authority_policy": {"automatic_promotion": False, "require_verified_parent": True,
            "novelty_policy": "REVIEW_REQUIRED", "changed_assumptions_policy": "REVIEW_REQUIRED",
            "unregistered_outcome_policy": "REVIEW_REQUIRED", "r_science_policy": "ADAPTIVE_ONLY_WITHIN_PREREGISTERED_BRANCHES"},
    })


def controller(root):
    c = ScientificChainController(root, engineering_only=True)
    c.register_stage_handler("fixture", stage_handler)
    return c


def bindings(root):
    return {"chain_id": "ENG", "registration_sha256": "1"*64, "stage_id": "ENG:S0", "handler_key": "fixture",
        "handler_ref": "infinity_grid.controller_only_fixture:stage_handler", "question_sha256": "2"*64,
        "source_sha256": "3"*64, "handler_source_sha256": "4"*64, "parameters_sha256": "5"*64,
        "authority_sha256": "6"*64, "dependencies_sha256": "7"*64,
        "evidence_store_id": digest({"chain_dir": str(root.resolve())}), "run_id": "a"*32}


@pytest.fixture
def signed(tmp_path):
    authority = EngineeringAuthority(True)
    b = bindings(tmp_path)
    result = {"outcome": "PASS", "answer": 7}
    r = authority.receipt(b, result)
    return authority, b, result, r


def verify(signed, *, role=ENGINEERING_ROLE):
    a, b, result, receipt = signed
    return verify_execution_receipt(receipt, expected_bindings=b, result=result,
        trusted_public_keys=a.public_keys, required_role=role)


def test_ed25519_valid_engineering_receipt(signed):
    assert verify(signed)["status"] == "VERIFIED"


def test_engineering_receipt_cannot_authorize_official_result(signed):
    with pytest.raises(ExecutionAuthorityError, match="ROLE_NOT_AUTHORIZED"):
        verify(signed, role="OFFICIAL_CONTROLLER")


@pytest.mark.parametrize("key", sorted(bindings(Path('/')).keys()))
def test_changed_binding_is_refused(signed, key):
    a, b, result, receipt = signed
    b[key] = "b"*len(b[key]) if key.endswith("sha256") or key in {"evidence_store_id", "run_id"} else "other"
    with pytest.raises(ExecutionAuthorityError, match="BINDING_MISMATCH"):
        verify(signed)


def test_changed_result_refused_even_with_recomputed_plain_hash(signed):
    a, b, result, r = signed
    result["answer"] = 8
    r["payload"]["result_sha256"] = digest(result)
    r["payload"]["science_sha256"] = science_digest(result)
    with pytest.raises(ExecutionAuthorityError, match="INVALID_EXECUTION_SIGNATURE"):
        verify(signed)


def test_no_trust_on_first_use(signed):
    a, b, result, r = signed
    with pytest.raises(ExecutionAuthorityError, match="UNTRUSTED_RECEIPT_KEY"):
        verify_execution_receipt(r, expected_bindings=b, result=result, trusted_public_keys={}, required_role=ENGINEERING_ROLE)


@pytest.mark.parametrize("field", ["signature", "payload", "role", "key_id", "algorithm", "schema_id"])
def test_missing_receipt_field_refused(signed, field):
    signed[3].pop(field)
    with pytest.raises(ExecutionAuthorityError, match="MALFORMED"):
        verify(signed)


def test_receipt_cannot_supply_its_own_trust_key(signed):
    signed[3]["public_key"] = next(iter(signed[0].public_keys.values())).hex()
    with pytest.raises(ExecutionAuthorityError, match="MALFORMED"):
        verify(signed)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_result_refused(value):
    with pytest.raises(ExecutionAuthorityError, match="NONFINITE_JSON"):
        science_digest({"value": value})


@pytest.mark.parametrize("field", ["wall_seconds", "pid", "run_id", "engine_cpu_seconds", "telemetry"])
def test_science_payload_rejects_telemetry_without_silently_dropping_fields(field):
    with pytest.raises(ExecutionAuthorityError, match="TELEMETRY_IN_SCIENCE_PAYLOAD"):
        science_digest({"nested": {field: 7}})


def test_science_digest_is_not_receipt_or_nonce_digest(tmp_path):
    a = EngineeringAuthority(True); b = bindings(tmp_path); result = {"outcome": "PASS"}
    r1 = a.receipt(b, result); b["run_id"] = "b"*32; r2 = a.receipt(b, result)
    assert r1["signature"] != r2["signature"]
    assert r1["payload"]["science_sha256"] == r2["payload"]["science_sha256"]


def test_default_official_entry_refuses_before_freeze_or_execution(tmp_path):
    c = ScientificChainController(tmp_path)
    with pytest.raises(ExecutionAuthorityError, match="OFFICIAL_AUTHORITY_NOT_CONFIGURED"):
        c.run(registration(hierarchy="G"))
    assert not (tmp_path/"LD-ENG"/"chain_registration.json").exists()


def test_engineering_cannot_run_g_science(tmp_path):
    c = controller(tmp_path)
    with pytest.raises(ExecutionAuthorityError, match="ENGINEERING_MODE_CANNOT_RUN_SCIENCE"):
        c.run(registration(hierarchy="G"))


def test_legacy_opt_in_cannot_reenable_private_executor(tmp_path):
    c = ScientificChainController(tmp_path, allow_legacy_new_registrations=True, engineering_only=True)
    with pytest.raises(ExecutionAuthorityError, match="LEGACY_EXECUTOR_DISABLED"):
        c.register_executor("legacy", lambda *_: ChainExecutionResult({"outcome": "PASS"}))


def test_unregistered_handler_refused(tmp_path):
    c = ScientificChainController(tmp_path, engineering_only=True)
    with pytest.raises(ExecutionAuthorityError, match="STAGE_HANDLER_UNREGISTERED"):
        c.run(registration())
    assert not list(tmp_path.rglob("partition.sqlite3"))


def test_alias_registration_not_in_inventory_refused(tmp_path):
    c = ScientificChainController(tmp_path, engineering_only=True)
    with pytest.raises(ExecutionAuthorityError, match="STAGE_HANDLER_NOT_IN_ACCEPTED_REGISTRY"):
        c.register_stage_handler("adhoc", stage_handler)


def test_unregistered_evaluator_refused_before_import():
    with pytest.raises(StageRuntimeError, match="EVALUATOR_NOT_IN_ACCEPTED_REGISTRY"):
        _resolve_ref("unregistered_fixture_module:run")


def test_direct_runtime_construction_refused_before_io(tmp_path):
    root = tmp_path/"not_created"
    with pytest.raises(ExecutionAuthorityError, match="CONTROLLER_PERMIT_REQUIRED"):
        StageScienceRuntime(chain_dir=root, chain_id="ENG", stage_id="ENG:S0", question_sha256="2"*64)
    assert not root.exists()


def test_runtime_permit_constructor_not_public(tmp_path):
    with pytest.raises(ExecutionAuthorityError, match="CONTROLLER_PERMIT_REQUIRED"):
        RuntimePermit(bindings(tmp_path))


@pytest.mark.parametrize("kind", ["wrong_question", "revoked"])
def test_runtime_permit_wrong_binding_or_revoked_refused(tmp_path, kind):
    a = EngineeringAuthority(True); permit = a.begin(bindings(tmp_path))
    if kind == "revoked": permit.revoke()
    with pytest.raises(ExecutionAuthorityError, match="PERMIT"):
        StageScienceRuntime(chain_dir=tmp_path, chain_id="ENG", stage_id="ENG:S0",
            question_sha256="9"*64 if kind=="wrong_question" else "2"*64, _execution_permit=permit)
    assert not (tmp_path/"decoder_stage_runtime").exists()


def test_revoked_runtime_cannot_publish(tmp_path):
    a = EngineeringAuthority(True); permit = a.begin(bindings(tmp_path))
    rt = StageScienceRuntime(chain_dir=tmp_path, chain_id="ENG", stage_id="ENG:S0", question_sha256="2"*64, _execution_permit=permit)
    permit.revoke()
    with pytest.raises(ExecutionAuthorityError, match="PERMIT_REVOKED"):
        rt.publish_json("fake", {"outcome": "PASS"})
    assert not list(tmp_path.rglob("fake.json"))


def test_manual_result_cannot_be_committed(tmp_path):
    c = controller(tmp_path); reg = c.freeze(registration()); d = c.chain_dir(reg["chain_id"])
    fake = ChainExecutionResult({"outcome": "PASS"}, {"lifecycle": "COMPLETE_VALID", "receipt": "made-up"})
    with pytest.raises(ExecutionAuthorityError, match="CONTROLLER_EXECUTION_TICKET_REQUIRED"):
        c._commit_stage(d, reg, reg["stages"][0], fake)
    assert not c._commit_path(d,"ENG:S0").exists()


def test_valid_shared_runtime_execution_and_signed_resume(tmp_path):
    c = controller(tmp_path); reg = registration(); state = c.run(reg)
    assert state["status"] == "COMPLETE" and state["authoritative"] is False
    d = c.chain_dir(reg["chain_id"]); commit = c._load_commit(d,"ENG:S0")
    assert commit["result"]["partition"]["class_count"] == 3
    assert commit["execution_receipt"]["role"] == ENGINEERING_ROLE
    cold = controller(tmp_path)
    assert cold.run(reg["chain_id"])["stage_execution_count"] == 1
    assert cold.status(reg["chain_id"])["authoritative"] is False


def test_fake_complete_state_with_only_recomputed_hash_refused(tmp_path):
    c = controller(tmp_path); reg = c.freeze(registration()); d = c.chain_dir(reg["chain_id"])
    st = c._load_state(reg,d); st.update(status="COMPLETE", current_stage=None)
    st["state_sha256"] = canonical_sha256({k:v for k,v in st.items() if k!="state_sha256"})
    c._state_path(d).write_text(json.dumps(st))
    with pytest.raises(ExecutionAuthorityError): c.run(reg["chain_id"])


@pytest.mark.parametrize("tamper", ["result", "outcome", "authoritative", "question", "unsigned"])
def test_committed_result_tampering_refused_on_reentry(tmp_path, tamper):
    c=controller(tmp_path); reg=registration(); c.run(reg); d=c.chain_dir(reg["chain_id"]); p=c._commit_path(d,"ENG:S0")
    obj=json.loads(p.read_text())
    if tamper=="result": obj["result"]["outcome"]="OTHER"; obj["result_sha256"]=canonical_sha256(obj["result"])
    elif tamper=="outcome": obj["outcome"]="OTHER"
    elif tamper=="authoritative": obj["authoritative"]=True
    elif tamper=="question": obj["question_sha256"]="9"*64
    else: obj.pop("execution_receipt")
    obj["commit_sha256"]=canonical_sha256({k:v for k,v in obj.items() if k!="commit_sha256"}); p.write_text(json.dumps(obj))
    with pytest.raises(ExecutionAuthorityError): c.run(reg["chain_id"])
    with pytest.raises(ExecutionAuthorityError): c.status(reg["chain_id"])


def test_mutating_completed_execution_before_commit_is_refused(tmp_path):
    c=controller(tmp_path); reg=c.freeze(registration()); d=c.chain_dir(reg["chain_id"]); stage=reg["stages"][0]
    out=c._execute_stage(stage,d,reg); out.result["extra"]=42
    with pytest.raises(ExecutionAuthorityError, match="EXECUTION_CHANGED_BEFORE_COMMIT"):
        c._commit_stage(d,reg,stage,out)


def test_source_file_change_between_execution_and_commit_is_refused(tmp_path):
    c=controller(tmp_path); src=tmp_path/"source-fixture"; src.mkdir(); f=src/"core.py"; f.write_text("VALUE = 1\n"); c._package_root=src
    reg=c.freeze(registration()); d=c.chain_dir(reg["chain_id"]); stage=reg["stages"][0]; out=c._execute_stage(stage,d,reg)
    f.write_text("VALUE = 2\n")
    with pytest.raises(ExecutionAuthorityError, match="EXECUTION_CHANGED_BEFORE_COMMIT"):
        c._commit_stage(d,reg,stage,out)


def test_source_digest_covers_binary_resources_and_is_path_independent(tmp_path):
    a=tmp_path/"a"; b=tmp_path/"b"; a.mkdir(); b.mkdir()
    for d in (a,b): (d/"fixture.zip").write_bytes(b"resource")
    assert source_tree_digest(a)==source_tree_digest(b)
    (b/"fixture.zip").write_bytes(b"changed")
    assert source_tree_digest(a)!=source_tree_digest(b)


@pytest.mark.parametrize("source", [
    "import multiprocessing\n", "from concurrent.futures import ProcessPoolExecutor\n",
    "import subprocess\n", "from os import write as w\nw(1,b'x')\n",
    "from pathlib import Path\nPath('x').open('w')\n", "open('x',mode_variable)\n",
    "from builtins import open as o\no('x','w')\n", "def main():\n    return 0\n",
    "if __name__ == '__main__':\n    pass\n", "import socket\n", "eval('1')\n",
])
def test_preexecution_lint_rejects_private_infrastructure(tmp_path, source):
    p=tmp_path/"stage.py"; p.write_text(source)
    assert audit_module_source(p)["status"] == "FAIL"


def test_read_only_module_not_rejected(tmp_path):
    p=tmp_path/"stage.py"; p.write_text("from pathlib import Path\ndef read(p):\n    return Path(p).read_text()\n")
    assert audit_module_source(p)["status"] == "PASS"


def test_one_and_four_worker_engineering_science_identical(tmp_path):
    outcomes=[]
    for n in (1,4):
        c=controller(tmp_path/str(n)); reg=registration(chain_id="WORKERS",workers=n); c.run(reg)
        commit=c._load_commit(c.chain_dir("WORKERS"),"ENG:S0")
        outcomes.append((commit["result"],commit["science_sha256"]))
    assert outcomes[0]==outcomes[1]


def test_unacknowledged_mirror_cannot_be_marked_complete(tmp_path):
    c=controller(tmp_path); reg=registration(mirror=True); st=c.run(reg); d=c.chain_dir(reg["chain_id"])
    assert st["status"]=="PAUSED"
    st.update(status="COMPLETE",current_stage=None)
    c._write_state(d,st)
    with pytest.raises(ExecutionAuthorityError,match="COMPLETE_WITHOUT_MIRROR"):
        c.run(reg["chain_id"])


def test_mirror_ack_then_signed_resume_works_for_engineering(tmp_path):
    c=controller(tmp_path); reg=registration(mirror=True); st=c.run(reg)
    c.acknowledge_external_mirror(reg["chain_id"],"ENG:S0",commit_sha256=st["stop"]["commit_sha256"],
        mirror_uri="engineering-fixture:mirror",mirror_sha256="8"*64)
    assert c.resume(reg["chain_id"])["status"]=="COMPLETE"


def test_official_status_does_not_bless_engineering_complete_state(tmp_path):
    c=controller(tmp_path); reg=registration(); c.run(reg)
    official=ScientificChainController(tmp_path)
    with pytest.raises(ExecutionAuthorityError,match="OFFICIAL_AUTHORITY_NOT_CONFIGURED"):
        official.status(reg["chain_id"])


def test_execution_ticket_is_consumed_once(tmp_path):
    c=controller(tmp_path); reg=c.freeze(registration()); d=c.chain_dir(reg["chain_id"]); stage=reg["stages"][0]
    out=c._execute_stage(stage,d,reg); c._commit_stage(d,reg,stage,out)
    with pytest.raises(ExecutionAuthorityError,match="CONTROLLER_EXECUTION_TICKET_REQUIRED"):
        c._commit_stage(d,reg,stage,out)


def test_bad_signature_bytes_refused(signed):
    signed[3]["signature"]="A"*88
    with pytest.raises(ExecutionAuthorityError,match="INVALID_EXECUTION_SIGNATURE"):
        verify(signed)


def test_question_scope_change_in_registration_refused_at_reentry(tmp_path):
    c=controller(tmp_path); reg=registration(); c.run(reg); d=c.chain_dir(reg["chain_id"])
    changed=copy.deepcopy(reg); changed["stages"][0]["question_sha256"]="9"*64
    changed=seal_chain_registration(changed)
    (d/"chain_registration.json").write_text(json.dumps(changed))
    state=json.loads(c._state_path(d).read_text()); state["registration_sha256"]=changed["registration_sha256"]; c._write_state(d,state)
    with pytest.raises(ExecutionAuthorityError,match="EXECUTION_BINDING_MISMATCH"):
        c.run(reg["chain_id"])


def test_packaged_resource_reads_still_allowed(tmp_path):
    p=tmp_path/"stage.py"; p.write_text("from importlib.resources import files\ndef read():\n    return files('infinity_grid').joinpath('_build_meta.json').read_text()\n")
    assert audit_module_source(p)["status"]=="PASS"


def test_dynamic_import_module_remains_disallowed(tmp_path):
    p=tmp_path/"stage.py"; p.write_text("from importlib import import_module\n")
    assert audit_module_source(p)["status"]=="FAIL"
