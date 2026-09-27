"""Registered compatibility tests. Original archived production bytes are read-only.
All negative mutations occur in disposable test copies; they are NOT save receipts
or acknowledgements for any live capture. No scientific evaluator is executed.
"""
from pathlib import Path
import json,hashlib,zipfile
import pytest
from infinity_grid import submission as sub, portable_registry as pr
from infinity_grid.canon import canonical_sha256

FIXTURE_SHA="462f1df3ef3898018a224a7d7487a7161d1db2bc00bd3b3e5790b30b2b38274c"
CAPSULE="b25aa7583b338d7887488f3501a82a1140d7b9d332af530161fc38daeb0f65d6"
COMPLETION="4f94f08c48bc675c9b05ad7bfa77cab3556e1f1838d08956ac13a89853391fb0"
ENGINE="a27badfc1a353fedac60ee7320295e02c1106c2b53254a3464ba414099f70a84"

@pytest.fixture(scope="module")
def blobs():
    p=Path(__file__).parent/"fixtures/legacy_adoption_objects.zip"
    assert hashlib.sha256(p.read_bytes()).hexdigest()==FIXTURE_SHA
    with zipfile.ZipFile(p) as z:
        out={n.split("/")[-1]:z.read(n) for n in z.namelist()}
    assert all(hashlib.sha256(b).hexdigest()==k for k,b in out.items())
    assert json.loads(out[CAPSULE])["completion_sha256"]==COMPLETION
    return out

@pytest.fixture
def ws(tmp_path,blobs):
    root=tmp_path/"original-capture"; root.mkdir()
    for n,d in json.loads(blobs[CAPSULE])["files"].items():
        assert not Path(n).is_absolute() and ".." not in Path(n).parts
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blobs[d])
    return root

def receipt_path(ws):
    files=list((ws/"durability/receipts"/ENGINE).glob("*.json")); assert len(files)==1
    return files[0]

def pending_engine(ws):
    return [r for r in sub.save_status(ws)["pending_objects"] if r["sha256"]==ENGINE]

def write_test_record(p,row,field):
    body={k:v for k,v in row.items() if k!=field}
    row=dict(body,**{field:canonical_sha256(body)})
    p.write_bytes(sub._json_bytes(row))

def test_original_capsule_restores_its_historical_save_contract(tmp_path,blobs):
    root=tmp_path/"objects-root";(root/"objects").mkdir(parents=True)
    for d,b in blobs.items():(root/"objects"/d).write_bytes(b)
    result=pr.verify_capsule(root,CAPSULE)
    assert result["completion_sha256"]==COMPLETION
    assert all((root/"objects"/d).read_bytes()==b for d,b in blobs.items())

def test_real_physical_receipt_satisfies_both_original_roles_without_rewrite(ws):
    before={p.relative_to(ws).as_posix():p.read_bytes() for p in (ws/"durability/receipts").rglob("*.json")}
    required=[r for r in sub.required_objects(ws) if r["sha256"]==ENGINE]
    assert len(required)==2 and len({r["obligation_id"] for r in required})==2
    assert sub.save_status(ws)["status"]=="SAVED"
    assert before=={p.relative_to(ws).as_posix():p.read_bytes() for p in (ws/"durability/receipts").rglob("*.json")}

def test_saved_evidence_can_be_verified_without_execution_environment(ws):
    rec=sub.capture_record(ws)
    assert sub.require_saved(ws,rec["job"]["job_id"],check_environment=False)==rec

def test_execution_environment_remains_strict(ws, monkeypatch):
    rec=sub.capture_record(ws)
    # Exercise the refusal on every host without changing historical evidence.
    # Only the observed interpreter in this disposable test process is varied.
    from collections import namedtuple
    version = namedtuple('VersionInfo', 'major minor micro')
    monkeypatch.setattr(sub.sys, 'version_info', version(3, 99, 0))
    with pytest.raises(Exception,match="PYTHON_ENVIRONMENT_MISMATCH"):
        sub.require_saved(ws,rec["job"]["job_id"])

@pytest.mark.parametrize("mutation",["version_only","wrong_archive","missing_producer","changed_source"])
def test_unverified_producer_does_not_enable_legacy_fallback(ws,mutation):
    p=ws/"CAPTURE.json"; rec=json.loads(p.read_text())
    if mutation=="version_only":rec["decoder_version"]="0.6.1"
    elif mutation=="wrong_archive":
        next(r for r in rec["objects"] if r["role"]=="engine_source")["sha256"]="0"*64
        # Retain ambiguity of the old digest, but not under a trusted producer.
        rec["objects"].append({"role":"input:other","sha256":ENGINE,"size_bytes":3815166,"object_name":ENGINE+".bin"})
    elif mutation=="missing_producer":
        rec["objects"]=[r for r in rec["objects"] if r["role"]!="engine_source"]
        # Keep the digest ambiguous in a non-producer role.
        rec["objects"].append({"role":"input:other","sha256":ENGINE,"size_bytes":3815166,"object_name":ENGINE+".bin"})
    else:(ws/"source/infinity_grid/_version.py").write_bytes(b"# changed producer source\n")
    if mutation!="changed_source":write_test_record(p,rec,"capture_id")
    assert pending_engine(ws)

@pytest.mark.parametrize("mutation",["digest","size","provider","drive_id","not_verified","bad_seal","missing","role_bound_v1","wrong_capture_v2"])
def test_bad_or_wrong_scope_receipt_never_satisfies_legacy_pair(ws,mutation):
    p=receipt_path(ws); row=json.loads(p.read_text())
    if mutation=="missing":p.unlink()
    elif mutation=="bad_seal":
        row["receipt_sha256"]="0"*64;p.write_bytes(sub._json_bytes(row))
    else:
        if mutation=="digest":row["sha256"]="0"*64
        elif mutation=="size":row["size_bytes"]+=1
        elif mutation=="provider":row["provider"]="not_google_drive"
        elif mutation=="drive_id":row["drive_file_id"]="!"
        elif mutation=="not_verified":row["raw_readback_verified"]=False
        elif mutation=="role_bound_v1":row["obligation_scope"]="CHECKPOINT_OUTBOX"
        elif mutation=="wrong_capture_v2":
            obj=next(r for r in sub.required_objects(ws) if r["sha256"]==ENGINE)
            row.update(schema_id="IG_DECODER_EXPLICIT_SAVE_RECEIPT_V2",obligation_id="0"*64,obligation_scope="CAPTURE_SAVE",role=obj["role"],logical_name=obj["logical_name"])
        write_test_record(p,row,"receipt_sha256")
    assert len(pending_engine(ws))==2

def test_modern_ambiguous_role_still_rejects_original_unbound_receipt(ws):
    obj=next(r for r in sub.required_objects(ws) if r["sha256"]==ENGINE)
    assert sub._valid_receipt(receipt_path(ws),obj,ambiguous=True) is False
    assert sub._valid_receipt(receipt_path(ws),obj,ambiguous=False) is True

def test_original_capsule_corrupt_object_refuses_before_save_fallback(tmp_path,blobs):
    root=tmp_path/"object-root";(root/"objects").mkdir(parents=True)
    for d,b in blobs.items():(root/"objects"/d).write_bytes(b)
    packet=json.loads(blobs[CAPSULE]);d=packet["files"]["CAPTURE.json"]
    (root/"objects"/d).write_bytes(blobs[d]+b" ")
    with pytest.raises(Exception,match="PROJECT_OBJECT_MISMATCH"):
        pr.verify_capsule(root,CAPSULE)
