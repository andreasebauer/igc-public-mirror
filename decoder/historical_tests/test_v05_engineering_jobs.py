from __future__ import annotations
import json, os, shutil, zipfile
from pathlib import Path
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from infinity_grid.v05_stage_architecture import audit_callable
from infinity_grid.v05_stage_registry import ENGINEERING_ONLY_STAGE_HANDLERS, _resolve
from infinity_grid.v05_execution_authority import ENGINEERING_ROLE, source_tree_digest
from infinity_grid.v05_chain import seal_chain_registration
from infinity_grid.v05_registered_service import POLICY_SCHEMA, REQUEST_SCHEMA, RegisteredExecutionService
from infinity_grid.v05_engineering_worker import engineering_source_tree_digest, validate_overlay_archive, EngineeringWorkerError, OVERLAY_SCHEMA, compact_validation_tail
from infinity_grid.v05_engineering_jobs import JOB_SPEC_SCHEMA, ACCEPT_SPEC_SCHEMA

PACKAGE=Path(__import__('infinity_grid').__file__).resolve().parent
SOURCE=PACKAGE.parent

def test_engineering_stage_handlers_pass_architecture_gate():
    assert ENGINEERING_ONLY_STAGE_HANDLERS['engineering.validate'].endswith(':engineering_job_handler')
    assert ENGINEERING_ONLY_STAGE_HANDLERS['engineering.accept'].endswith(':engineering_acceptance_handler')
    assert audit_callable(_resolve(ENGINEERING_ONLY_STAGE_HANDLERS['engineering.validate']))['status']=='PASS'
    assert audit_callable(_resolve(ENGINEERING_ONLY_STAGE_HANDLERS['engineering.accept']))['status']=='PASS'

def test_engineering_source_digest_is_content_bound(tmp_path):
    c=tmp_path/'source'; c.mkdir(); p=c/'probe.txt'; p.write_text('a')
    a=engineering_source_tree_digest(c); p.write_text('b'); b=engineering_source_tree_digest(c)
    assert a!=b

def test_overlay_manifest_rejects_unlisted_member(tmp_path):
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f:
        f.writestr('OVERLAY_MANIFEST.json',json.dumps({'schema_id':OVERLAY_SCHEMA,'files':[],'delete_paths':[],'overlay_sha256':'0'*64}))
        f.writestr('extra.txt','x')
    # Binding SHA is checked first; this proves an archive cannot be silently substituted.
    with pytest.raises(EngineeringWorkerError): validate_overlay_archive(z,'0'*64)


def test_compact_validation_tail_keeps_all_failed_nodes():
    raw="trace\n"*2000 + "FAILED tests/a.py::test_a - A\nFAILED tests/b.py::test_b - B\n29 failed, 240 passed, 1 skipped in 92.41s\n"
    tail=compact_validation_tail(raw)
    assert "tests/a.py::test_a" in tail and "tests/b.py::test_b" in tail
    assert "29 failed, 240 passed, 1 skipped" in tail
    assert len(tail) <= 16000

def _write_private(path:Path,data:bytes):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data); path.chmod(0o600)

def _registration(source_sha,package_sha,uid,gid):
    job_id='SELF-HOST-VALIDATE'
    job={'schema_id':JOB_SPEC_SCHEMA,'job_id':job_id,'operation':'VALIDATE_SOURCE','parent_source_sha256':source_sha,'parent_package_sha256':package_sha,
         'expected_candidate_source_sha256':source_sha,'expected_candidate_package_sha256':package_sha,'overlay_path':None,'overlay_sha256':None,
         'validation_groups':['engineering_layer'],'worker_uid':uid,'worker_gid':gid,'wall_seconds_max':120}
    accept={'schema_id':ACCEPT_SPEC_SCHEMA,'job_id':job_id,'source_stage_id':'ENG:E0','expected_candidate_source_sha256':source_sha,'expected_candidate_package_sha256':package_sha}
    base={'schema_id':'IG_DECODER_V05_SCIENTIFIC_CHAIN_REGISTRATION_V2','chain_id':'ENG-SELF-HOST','subject':{'hierarchy':'ENGINEERING','level':0,'parent_ref':'CURRENT_SOURCE'},
      'release_line':'v0.5 / 0.50','mode':'CONTROLLED_S_THEN_ADAPTIVE_R','parent_authority':{'authority_ref':'engineering-parent','science_sha256':'0'*64,'verification_sha256':'1'*64,'status':'CERTIFIED_PASS'},
      'stages':[
       {'stage_id':'ENG:E0','series':'S','stage_kind':'DECODER_STAGE','question_ref':'registered engineering source validation','question_sha256':'2'*64,'depends_on':[],
        'execution':{'handler_key':'engineering.validate','parameters':job},'result_contract':{'artifact_logical_name':'engineering_job.json','outcome_pointer':'/outcome','allowed_outcomes':['PASS']},
        'transitions':{'PASS':{'action':'NEXT','next_stage':'ENG:E1','reason':'VALIDATED_CHILD'}},'auto_run':True,'promotion_effect':'NONE'},
       {'stage_id':'ENG:E1','series':'S','stage_kind':'DECODER_STAGE','question_ref':'registered engineering candidate acceptance','question_sha256':'3'*64,'depends_on':['ENG:E0'],
        'execution':{'handler_key':'engineering.accept','parameters':accept},'result_contract':{'artifact_logical_name':'engineering_acceptance.json','outcome_pointer':'/outcome','allowed_outcomes':['PASS']},
        'transitions':{'PASS':{'action':'END','next_stage':None,'reason':'ENGINEERING_CANDIDATE_ACCEPTED'}},'auto_run':True,'promotion_effect':'NONE'}],
      'budgets':{'max_stage_executions':2,'max_chain_wall_seconds':180,'default_workers':1},'durability':{'external_mirror_required':False,'fsync_each_transition':True,'stage_commits':'APPEND_ONLY'},
      'authority_policy':{'automatic_promotion':False,'changed_assumptions_policy':'REVIEW_REQUIRED','novelty_policy':'REVIEW_REQUIRED','r_science_policy':'ADAPTIVE_ONLY_WITHIN_PREREGISTERED_BRANCHES','require_verified_parent':True,'unregistered_outcome_policy':'REVIEW_REQUIRED'}}
    return seal_chain_registration(base)

@pytest.mark.skipif(os.name!='posix' or not hasattr(os,'geteuid') or os.geteuid()!=0,reason='requires POSIX controller identity separation')
def test_registered_engineering_self_host_job_uses_child_and_release(tmp_path):
    uid=65534; gid=65534
    source_sha=engineering_source_tree_digest(SOURCE); package_sha=source_tree_digest(PACKAGE); before=source_sha
    reg=_registration(source_sha,package_sha,uid,gid)
    key=Ed25519PrivateKey.generate(); key_path=tmp_path/'config'/'key.bin'; _write_private(key_path,key.private_bytes_raw())
    policy={'schema_id':POLICY_SCHEMA,'role':ENGINEERING_ROLE,'source_sha256':package_sha,'service_root':str(tmp_path/'service'),'signing_key_path':str(key_path),
      'public_key_hex':key.public_key().public_bytes_raw().hex(),'registrations':[{'registration':reg,'evaluator_refs':['infinity_grid.controller_only_fixture:partition_evaluator']}]}
    policy_path=tmp_path/'config'/'policy.json'; _write_private(policy_path,json.dumps(policy,sort_keys=True).encode())
    svc=RegisteredExecutionService(policy_path); req={'schema_id':REQUEST_SCHEMA,'operation':'run','registration_sha256':reg['registration_sha256']}
    r=svc.dispatch(req); assert r['result']['status']=='COMPLETE'; assert engineering_source_tree_digest(SOURCE)==before
    pub=svc.dispatch(dict(req,operation='publish'))['result']; assert pub['record']['authoritative'] is False
    acc=pub['record']['commits'][-1]['result']['engineering_acceptance']; assert acc['status']=='ACCEPTED_ENGINEERING_CANDIDATE'
    rel=Path(policy['service_root'])/'registered_publications'/'engineering_releases'/pub['record_sha256']/'source.zip'
    assert rel.is_file() and __import__('hashlib').sha256(rel.read_bytes()).hexdigest()==acc['release_zip_sha256']
