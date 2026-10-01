"""Prospective bootstrap checks, executed only by the registered old-engine validator.
Real immutable metadata fixtures are inputs, not fabricated execution receipts.
Positive unit construction does not claim actual adoption or science completion.
"""
from pathlib import Path
import copy,json,hashlib
import pytest
from infinity_grid import change_preservation as fix, change_sessions as cs, result_contracts as rc, submission as sub

F=Path(__file__).parent/'fixtures/preservation_rebind'
def record():return json.loads((F/'CAPTURE.json').read_text())
def session():return json.loads((F/'SESSION.json').read_text())

def test_frozen_real_capture_and_review_scope():
    r=record();s=session();v=json.loads((F/'REVISION.json').read_text())
    assert r['capture_id']=='27cfad529020e830fa3a987822ac56e5514f8fbf32f57a0199eedbb555b746cc'
    assert sub._sha((F/'CAPTURE.json').read_bytes())=='eb8e90197114b032aa07d5a16e8ae6544a82e801c6291a7e9320a6f7e40a2118'
    assert v['record_sha256']=='8e3b18395db3e2b5155b32bc37a196b4ae6d31042103a71505f8c95a8a84efe0'
    assert v['requirements']==s['requirements'] and len(v['requirements']['nodes'])==4
    assert v['source_object']['sha256']=='2dab5fbdaade8c6cde70880424e2972295dc8b8ac4572976581cada4a7552704'

def test_allowance_amendment_changes_only_pending_ceiling():
    r=record();before=rc.policy(r['result_contract']);after=rc.policy(fix.revised_contract(r,1073741824))
    assert before['max_pending_bytes']==536870912
    # This retained backlog triggered the historical admission pause under the
    # smaller pre-dev15 defaults.  The reviewed dev15 defaults already fit it;
    # do not manufacture a current obstruction to justify the old amendment.
    assert 144916043+before['max_commit_bytes']<=before['max_pending_bytes']
    assert 144916043+after['max_commit_bytes']<after['max_pending_bytes']
    assert {k for k in before if before[k]!=after[k]}=={'max_pending_bytes'}

def test_contract_changes_no_result_or_input_semantics():
    old=record();new=copy.deepcopy(old);new['result_contract']=fix.revised_contract(old,1073741824)
    assert fix.semantics(old)==fix.semantics(new)
    assert rc.normalize({k:v for k,v in new['result_contract'].items() if k!='declared_outcomes'},old['job']['execution'],old['job']['question'])==new['result_contract']

def test_no_mutation_of_supplied_record():
    r=record();prior=copy.deepcopy(r);fix.revised_contract(r,1073741824);assert r==prior

@pytest.mark.parametrize('value',[0,-1,536870912,536870911,True,1073741824.0,'1073741824'])
def test_invalid_allowance_refused(value):
    with pytest.raises(sub.SubmissionError,match='PRESERVATION_INCREASE_REQUIRED'):fix.revised_contract(record(),value)

def test_workspace_limit_still_enforced():
    with pytest.raises(sub.SubmissionError,match='PRESERVATION_EXCEEDS_WORKSPACE'):fix.revised_contract(record(),8589934593)

@pytest.mark.parametrize('field,value',[('wall_seconds_max',120),('execution_policy','AUTOMATIC_RUNTIME_DEADLINE_V1')])
def test_no_automatic_deadline_allowed(field,value):
    r=record();r['job']['resources'][field]=value
    with pytest.raises(sub.SubmissionError,match='PRESERVATION_NO_DEADLINE_REQUIRED'):fix.revised_contract(r,1073741824)

def test_project_science_capture_cannot_be_rebound():
    r=record();r['objects'].append({'role':'project_source','sha256':'f'*64})
    with pytest.raises(sub.SubmissionError,match='PRESERVATION_ENGINE_CHANGE_ONLY'):fix.revised_contract(r,1073741824)

def test_nonengineering_handler_cannot_be_rebound():
    r=record();r['job']['execution']['handler_ref']='project.m2_native_dispatch:handler'
    with pytest.raises(sub.SubmissionError,match='PRESERVATION_ENGINE_CHANGE_ONLY'):fix.revised_contract(r,1073741824)

def test_initial_pointer_reconstructs_exact_saved_hash():
    s=session();p=fix.anchor(s)
    assert p['record_sha256']=='8b0d089ee3104b824b7d31dca86cf3a2e0daf5562319b31587714afaa03321e8'
    assert p['target']['completion_sha256']=='ced6ba8aee20e6d94936d264ad92d43142d9d1649706b99cb7379b454c3eeaa4'

def test_noninitial_or_foreign_pointer_refuses():
    s=session();s['parent_pointer_sha256']='0'*64
    with pytest.raises(sub.SubmissionError,match='RECOVERY_ORIGINAL_POINTER_REQUIRED'):fix.anchor(s)

@pytest.mark.parametrize('where',[('workspace','source_sha256'),('question','stopping_rule'),('execution','handler_ref'),('resources','workers'),('result','result_checks')])
def test_semantic_changes_are_detected(where):
    a=record();b=copy.deepcopy(a);group,key=where
    if group=='workspace':b['workspace'][key]='0'*64
    elif group=='result':b['result_contract'][key]=[]
    else:b['job'][group][key]='changed'
    assert fix.semantics(a)!=fix.semantics(b)

def test_legacy_binding_remains_exact_when_no_amendment(tmp_path):
    original={'revision_id':'r','capture_id':'c'};ws=tmp_path/'old'
    assert fix.selected_binding(tmp_path,tmp_path/'change','r',original,ws)==(original,ws)

def test_corrupted_amendment_seal_refuses(tmp_path):
    folder=tmp_path/'change';p=folder/'execution_amendments/r.json';p.parent.mkdir(parents=True)
    p.write_text('{"record_sha256":"bad"}')
    with pytest.raises(sub.SubmissionError,match='CHANGE_RECORD_MISMATCH'):fix.selected_binding(tmp_path,folder,'r',{},tmp_path/'old')

def test_foreign_amendment_binding_refuses(tmp_path):
    folder=tmp_path/'change';p=folder/'execution_amendments/r.json'
    cs._immutable(p,cs._sealed({'schema_id':fix.SCHEMA,'original_binding':{'capture_id':'foreign'},'revision_id':'r'}))
    with pytest.raises(sub.SubmissionError,match='PRESERVATION_AMENDMENT_BINDING'):fix.selected_binding(tmp_path,folder,'r',{},tmp_path/'old')

def test_immutable_byte_writer_never_overwrites(tmp_path):
    p=tmp_path/'record';cs._immutable_bytes(p,b'original');cs._immutable_bytes(p,b'original')
    with pytest.raises(sub.SubmissionError,match='RETAINED_RECORD_CONFLICT'):cs._immutable_bytes(p,b'changed')
    assert p.read_bytes()==b'original'

def test_qualification_is_mandatory_before_mutation(tmp_path):
    with pytest.raises(Exception):fix.recover_store(tmp_path,tmp_path,tmp_path/'no-proof')
    assert not (tmp_path/'CURRENT_LOCAL.json').exists()

def test_absent_completion_cannot_be_recorded(tmp_path):
    with pytest.raises(Exception):cs.record_test_completion(tmp_path,'NO.SESSION','0'*64)
    assert not list(tmp_path.rglob('*RECEIPT*'))

def test_activation_packet_preserves_amendment_bytes(tmp_path):
    p=tmp_path/'execution_amendments/r.json';p.parent.mkdir();p.write_bytes(b'{"scope":"fixture"}\n')
    assert fix.activation_amendment_files(tmp_path,'r')=={'PRESERVATION_AMENDMENT.json':p.read_bytes()}

def test_recovery_retains_reviewed_core_byte_boundaries():
    # Keep the historical baseline immutable. Explicitly account for the reviewed
    # controller evolution; all other pins and exact-byte checks remain required.
    expected=json.loads((F/'UNCHANGED_CORE_SHA256.json').read_text())
    amendment=json.loads((F/'APPROVED_08LIB_CORE_CHANGES.json').read_text())
    assert amendment['schema']=='IG_08LIB_REVIEWED_CORE_PIN_AMENDMENT_V1'
    assert set(amendment['changes'])=={
        'infinity_grid/v05_controller_event_loop.py',
        'infinity_grid/v05_origin_guard.py',
        'infinity_grid/v05_validation_runtime.py',
        'infinity_grid/preservation.py',
        'infinity_grid/result_contracts.py',
    }
    for name,row in amendment['changes'].items():
        assert row['historical_sha256']==expected[name] and row['reason']
        expected[name]=row['reviewed_sha256']
    successor=json.loads((F/'DEV83_REVIEWED_CORE_CHANGES.json').read_text())
    assert successor['schema']=='IG_08LIB_REVIEWED_CORE_PIN_SUCCESSOR_V1'
    assert set(successor['changes'])=={
        'infinity_grid/v05_controller_event_loop.py',
        'infinity_grid/preservation.py',
        'infinity_grid/result_contracts.py',
    }
    for name,row in successor['changes'].items():
        assert row['previous_reviewed_sha256']==expected[name] and row['reason']
        expected[name]=row['reviewed_sha256']
    recovery=json.loads((F/'ATTEMPT_RECOVERY_REVIEWED_CORE_CHANGES.json').read_text())
    assert recovery['schema']=='IG_08LIB_REVIEWED_CORE_PIN_SUCCESSOR_V1'
    assert set(recovery['changes'])=={'infinity_grid/v05_controller_event_loop.py'}
    for name,row in recovery['changes'].items():
        assert row['previous_reviewed_sha256']==expected[name] and row['reason']
        expected[name]=row['reviewed_sha256']
    sealed=json.loads((F/'SQLITE_SEPARATE_EVIDENCE_CORE_CHANGES.json').read_text())
    assert sealed['schema']=='IG_08LIB_REVIEWED_CORE_PIN_SUCCESSOR_V1'
    assert set(sealed['changes'])=={'infinity_grid/v05_controller_event_loop.py','infinity_grid/preservation.py'}
    for name,row in sealed['changes'].items():
        assert row['previous_reviewed_sha256']==expected[name] and row['reason']
        expected[name]=row['reviewed_sha256']
    publication=json.loads((F/'TERMINAL_JSON_EQUIVALENCE_CORE_CHANGES.json').read_text())
    assert publication['schema']=='IG_08LIB_REVIEWED_CORE_PIN_SUCCESSOR_V1'
    assert set(publication['changes'])=={'infinity_grid/preservation.py','infinity_grid/v05_controller_event_loop.py'}
    for name,row in publication['changes'].items():
        assert row['previous_reviewed_sha256']==expected[name] and row['reason']
        expected[name]=row['reviewed_sha256']
    snapshot=json.loads((F/'SNAPSHOT_EVIDENCE_FILTER_CORE_CHANGES.json').read_text())
    assert snapshot['schema']=='IG_08LIB_REVIEWED_CORE_PIN_SUCCESSOR_V1'
    assert set(snapshot['changes'])=={'infinity_grid/v05_controller_event_loop.py'}
    for name,row in snapshot['changes'].items():
        assert row['previous_reviewed_sha256']==expected[name] and row['reason']
        expected[name]=row['reviewed_sha256']
    reviewed=json.loads((F/'DEV140_REVIEWED_CORE_CHANGES.json').read_text())
    assert reviewed['schema']=='IG_08LIB_REVIEWED_CORE_PIN_SUCCESSOR_V1'
    assert set(reviewed['changes'])=={'infinity_grid/submission.py', 'infinity_grid/result_contracts.py', 'infinity_grid/portable_registry.py', 'infinity_grid/v05_controller_event_loop.py', 'infinity_grid/preservation.py', 'infinity_grid/v05_validation_runtime.py'}
    for name,row in reviewed['changes'].items():
        assert row['previous_reviewed_sha256']==expected[name] and row['reason']
        expected[name]=row['reviewed_sha256']
    cleanup=json.loads((F/'DEV146_REVIEWED_CORE_CHANGES.json').read_text())
    assert cleanup['schema']=='IG_08LIB_REVIEWED_CORE_PIN_SUCCESSOR_V1'
    assert set(cleanup['changes'])=={'infinity_grid/v05_controller_event_loop.py','infinity_grid/preservation.py','infinity_grid/v05_validation_runtime.py'}
    for name,row in cleanup['changes'].items():
        assert row['previous_reviewed_sha256']==expected[name] and row['reason']
        expected[name]=row['reviewed_sha256']
    root=Path(__file__).resolve().parents[1]
    for name,digest in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
