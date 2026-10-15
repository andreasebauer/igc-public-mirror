"""Qualified continuation: reuse Q05 observations, correct downstream metadata scope."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

from infinity_grid import direct_v2_observation as direct
from infinity_grid import semantic_sentinel as ss, theorem_registry as tr
from infinity_grid.scientific_architecture import ScientificProtocolRegistry
from infinity_grid.canon import canonical_sha256, write_json_atomic


def main():
    started = time.monotonic()
    inputs = json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params = json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role, sha in params['input_hashes'].items():
        assert hashlib.sha256(Path(inputs[role]).read_bytes()).hexdigest() == sha
    preservation = json.loads(Path(inputs['q05_preserved']).read_text())
    assert preservation['outbox']['status'] == 'PRESERVED' and preservation['outbox']['pending_bytes'] == 0
    with zipfile.ZipFile(inputs['q05_partial']) as z:
        items = {name:z.read(name) for name in z.namelist()}
    parent = json.loads(items['COMPLETION.json'])
    capture = json.loads(items['CAPTURE.json'])
    assert parent['completion_sha256'] == params['q05_completion_sha256']
    assert parent['completion_sha256'] == canonical_sha256({k:v for k,v in parent.items() if k != 'completion_sha256'})
    assert parent['status'] == 'RESULT_REJECTED' and parent['evidence_status'] == 'REJECTED'
    assert parent['result']['exit_code'] == 1 and parent['result']['outcome'] == 'PROCESS_FAILED'
    assert capture['capture_id'] == params['q05_capture_id']
    assert capture['capture_id'] == canonical_sha256({k:v for k,v in capture.items() if k != 'capture_id'})
    assert parent['source_sha256'] == capture['workspace']['source_sha256']
    assert parent['registration_sha256'] == capture['job']['registration_sha256']
    evidence = {row['path']:row for row in parent['evidence']}
    for name, raw in items.items():
        path = 'script/stderr.log' if name == 'SCRIPT_STDERR.txt' else 'script/work/' + name
        if name in ('CAPTURE.json', 'COMPLETION.json'):
            continue
        row = evidence[path]
        assert hashlib.sha256(raw).hexdigest() == row['sha256'] and len(raw) == row['size_bytes']
    assert 'assert missing' in items['SCRIPT_STDERR.txt'].decode()
    comparisons = json.loads(items['DIRECT_ROUTE_COMPARISONS.json'])
    trace = json.loads(items['DIRECT_EXECUTION_TRACE.json'])
    exports = json.loads(items['DIRECT_ROUTE_EXPORTS.json'])['exports']
    assert comparisons['status'] == 'PASS' and comparisons['packets'] == len(exports) == 10
    assert trace['status'] == 'PASS' and trace['forbidden_calls'] == []
    assert trace['transport_or_sampled_future_calls'] == 0 and trace['profile_restored'] is True
    binding = json.loads(Path(inputs['saved_binding']).read_text())
    contract = json.loads(Path(inputs['direct_contract']).read_text())
    direct.validate_contract(contract)
    baseline = ss.verify_native_semantics(raise_on_change=False)
    theorems = tr.verify_all_theorems(raise_on_stale=False)
    assert baseline['status'] == 'REOPEN_REQUIRED' and len(theorems['stale_theorems']) == 4
    # Temporary reconstruction of input FILES only. No state DAG reconstruction,
    # scanner calls, recipe generation or Q05 packet observation is repeated.
    with tempfile.TemporaryDirectory(prefix='ig_direct_q06_') as directory:
        root = Path(directory)
        with zipfile.ZipFile(inputs['saved_inputs']) as z:
            names = z.namelist()
            assert len(names) == len(set(names)) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
            z.extractall(root)
        session = direct.SavedDirectSession(root, binding, contract)
        rows = []
        for exported, row in zip(exports, comparisons['rows']):
            p = exported['provenance']
            assert p['kind'] == 'NATIVE_MATERIALIZED_OBSERVATION'
            saved = session.dispatch(row['domain'], row['level'])
            assert exported['observations'] == saved['observations']
            assert exported['packet_sha256'] == saved['packet_sha256'] == row['packet_sha256']
            assert exported['science_sha256'] == canonical_sha256({k:v for k,v in exported.items() if k != 'science_sha256'})
            assert exported['science_sha256'] == row['native_result_sha256']
            assert exported['theorem_transport'] == 'NOT_CONSUMED'
            assert saved['provenance']['producing_capture_id'] == binding['prefix_capture_id']
            assert saved['provenance']['producing_completion_sha256'] == binding['prefix_completion_sha256']
            rows.append({'domain':row['domain'], 'level':row['level'], 'state_count':row['states'],
                         'packet_sha256':row['packet_sha256'], 'observations_reused':True})
        refusals = []
        def refuse(name, callback):
            try:
                callback()
            except direct.DirectObservationError as exc:
                refusals.append({'name':name, 'reason':str(exc)})
            else:
                raise AssertionError('Missing refusal:' + name)
        for c in ('TRANSITION_SYSTEM','ACTION_MULTIPLICITY','MATCHED_BACKBONE',
                  'INHERITED_NO_READ_THEOREM','THEOREM_GROWTH_BOUND'):
            refuse(c, lambda c=c: direct.compile_requests(contract,[c]))
        for d,l in [('selected',14),('selected',100),('full_pool',10),('other',13)]:
            refuse(str((d,l)), lambda d=d,l=l:session.dispatch(d,l))
        invalid = deepcopy(contract);invalid['production_authority']='PRODUCTION'
        refuse('authority_escalation', lambda:direct.compile_requests(invalid,list(direct.CAPABILITIES)))
        invalid = deepcopy(contract);invalid['package_payload_sha256']='0'*64
        invalid['science_sha256']=canonical_sha256({k:v for k,v in invalid.items() if k!='science_sha256'})
        refuse('resealed_wrong_source', lambda:direct.compile_requests(invalid,list(direct.CAPABILITIES)))
        invalid = deepcopy(contract);invalid['read_set'].remove('typed_edges')
        invalid['science_sha256']=canonical_sha256({k:v for k,v in invalid.items() if k!='science_sha256'})
        refuse('hidden_read_omitted', lambda:direct.compile_requests(invalid,list(direct.CAPABILITIES)))
        refuse('invalid_native_population', lambda:direct.observe_materialized([],13,[],contract))
        got=session.dispatch('selected',13);got['observations']['V2_BRANCHING_DIAGNOSTIC'].clear()
        assert session.dispatch('selected',13)['observations']['V2_BRANCHING_DIAGNOSTIC']
        artifact=root/binding['artifacts']['materialized_audit']['file_name'];old=artifact.read_bytes()
        artifact.write_bytes(old+b' ')
        try:session.dispatch('selected',13)
        except Exception as exc:
            assert 'SAVED_INPUT_BYTES_CHANGED' in str(exc)
            refusals.append({'name':'input_changed_after_admission','reason':str(exc)})
        else:raise AssertionError('modified input admitted')
        finally:artifact.write_bytes(old)
        downstream=[]
        for test in ScientificProtocolRegistry().tests():
            required=test['required_capabilities'];missing=sorted(set(required)-set(direct.CAPABILITIES))
            if required:
                assert missing
                refuse('legacy_test:'+test['test_id'],lambda required=required:direct.compile_requests(contract,required))
                status='NEEDS_EXPLICIT_PROTOCOL_ADAPTER'
            else:
                # Zero direct capabilities is not permission to run a boundary
                # sentinel without its dependency Test findings and Regime context.
                status='METADATA_ONLY_DEPENDENCIES_REQUIRE_REVIEW'
            downstream.append({'test_id':test['test_id'],'required_capabilities':required,
                'missing':missing,'status':status,'dependency_test_refs':test.get('dependency_test_refs',[])})
        assert len(downstream)==12
        assert sum(bool(r['required_capabilities']) for r in downstream)==11
        session.close();refuse('closed_session',lambda:session.dispatch('selected',13))
    for ident in theorems['stale_theorems']:
        try:tr.require_current_theorems((ident,))
        except tr.StaleTheoremError:pass
        else:raise AssertionError('stale theorem admitted')
    assert ss.verify_native_semantics(raise_on_change=False)==baseline
    assert tr.verify_all_theorems(raise_on_stale=False)==theorems
    assert len(refusals)==26
    write_json_atomic(Path('PARENT_REUSE.json'),{'status':'PASS','parent_status_retained':'RESULT_REJECTED',
        'parent_capture_id':capture['capture_id'],'parent_completion_sha256':parent['completion_sha256'],
        'native_observations_reused':10,'new_scanner_calls':0,'rows':rows,
        'scope':'New continuation verifies/reuses sealed partial observations; no parent success retag'})
    write_json_atomic(Path('DIRECT_EXECUTION_TRACE_REUSED.json'),trace)
    write_json_atomic(Path('DOWNSTREAM_COMPATIBILITY.json'),{'status':'EXPLICIT_ADAPTER_REQUIRED',
        'legacy_tests_checked':12,'tests_requiring_data_adapter':11,'metadata_only_tests':1,
        'tests':downstream,'missing_requirements_silently_ignored':False,'full_pipeline_qualified':False})
    write_json_atomic(Path('REFUSAL_CHECKS.json'),{'status':'PASS','check_count':len(refusals),
        'checks':refusals,'caller_mutation_isolated':True})
    write_json_atomic(Path('QUALIFICATION_RESULT.json'),{'status':'PASS',
        'scientific_outcome':'DIRECT_V2_ROUTE_CONTINUATION_QUALIFIED_WITH_EXPLICIT_DOWNSTREAM_GAPS',
        'packets_compared':10,'observations_reexecuted':0,'transport_or_sampled_future_calls':0,
        'stale_theorems':4,'legacy_data_adapters_required':11,'metadata_only_tests':1,'refusal_checks':26,
        'production_authority':'NONE','generation_allowed':False,
        'package2_status':'OPEN_DOWNSTREAM_PROTOCOL_INTEGRATION','wall_seconds':time.monotonic()-started})


if __name__=='__main__':main()
