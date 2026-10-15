"""Native saved-packet diagnostic Test integration; no science regeneration."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

from infinity_grid import direct_v2_observation as direct, direct_v2_tests as tests
from infinity_grid import o7_science_compat as legacy
from infinity_grid import semantic_sentinel as ss, theorem_registry as tr
from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid.scientific_architecture import UndeclaredCapabilityRead, ObservationView


def main():
    started = time.monotonic()
    inputs = json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params = json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role, sha in params['input_hashes'].items():
        assert hashlib.sha256(Path(inputs[role]).read_bytes()).hexdigest() == sha
    rejected=json.loads(Path(inputs['q07_completion']).read_text())
    assert rejected['status']=='RESULT_REJECTED' and rejected['evidence_status']=='REJECTED'
    assert rejected['completion_sha256']==params['q07_completion_sha256']
    assert canonical_sha256({k:v for k,v in rejected.items() if k!='completion_sha256'})==rejected['completion_sha256']
    contract = json.loads(Path(inputs['test_contract']).read_text())
    direct_contract = json.loads(Path(inputs['direct_contract']).read_text())
    requested = list(contract['test_specs'])
    assert len(requested) == 10
    baseline = ss.verify_native_semantics(raise_on_change=False)
    theorems = tr.verify_all_theorems(raise_on_stale=False)
    assert baseline['status'] == 'REOPEN_REQUIRED' and len(theorems['stale_theorems']) == 4
    parent = json.loads(Path(inputs['q06_completion']).read_text())
    assert parent['completion_sha256'] == params['q06_completion_sha256']
    assert canonical_sha256({k:v for k,v in parent.items() if k!='completion_sha256'}) == parent['completion_sha256']
    assert parent['status'] == 'COMPLETED' and parent['evidence_status'] == 'VERIFIED'
    preserved = json.loads(Path(inputs['q06_preserved']).read_text())
    assert preserved['outbox']['status']=='PRESERVED' and preserved['outbox']['pending_bytes']==0
    calls = set()
    def profile(frame, event, arg):
        if event == 'call':calls.add((str(frame.f_globals.get('__name__','')),frame.f_code.co_name))
    assert sys.getprofile() is None
    refusals = []
    def refuse(name, callback, exception=tests.DiagnosticTestError):
        try:callback()
        except exception as exc:refusals.append({'name':name,'reason':str(exc)})
        else:raise AssertionError('Missing refusal:' + name)
    rows = []
    with tempfile.TemporaryDirectory(prefix='ig_direct_q07_') as directory:
        root = Path(directory)
        with zipfile.ZipFile(inputs['saved_inputs']) as z:
            names=z.namelist()
            assert len(names)==len(set(names)) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
            z.extractall(root)
        binding=json.loads(Path(inputs['saved_binding']).read_text())
        session=direct.SavedDirectSession(root,binding,direct_contract)
        sys.setprofile(profile)
        try:
            for domain,level in sorted(session.packets):
                result=tests.execute(session,domain,level,contract,direct_contract,requested)
                packet=session.packets[domain,level]
                assert len(result['executions'])==10
                for exe in result['executions']:
                    ev=exe['evidence_payload'];old=ev['legacy_projection_ref']
                    # Independent expected projection by frozen packet field path.
                    expected = {
                        legacy.TEST_GRAMMAR:{'grammar_sha256':packet['grammar_sha256']},
                        legacy.TEST_DIVERSITY:packet['diversity'],
                        legacy.TEST_BRANCHING:packet['branching'],
                        legacy.TEST_SYMMETRY:packet['symmetry'],
                        legacy.TEST_OVERLAP:packet['overlap_gluing'],
                        legacy.TEST_LINEAGE:packet['lineage'],
                        legacy.TEST_QUOTIENT:packet['quotient_observer'],
                        legacy.TEST_TOPOLOGY:packet['topology_services'],
                        legacy.TEST_OBSTRUCTION:packet['obstruction_relief'],
                        legacy.TEST_SATURATION:{'states':packet['states'],'raw_growth':packet['raw_growth'],
                            'normalized_signature':packet['normalized_signature'],
                            'normalized_signature_sha256':packet['normalized_signature_sha256']},
                    }[old]
                    assert ev['projection']==expected
                    assert exe['evidence_sha256']==canonical_sha256(ev)
                    assert exe['finding']['test_ref'] not in legacy.LEAF_TEST_REFS
                    assert exe['finding']['evidence_class']==exe['finding']['claim_ceiling']=='RECONNAISSANCE'
                    assert exe['finding']['scientific_status']=='NULL' and not exe['finding']['reopen_triggered']
                    assert set(exe['observation_view']['capability_bindings'])=={'V2_DIAGNOSTIC_PACKET'}
                    assert set(exe['actual_shape_reads'])==set(tests.READS[old])
                rows.append(result)
        finally:sys.setprofile(None)
        for ref in legacy.LEAF_TEST_REFS+(legacy.TEST_BOUNDARY_SENTINEL,):
            refuse('legacy_identity:'+ref,lambda ref=ref:tests.compile_tests(contract,direct_contract,[ref]))
        for ref in (tests.diagnostic_ref(legacy.TEST_REPRESENTATION),'IG_TEST_DIRECT_V2_BOUNDARY_DIAGNOSTIC@1.0.0'):
            refuse('missing_evidence:'+ref,lambda ref=ref:tests.compile_tests(contract,direct_contract,[ref]))
        for key,value in [('production_authority','PRODUCTION'),('generation_allowed',True),
                          ('legacy_regime_gate_bypass',True),('quantity_policy','COMPLETE_SUCCESSOR_COUNTER')]:
            changed=deepcopy(contract);changed[key]=value
            changed['science_sha256']=canonical_sha256({k:v for k,v in changed.items() if k!='science_sha256'})
            refuse(key,lambda changed=changed:tests.compile_tests(changed,direct_contract,requested))
        changed=deepcopy(contract);changed['read_bindings'][legacy.TEST_BRANCHING]={}
        changed['science_sha256']=canonical_sha256({k:v for k,v in changed.items() if k!='science_sha256'})
        refuse('omitted_read',lambda:tests.compile_tests(changed,direct_contract,requested))
        for req in ([],[requested[0],requested[0]],['COMPLETE_COUNTER']):
            refuse('invalid_request:'+str(req),lambda req=req:tests.compile_tests(contract,direct_contract,req))
        refuse('unqualified_depth',lambda:tests.execute(session,'selected',100,contract,direct_contract,requested),direct.DirectObservationError)
        refuse('unbound_session',lambda:tests.execute(object(),'selected',13,contract,direct_contract,requested))
        facade=tests._ProjectionReads(legacy.TEST_GRAMMAR,session.packets['selected',13])
        refuse('hidden_shape_read',lambda:facade.read('RESOURCE_COUNTERS'))
        view=ObservationView(entity_instance_ref='refusal-check',entity_class_ref=tests.ENTITY,
            regime_ref=tests.REGIME,regime_depth=13,test_ref=requested[0],
            required_capabilities=['V2_DIAGNOSTIC_PACKET'],capability_payloads={'V2_DIAGNOSTIC_PACKET':{}},
            source_science_sha256='0'*64,provider_ref=tests.PROVIDER)
        refuse('undeclared_capability_read',lambda:view.read('RESOURCE_COUNTERS'),UndeclaredCapabilityRead)
        # Dispatch results cannot mutate the admitted source population.
        before=deepcopy(rows[0]['executions'][0]['evidence_payload']['projection'])
        rows[0]['executions'][0]['evidence_payload']['projection'].clear()
        first=tests.execute(session,'full_pool',11,contract,direct_contract,requested)
        assert first['executions'][0]['evidence_payload']['projection']==before
        rows[0]=first
        session.close()
        refuse('closed_session',lambda:tests.execute(session,'selected',13,contract,direct_contract,requested),direct.DirectObservationError)
    forbidden=[list(c) for c in sorted(calls) if
        c[0].endswith('fixed_grammar_transport') or
        c[1] in ('_scan_level','_action_aggregate_v2','_states_from_dag','_read_probe',
                 '_resource_future_signature','reserve_external','_build_next_panel','_farthest_select','load_earned_regime_laws')]
    assert not forbidden and sys.getprofile() is None
    for ident in theorems['stale_theorems']:
        try:tr.require_current_theorems((ident,))
        except tr.StaleTheoremError:pass
        else:raise AssertionError('stale theorem admitted')
    assert ss.verify_native_semantics(raise_on_change=False)==baseline
    assert tr.verify_all_theorems(raise_on_stale=False)==theorems
    assert len(refusals)==27
    write_json_atomic(Path('TEST_DISPATCHES.json'),{'dispatches':rows,'packet_count':10,'projection_count':100})
    write_json_atomic(Path('COMPATIBILITY.json'),tests.compatibility_report())
    write_json_atomic(Path('REFUSALS.json'),{'status':'PASS','count':len(refusals),'checks':refusals})
    write_json_atomic(Path('EXECUTION_TRACE.json'),{'status':'PASS','forbidden_calls':forbidden,
        'measured_calls':[list(c) for c in sorted(calls)],'profile_restored':True,
        'new_scanner_calls':0,'new_DAG_reconstructions':0})
    write_json_atomic(Path('QUALIFICATION_RESULT.json'),{'status':'PASS',
        'scientific_outcome':'DIRECT_V2_DIAGNOSTIC_TEST_DISPATCH_QUALIFIED_WITH_MATCHED_CONTEXT_REFUSALS',
        'rejected_predecessor_completion_sha256':rejected['completion_sha256'],
        'packet_count':10,'diagnostic_test_count':10,'projection_count':100,'refusal_checks':27,
        'stale_theorems':4,'legacy_tests_authorized':0,'explicit_evidence_refusals':2,
        'production_authority':'NONE','generation_allowed':False,'full_pipeline_qualified':False,
        'package2_status':'OPEN_COUNTER_READ_AND_MATCHED_LONGITUDINAL_QUALIFICATION',
        'wall_seconds':time.monotonic()-started})


if __name__=='__main__':main()
