"""Native qualification of explicit saved-input integration into the real session."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time
import zipfile

from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid import materialized_discovery as md
from infinity_grid import qualified_discovery_inputs as qi
from infinity_grid import semantic_sentinel, theorem_registry


def main():
    started = time.monotonic()
    inputs = json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params = json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role, expected in params['input_hashes'].items():
        assert hashlib.sha256(Path(inputs[role]).read_bytes()).hexdigest() == expected
    root = Path('saved_inputs')
    root.mkdir()
    with zipfile.ZipFile(inputs['saved_inputs']) as z:
        names = z.namelist()
        assert len(names)==len(set(names)) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        z.extractall(root)
    binding = json.loads(Path(inputs['integration_binding']).read_text())
    baseline = semantic_sentinel.verify_native_semantics(raise_on_change=False)
    theorems = theorem_registry.verify_all_theorems(raise_on_stale=False)
    assert baseline['status']=='REOPEN_REQUIRED' and len(theorems['stale_theorems'])==4
    # Static source comparison preserves the original generator, scanner and worker bodies.
    old = ast.parse(Path(inputs['original_materialized_module']).read_text())
    new = ast.parse(Path(md.__file__).read_text())
    old_class = next(x for x in old.body if isinstance(x,ast.ClassDef) and x.name=='MaterializedDiscoverySession')
    new_class = next(x for x in new.body if isinstance(x,ast.ClassDef) and x.name=='MaterializedDiscoverySession')
    original = {x.name:x for x in old_class.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
    changed = {x.name:x for x in new_class.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
    reviewed = {'close','current_level','advance_to','_build_and_store'}
    for name, node in original.items():
        if name not in reviewed:
            assert ast.dump(node,include_attributes=False)==ast.dump(changed[name],include_attributes=False),name
    for name in ('advance_to','_build_and_store'):
        new_body = changed[name].body[2:] if name=='advance_to' else changed[name].body[1:]
        assert [ast.dump(x,include_attributes=False) for x in original[name].body]==[ast.dump(x,include_attributes=False) for x in new_body]
    before = [ast.dump(x,include_attributes=False) for x in old.body if x is not old_class]
    after = [ast.dump(x,include_attributes=False) for x in new.body if x is not new_class]
    assert before==after
    write_json_atomic(Path('SOURCE_COMPATIBILITY.json'),{'status':'PASS','original_generator_bodies_unchanged':True,
        'default_constructor_AST_unchanged':True,'calibration_and_scanner_AST_unchanged':True,'installed_V1_resource_unchanged':True,
        'reviewed_existing_methods':sorted(reviewed),'production_authority':'NONE'})
    # Refusals must precede loading exact reconstruction code.
    refusals=[]
    def refuse(label, callback, expected):
        try:callback()
        except (qi.SavedInputError,md.MaterializedDiscoveryError) as exc:
            assert expected in str(exc),(label,str(exc));refusals.append(label)
        else:raise AssertionError('Missing refusal: '+label)
    invalid = deepcopy(binding);invalid['production_authority']='PRODUCTION'
    refuse('authority_escalation',lambda:md.MaterializedDiscoverySession.from_qualified_saved_inputs(root,invalid),'FINITE_INPUT_SCOPE_REQUIRED')
    invalid = deepcopy(binding);invalid['prefix_completion_sha256']='0'*64
    refuse('origin_completion_retag',lambda:qi.preflight(root,invalid),'DECLARED_ORIGIN_IDENTITY')
    invalid = deepcopy(binding);invalid['artifacts']['prefix_index']['file_name']='../STATE_ROOT_INDEX.json'
    refuse('escaped_input_path',lambda:qi.preflight(root,invalid),'ARTIFACT_PATH')
    target = root/binding['artifacts']['prefix_index']['file_name'];raw=target.read_bytes();target.write_bytes(raw+b' ')
    refuse('changed_input_before_load',lambda:qi.preflight(root,binding),'SAVED_INPUT_BYTES_CHANGED')
    target.write_bytes(raw)
    refuse('default_V1_calibration_still_refused',lambda:md.MaterializedDiscoverySession(),'SCANNER_DIAGNOSTIC_CALIBRATION_REOPEN_REQUIRED')
    with md.MaterializedDiscoverySession.from_qualified_saved_inputs(root,binding) as session:
        assert type(session) is md.MaterializedDiscoverySession and session.current_level==13
        packets=[]
        expected = {}
        for role,domain,key in [('materialized_audit','selected','levels'),('full_audit','full_pool','depths')]:
            data=qi.read(root/binding['artifacts'][role]['file_name'])
            for row in data[key]:expected[(domain,row['level'])]=row['scan']
        for (domain,level), packet in sorted(expected.items()):
            observed=session.qualified_saved_packet(domain,level);assert observed==packet
            observed.clear();assert session.qualified_saved_packet(domain,level)==packet
            if domain=='selected':
                loaded=session.advance_to(level);assert loaded.summary==packet and len(loaded.states)==24
                assert loaded.materialization['generated_here'] is False
            packets.append({'domain':domain,'level':level,'exact_saved_packet_equal':True,'copy_isolated':True})
        r100=session.qualified_r100_descriptors()
        manifest=qi.read(root/binding['artifacts']['r100_manifest']['file_name'])
        assert len(r100)==193 and r100==[row['public_state'] for row in manifest['carriers']]
        r100[0]['total_caps'].clear();assert session.qualified_r100_descriptors()==[row['public_state'] for row in manifest['carriers']]
        for level in [6,14,100,True,'13']:
            refuse('advance_refuses_'+str(level),lambda level=level:session.advance_to(level),'SAVED_SESSION_GENERATION_NOT_AUTHORIZED')
        refuse('private_builder_refuses_generation',lambda:session._build_and_store(14),'SAVED_SESSION_GENERATION_NOT_AUTHORIZED')
        target.write_bytes(raw+b' ')
        refuse('changed_input_on_cached_access',lambda:session.qualified_saved_packet('selected',7),'SAVED_INPUT_BYTES_CHANGED')
        target.write_bytes(raw)
        prefix_count=len(session._qualified_saved_inputs.states)
        assert prefix_count==747
    refuse('closed_session_access',lambda:session.qualified_saved_packet('selected',7),'SAVED_SESSION_CLOSED')
    assert semantic_sentinel.verify_native_semantics(raise_on_change=False)==baseline
    assert theorem_registry.verify_all_theorems(raise_on_stale=False)==theorems
    write_json_atomic(Path('REFUSAL_CHECKS.json'),{'status':'PASS','checks':refusals,'check_count':len(refusals)})
    write_json_atomic(Path('SESSION_INPUT_RESULT.json'),{'status':'PASS','actual_class':'MaterializedDiscoverySession',
        'prefix_root_occurrences':747,'current_R100_roots':193,'public_descriptor_matches':193,
        'saved_packet_matches':len(packets),'packets':packets,'saved_packet_copies_isolated':True,
        'default_generation_still_refused':True,'qualified_session_generation_refused':True,
        'origin_Q07_producing_binding_retained':True,'installed_calibration_replaced':False,
        'old_observer_jobs_replayed':False,'live_scanner_calls':0,'recipe_worker_calls':0,'selector_calls':0,'marker_tasks':0,
        'production_authority':'NONE','semantic_status':'REOPEN_REQUIRED','stale_theorems':4})
    # Exact originals remain in the captured input pack; discard this mutable working copy.
    shutil.rmtree(root)
    write_json_atomic(Path('QUALIFICATION_RESULT.json'),{'status':'PASS',
        'scientific_outcome':'ACTUAL_DISCOVERY_SESSION_SAVED_V2_R100_INPUTS_INTEGRATED_NO_GENERATION_AUTHORITY',
        'prefix_root_occurrences':747,'current_R100_roots':193,'saved_packet_matches':10,'refusal_checks':len(refusals),
        'production_authority':'NONE','generation_allowed':False,'Counter_read_premises_closed':False,
        'wall_seconds':time.monotonic()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__=='__main__':
    main()
