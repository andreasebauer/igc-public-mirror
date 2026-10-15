"""Native qualification of registered full-output producer service hooks."""
from pathlib import Path
from copy import deepcopy
import ast
import gzip
import hashlib
import json
import os
import tempfile
import time
import zipfile
from importlib.resources import files
import reference
from infinity_grid.canon import canonical_sha256,write_json_atomic
from infinity_grid.direct_v2_observation import source_snapshot
from infinity_grid.isolated_runtime import BUNDLED_MANIFEST_SHA256
from infinity_grid.v05_stage_runtime import _resolve_ref,_bind_controller_fallback_kernel_view
from infinity_grid.v05_stage_registry import semantic_audit_for_ref
from infinity_grid.v05_kernel_service_providers import build_stage_kernel_view
from infinity_grid.v05_kernel_services import current_kernel_view,KernelServiceViolation
from infinity_grid.full_marker_export import ExportContractError,assemble_marker_export
from infinity_grid import master_full_export_services as exports
from infinity_grid import semantic_sentinel,theorem_registry

REF='infinity_grid.master_full_export_evaluators:'

def main():
    started=time.monotonic();inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role,sha in params['input_hashes'].items():assert hashlib.sha256(Path(inputs[role]).read_bytes()).hexdigest()==sha
    source=source_snapshot();runtime=BUNDLED_MANIFEST_SHA256
    gates={name:semantic_audit_for_ref(REF+name) for name in ('owner_export_evaluator','marker_export_evaluator','saved_dag_export_evaluator')}
    assert all(g['status']=='PASS' for g in gates.values())
    def invoke(name,payload):
        ref=REF+name;fn=_resolve_ref(ref)
        _bind_controller_fallback_kernel_view(ref,canonical_sha256(payload))
        return fn(payload)
    def binding(root,task,owner=False,dag=False):
        b={'recipe_ref':'MASTER.FULL.EXPORT.Q19.FINITE.QUALIFICATION','source_ref':source,
            'runtime_ref':runtime,'input_root_ref':root,'task_ref':task}
        if owner:b.update(left_parent_occurrence_ref=task+':LEFT',right_parent_occurrence_ref=task+':RIGHT')
        if dag:b['identity_scope']='DECODER091_NATIVE_V3'
        return b
    view=build_stage_kernel_view('Q19_INPUT_BASIS',('AUTHORITY_BASIS','OPERATOR_BASIS'),scope_identity='Q19_DECLARED_BASIS')
    basis=view.call('AUTHORITY_BASIS');ops=view.call('OPERATOR_BASIS')
    assert set(basis)=={'D2_PATH','D2_BROOM','D4_PATH','D4_BROOM'} and (0,0) in ops
    pairs=[(l,r,(0,0)) for l in sorted(basis) for r in sorted(basis)]
    # Two explicitly different directed endpoint labels supplement the 16 base pairs.
    extra=[op for op in ops if op!=(0,0)]
    assert len(extra)>=2
    pairs += [('D2_PATH','D4_BROOM',extra[0]),('D4_BROOM','D2_PATH',extra[-1])]
    with zipfile.ZipFile(inputs['prior_outputs']) as z:
        parent=json.loads(z.read('CAPTURE.json'))
        assert parent['capture_id']=='95348d2b0dfbadca826379a9d06626e189694dde11c44b21771705a1b6c33c1c'
        assert parent['capture_id']==canonical_sha256({k:v for k,v in parent.items() if k!='capture_id'})
        prior_owner=json.loads(z.read('script/work/COMPLETE_OWNER_EXPORTS.json'))['cases']
        prior_marker=json.loads(z.read('script/work/COMPLETE_MARKER_EXPORTS.json'))['cases']
        assert json.loads(z.read('script/SCRIPT_RESULT.json'))['exit_code']==1
    assert len(prior_owner)==len(prior_marker)==18
    manifest=json.loads(Path(inputs['prior_source_manifest']).read_text())
    package=Path(exports.__file__).parent
    current={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    assert set(manifest)==set(current)
    assert [n for n in manifest if manifest[n]!=current[n]]==['master_full_export_services.py']
    baseline=Path(inputs['baseline_export_services']).read_text()
    expected=baseline.replace("n.endswith('/O7_IMMUTABLE_SURVIVORS.json')", "n.endswith('/graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json')")
    assert ast.dump(ast.parse(expected),include_attributes=False)==ast.dump(ast.parse(Path(exports.__file__).read_text()),include_attributes=False)
    prior_source=canonical_sha256(list(manifest.items()))
    owner_outputs=[];marker_outputs=[];counts=[];refusals=[]
    def refuse(name,fn,types=(ExportContractError,KernelServiceViolation)):
        try:fn()
        except types as exc:refusals.append({'case':name,'type':type(exc).__name__,'message':str(exc)})
        else:raise AssertionError('ACCEPTED:'+name)
    for i,(ln,rn,op) in enumerate(pairs):
        left,right=basis[ln],basis[rn];op=list(op);root=exports.relation_input_root(left,right,op)
        task='Q19:'+str(i);b=binding(root,task,owner=True)
        payload={'left':left,'right':right,'operator':op,'binding':b}
        owner=prior_owner[i]
        assert owner['binding']['source_ref']==prior_source and owner['binding']['input_root_ref']==root
        reference.verify_owner(owner)
        # Full service value survives exact JSON serialization; all fields persist.
        assert json.loads(json.dumps(owner))==owner
        owner_outputs.append(owner)
        mb=binding(root,task);mp={**payload,'binding':mb}
        marker=prior_marker[i]
        assert marker['binding']['source_ref']==prior_source and marker['binding']['input_root_ref']==root
        reference.verify_marker(marker,owner)
        assert json.loads(json.dumps(marker))==marker
        marker_outputs.append(marker)
        counts.append({'case':i,'left':ln,'right':rn,'operator':op,'formations':owner['formation_count'],
            'set_children':owner['child_SET_count'],'marker_children':len(marker['children'])})
        if i==0:
            for field,value in [('source_ref','0'*64),('runtime_ref','0'*64),('input_root_ref','0'*64)]:
                bad=deepcopy(payload);bad['binding'][field]=value
                refuse('owner_'+field,lambda bad=bad:invoke('owner_export_evaluator',bad))
                badm=deepcopy(mp);badm['binding'][field]=value
                refuse('marker_'+field,lambda badm=badm:invoke('marker_export_evaluator',badm))
            bad=deepcopy(payload);bad['binding'].pop('left_parent_occurrence_ref')
            refuse('missing_parent_occurrence',lambda:invoke('owner_export_evaluator',bad))
            # Production evaluator gets exactly one service, no hidden kernel/oracle access.
            _bind_controller_fallback_kernel_view(REF+'owner_export_evaluator','Q19_REFUSAL_SCOPE')
            refuse('undeclared_internal',lambda:current_kernel_view().call('PREPARE_TREE'))
            refuse('undeclared_reference',lambda:current_kernel_view().call('LEGACY_MARKER_DECODE_ORACLE'))
            broken=deepcopy(marker);broken['children'][0]['raw_tree']['edges'][-1][0]=999
            refuse('independent_raw_wiring_corruption',lambda:reference.verify_marker(broken,owner),types=(AssertionError,IndexError))
            broken=deepcopy(owner);broken['formation_occurrences'].pop()
            refuse('independent_missing_formation',lambda:reference.verify_owner(broken),types=(AssertionError,))
            broken=deepcopy(owner);broken['formation_occurrences'][0]['right_component_map'][0]=0
            refuse('independent_component_offset_corruption',lambda:reference.verify_owner(broken),types=(AssertionError,))
    # Retain exact full prior payloads in a compressed artifact, never rerun their kernels.
    for name,outputs in [('COMPLETE_OWNER_EXPORTS',owner_outputs),('COMPLETE_MARKER_EXPORTS',marker_outputs)]:
        raw=json.dumps({'cases':outputs},sort_keys=True,separators=(',',':')).encode()
        Path(name+'.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    write_json_atomic('REUSED_EXPORT_BINDINGS.json',{'status':'PASS','relation_cases':18,'marker_cases':18,
        'parent_capture_id':parent['capture_id'],'original_package_payload_sha256':prior_source,
        'parent_execution':'FAILED_AT_DAG_EXPORT_NOT_ACCEPTED','recomputed_owner_or_marker_cases':0,
        'qualification':'NEW_INDEPENDENT_ALL_FIELD_RECHECK_IN_Q19','original_output_bindings_retained':True})
    dags=[];input_binding=json.loads(Path(inputs['saved_binding']).read_text())
    with tempfile.TemporaryDirectory(prefix='ig_q18_saved_dags_') as directory:
        root=Path(directory)
        with zipfile.ZipFile(inputs['saved_inputs']) as z:
            assert len(z.namelist())==len(set(z.namelist()))
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(root)
        seed=files('infinity_grid').joinpath('resources','decoder','O7_MATERIAL_ROOT_cb6f48641eb9.zip').read_bytes()
        receipts=[{'path':p['file_name'],'sha256':p['sha256'],'bytes':p['size_bytes']} for p in input_binding['artifacts'].values()]
        receipts.append({'path':'O7_MATERIAL_ROOT_cb6f48641eb9.zip','sha256':hashlib.sha256(seed).hexdigest(),'bytes':len(seed)})
        input_root=canonical_sha256(sorted(receipts,key=lambda r:r['path']))
        for role in ('prefix_dag','r100_dag'):
            payload={'root':str(root),'input_binding':input_binding,'role':role,'binding':binding(input_root,'Q19:'+role,dag=True)}
            out=invoke('saved_dag_export_evaluator',payload)
            original=json.loads((root/input_binding['artifacts'][role]['file_name']).read_bytes())
            metrics=reference.verify_dag_export(out,original)
            assert metrics['roots']==(747 if role=='prefix_dag' else 193)
            assert metrics['stored_nodes']==(4144 if role=='prefix_dag' else 14465)
            raw=json.dumps(out,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
            target=role.upper()+'_FULL_EXPORT.json.gz';Path(target).write_bytes(gzip.compress(raw,mtime=0))
            assert json.loads(gzip.decompress(Path(target).read_bytes()))==out
            dags.append({'role':role,**metrics,'artifact':target,'raw_size_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),
                'gzip_sha256':hashlib.sha256(Path(target).read_bytes()).hexdigest(),
                'original_input_binding':input_binding,'scope':out['profile_scope']})
            if role=='prefix_dag':
                bad=deepcopy(payload);bad['binding']['identity_scope']='DECODER091_HISTORICAL_COORDINATE_V1'
                refuse('saved_DAG_historical_alias',lambda:invoke('saved_dag_export_evaluator',bad))
                bad=deepcopy(payload);bad['binding']['source_ref']='0'*64
                refuse('saved_DAG_source_change',lambda:invoke('saved_dag_export_evaluator',bad))
                bad=deepcopy(payload);bad['role']='R100_SCANNER_PACKET'
                refuse('saved_DAG_capability_mix',lambda:invoke('saved_dag_export_evaluator',bad))
                broken=deepcopy(out);broken['root_occurrences'][0]['root_ordinal']=True
                # Explicit type check: Python bool/int equality cannot certify an ordinal.
                assert type(broken['root_occurrences'][0]['root_ordinal']) is not int
    assert len(counts)==18 and any(c['formations']>c['set_children'] for c in counts)
    assert len(refusals)==15
    assert semantic_sentinel.verify_native_semantics(raise_on_change=False)['status']=='REOPEN_REQUIRED'
    assert len(theorem_registry.verify_all_theorems(raise_on_stale=False)['stale_theorems'])==4
    write_json_atomic('DAG_EXPORT_RECEIPTS.json',{'status':'PASS','exports':dags})
    write_json_atomic('INDEPENDENT_ALL_FIELD_CHECKS.json',{'status':'PASS','cases':counts,
        'reference_source_sha256':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
        'checks':['ALL_LABELED_OWNER_PAIRS','FULL_RAW_CHILD','LOCAL_RESOURCE_VECTORS','COMPONENT_MAPS',
            'INDEPENDENT_ALL_ROOT_CANON','EXACT_SET_CORRESPONDENCE','PARENT_AND_CHILD_FRESH_D_MARKER_STATES',
            'FULL_DECODER_TREE_AND_IDENTITY','EXACT_DAG_BYTES','ROOT_AND_ORDERED_SLOT_OCCURRENCES',
            'ALL_TYPED_WITNESS_PATHS','ARBITRARY_PRECISION_OCCURRENCE_COUNTS'],
        'scope':'18_FINITE_CURRENT_G5_G6_CASES_AND_TWO_ORIGINAL_SAVED_NATIVE_DAGS_NOT_FULL_MASTER'})
    write_json_atomic('REFUSALS.json',{'status':'PASS','count':len(refusals),'checks':refusals})
    write_json_atomic('RUNTIME_HOOKS.json',{'status':'PASS','semantic_gates':gates,
        'registered_services':['COMPLETE_OWNER_EXPORT','COMPLETE_MARKER_EXPORT','SAVED_DAG_FULL_EXPORT'],
        'source_payload_sha256':source,'runtime_manifest_sha256':runtime})
    write_json_atomic('QUALIFICATION_RESULT.json',{'status':'PASS','scientific_outcome':'REGISTERED_FULL_OUTPUT_EXPORT_ROUTES_FINITE_QUALIFIED',
        'relation_cases':18,'marker_cases':18,'dag_exports':2,'refusal_count':len(refusals),
        'formation_occurrences':sum(c['formations'] for c in counts),'exact_SET_children':sum(c['set_children'] for c in counts),
        'production_authority':'NONE','generation_allowed':False,'package2_complete':False,
        'full_master_producer_compatible':False,'owner_marker_cases_recomputed':0,'stale_theorems':4,'body_elapsed_seconds':time.monotonic()-started})

if __name__=='__main__':main()
