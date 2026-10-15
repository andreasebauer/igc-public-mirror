"""Decoder-owned complete exports under explicit source/runtime/input bindings.

These services retain full scientific payloads. They grant no master admission,
theorem authority, historical correspondence, or generation permission.
"""
import hashlib
import json
from importlib.resources import files
import zipfile
from .canon import canonical_sha256
from .direct_v2_observation import source_snapshot
from .isolated_runtime import BUNDLED_MANIFEST_SHA256, verified_runtime
from .full_marker_export import assemble_marker_export, require
from .full_owner_export import export_owner_occurrences
from .full_g3_export import assemble_g3_dag_export, canonical_hash
from . import qualified_discovery_inputs as qi


def check_binding(binding, input_root, *, owner=False, dag=False):
    fields={'recipe_ref','source_ref','runtime_ref','input_root_ref','task_ref'}
    if owner:fields |= {'left_parent_occurrence_ref','right_parent_occurrence_ref'}
    if dag:fields.add('identity_scope')
    require(type(binding) is dict and set(binding)==fields, 'EXPORT_BINDING_FIELDS')
    require(all(type(v) is str and v for v in binding.values()), 'EXPORT_BINDING_VALUE')
    require(binding['source_ref']==source_snapshot(), 'EXPORT_SOURCE_CHANGED')
    require(binding['runtime_ref']==BUNDLED_MANIFEST_SHA256, 'EXPORT_RUNTIME_CHANGED')
    require(binding['input_root_ref']==input_root, 'EXPORT_INPUT_CHANGED')
    verified_runtime()


def relation_input_root(left,right,operator):
    return canonical_sha256({'left':left,'right':right,'operator':operator})


def complete_owner_export(left_record,right_record,operator,binding):
    check_binding(binding,relation_input_root(left_record,right_record,operator),owner=True)
    from .exact_tree_relation_kernel import get_relation_kernel,tree_from_record,tree_to_record,graft_pair_raw
    return export_owner_occurrences(kernel=get_relation_kernel(),graft_pair_raw=graft_pair_raw,
        tree_to_record=tree_to_record,left=tree_from_record(left_record),right=tree_from_record(right_record),
        operator=operator,binding=binding)


def complete_marker_export(left_record,right_record,operator,binding):
    check_binding(binding,relation_input_root(left_record,right_record,operator))
    from .exact_tree_relation_kernel import get_relation_kernel,tree_from_record,tree_to_record
    from .g6_fresh_marker_kernel import marker_observer_state,write_marker_states,decode_marker_observer_state
    k=get_relation_kernel();left=tree_from_record(left_record);right=tree_from_record(right_record)
    ql=marker_observer_state(left,kernel=k);qr=marker_observer_state(right,kernel=k)
    diagnostics={};states=write_marker_states(ql,qr,tuple(operator),kernel=k,diagnostics=diagnostics)
    decoded=[]
    for record,q,can in diagnostics['raw_child_rows']:
        child,stats=decode_marker_observer_state(q,kernel=k)
        identity=k.prepare(child).unrooted_canon
        decoded.append({'status':'PASS','candidate_count':1,'candidates':(identity,),
            'tree':tree_to_record(child),'unrooted_canon':identity,'stats':stats})
    return assemble_marker_export(binding=binding,left_tree=left_record,right_tree=right_record,
        operator=operator,left_marker_state=ql,right_marker_state=qr,
        write_result={'states':states,'diagnostics':diagnostics},decoded_children=decoded)


def saved_dag_export(root,input_binding,role,binding):
    require(role in ('prefix_dag','r100_dag'), 'EXPORT_DAG_DOMAIN')
    # This original admission verifies native captures, completions and all bytes;
    # it does not construct an O7 runtime, select panels or reconstruct states.
    admission=qi.preflight(root,input_binding)
    seed=files('infinity_grid').joinpath('resources','decoder','O7_MATERIAL_ROOT_cb6f48641eb9.zip')
    seed_raw=seed.read_bytes()
    raw={pin['file_name']:(admission.root/pin['file_name']).read_bytes()
         for pin in input_binding['artifacts'].values()}
    raw['O7_MATERIAL_ROOT_cb6f48641eb9.zip']=seed_raw
    required={name:{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for name,b in raw.items()}
    receipts=[{'path':n,**p} for n,p in sorted(required.items())]
    check_binding(binding,canonical_hash(receipts),dag=True)
    require(binding['identity_scope']=='DECODER091_NATIVE_V3', 'EXPORT_NATIVE_SCOPE_REQUIRED')
    import io
    with zipfile.ZipFile(io.BytesIO(seed_raw)) as z:
        names=[n for n in z.namelist() if n.endswith('/graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json')]
        require(len(names)==1,'EXPORT_AUTHORITY_FILE')
        ids=[r['state_digest'] for r in json.loads(z.read(names[0]))['records']]
    # Authority may have several retained rows for one immutable source ID.
    ids=sorted(set(ids))
    dag=json.loads(raw[input_binding['artifacts'][role]['file_name']])
    out=assemble_g3_dag_export(dag=dag,binding=binding,required_inputs=required,
        input_bytes=raw,authority_source_ids=ids)
    out['original_input_binding']=input_binding
    out['profile_scope']='CURRENT_NATIVE_RECURSIVE_CARRIER_AND_AUTHORITY_BYTES_ONLY'
    out['historical_V1_correspondence']='NOT_GRANTED'
    out['microscopic_expansion']='NOT_CLAIMED'
    return out
