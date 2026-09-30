"""Legacy raw bytes, original seals and explicit recovery bindings."""
from copy import deepcopy
import hashlib,json
import pytest
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_legacy import legacy_dependencies,LegacyBindingError
from infinity_grid.storage_closure import verify_declared_closure,ClosureError
from infinity_grid.storage_collections import ContentDirectory


def put(root,raw):
    ref={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))};(root/(ref['sha256']+'.blob')).write_bytes(raw);return ref


def fixture(tmp_path):
    root=tmp_path/'content';root.mkdir();leaf=put(root,b'engineering source and payload bytes')
    def record(kind,ident,payload,deps=[]):
        return seal_reference_record({'schema_id':'IG_REPLAY_REFERENCE_RECORD_V1','record_id':ident,'record_type':kind,
            'layer':'L0','payload':payload,'provenance':{'classification':'FINITE_COMPUTATIONAL_OBSERVATION','status':'PINNED',
            'source_hashes':[{'ref':'original/source','sha256':leaf['sha256']}],'explanation':'synthetic adapter fixture'},
            'scope':{'fixture':'ENGINEERING_ONLY'},'nonclaims':['NO_SCIENCE_AUTHORITY'],'dependencies':deps,
            'epistemic_status':'HISTORICAL','science_execution':'NONE','authority_effect':'NONE'})
    records=[record('CANONICAL_OBJECT','IGRD/L0/OBJECT/'+n,{'object_identity':n,'carrier_schema':'FIXTURE_CARRIER','canonical_bytes_sha256':leaf['sha256'],'formation_provenance':'FIXTURE_FORMATION'}) for n in ['A','B']]
    records.append(record('MECHANISM','IGRD/L0/MECHANISM/M',{'mechanism_id':'M','input_record_ids':[records[0]['record_id']],
        'output_record_ids':[records[1]['record_id']],'determinism':'DETERMINISTIC',
        'implementation_hashes':[{'ref':'original/impl.py','sha256':leaf['sha256']}]},[records[0]['record_id']]))
    # Pretty JSON is deliberate: raw-file hashes differ from semantic seals.
    raws=[json.dumps(r,indent=2).encode()+b'\n' for r in records];refs=[put(root,r) for r in raws]
    triples=[{'record_id':r['record_id'],'record_sha256':r['record_sha256'],'content_ref':ref} for r,ref in zip(records,refs)]
    bindings=[]
    for i,r in enumerate(records):
        paths=['/provenance/source_hashes/0/sha256','/payload/canonical_bytes_sha256' if i<2 else '/payload/implementation_hashes/0/sha256']
        bindings.append({'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1','record_ref':triples[i],
            'record_links':[] if i<2 else triples[:2], 'digest_bindings':[{'path':p,'content_ref':leaf} for p in paths],
            'symbol_bindings':[{'path':'/payload/'+p,'value':r['payload'][p],'content_ref':leaf} for p in ['carrier_schema','formation_provenance']] if i<2 else []})
    brefs=[put(root,canonical_bytes(b)) for b in bindings]
    req={'schema_id':'IG_STORAGE_DECLARED_CLOSURE_REQUEST_V2','purpose':'SCHEMA_FIXTURE','roots':[refs[2]],
        'interpretations':[{'content_ref':r,'format':'NATIVE_REFERENCE_RECORD_V1'} for r in refs]+[{'content_ref':leaf,'format':'OPAQUE_LEAF_V1'}],
        'required_content':refs+brefs+[leaf],'legacy_bindings':[{'content_ref':r,'binding_ref':b} for r,b in zip(refs,brefs)]}
    return root,req,records,raws,bindings


def run(root,req):return verify_declared_closure(ContentDirectory(root),put(root,canonical_bytes(req)))
def replace_binding(root,req,bindings,i):
    old=req['legacy_bindings'][i]['binding_ref'];new=put(root,canonical_bytes(bindings[i]));req['legacy_bindings'][i]['binding_ref']=new
    req['required_content']=[new if r==old else r for r in req['required_content']]


def test_legacy_recursive_inputs_outputs_and_files(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);before={p.name:p.read_bytes() for p in root.iterdir()}
    out=run(root,req)
    assert out['status']=='STRUCTURAL_PASS' and len(out['legacy_records_verified'])==3
    assert out['scientific_acceptance']=='NOT_GRANTED'
    assert {r['record_id'] for r in out['legacy_records_verified']}=={r['record_id'] for r in records}
    assert all((root/name).read_bytes()==raw for name,raw in before.items())


def test_raw_digest_and_semantic_seal_are_distinct(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path)
    rr=bindings[0]['record_ref'];assert rr['content_ref']['sha256']!=rr['record_sha256']
    out=legacy_dependencies(raws[0],canonical_bytes(bindings[0]));assert out['record_ref']==rr
    assert out['epistemic_status']=='HISTORICAL' and out['science_execution']=='NONE'
    assert out['authority_effect']=='NONE'


def test_corrupted_original_seal_refused(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);records[0]['payload']['object_identity']='CHANGED'
    with pytest.raises(Exception,match='hash mismatch'):legacy_dependencies(canonical_bytes(records[0]),canonical_bytes(bindings[0]))


def test_missing_mechanism_output_binding_refused(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);bindings[2]['record_links'].pop();replace_binding(root,req,bindings,2)
    with pytest.raises(LegacyBindingError,match='LINK_CLOSURE'):run(root,req)


def test_missing_provenance_bytes_binding_refused(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);bindings[0]['digest_bindings'].pop(0)
    with pytest.raises(LegacyBindingError,match='MISSING_DIGEST'):legacy_dependencies(raws[0],canonical_bytes(bindings[0]))


def test_wrong_digest_binding_refused(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);bindings[0]['digest_bindings'][0]['content_ref']=put(root,b'wrong')
    with pytest.raises(LegacyBindingError,match='DIGEST_BINDING_MISMATCH'):legacy_dependencies(raws[0],canonical_bytes(bindings[0]))


def test_symbolic_schema_binding_must_preserve_value(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);bindings[0]['symbol_bindings'][0]['value']='DIFFERENT_SCHEMA'
    with pytest.raises(LegacyBindingError,match='SYMBOL_BINDING'):legacy_dependencies(raws[0],canonical_bytes(bindings[0]))


def test_missing_formation_definition_refused(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);bindings[0]['symbol_bindings'].pop()
    with pytest.raises(LegacyBindingError,match='MISSING_SYMBOL'):legacy_dependencies(raws[0],canonical_bytes(bindings[0]))


def test_legacy_link_semantic_seal_mismatch_refused(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);bindings[2]['record_links'][0]=dict(bindings[2]['record_links'][0],record_sha256='0'*64)
    replace_binding(root,req,bindings,2)
    with pytest.raises(ClosureError,match='LEGACY_REFERENCE_BINDING'):run(root,req)


def test_adapter_inventory_and_registration_required(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);req['required_content'].remove(req['legacy_bindings'][0]['binding_ref'])
    with pytest.raises(ClosureError,match='INVENTORY_MISMATCH'):run(root,req)
    req['legacy_bindings'].pop(0)
    with pytest.raises(ClosureError,match='ADAPTER_REQUIRED'):run(root,req)


def test_duplicate_adapter_and_record_links_refused(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);req['legacy_bindings'].append(req['legacy_bindings'][0])
    with pytest.raises(ClosureError,match='DUPLICATE_LEGACY_BINDING'):run(root,req)
    bindings[2]['record_links'].append(bindings[2]['record_links'][0])
    with pytest.raises(LegacyBindingError,match='DUPLICATE_LEGACY_RECORD'):legacy_dependencies(raws[2],canonical_bytes(bindings[2]))


def test_unsupported_type_not_guessed(tmp_path):
    root,req,records,raws,bindings=fixture(tmp_path);r=deepcopy(records[0]);r['record_type']='RESUME_FRONTIER'
    r['payload']={'root_run_id':'FIXTURE','manifest_dag_sha256':'0'*64,'runner_state_sha256':'1'*64,'completed_node_ids':[],'checkpoint_sha256_by_node':{},'next_node_id':'FIXTURE_NEXT','frontier_status':'READY'}
    r['dependencies']=[];r=seal_reference_record(r)
    with pytest.raises(LegacyBindingError,match='UNSUPPORTED_LEGACY'):legacy_dependencies(canonical_bytes(r),canonical_bytes(bindings[0]))
