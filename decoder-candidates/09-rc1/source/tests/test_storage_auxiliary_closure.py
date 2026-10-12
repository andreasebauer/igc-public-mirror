import json,zipfile,hashlib
from pathlib import Path
import pytest
from test_storage_legacy import put,run
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_closure import verify_declared_closure,ClosureLimits
from infinity_grid.storage_collections import ContentDirectory


def fixture(tmp_path):
    S=Path(__file__).resolve().parents[1];root=tmp_path/'content';root.mkdir();allrefs={};formats={}
    def add(raw,fmt='OPAQUE_LEAF_V1'):
        ref=put(root,raw);allrefs[ref['sha256']]=ref;formats[ref['sha256']]={'content_ref':ref,'format':fmt};return ref
    with zipfile.ZipFile(S/'tests/fixtures/dev79_index_reproduction/P9_CAPTURED_STATE_OBJECT_V1.zip') as z:
        records={}
        for n in z.namelist():
            if '/records/' not in n or not n.endswith('.json'):continue
            raw=z.read(n);r=json.loads(raw)
            if r['record_type'] in {'CANONICAL_OBJECT','GENERATION_RECIPE','NEGATIVE_RESULT'}:records[r['record_id']]=(raw,r)
    triples={rid:{'record_id':rid,'record_sha256':r['record_sha256'],'content_ref':add(raw,'NATIVE_REFERENCE_RECORD_V1')} for rid,(raw,r) in records.items()}
    bindings={};maps=[]
    for rid,(raw,r) in records.items():
        p=r['payload'];digests=[];symbols=[]
        for i,row in enumerate(r['provenance']['source_hashes']):
            ref=add((S/row['ref']).read_bytes());assert ref['sha256']==row['sha256'];digests.append({'path':f'/provenance/source_hashes/{i}/sha256','content_ref':ref})
        if r['record_type']=='CANONICAL_OBJECT':
            ref=next(x['content_ref'] for x in digests if x['content_ref']['sha256']==p['canonical_bytes_sha256']);digests.append({'path':'/payload/canonical_bytes_sha256','content_ref':ref})
            producer={'L0':'replay_l0_executor.py','C0_HISTORICAL':'replay_c0_historical_executor.py','L2J3':'replay_l2j3_executor.py'}[r['layer']]
            definition=add((S/'infinity_grid'/producer).read_bytes())
            symbols=[{'path':'/payload/'+k,'value':p[k],'content_ref':definition} for k in ['carrier_schema','formation_provenance']]
        if r['record_type']=='GENERATION_RECIPE':
            ref=add((S/(p['implementation_ref'].split(':')[0].replace('.','/')+'.py')).read_bytes());assert ref['sha256']==p['implementation_sha256'];digests.append({'path':'/payload/implementation_sha256','content_ref':ref})
        oracle=r['record_type']=='NEGATIVE_RESULT' and r['layer']=='L2J3'
        if r['record_type']=='NEGATIVE_RESULT' and not oracle:
            for i,w in enumerate(p['witnesses']):
                if isinstance(w,dict):symbols.append({'path':f'/payload/witnesses/{i}/source','value':w['source'],'content_ref':add((S/w['source']).read_bytes())})
        b={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V3' if oracle else 'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1','record_ref':triples[rid],'record_links':[triples[n] for n in r['dependencies']],'digest_bindings':digests,'symbol_bindings':symbols}
        if oracle:
            b['semantic_bindings']=[];b['oracle_witness_binding']={'oracle_ref':digests[0]['content_ref'],'files':[{'file':label,'content_ref':add((S/'tests/fixtures/storage_witnesses'/Path(label).name).read_bytes())} for label in sorted({w['file'] for w in p['witnesses']})]}
        br=put(root,canonical_bytes(b));allrefs[br['sha256']]=br;maps.append({'content_ref':triples[rid]['content_ref'],'binding_ref':br});bindings[rid]=b
    req={'schema_id':'IG_STORAGE_DECLARED_CLOSURE_REQUEST_V2','purpose':'SCIENCE','roots':[t['content_ref'] for t in triples.values()],'interpretations':list(formats.values()),'required_content':list(allrefs.values()),'legacy_bindings':maps}
    return root,req,bindings


def change(root,req,b):
    old=next(row['binding_ref'] for row in req['legacy_bindings'] if row['content_ref']==b['record_ref']['content_ref']);new=put(root,canonical_bytes(b))
    for row in req['legacy_bindings']:
        if row['binding_ref']==old:row['binding_ref']=new
    req['required_content']=[new if ref==old else ref for ref in req['required_content']]


def test_all_twenty_original_auxiliary_records_close(tmp_path):
    root,req,bs=fixture(tmp_path);out=run(root,req)
    assert out['status']=='STRUCTURAL_PASS' and len(out['legacy_records_verified'])==20
    assert {r['record_id'] for r in out['legacy_records_verified']}==set(bs)
    assert out['scientific_acceptance']=='NOT_GRANTED'


def test_oracle_witness_leaf_cannot_be_omitted_from_inventory(tmp_path):
    root,req,bs=fixture(tmp_path);ref=bs['IGRD/L2J3/NEGATIVE/F']['oracle_witness_binding']['files'][0]['content_ref'];req['required_content'].remove(ref)
    with pytest.raises(ValueError,match='INVENTORY_MISMATCH'):run(root,req)


def test_changed_oracle_witness_provider_bytes_refused(tmp_path):
    root,req,bs=fixture(tmp_path);ref=bs['IGRD/L2J3/NEGATIVE/F']['oracle_witness_binding']['files'][0]['content_ref'];(root/(ref['sha256']+'.blob')).write_bytes(b'bad')
    with pytest.raises(Exception):run(root,req)


def test_wrong_oracle_bytes_cannot_pass_binding(tmp_path):
    root,req,bs=fixture(tmp_path);b=bs['IGRD/L2J3/NEGATIVE/F'];b['oracle_witness_binding']['oracle_ref']=put(root,b'{}');change(root,req,b)
    with pytest.raises(ValueError,match='RAW_BINDING_MISMATCH'):run(root,req)


def test_duplicate_witness_file_binding_refused(tmp_path):
    root,req,bs=fixture(tmp_path);b=bs['IGRD/L2J3/NEGATIVE/F'];b['oracle_witness_binding']['files'].append(b['oracle_witness_binding']['files'][0]);change(root,req,b)
    with pytest.raises(ValueError,match='INVALID_ORACLE_WITNESS_FILE_BINDING'):run(root,req)


def test_missing_witness_binding_refused(tmp_path):
    root,req,bs=fixture(tmp_path);b=bs['IGRD/L2J3/NEGATIVE/F'];b['oracle_witness_binding']['files'].pop();change(root,req,b)
    with pytest.raises(ValueError,match='ORACLE_WITNESS_FILE_INVENTORY'):run(root,req)


def test_old_negative_profile_cannot_silently_skip_oracle_witnesses(tmp_path):
    root,req,bs=fixture(tmp_path);b=bs['IGRD/L2J3/NEGATIVE/F'];b['schema_id']='IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1';b.pop('semantic_bindings');b.pop('oracle_witness_binding');change(root,req,b)
    with pytest.raises(ValueError,match='UNSUPPORTED_NEGATIVE_WITNESS_PROFILE'):run(root,req)


def test_witness_reads_count_against_closure_budget(tmp_path):
    root,req,bs=fixture(tmp_path);out=run(root,req)
    with pytest.raises(ValueError,match='BYTE_BUDGET'):verify_declared_closure(ContentDirectory(root),put(root,canonical_bytes(req)),limits=ClosureLimits(max_total_bytes=out['bytes_read']-1))
