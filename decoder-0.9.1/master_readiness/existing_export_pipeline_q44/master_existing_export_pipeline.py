"""Attach preserved complete stores without recomputing scientific outputs.

Records keep their original native scopes; cross-store equality is not inferred.
Sink rows are provisional until the complete returned manifest is accepted.
"""
from io import BytesIO
from pathlib import Path
import hashlib,json,zipfile
from .canon import canonical_sha256
from .storage_schema import canonical_bytes
from .storage_collections import CollectionLimits,collection_records,CollectionReadError
from .storage_catalog import IndexRecord,inventory

SCHEMA='IG_MASTER_EXISTING_EXPORT_ATTACH_PLAN_V1'
ROUTE='IG_MASTER_EXISTING_EXPORT_ATTACH@1.0.0'
COMPONENTS=['G3_CURRENT_SAVED_DAGS','G5_G6_FINITE18','G6_S1_INITIAL496']
DATA_PINS={'g3':'f8d705f8e6008187549f5362524a8628663229c2f6e3d94bfef7e587e7c04d19','g6':'41548789f10c5dca4ceb4ee6e0b302833600f92c9c8c46c25a6dae3db9f0ce60','q22':'83b534e30635a0da4884ce52bf9bd503decab9a3f1a563d3f96382889c38a6c2','q26':'a38dcf8f9a1592167f63929099d15cb8a91efde218328e229ab4a1b733dc4938','q32':'d70ca7cc87b5ae6428fb6c31b5ec568f03dd1a35de941764ec06e55ae4414235'}
COMPLETIONS={'q22':'d2ad525ef8f6a8d270dd95a663d0130501ca3f28737b854a298ac4d2d5cbdff8','q26':'c5a59cdba0d3108b54de3f6147bfe582aa0049751ccb13f1d4b7997f8c49129c','q32':'76754b3671a26acbd687a8de3be2133bf55687ceac11fc1d918812316a4f4a7d'}
class ExistingExportError(ValueError):pass
def require(ok,reason):
    if not ok:raise ExistingExportError(reason)
def sha(b):return hashlib.sha256(b).hexdigest()
def compile_plan(plan):
    require(type(plan) is dict and set(plan)=={'schema_id','route','components','g4_fixture_policy','generation_allowed','production_authority'},'EXISTING_EXPORT_PLAN_FIELDS')
    require(plan['schema_id']==SCHEMA and plan['route']==ROUTE and plan['components']==COMPONENTS and plan['g4_fixture_policy']=='EVIDENCE_ONLY' and plan['generation_allowed'] is False and plan['production_authority']=='NONE','EXISTING_EXPORT_SCOPE_NOT_ADMITTED')
    return {'schema_id':'IG_COMPILED_EXISTING_EXPORT_ATTACH_V1','route':ROUTE,'components':list(COMPONENTS),'plan_sha256':canonical_sha256(plan),'required_theorems':[],'generation_allowed':False,'production_authority':'NONE'}
def _snapshot(path,expected):
    raw=Path(path).read_bytes();require(sha(raw)==expected,'EXISTING_EXPORT_INPUT_BYTES')
    return _zip(raw)
def _zip(raw):
    z=zipfile.ZipFile(BytesIO(raw));require(len(z.namelist())==len(set(z.namelist())),'EXISTING_EXPORT_DUPLICATE_MEMBER');return z
def _json(z,n):return json.loads(z.read(n))
def authority(z,pin):
    cap=_json(z,'native/CAPTURE.json');done=_json(z,'native/COMPLETION.json');proof=_json(z,'native/TERMINAL_COMPLETION_PROOF.json')
    require(cap['capture_id']==canonical_sha256({k:v for k,v in cap.items() if k!='capture_id'}),'EXISTING_EXPORT_CAPTURE_SEAL')
    require(done['completion_sha256']==pin==canonical_sha256({k:v for k,v in done.items() if k!='completion_sha256'}),'EXISTING_EXPORT_COMPLETION_SEAL')
    require(done['status']=='COMPLETED' and done['evidence_status']=='VERIFIED' and done['source_sha256']==cap['workspace']['source_sha256'] and done['registration_sha256']==cap['job']['registration_sha256'],'EXISTING_EXPORT_NATIVE_BINDING')
    require(proof['capture_id']==cap['capture_id'] and proof['completion_sha256']==pin and (proof.get('terminal_completion_proof') is True or proof.get('terminal_completion_preserved') is True),'EXISTING_EXPORT_PRESERVATION')
    return cap,done
def accepted(z,done,name):
    raw=z.read('accepted_outputs/'+name);row=next(e for e in done['evidence'] if e['path']=='script/work/'+name)
    require(sha(raw)==row['sha256'] and len(raw)==row['size_bytes'],'EXISTING_EXPORT_ACCEPTED_BYTES');return raw
class ArchiveContent:
    def __init__(self,z,prefix):self.z=z;self.prefix=prefix
    def read(self,ref,max_bytes):
        require(type(ref) is dict and set(ref)=={'sha256','size_bytes'} and type(ref['size_bytes']) is str and ref['size_bytes'].isdigit(),'EXISTING_EXPORT_CONTENT_REF')
        n=int(ref['size_bytes']);require(n<=max_bytes,'EXISTING_EXPORT_CONTENT_BUDGET')
        raw=self.z.read(self.prefix+'content/'+ref['sha256']+'.blob')
        require(len(raw)==n and sha(raw)==ref['sha256'],'EXISTING_EXPORT_CONTENT_BYTES');return raw
def execute(plan,paths,*,sink=None):
    compiled=compile_plan(plan)
    require(type(paths) is dict and set(paths)=={'g3','g6','q22','q26','q32'},'EXISTING_EXPORT_INPUT_SET')
    # Exact verified byte snapshots stay open throughout consumption.
    handles=[]
    try:
        z={k:_snapshot(paths[k],v) for k,v in DATA_PINS.items()};handles.extend(z.values())
        authorities={k:authority(z[k],v) for k,v in COMPLETIONS.items()}
        for k in authorities:accepted(z[k],authorities[k][1],'QUALIFICATION_RESULT.json')
        receipt=json.loads(accepted(z['q26'],authorities['q26'][1],'DATA_ARCHIVE_RECEIPT.json'))
        require(receipt['sha256']==DATA_PINS['g6'] and receipt['source_bytes_preserved'] is True,'EXISTING_EXPORT_G6_DATA_BINDING')
        spec=_json(z['q32'],'native/SPECIFICATION.json');q32cap=authorities['q32'][0]
        require(any(i['logical_name']=='core_capsule' and i['sha256']==DATA_PINS['g3'] for i in q32cap['job']['input_artifacts']),'EXISTING_EXPORT_G3_CAPTURE_INPUT')
        finite_raw=accepted(z['q22'],authorities['q22'][1],'SCIENTIFIC_STORAGE_FULL.zip');finite=_zip(finite_raw);handles.append(finite)
        terms=json.loads(accepted(z['q22'],authorities['q22'][1],'STRICT_G4_TERM_READS.json'))
        require(terms['scope']=='12_SMALL_ATOMIC_TERM_QUALIFICATION_INPUTS_NOT_HISTORICAL_CORPUS_ADOPTION' and terms['term_count']==12,'EXISTING_EXPORT_FIXTURE_SCOPE')
        sources=[(COMPONENTS[0],z['g3'],'','q32'),(COMPONENTS[1],finite,'','q22'),(COMPONENTS[2],z['g6'],'partial_store/','q26')]
        # Preflight all stores before emitting any provisional record.
        candidates=[]
        for name,archive,prefix,q in sources:
            raw=archive.read(prefix+'CANDIDATE.json');candidate=json.loads(raw);index=_json(archive,prefix+'INDEX_RECEIPT.json')
            require(index['candidate_ref']=={'sha256':sha(raw),'size_bytes':str(len(raw))} and candidate['counts']==index['counts'],'EXISTING_EXPORT_CANDIDATE_BINDING')
            require(candidate['production_authority']=='NONE' and candidate['release_acceptance']=='NOT_GRANTED','EXISTING_EXPORT_NOT_RELEASE_ADMISSION')
            require(set(candidate['collections'])=={'objects','occurrences','relations','incidences'},'EXISTING_EXPORT_COLLECTION_DOMAIN')
            require(int(index['inventory']['record_count'])==sum(candidate['counts'].values()),'EXISTING_EXPORT_INVENTORY_COUNT')
            candidates.append((name,archive,prefix,q,candidate,index,raw))
        attached=[]
        for name,archive,prefix,q,candidate,index,raw in candidates:
            store=ArchiveContent(archive,prefix);seen={};stream=hashlib.sha256()
            def records():
                for kind in sorted(candidate['collections']):
                    count=0;collection=candidate['collections'][kind]
                    budget=candidate.get('collection_io_limits',{}).get(kind,{})
                    limits=CollectionLimits(max_total_bytes=budget.get('exact_read_bytes',67108864),max_records=max(1,candidate['counts'][kind]))
                    for row in collection_records(store,collection,collection_kind=kind,purpose='SCIENCE',limits=limits):
                        require(row.raw==canonical_bytes(json.loads(row.raw)),'EXISTING_EXPORT_EXACT_TYPED_BYTES')
                        # Component namespace is a pipeline address, never a replacement native identity.
                        if sink is not None:sink(name,kind,row.key,row.raw)
                        stream.update(len(row.raw).to_bytes(8,'big'));stream.update(row.raw);count+=1
                        yield IndexRecord(kind+'/'+row.key,row.raw)
                    require(count==candidate['counts'][kind],'EXISTING_EXPORT_COLLECTION_COUNT');seen[kind]=count
            inv=inventory(records());require(inv==index['inventory'],'EXISTING_EXPORT_RECORD_INVENTORY')
            attached.append({'component':name,'qualifying_capture_id':authorities[q][0]['capture_id'],'qualifying_completion_sha256':COMPLETIONS[q],'original_storage_binding':candidate['binding'],'candidate_ref':index['candidate_ref'],'collections':candidate['collections'],'collection_io_limits':candidate.get('collection_io_limits',{}),'counts':seen,'inventory':inv,'full_record_stream_sha256':stream.hexdigest(),'native_references_rewritten':False,'cross_store_identity_merge':False})
        return {'schema_id':'IG_MASTER_EXISTING_EXPORT_PIPELINE_RESULT_V1','compiled':compiled,'components':attached,'total_records':sum(sum(c['counts'].values()) for c in attached),'g4_fixture_terms':12,'g4_fixtures_added_to_science':0,'g4_historical_corpus_coverage':False,'generation_calls':0,'storage_rebuilds':0,'index_rebuilds':0,'production_authority':'NONE','full_master_producer_compatible':False,'required_theorems':[],'commit_scope':'ALL_THREE_READS_COMPLETE; SINK_ROWS_PROVISIONAL_BEFORE_SUCCESS'}
    finally:
        for z in handles:z.close()
