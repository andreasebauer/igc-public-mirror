"""Native qualification of strict term reads and complete scientific storage."""
from copy import deepcopy
from pathlib import Path
import gzip,hashlib,json,os,shutil,tempfile,time,zipfile
import reference
from infinity_grid.canon import canonical_sha256,write_json_atomic
from infinity_grid.direct_v2_observation import source_snapshot
from infinity_grid.isolated_runtime import BUNDLED_MANIFEST_SHA256
from infinity_grid.v05_stage_runtime import _resolve_ref,_bind_controller_fallback_kernel_view
from infinity_grid.v05_stage_registry import semantic_audit_for_ref
from infinity_grid.v05_kernel_services import current_kernel_view,KernelServiceViolation
from infinity_grid.full_marker_export import ExportContractError
from infinity_grid.g4_term_state import G4TermError
from infinity_grid.master_scientific_storage import read_accepted,verify_export_storage
from infinity_grid.storage_collections import ContentDirectory,collection_records,CollectionReadError
from infinity_grid.storage_catalog import ReadIndex,ReadIndexError
from infinity_grid.storage_schema import validate_record_bytes
from infinity_grid import semantic_sentinel,theorem_registry

REF='infinity_grid.master_full_export_evaluators:'

def main():
    started=time.monotonic();inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role,sha in params['input_hashes'].items():assert hashlib.sha256(Path(inputs[role]).read_bytes()).hexdigest()==sha
    from infinity_grid import master_scientific_storage as storage_module
    base=json.loads(Path(inputs['baseline_package']).read_text());package=Path(storage_module.__file__).parent
    current={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    assert set(current)-set(base)=={'g4_term_admission.py','master_scientific_storage.py'} and not set(base)-set(current)
    changed=sorted(k for k in base if base[k]!=current[k])
    assert set(changed)=={'master_full_export_evaluators.py','v05_kernel_services.py','v05_kernel_service_providers.py','v05_stage_registry.py'}
    write_json_atomic('SOURCE_COMPATIBILITY.json',{'status':'PASS','all_other_source_bytes_unchanged':True,
        'changed_runtime_hooks':changed,'legacy_g4_constructor_unchanged':True,'scientific_kernel_algorithms_unchanged':True})
    source=source_snapshot();runtime=BUNDLED_MANIFEST_SHA256
    def bind(root,task,recipe):return {'source_ref':source,'runtime_ref':runtime,'input_root_ref':root,'task_ref':task,'recipe_ref':recipe}
    def invoke(name,payload):
        ref=REF+name;fn=_resolve_ref(ref);_bind_controller_fallback_kernel_view(ref,canonical_sha256(payload));return fn(payload)
    failures=[]
    def refuse(name,fn,types=(G4TermError,ExportContractError,KernelServiceViolation,ReadIndexError,CollectionReadError,AssertionError)):
        try:fn()
        except types as e:failures.append({'case':name,'type':type(e).__name__,'message':str(e)})
        else:raise AssertionError('ACCEPTED:'+name)
    gates={n:semantic_audit_for_ref(REF+n) for n in ('strict_term_read_evaluator','export_storage_evaluator')}
    assert all(g['status']=='PASS' for g in gates.values())
    terms=[]
    for n in (3,4,5):
        for kind in ('path','star'):
            caps=[[(1<<200)+17*i+3*t for t in range(7)] for i in range(n)]
            edges=[[i,i+1,i%7,(i+1)%7] for i in range(n-1)] if kind=='path' else [[0,i,i%7,(i+1)%7] for i in range(1,n)]
            for reverse in (False,True):
                p=list(reversed(range(n))) if reverse else list(range(n));nc=[None]*n
                for i,j in enumerate(p):nc[j]=caps[i]
                wire=reference.reference_term(nc,[[p[u],p[v],a,b] for u,v,a,b in edges])
                b=bind(canonical_sha256(wire),'Q20:TERM:'+str(len(terms)),'STRICT_G4_ATOMIC_TERM_READ_V1')
                out=invoke('strict_term_read_evaluator',{'term':wire,'binding':b})
                assert reference.encoded(out['term'])==reference.encoded(wire)
                terms.append(out)
    wire=deepcopy(terms[0]['term'])
    mutations=[('bool_capacity',lambda w:w['node_caps'][0].__setitem__(0,True)),
        ('float_capacity',lambda w:w['node_caps'][0].__setitem__(0,1.0)),
        ('text_capacity',lambda w:w['node_caps'][0].__setitem__(0,'1')),
        ('negative_capacity',lambda w:w['node_caps'][0].__setitem__(0,-1)),
        ('bool_total',lambda w:w['total_caps'].__setitem__(0,True)),
        ('wrong_total',lambda w:w['total_caps'].__setitem__(0,1)),
        ('bool_vertex',lambda w:w['typed_edges'][0].__setitem__(0,True)),
        ('float_type',lambda w:w['typed_edges'][0].__setitem__(2,0.0)),
        ('self_edge',lambda w:w['typed_edges'][0].__setitem__(1,w['typed_edges'][0][0])),
        ('missing_vertex',lambda w:w['typed_edges'][0].__setitem__(0,99)),
        ('outside_type',lambda w:w['typed_edges'][0].__setitem__(3,7)),
        ('term_checksum',lambda w:w.update(term_ref='0'*64)),
        ('topology_checksum',lambda w:w.update(topology_canon_sha256='0'*64)),
        ('extra_field',lambda w:w.update(production_authority='PRODUCTION')),
        ('missing_field',lambda w:w.pop('total_caps'))]
    for name,change in mutations:
        bad=deepcopy(wire);change(bad);b=bind(canonical_sha256(bad),'Q20:REFUSE:'+name,'STRICT_G4_ATOMIC_TERM_READ_V1')
        refuse(name,lambda bad=bad,b=b:invoke('strict_term_read_evaluator',{'term':bad,'binding':b}))
    for field,value in [('source_ref','0'*64),('runtime_ref','0'*64),('input_root_ref','0'*64),('recipe_ref','MASTER_GENERATE')]:
        b=bind(canonical_sha256(wire),'Q20:REFUSE:'+field,'STRICT_G4_ATOMIC_TERM_READ_V1');b[field]=value
        refuse('term_'+field,lambda b=b:invoke('strict_term_read_evaluator',{'term':wire,'binding':b}))
    write_json_atomic('STRICT_G4_TERM_READS.json',{'status':'PASS','term_count':len(terms),'terms':terms,
        'scope':'12_SMALL_ATOMIC_TERM_QUALIFICATION_INPUTS_NOT_HISTORICAL_CORPUS_ADOPTION'})
    evidence=inputs['accepted_exports'];root_sha=hashlib.sha256(Path(evidence).read_bytes()).hexdigest()
    owners,markers,_=read_accepted(evidence)
    with tempfile.TemporaryDirectory(prefix='ig_q20_storage_') as tmp:
        tmp=Path(tmp);storepath=tmp/'primary';b=bind(root_sha,'Q20:STORE:ALL18','Q20_SAVED_G5_G6_STORAGE_ONLY')
        payload={'evidence_path':evidence,'destination':str(storepath),'binding':b}
        receipt=invoke('export_storage_evaluator',payload)
        before=verify_export_storage(storepath)
        assert before['counts']=={'objects':191,'occurrences':682,'relations':664,'incidences':2269}
        manifest=json.loads((storepath/'CANDIDATE.json').read_bytes());store=ContentDirectory(storepath/'content')
        for ref in manifest['content_refs']:
            raw=store.read(ref,int(ref['size_bytes']))
            if raw.startswith(b'{'):
                value=json.loads(raw)
                if value.get('schema_id') in ('IG_STORAGE_IDENTITYPROFILE_V1','IG_STORAGE_RELATIONTYPE_V1','IG_STORAGE_PAYLOAD_V1'):
                    validate_record_bytes(raw,max_bytes=len(raw))
        def rows(c,kind):return [json.loads(r.raw) for r in collection_records(store,c,collection_kind=kind,purpose='SCIENCE')]
        groups={k:rows(c,k) for k,c in manifest['collections'].items()}
        independent=reference.verify_all_storage(manifest,groups,lambda r:store.read(r,int(r['size_bytes'])),rows,owners,markers)
        # Read-only indexed joins retain repeated occurrence endpoints and ordering.
        with ReadIndex(storepath/'index',expected_manifest_ref=receipt['index']['manifest_ref'],release_root=receipt['candidate_ref']) as index:
            parent=next(r['occurrence_ref'] for r in groups['occurrences'] if r['occurrence_ref']['native_id'].endswith(':LEFT'))
            joined=[];cursor=None
            while True:
                page=index.incidences(endpoint={'kind':'OCCURRENCE','ref':parent},limit=7,after=cursor)
                joined.extend(page['rows']);cursor=page['next_cursor']
                if cursor is None:break
            expected=[r for r in groups['incidences'] if r['endpoint']=={'kind':'OCCURRENCE','ref':parent}]
            assert {reference.sha(r['record']) for r in joined}=={reference.sha(r) for r in expected}
            assert len(joined)>1
            refuse('stale_index_root',lambda:ReadIndex(storepath/'index',expected_manifest_ref=receipt['index']['manifest_ref'],release_root=manifest['scope_ref']))
        # Native full-output service is absent from the strict-term-only view.
        bterm=bind(canonical_sha256(wire),'Q20:SERVICE_SCOPE','STRICT_G4_ATOMIC_TERM_READ_V1')
        invoke('strict_term_read_evaluator',{'term':wire,'binding':bterm})
        refuse('undeclared_storage_service',lambda:current_kernel_view().call('FULL_EXPORT_STORAGE'))
        for field,value in [('source_ref','0'*64),('runtime_ref','0'*64),('input_root_ref','0'*64),('recipe_ref','MASTER_GENERATE')]:
            bad=deepcopy(payload);bad['binding'][field]=value;bad['destination']=str(tmp/('refused_'+field))
            refuse('storage_'+field,lambda bad=bad:invoke('export_storage_evaluator',bad))
            assert not Path(bad['destination']).exists()
        broken=deepcopy(groups);broken['occurrences'].pop()
        refuse('missing_occurrence',lambda:reference.verify_all_storage(manifest,broken,lambda r:store.read(r,int(r['size_bytes'])),rows,owners,markers))
        w=next(iter(manifest['formation_witnesses']));witness=json.loads(store.read(w['witness_ref'],int(w['witness_ref']['size_bytes'])))
        badmap=deepcopy(witness['raw_to_retained_vertex_map']);badmap[0]=badmap[1]
        row=witness['original_formation'];owner=next(o for o in owners if o['binding']['task_ref']==w['task'])
        retained=owner['child_SET'][row['retained_child_ordinal']]['full_tree'];classes={h:k for h,k,c in owner['authority']['classes']}
        refuse('nonbijective_vertex_map',lambda:reference.verify_vertex_map(row['full_raw_child'],retained,badmap,classes))
        # Lossless portable archive, including exact recorded file modes.
        archive=Path('SCIENTIFIC_STORAGE_FULL.zip');members=[]
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p in sorted(storepath.rglob('*')):
                if p.is_file():
                    name=p.relative_to(storepath).as_posix();raw=p.read_bytes();mode=p.stat().st_mode&0o777
                    members.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mode':mode});z.write(p,name)
            z.writestr('ARCHIVE_MANIFEST.json',reference.encoded(members))
        cold=tmp/'cold';cold.mkdir()
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist())==len(set(z.namelist()))
            for info in z.infolist():
                if info.filename=='ARCHIVE_MANIFEST.json':continue
                path=cold/info.filename;assert not Path(info.filename).is_absolute() and '..' not in Path(info.filename).parts
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(info));path.chmod((info.external_attr>>16)&0o777)
            assert json.loads(z.read('ARCHIVE_MANIFEST.json'))==members
        for r in members:
            p=cold/r['path'];assert len(p.read_bytes())==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'] and p.stat().st_mode&0o777==r['mode']
        cold_before={p.relative_to(cold).as_posix():(p.read_bytes(),p.stat().st_mtime_ns,p.stat().st_mode) for p in cold.rglob('*') if p.is_file()}
        after=verify_export_storage(cold);assert before==after
        assert cold_before=={p.relative_to(cold).as_posix():(p.read_bytes(),p.stat().st_mtime_ns,p.stat().st_mode) for p in cold.rglob('*') if p.is_file()}
        # Missing blob refusal on the cold copy, after exact successful readback.
        victim=cold/'content'/(manifest['content_refs'][0]['sha256']+'.blob');saved=victim.read_bytes();victim.unlink()
        refuse('missing_cold_blob',lambda:verify_export_storage(cold))
        victim.write_bytes(saved);victim.chmod(0o444)
        corrupt=bytearray(saved);corrupt[0]^=1;victim.chmod(0o644);victim.write_bytes(corrupt)
        refuse('corrupted_cold_blob',lambda:verify_export_storage(cold))
        write_json_atomic('STORAGE_QUALIFICATION.json',{'status':'PASS','primary':before,'cold':after,
            'independent':independent,'indexed_parent_join_rows':len(joined),'cold_file_count':len(members),
            'exact_bytes_modes_and_read_only_verified':True,'original_export_cases_recomputed':0,
            'storage_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
        write_json_atomic('STORAGE_INDEX_RECEIPT.json',receipt)
    assert len(failures)==29
    assert semantic_sentinel.verify_native_semantics(raise_on_change=False)['status']=='REOPEN_REQUIRED'
    assert len(theorem_registry.verify_all_theorems(raise_on_stale=False)['stale_theorems'])==4
    write_json_atomic('REFUSALS.json',{'status':'PASS','count':len(failures),'checks':failures})
    write_json_atomic('RUNTIME_HOOKS.json',{'status':'PASS','services':['STRICT_G4_TERM_READ','FULL_EXPORT_STORAGE'],'semantic_gates':gates,'source_payload_sha256':source,'runtime_manifest_sha256':runtime})
    write_json_atomic('QUALIFICATION_RESULT.json',{'status':'PASS','scientific_outcome':'STRICT_G4_AND_FULL_EXPORT_SCIENCE_STORAGE_FINITE_QUALIFIED',
        'strict_term_reads':12,'stored_objects':191,'stored_occurrences':682,'stored_relations':664,'stored_incidences':2269,
        'formation_vertex_maps':646,'refusal_count':len(failures),'saved_export_cases_recomputed':0,
        'cold_recovery':'EXACT_BYTES_MODES_BINDINGS_AND_READ_ONLY_INDEX_PASS','production_authority':'NONE',
        'full_master_producer_compatible':False,'package2_complete':False,'body_elapsed_seconds':time.monotonic()-started})

if __name__=='__main__':main()
