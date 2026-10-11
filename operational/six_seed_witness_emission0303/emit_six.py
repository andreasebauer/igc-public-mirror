import json,hashlib,zipfile,sys,os,types,importlib.util,resource
from pathlib import Path
B=Path(__file__).parent
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
assert os.statvfs(B).f_bavail*os.statvfs(B).f_frsize>=10*1024**3
m=json.loads((B/'SOURCE_MANIFEST.json').read_text());local=Path('/tmp/ig_p0_native161_r1/cold-final/source');dest=B/'source';pins={}
with zipfile.ZipFile(B/'DEV156_CODE.zip') as z:
 for n in ['infinity_grid/o7_edge_canonicalization.py','infinity_grid/canon.py','infinity_grid/core/canonical.py']:
  key='dev156/source/'+n;data=z.read(key) if key in z.namelist() else (local/n).read_bytes();assert sha(data)==m['candidate_files'][n]['sha256']
  p=dest/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);pins[n]=sha(data)
for name,path in [('infinity_grid',dest/'infinity_grid'),('infinity_grid.core',dest/'infinity_grid/core')]:
 mod=types.ModuleType(name);mod.__path__=[str(path)];sys.modules[name]=mod
from infinity_grid.o7_edge_canonicalization import state_identity,verify_relabeling_witness
from infinity_grid.canon import canonical_sha256
original=local/'infinity_grid/resources/replay/O7_C1_TYPED_ORIGINAL_MINIMAL.zip';assert sha(original.read_bytes())=='08df93f3b23f71bf33d14a033b7d1117c4cdb3c4ca10cb39ac85ef1b3a485dfd'
fixture=B/'fixture';fixture.mkdir(exist_ok=True)
with zipfile.ZipFile(original) as z:
 for n,h in json.loads(z.read('MANIFEST_SHA256.json')).items():assert sha(z.read(n))==h
 z.extractall(fixture)
os.environ['OSCOUT_DATA_ROOT']=str(fixture)
spec=importlib.util.spec_from_file_location('_six_seed_original',fixture/'02_CODE/o7_live_engine.py');engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine);engine.O6=engine.import_o6();parents=engine.load_parent_records()
prior=json.loads(Path('/workspace/scratch/87702f15343f/seed_binding/SIX_SEED_BINDING.json').read_text());mapping=json.loads(Path('/workspace/scratch/87702f15343f/v3_binding/V3_BINDING_RESULT.json').read_text());byalias={r['historical_state_alias']:r for r in mapping['seeds']};out=[]
for seed in prior['mapped_seeds']:
 row=seed['immutable_row'];saved=byalias[row['state_digest']];ctx=engine._profile_row_context(row,parents);edges=tuple(tuple(e) for e in row['edges'])
 assert canonical_sha256([p.h6 for p in ctx.parents])==saved['parent_exact_coordinates_sha256']
 result=state_identity(ctx,edges)
 assert result['object_identity']==saved['new_global_identity'],'V3_MISMATCH'
 assert canonical_sha256(result['edges'])==saved['new_edges_sha256'],'EDGE_MISMATCH'
 assert canonical_sha256(result['witness'])==saved['witness_sha256'],'WITNESS_MISMATCH'
 assert verify_relabeling_witness(ctx,edges,result['edges'],result['witness']),'WITNESS_VERIFY'
 record={'source_row':row,'saved_mapping':saved,'result':result,'witness_verified':True,'three_saved_hashes_exact':True}
 p=B/f"WITNESS_{seed['saved_native_validation']['index']:03d}.json";p.write_text(json.dumps(record,indent=2)+'\n');out.append({'file':p.name,'sha256':sha(p.read_bytes()),'saved_witness_sha256':saved['witness_sha256']});print('PASS',p.name,flush=True)
import igraph
summary={'status':'PASS_SIX_FULL_WITNESSES_EMITTED_EXACT_SAVED_HASHES','source_files':pins,'runtime_python':sys.version,'igraph_version':igraph.__version__,'runtime_scope':'Existing V5 runtime used for bounded engineering witness emission; no full RC qualification claim','original_fixture_sha256':sha(original.read_bytes()),'witnesses':out,'bounded_seed_identity_evaluations':6,'observer_runs':0,'full_cohort_regeneration':False,'migration_activated':False}
(B/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
