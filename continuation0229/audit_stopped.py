"""Cold-restoration audit of the stopped capture; no candidate generation."""
from pathlib import Path
import sys,json,hashlib,sqlite3
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import preservation as pr
E=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());M=json.loads((B/'READBACKS.json').read_text());O=Path('/tmp/ig_cold0229_objects');O.mkdir(exist_ok=True)
for x in E['dependencies']:
 p=Path(M[x['sha256']]['path']);assert p.stat().st_size==x['size_bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==x['sha256'];q=O/(x['sha256']+'.bin');q.symlink_to(p) if not q.exists() else None
C=Path('/tmp/ig_native0229/cold_stopped');assert not C.exists();pr.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,E['sha256'],objects=O)
for n in list(sys.modules):
 if n=='infinity_grid' or n.startswith('infinity_grid.') or n=='project' or n.startswith('project.'):del sys.modules[n]
sys.path[0]=str(C/'source')
from infinity_grid.structural_encoding import structural_canonical_bytes
from project.partitions import read,assemble
from project.historical import maturation_parallel as mp
S=json.loads((B/'SPEC.json').read_text());h=S['execution']['parameters']['bindings']['bootstrap'];base=read(C/'runtime/intake/artifacts'/(h+'.bin'),h);R=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'));A=R/'artifacts';man=json.loads((A/'historical_g1_depth_99_manifest.json').read_text());nodes=dict(base['dag']['nodes'])
for ref in man['partitions']:
 p=A/Path(ref['path']).name;d=read(p,ref['sha256'])
 for k,v in d['nodes'].items():assert k not in nodes or nodes[k]==v;nodes[k]=v
assert d['level']==99 and d['selected_count']==24 and d['candidate_count']==193 and d['parent_science_sha256']==base['dag']['science_sha256'];dag=assemble(nodes,man['roots'],man['science_sha256'])
counts={}
for n in (99,100):
 p=R/'phases'/('historical_g1_depth_'+str(n))/'state_store.sqlite3'
 with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as c:
  counts[n]=dict(tasks=c.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0],states=c.execute('SELECT COUNT(*) FROM states').fetchone()[0])
  if n==99:
   digest,exact,sj=c.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchone();assert json.loads(sj)==d and bytes(exact)==structural_canonical_bytes(d) and hashlib.sha256(exact).hexdigest()==digest
assert counts[99]==dict(tasks=1,states=1) and counts[100]==dict(tasks=0,states=0);print('PASS cold exact depth99 committed; depth100 uncommitted',flush=True)
_,states=mp._states_from_dag(dag);assert len(states)==24 and mp.state_dag_wire(states)==dag;print('PASS cold24 depth99 DAG roundtrips',flush=True)
raw=json.dumps(dict(level=99,dag=dag,candidate_count=193,selected_count=24),sort_keys=True,separators=(',',':')).encode();p=Path('/tmp/ig_verified0229/BOOTSTRAP99_HISTORICAL.json');p.write_bytes(raw);(B/'BOOTSTRAP99_META.json').write_text(json.dumps(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),science_sha256=dag['science_sha256']),indent=2))
r=dict(status='PASS_COLD_STOPPED_CAPTURE_DEPTH99',native_completion_status='FAILED_TERMINAL_COMPARISON',accepted_historical_depth=98,depth99_exact_saved_identity=True,depth99_DAG_roundtrip_roots=24,depth100_committed=False,counts=counts,candidate_generation_during_audit=0,original_workspace_scientific_state_read=False,terminal_comparison='FAILED_UNDIAGNOSED',observed_terminal_population_persisted=False,master_slices=151,new_admissions=0);(B/'STOPPED_AUDIT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
