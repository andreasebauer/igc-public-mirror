"""Independent cold audit of persisted diagnostic bytes; zero generation."""
from pathlib import Path
import sys,json,hashlib,sqlite3,shutil
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import preservation as pr
E=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());M=json.loads((B/'READBACKS.json').read_text());O=Path('/tmp/ig_cold0230_objects');O.mkdir(exist_ok=True)
for x in E['dependencies']:
 p=Path(M[x['sha256']]['path']);assert p.stat().st_size==x['size_bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==x['sha256'];q=O/(x['sha256']+'.bin')
 if not q.exists():q.symlink_to(p)
C=Path('/tmp/ig_native0230/cold_diagnostic');assert not C.exists();pr.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,E['sha256'],objects=O)
for n in list(sys.modules):
 if n=='infinity_grid' or n.startswith('infinity_grid.') or n=='project' or n.startswith('project.'):del sys.modules[n]
sys.path[0]=str(C/'source')
from infinity_grid.structural_encoding import structural_canonical_bytes
from project.partitions import read,assemble
from project.handler import differences
R=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'));A=R/'artifacts';data=json.loads((A/'terminal100_diagnostic_state.json').read_text());observed=json.loads((A/'terminal100_observed_population.json').read_text());report=json.loads((A/'terminal100_diagnostic_report.json').read_text());assert data['terminal_interface_population']==observed
with sqlite3.connect((R/'phases/terminal100_diagnostic/state_store.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as c:
 assert c.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]==1 and c.execute('SELECT COUNT(*) FROM states').fetchone()[0]==1
 digest,exact,sj=c.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchone();assert json.loads(sj)==data and bytes(exact)==structural_canonical_bytes(data) and hashlib.sha256(exact).hexdigest()==digest
S=json.loads((B/'SPEC.json').read_text());h=S['execution']['parameters']['bindings'];reference=read(C/'runtime/intake/artifacts'/(h['reference']+'.bin'),h['reference']);base=read(C/'runtime/intake/artifacts'/(h['bootstrap']+'.bin'),h['bootstrap']);assert base['level']==99 and data['level']==100 and data['candidate_count']==193 and data['selected_count']==193 and len(data['roots'])==193
nodes=dict(base['dag']['nodes'])
for k,v in data['nodes'].items():assert k not in nodes or nodes[k]==v;nodes[k]=v
assemble(nodes,data['roots'],data['science_sha256'])
counts,examples=differences(observed,reference);assert counts==report['difference_counts'] and examples==report['examples'];assert report['interface_rows_exact_equal']==(observed['interfaces']==reference['interfaces']);assert report['top_level_differing_keys']==[k for k in sorted(set(observed)|set(reference)) if observed.get(k)!=reference.get(k)]
assert len(observed['interfaces'])==193;assert report['accepted_historical_depth']==98 and report['new_admissions']==0
for name in ['terminal100_observed_population','terminal100_diagnostic_report']:
 shutil.copy2(A/(name+'.json'),B/(name.upper()+'.json'))
r=dict(status='PASS_COLD_DIAGNOSTIC_SAVED_BYTES',diagnostic_only=True,native_exact_state_identity=True,terminal_DAG_science_checked=True,interface_rows=193,interface_rows_exact_equal=report['interface_rows_exact_equal'],difference_counts=counts,top_level_differing_keys=report['top_level_differing_keys'],beam_gate=data['historical_depth100_beam_anchor'],terminal_comparison=data['terminal_comparison'],candidate_generation_during_audit=0,original_workspace_scientific_state_read=False,full193_DAG_state_roundtrip_performed=False,accepted_historical_depth=98,master_slices=151,new_admissions=0);(B/'AUDIT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
