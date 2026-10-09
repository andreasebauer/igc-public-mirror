"""Independent cold-restored terminal requalification; no candidate generation."""
from pathlib import Path
import sys,json,hashlib,sqlite3,shutil
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import preservation as pr
E=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());M=json.loads((B/'READBACKS.json').read_text());O=Path('/tmp/ig_cold0231_objects');O.mkdir(exist_ok=True)
for x in E['dependencies']:
 p=Path(M[x['sha256']]['path']);assert p.stat().st_size==x['size_bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==x['sha256'];q=O/(x['sha256']+'.bin')
 if not q.exists():q.symlink_to(p)
C=Path('/tmp/ig_native0231/cold_requalified');assert not C.exists();pr.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,E['sha256'],objects=O)
for n in list(sys.modules):
 if n=='infinity_grid' or n.startswith('infinity_grid.') or n=='project' or n.startswith('project.'):del sys.modules[n]
sys.path[0]=str(C/'source')
from infinity_grid.structural_encoding import structural_canonical_bytes
from project.worker import read_bound
from project.historical import maturation_parallel as mp
from project.public_interface import extract_interface_population
from project.current_interface import extract_interface_population as current_population
R=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'));A=R/'artifacts';data=json.loads((A/'terminal100_requalification.json').read_text())
with sqlite3.connect((R/'phases/terminal100_historical_v1_requalification/state_store.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
 assert db.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]==1 and db.execute('SELECT COUNT(*) FROM states').fetchone()[0]==1
 digest,exact,sj=db.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchone();assert json.loads(sj)==data and bytes(exact)==structural_canonical_bytes(data) and hashlib.sha256(exact).hexdigest()==digest
h=json.loads((B/'SPEC.json').read_text())['execution']['parameters']['bindings'];inp={k:read_bound(C/'runtime/intake/artifacts'/(v+'.bin'),v) for k,v in h.items()}
parent=inp['bootstrap'];reference=inp['reference'];_,states=mp._states_from_dag(parent['dag']);assert len(states)==193 and mp.state_dag_wire(states)==parent['dag'];print('PASS cold193 DAG roundtrips',flush=True)
observed=extract_interface_population(states,source_stage='G1:R100',source_authority_sha256=reference['source_authority_sha256']);assert observed==reference==data['historical_interface_population']
assert current_population(states,source_stage='G1:R100',source_authority_sha256=reference['source_authority_sha256'])==inp['diagnostic_v2'];print('PASS exact193 historical V1 and current V2 populations',flush=True)
by_ref={x.construction_digest:x for x in states};beam={k:{'skin':str(by_ref[k].skin),'caps':list(by_ref[k].total_caps)} for k in parent['beam_roots']};expected={x['construction_digest']:{'skin':x['resource_skin_sha256'],'caps':x['total_free_by_type']} for x in inp['anchor']['materialized_discovery_evidence']['state_probes']};assert len(beam)==24 and beam==expected
assert data['candidate_generation_calls']==0 and data['new_admissions']==0
shutil.copy2(A/'terminal100_requalification.json',B/'TERMINAL100_REQUALIFICATION.json');shutil.copy2(A/'historical_v1_interface_population.json',B/'HISTORICAL_V1_INTERFACE_POPULATION.json')
r=dict(status='PASS_COLD_HISTORICAL_V1_TERMINAL100_REQUALIFICATION',accepted_historical_depth=100,qualification_scope='HISTORICAL_V1_G1_TERMINAL_REPLAY_ONLY',native_exact_state_identity=True,restored_roots=193,all193_DAG_roundtrip=True,terminal_comparison='PASS193_EXACT_HISTORICAL_V1_PUBLIC_INTERFACES',beam_gate='PASS24_EXACT_ROOT_SKIN_CAPS',current_v2_population_reproduced=True,current_v2_engine_unchanged=True,candidate_generation_during_native=0,candidate_generation_during_audit=0,original_live_workspace_scientific_state_read=False,master_slices=151,new_admissions=0,G2_promotion=False)
(B/'AUDIT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
