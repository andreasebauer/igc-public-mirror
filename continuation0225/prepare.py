"""Freeze a separate recovery capture; never amend or regenerate checkpoint0224."""
from pathlib import Path
import json,hashlib,sqlite3,sys
B=Path(__file__).resolve().parent;P=B.parent/'continuation0224'
J=Path(json.loads((P/'POINTER.json').read_text())['workspace'])
sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid.structural_encoding import structural_canonical_bytes
rows=[]
for n in range(75,81):
 f=next(J.glob(f'runtime/runs/*/chain/decoder_stage_runtime/*/phases/historical_g1_depth_{n}/state_store.sqlite3'))
 with sqlite3.connect(f.resolve().as_uri()+'?mode=ro',uri=True) as c:
  saved=c.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchall();count=c.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]
 assert len(saved)==1 and count==1
 digest,exact,sj=saved[0];state=json.loads(sj)
 assert structural_canonical_bytes(state)==bytes(exact) and hashlib.sha256(exact).hexdigest()==digest
 rows.append({'level':n,'state':state,'index_digest':digest,'generation_tasks':count})
export=json.loads((P/'CHECKPOINT_EXPORT.json').read_text())
capture=json.loads((P/'POINTER.json').read_text())['capture_id']
evidence={'schema_id':'IG_SAVED_HISTORICAL_EXACT_PHASES_V1','prior_capture_id':capture,'prior_checkpoint_sha256':export['sha256'],'phases':rows}
(B/'SAVED_PHASES.json').write_text(json.dumps(evidence,sort_keys=True,separators=(',',':')))
spec=json.loads((P/'SPEC.json').read_text());spec['job_id']='MASTER.G1.HISTORICAL.COMPLETION.RECOVERY.0225';spec['project_source']=str(B/'project')
spec['question']={'stage_id':spec['job_id'],'description':'Verify saved depths75–80 exact native identities and restore144 DAG roots; separately seal recovery without candidate regeneration','outcomes':['PASS_HISTORICAL_DEPTH80_NATIVE_RECOVERY'],'stopping_rule':'Stop first saved identity, DAG, anchor, resource or preservation mismatch'}
files={'bootstrap':Path(spec['inputs'][-1]['path']),'phases':B/'SAVED_PHASES.json','anchor':P/'HISTORICAL_DEPTH80_SOURCE_INPUT.json','prior_audit':P/'AUDIT.json','prior_export':P/'CHECKPOINT_EXPORT.json'}
hashes={k:hashlib.file_digest(v.open('rb'),'sha256').hexdigest() for k,v in files.items()}
spec['inputs']=[{'logical_name':k,'path':str(v),'sha256':hashes[k]} for k,v in files.items()]
spec['execution']['parameters']={'bindings':hashes,'prior_capture_id':capture,'final_science_sha256':rows[-1]['state']['science_sha256']}
spec['resources']['workspace_budget_bytes']=4294967296
spec['output_contract']['result_checks']=[{'pointer':'/'+k,'equals':v} for k,v in {'outcome':'PASS_HISTORICAL_DEPTH80_NATIVE_RECOVERY','completed_depth':80,'roots':24,'historical_depth80_anchor':'PASS24_EXACT_ROOT_SKIN_CAPS','candidate_regeneration_calls':0,'saved_native_exact_identity_phases':6,'DAG_roundtrip_roots':144,'master_scientific_slices':151,'new_admissions':0,'prior_capture_status':'PAUSED_WORKSPACE_BUDGET','terminal_comparison':'NOT_RUN'}.items()]
(B/'SPEC.json').write_text(json.dumps(spec,indent=2))
for script in ('preflight.py','transfer.py'):
 p=B/script;p.write_text(p.read_text().replace('0224','0225'))
attest={'status':'PASS_SAVED_PHASE_NATIVE_IDENTITY_EXTRACTION','prior_capture_id':capture,'prior_checkpoint_sha256':export['sha256'],'prior_capture_modified':False,'candidate_generation_calls':0,'phase_count':6,'frozen_bindings':hashes,'historical_helpers_unchanged':True}
for f in (B/'project/historical').glob('*.py'):
 assert f.read_bytes()==(P/'project/historical'/f.name).read_bytes()
(B/'SOURCE_ATTESTATION.json').write_text(json.dumps(attest,indent=2));print(json.dumps(attest))
