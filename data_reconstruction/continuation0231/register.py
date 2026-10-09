"""Register historical V1 restoration separately from current V2 operations."""
from pathlib import Path
import json,hashlib,ast,sys,sqlite3,shutil
B=Path(__file__).resolve().parent;P=Path('/workspace/scratch/75d85ae95659/continuation0230')
sys.path.insert(0,'/tmp/ig_engine0204')
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert sys.version_info[:3]==(3,13,5) and sqlite3.sqlite_version=='3.51.3'
r=json.loads((B/'SOURCE_INPUT_RECOVERY.json').read_text());s=json.loads((P/'SPEC.json').read_text())
for p in (B/'project/historical').glob('*.py'):assert p.read_bytes()==(P/'project/historical'/p.name).read_bytes()
assert (B/'project/current_interface.py').read_bytes()==(P/'project/public_interface.py').read_bytes()
assert (B/'project/public_interface.py').read_text()==(B/'project/current_interface.py').read_text().replace('G_UPLIFT_S0_S1_IMPLEMENTATION_SPEC_V2.json','G_UPLIFT_S0_S1_IMPLEMENTATION_SPEC_V1.json')
f=Path('/tmp/ig_engine0204/infinity_grid/uplift_structural.py');assert sha(f)=='3eb7e414c918e76deb176979abc8cfec1e278f90f70273bbeda6da71f68c13e6'
shutil.copy2('/tmp/ig_engine0204/infinity_grid/resources/uplift/G_UPLIFT_S0_S1_IMPLEMENTATION_SPEC_V1.json',B/'HISTORICAL_IMPLEMENTATION_SPEC_V1.json')
shutil.copy2(P/'HISTORICAL_DEPTH100_SOURCE_INPUT.json',B/'HISTORICAL_DEPTH100_SOURCE_INPUT.json')
shutil.copy2(P/'SAVE_RECEIPT.json',B/'PREDECESSOR_HANDOFF_REFERENCE.json')
inputs=dict(bootstrap=B/'BOOTSTRAP100_HISTORICAL.json.gz',reference=Path('/tmp/ig_verified0228/6b7405f8e3374061f167cecd20225db4f54f22ad45ed8ac97cace64f1fe8f666.bin'),anchor=B/'HISTORICAL_DEPTH100_SOURCE_INPUT.json',v1_spec=B/'HISTORICAL_IMPLEMENTATION_SPEC_V1.json',diagnostic_v2=B/'DIAGNOSTIC_V2_POPULATION.json')
h={k:sha(p) for k,p in inputs.items()};v1=json.loads(inputs['v1_spec'].read_text())
s.update(job_id='MASTER.G1.HISTORICAL.V1.REQUALIFICATION.0231',project_source=str(B/'project'))
s['question']=dict(stage_id=s['job_id'],description='Restore saved terminal100; exact historical V1 population and all193 DAG roundtrips; current V2 observation preserved',outcomes=['PASS_HISTORICAL_V1_TERMINAL_REQUALIFICATION'],stopping_rule='Stop first input identity, spec, DAG, census, beam, interface, resource or preservation mismatch. No candidate generation. No master admission or G2 promotion.')
s['execution']['parameters']=dict(bindings=h,historical_spec_identity=v1['spec_sha256'],bootstrap_raw_sha256=r['bootstrap_raw_sha256'])
s['inputs']=[dict(logical_name=k,path=str(p),sha256=h[k]) for k,p in inputs.items()]
values=dict(outcome='PASS_HISTORICAL_V1_TERMINAL_REQUALIFICATION',completed_depth=100,restored_roots=193,all193_DAG_roundtrip=True,terminal_comparison='PASS193_EXACT_HISTORICAL_V1_PUBLIC_INTERFACES',historical_depth100_beam_anchor='PASS24_EXACT_ROOT_SKIN_CAPS',candidate_generation_calls=0,current_v2_population_reproduced=True,master_scientific_slices=151,new_admissions=0)
s['output_contract']['result_checks']=[dict(pointer='/'+k,equals=v) for k,v in values.items()]
(B/'SPEC.json').write_text(json.dumps(s,indent=2)+'\n')
a=dict(status='PASS_HISTORICAL_V1_RESTORATION_SOURCE_BINDING',parent_checkpoint=230,historical_helpers_byte_identical=True,current_extractor_byte_identical=True,historical_extractor_only_change='Explicit V1 spec resource constant',historical_spec_identity=v1['spec_sha256'],current_engine_file_sha256=sha(f),frozen_engine_modified=False,operator_executed_candidate_generation=False,source_recovery=r,sha256_by_file={str(p.relative_to(B)):sha(p) for p in (B/'project').rglob('*.py')})
(B/'SOURCE_ATTESTATION.json').write_text(json.dumps(a,indent=2));(B/'READBACKS.json').write_bytes((P/'READBACKS.json').read_bytes())
print('PASS registered historical V1 restoration spec; no candidate generation')
