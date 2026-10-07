from pathlib import Path
import sys,json,hashlib,tempfile,shutil
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B));sys.path.insert(0,'/workspace/scratch/3ea7ce26e49e/engine')
from project.unified import G2SavedRecordCandidateReader
from project.arguments import predecessor_arguments
from project.checks import bounded_routes
from infinity_grid.workflow_guard import preflight
from infinity_grid.result_contracts import normalize
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
spec=json.loads((B/'SPEC.json').read_text());i={x['logical_name']:x['path'] for x in spec['inputs']}
for x in spec['inputs']+spec['environment']['artifacts']:
 if sha(x['path'])!=x['sha256']:raise ValueError('INPUT_HASH:'+x['logical_name'])
with tempfile.TemporaryDirectory(prefix='ig_register0186_',dir='/tmp') as tmp:
 with G2SavedRecordCandidateReader(i['s1_candidate'],i['s1_proposal'],Path(i['s1_manifest']).parent,previous_catalog=i['new_catalog'],previous_arguments=predecessor_arguments(i,tmp)) as r:
  result=bounded_routes(r,i);assert r.coverage_report()['scientific_slices']==150
  h=r.saved.db.execute('SELECT outcome FROM records GROUP BY outcome HAVING count(*)=3 LIMIT 1').fetchone()[0]
  assert len(r.g2_s1_outcome_members(h.hex()))==3
  p=r.g2_s1_provenance();p['counts']['record_rows']=0;assert r.g2_s1_provenance()['counts']['record_rows']==580351
  assert r.saved.db.execute('PRAGMA query_only').fetchone()==(1,)
 for call in [lambda:r.g2_s1_record(0),r.coverage_report,lambda:r.lookup_g1_public_interface('missing')]:
  try:call()
  except ValueError:pass
  else:raise AssertionError('CLOSED_READER')
normalized=normalize(spec['output_contract'],spec['execution'],spec['question'])
gate=preflight(B,[B/'project/handler.py'])
(B/'NATIVE_PREFLIGHT.json').write_text(json.dumps({'status':'PASS','result_contract_normalized':True,'architecture':gate,'capture_called':False,'handler_called':False},indent=2)+'\n')
result.update(status='PASS_BOUNDED_REGISTRATION_AND_NATIVE_PREFLIGHT',master_release='MASTER_DATA_V1_0150',scientific_slices=150,maximum_class_membership_test=3,read_only_SQLite=True,previous_reader_bytes_unchanged=True,saved_reader_bytes_unchanged=sha(B/'project/saved_reader.py')=='e261c774b3193e29af58a1b23e272b14dd7bea59ac1c03ee476f7b275c550f95',pinned_inputs_verified=len(spec['inputs']),native_capture_started=False,native_handler_called=False,full_native_census_executed=False,generation_calls=0,new_admissions=0,next_scope='WP6_NATIVE_SAVED_G2_S1_SCOPED_INTEGRATION_CAPTURE_AND_EXECUTION')
(B/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
