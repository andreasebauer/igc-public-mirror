"""Prospective finite saved-domain correspondence audit; no producer execution."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import ast,gzip,hashlib,json,os,tempfile,time,zipfile
import reference
from infinity_grid.canon import canonical_sha256,write_json_atomic
from infinity_grid import saved_g1_scope_compatibility as compat
from infinity_grid import r100_reuse,semantic_sentinel,theorem_registry

def main():
 start=time.monotonic();inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text());params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
 for k,h in params['input_hashes'].items():assert hashlib.sha256(Path(inputs[k]).read_bytes()).hexdigest()==h
 assert params['input_hashes']['historical']=='c02849020a4b5a6e9bd755da3cf405ad1e422fa51fb5451c1ad7af26760f087e'
 assert params['input_hashes']['current']=='51cc2df89254cb94a22307ace944e4edaa29d5aa117febdd6353ba44bb3e8fb2'
 # Bind current source delta prospectively; existing scientific implementation stays exact.
 pkg=Path(compat.__file__).parent;baseline=json.loads(Path(inputs['baseline_manifest']).read_text())
 current={p.relative_to(pkg).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in pkg.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
 assert sorted(n for n in set(current)|set(baseline) if current.get(n)!=baseline.get(n))==['saved_g1_scope_compatibility.py']
 hb={'scope':compat.HISTORICAL,'artifact_sha256':params['input_hashes']['historical'],'producer_ref':'1274f1f90fede4238a19fca63eb3059857a696748aa07ced627b71b59e8a3603','production_authority':'NONE'}
 cb={'scope':compat.CURRENT,'artifact_sha256':params['input_hashes']['current'],'producer_ref':'76f3bfee70c13d34ac2eafc91060d96b199ee7d9fa794dd20d938417179ce2a2','production_authority':'NONE'}
 ha=compat.admit(inputs['historical'],hb,consumer_scope=compat.HISTORICAL);ca=compat.admit(inputs['current'],cb,consumer_scope=compat.CURRENT)
 refusals=[]
 def refuse(name,fn):
  try:fn()
  except (compat.ScopeCompatibilityError,r100_reuse.R100ReuseError,ValueError) as e:refusals.append({'case':name,'error':str(e)})
  else:raise AssertionError('REFUSAL_NOT_ENFORCED:'+name)
 refuse('historical_as_current',lambda:compat.admit(inputs['historical'],hb,consumer_scope=compat.CURRENT))
 refuse('current_as_historical',lambda:compat.admit(inputs['current'],cb,consumer_scope=compat.HISTORICAL))
 for field,value in [('production_authority','MASTER'),('artifact_sha256','0'*64),('scope','GLOBAL_G1'),('producer_ref',True)]:
  bad=dict(hb);bad[field]=value;refuse('binding_'+field,lambda bad=bad:compat.admit(inputs['historical'],bad,consumer_scope=compat.HISTORICAL))
 bad=dict(hb,adopt_old_authority=True);refuse('extra_authority_field',lambda:compat.admit(inputs['historical'],bad,consumer_scope=compat.HISTORICAL))
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'changed';p.write_bytes(b'first');b=dict(hb,artifact_sha256=hashlib.sha256(b'first').hexdigest());ad=compat.admit(p,b,consumer_scope=compat.HISTORICAL);p.write_bytes(b'second');refuse('bytes_changed_after_admission',ad.read)
 with zipfile.ZipFile(inputs['historical']) as z:
  receipt=json.loads(Path(inputs['historical_receipt']).read_text());assert receipt['archive_sha256']==hb['artifact_sha256']
  for r in receipt['manifest_members_verified']:assert len(z.read(r['member']))==r['bytes'] and hashlib.sha256(z.read(r['member'])).hexdigest()==r['sha256']
  old=json.loads(gzip.decompress(z.read('BOOTSTRAP100_HISTORICAL.json.gz')))['dag']
  hp=json.loads(z.read('HISTORICAL_V1_INTERFACE_POPULATION.json'));diag=json.loads(z.read('DIAGNOSTIC_V2_POPULATION.json'));hspec=json.loads(z.read('HISTORICAL_IMPLEMENTATION_SPEC_V1.json'))
 assert old['schema_id']=='IG_MATURATION_STATE_DAG_V1' and len(old['nodes'])==16528 and len(old['roots'])==193
 assert canonical_sha256({k:v for k,v in old.items() if k!='science_sha256'})==old['science_sha256']==receipt['DAG_science_sha256_read_from_source']
 for population in (hp,diag):
  assert canonical_sha256({k:v for k,v in population.items() if k!='science_sha256'})==population['science_sha256']
  for row in population['interfaces']:assert canonical_sha256({k:v for k,v in row.items() if k!='science_sha256'})==row['science_sha256']
 assert {r['carrier_ref'] for r in hp['interfaces']}==set(old['roots'])
 historical_rows={r['carrier_ref']:r for r in hp['interfaces']};diagnostic_rows={r['carrier_ref']:r for r in diag['interfaces']}
 historical_spec_only=[]
 for ref,row in historical_rows.items():
  other=diagnostic_rows[ref];diff=sorted(k for k in set(row)|set(other) if reference.enc(row.get(k))!=reference.enc(other.get(k)))
  assert diff==['implementation_spec_sha256','science_sha256'];historical_spec_only.append({'carrier_ref':ref,'differing_fields':diff})
 with zipfile.ZipFile(inputs['current']) as z:
  new=json.loads(z.read('EXACT_STATE_DAG_V1.json'));cert=json.loads(z.read('CERTIFIED_CARRIER_SET_V1.json'));binding=json.loads(z.read('R100_BINDING.json'))
  cap=json.loads(z.read('R100_CAPTURE.json'));done=json.loads(z.read('R100_COMPLETION.json'))
  for row,field in ((cap,'capture_id'),(done,'completion_sha256')):assert canonical_sha256({k:v for k,v in row.items() if k!=field})==row[field]
  # Original native capture/completion identity stays in outputs, never rebound.
  assert done['status']=='COMPLETED' and done['evidence_status']=='VERIFIED'
  assert cap['workspace']['source_sha256']==done['source_sha256']
  origin=cap['job']['execution']['parameters']
  assert origin['origin_engine_sha256']==binding['source_sha256']=='76f3bfee70c13d34ac2eafc91060d96b199ee7d9fa794dd20d938417179ce2a2'
  assert cap['job']['job_id']=='MASTER.STEP3.CURRENT.R100.Q10.PRESERVATION'
  assert done['scientific_outcome']=='Q07_VERIFIED_OUTPUTS_REUSED_NO_SCIENTIFIC_REEXECUTION'
  assert done['registration_sha256']==cap['job']['registration_sha256']
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for n in ('CERTIFIED_CARRIER_SET_V1.json','EXACT_STATE_DAG_V1.json'): (root/n).write_bytes(z.read(n))
   admitted=r100_reuse.preflight(root,binding);assert len(admitted.ordered_roots)==193
   (root/'EXACT_STATE_DAG_V1.json').write_text(json.dumps(old,separators=(',',':')))
   refuse('actual_current_r100_preflight_historical_substitution',lambda:r100_reuse.preflight(root,binding))
 assert new['execution_policy']=='NATIVE_V3' and len(new['nodes'])==14465 and len(new['roots'])==193
 comparison=compat.compare_ordered_dags(old,new)
 old_motifs={};new_motifs={}
 for dag,dest in ((old,old_motifs),(new,new_motifs)):
  for key in dag['roots']:
   row=dag['nodes'][key];label=(row['level'],row['lane'],row['motif_id']);assert label not in dest;dest[label]=key
 assert set(old_motifs)==set(new_motifs)
 # Motifs define prospective comparison coordinates only, never carrier identity.
 cr={r['construction_digest']:r['public_state'] for r in cert['carriers']}
 pairs=[];witnesses=[]
 for motif,oldkey in sorted(old_motifs.items()):
  newkey=new_motifs[motif];a=historical_rows[oldkey]['total_free_by_type'];b=cr[newkey]['total_caps']
  assert len(a)==len(b)==7 and all(type(v) is int for v in a+b)
  delta=[i for i in range(7) if a[i]!=b[i]]
  pairs.append({'motif_comparison_coordinate':list(motif),'historical_alias':oldkey,'current_alias':newkey,'historical_total_caps':a,'current_total_caps':b,'unequal_types':delta})
  if delta and len(witnesses)<3:witnesses.append({'motif_coordinate':list(motif),'caps7_direct_difference':pairs[-1],'ordered_carrier_direct_difference':reference.first_difference(oldkey,newkey,old['nodes'],new['nodes'])})
 capsets={'historical':Counter(tuple(r['total_free_by_type']) for r in hp['interfaces']),'current':Counter(tuple(r['public_state']['total_caps']) for r in cert['carriers'])}
 public_equal=capsets['historical']==capsets['current']
 direct={'schema_id':'IG_SAVED_R100_PUBLIC_COORDINATE_COMPARISON_V1','motif_coordinates_are_identity':False,'paired_motifs':len(pairs),'paired_caps7_equal':sum(not p['unequal_types'] for p in pairs),'caps7_multisets_equal':public_equal,'common_caps7_values':len(set(capsets['historical'])&set(capsets['current'])),'rows':pairs,'independent_direct_witnesses':witnesses}
 # Independent tiny controls require repeated slots and exact JSON scalar types.
 leaf={'kind':'O7','construction_digest':'a','source_id':'s','reserve_counts':[0]*7,'lane':'L'}
 tiny={'roots':['a'],'nodes':{'a':leaf}}
 assert compat.compare_ordered_dags(tiny,tiny)['ordered_root_sequences_equal'] is True
 tamper=deepcopy(tiny);tamper['nodes']['a']['reserve_counts'][0]=True
 refuse('boolean_reservation',lambda:compat.compare_ordered_dags(tiny,tamper))
 tamper=deepcopy(tiny);tamper['nodes']['a']['reserve_counts'][0]=0.0
 refuse('float_reservation',lambda:compat.compare_ordered_dags(tiny,tamper))
 missing=deepcopy(tiny);missing['roots']=['missing'];refuse('unresolved_root',lambda:compat.compare_ordered_dags(tiny,missing))
 def lift(children):return {'kind':'LIFT','construction_digest':'r','level':8,'lane':'L','motif_id':'M','top_edges_full':[],'children':children}
 repeated={'roots':['r'],'nodes':{'a':leaf,'r':lift(['a','a'])}}
 single={'roots':['r'],'nodes':{'a':leaf,'r':lift(['a'])}}
 assert compat.compare_ordered_dags(repeated,single)['ordered_root_sequences_equal'] is False
 # Actual producer dependency routes are source-read, never invoked.
 consumers=[]
 for name,functions in [('uplift_g3_s0.py',['run_g3_s0']),('uplift_g3_s1.py',['reproduce_s0_challenge_states'])]:
  p=pkg/name;tree=ast.parse(p.read_text());found=[]
  for node in ast.walk(tree):
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in functions:
    found.append({'function':node.name,'line':node.lineno,'calls':sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)})})
  assert len(found)==len(functions);consumers.append({'source':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'functions':found})
 assert semantic_sentinel.verify_native_semantics(raise_on_change=False)['status']=='REOPEN_REQUIRED'
 assert len(theorem_registry.verify_all_theorems(raise_on_stale=False)['stale_theorems'])==4
 write_json_atomic('ORDERED_DAG_COMPARISON.json',comparison);write_json_atomic('PUBLIC_CAPS7_COMPARISON.json',direct)
 write_json_atomic('HISTORICAL_V1_V2_SPEC_DIAGNOSTIC.json',{'scope':'SAME_HISTORICAL_SAVED_COHORT_ONLY','rows':historical_spec_only,'new_current_cohort_equivalence':False,'population_rewritten':False})
 write_json_atomic('CONSUMER_DEPENDENCY_TRACE.json',{'consumers':consumers,'rule':'Historical alias witness consumers require actual correspondence; current saved input admission is not historical witness adoption.'})
 classification='ORDERED_AND_PUBLIC_DIFFERENCE' if not public_equal else ('ORDERED_DIFFERENCE_PUBLIC_MULTISET_EQUAL' if not comparison['ordered_root_sequences_equal'] else 'ORDERED_AUDIT_PROJECTION_EQUAL_POLICY_NOT_PROVEN')
 write_json_atomic('QUALIFICATION_RESULT.json',{'status':'PASS','scientific_outcome':'FINITE_SAVED_G1_SCOPE_CORRESPONDENCE_AUDITED','classification':classification,'historical_nodes':16528,'current_nodes':14465,'historical_roots':193,'current_roots':193,'common_exact_root_values':comparison['common_exact_root_values'],'paired_caps7_equal':direct['paired_caps7_equal'],'caps7_multisets_equal':public_equal,'refusal_count':len(refusals),'scope_gate_qualified':True,'actual_current_preflight_checked':True,'historical_current_equivalence_granted':False,'generation_calls':0,'production_authority':'NONE','revised_package2_complete':False,'body_seconds':time.monotonic()-start,'current_preservation_capture_id':cap['capture_id'],'current_preservation_completion_sha256':done['completion_sha256'],'original_current_capture_id':origin['origin_capture_id'],'original_current_completion_sha256':origin['origin_completion_sha256'],'original_current_engine_sha256':origin['origin_engine_sha256'],'source_delta':['saved_g1_scope_compatibility.py']})
 write_json_atomic('REFUSALS.json',{'status':'PASS','cases':refusals})
if __name__=='__main__':main()
