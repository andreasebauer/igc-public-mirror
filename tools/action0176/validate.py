"""Validate saved projected evidence and recipe definitions without scientific imports."""
from pathlib import Path
import ast,collections,gzip,hashlib,io,json,sys,zipfile
B=Path(__file__).resolve().parent;I=B/'inputs'
sha=lambda b:hashlib.sha256(b).hexdigest()
js=lambda d:json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
h=lambda d:sha(js(d))
load=lambda n:json.loads((I/n).read_bytes())
def function_ast(n,name):
 t=ast.parse((I/n).read_text());f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name)
 return ast.dump(f,include_attributes=False)
def validate():
 for n,p in json.loads((B/'INPUT_PINS.json').read_bytes()).items():
  b=(I/n).read_bytes();assert sha(b)==p['sha256'] and len(b)==p['bytes']
 pop=load('G2_S0_INTERFACE_POPULATION.json');rows=pop['interfaces'];refs=[r['carrier_ref'] for r in rows]
 assert refs==sorted(refs) and len(set(refs))==193
 assert h(refs)==pop['carrier_ref_set_sha256']==load('G1_R100_COMPLETE_COHORT.json')['carrier_refs_sha256']
 assert h({k:v for k,v in pop.items() if k!='science_sha256'})==pop['science_sha256']
 for r in rows:
  caps=r['total_free_by_type'];res=r['one_endpoint_reservations']
  assert len(caps)==len(res)==7 and all(type(x)==int and x>0 for x in caps)
  for t,x in enumerate(res):
   assert x['endpoint_type']==t and x['available'] is True
   assert x['successor_total_free_by_type']==[v-(j==t) for j,v in enumerate(caps)]
  payload={'schema_id':'IG_G1_PUBLIC_ONE_ENDPOINT_INTERFACE_SEMANTICS_V1','boundary_resource_skin_sha256':r['boundary_resource_skin_sha256'],'total_free_by_type':caps,'one_endpoint_reservations':res,'scope':'ONE_EXTERNAL_ENDPOINT_RESERVATION_FOR_WHOLE_CARRIER_PAIR_CONNECTION'}
  assert h(payload)==r['interface_sha256']
 classes=collections.Counter(r['interface_sha256'] for r in rows)
 assert len(classes)==192 and sorted(classes.values())==[1]*191+[2]
 assert pop['interface_class_histogram']==[{'interface_sha256':k,'carrier_count':classes[k]} for k in sorted(classes)]
 assert function_ast('maturation_parallel.py','enumerate_candidate_recipes')==function_ast('historical_maturation_parallel.py','enumerate_candidate_recipes')
 for name in ['_semantic_interface_payload','_outcome_semantics']:
  assert function_ast('uplift_structural.py',name)==function_ast('historical_uplift_structural.py',name)
 assert load('O_REGIME_MOTIF_LIBRARY_v1.json')==load('historical_O_REGIME_MOTIF_LIBRARY_v1.json')
 motifs=load('O_REGIME_MOTIF_LIBRARY_v1.json')['motifs'];recipes=[]
 for mi,m in enumerate(motifs):
  n=m['n']
  for lane,seed in [('HOM',mi%31)]+([('HET',(mi*3+1)%31)] if n in (4,5) or n==6 and mi%6==0 else [])+([('MIX',(mi*5+2)%31)] if n==4 else []):
   recipes.append({'lane':lane,'motif_id':f'{lane}:{n}:{mi}','n':n,'edges':m['edges'],'schedule_seed':seed,'force_pair':None})
 recipes += [{'lane':'TWIN','motif_id':'TWIN:'+label,'n':6,'edges':edges,'schedule_seed':0,'force_pair':[0,0]} for label,edges in [('A',[[0,1],[1,2],[1,3],[2,4],[3,5]]),('B',[[0,4],[0,5],[1,3],[2,3],[3,4]])]]
 cohort=load('R100_SOURCE_INPUT.json')['materialized_discovery_evidence']['candidate_cohort']
 assert len(recipes)==193 and {x['motif_id'] for x in recipes}=={x['motif_id'] for x in cohort['motif_structural_signatures']}
 failed=load('REALIZATION_RESULT.json')
 assert failed['status']=='FAIL' and failed['classification']=='S1_PROJECTION_NOT_CONGRUENT_REPAIR_BEFORE_S2'
 assert failed['projected_classes_split_by_realized_public_observer_count']==4844
 assert failed['projection_immediate_mismatch_count']==failed['realization_failure_count']==0
 out={'schema':'IG_G1_PROJECTED_SAVED_RECOVERY_V1','status':'PASS_FOR_DECLARED_SAVED_PUBLIC_SCOPE','master_release':'MASTER_DATA_V1_0149','new_admissions':0,'generation_calls':0,'saved_G1_public_interfaces':193,'saved_interface_classes':192,'saved_one_endpoint_reservation_rows':1351,'exact_capacity_arithmetic_checks':1351,'current_historical_recipe_enumerator_AST_equal':True,'motif_library_equal':True,'recipes_enumerated_without_construction':193,'recipe_motif_ids_match_saved_R100':True,'full_exact_carrier_DAG_available':False,'complete_construction_recipe_closure':False,'historical_projected_S1_claim':'PROJECTED_CONNECTION_CENSUS_ONLY','later_realization_status':'FAIL_REPAIR_REQUIRED','historical_realization_split_classes':4844,'historical_realization_failed_constructions':0,'source_skin_hashes':'SAVED_OPAQUE_EVIDENCE_NOT_RECOMPUTED_FROM_EXACT_CARRIERS','next_scope':'WP6_G1_G2_REPAIRED_S1_EVIDENCE_AND_EXACT_PARENT_DAG_RECOVERY','full_l0_to_g8_complete':False}
 return out,recipes,rows

def validate_stream(path,rows):
 with zipfile.ZipFile(path) as outer:
  assert sha(Path(path).read_bytes())=='9e7eb190fe63336e46c69b9e076e859f1679f73aad18bad35c2b817b7e66ff84'
  n=next(n for n in outer.namelist() if n.startswith('science_original/') and n.endswith('.zip'))
  with zipfile.ZipFile(io.BytesIO(outer.read(n))) as z:
   H=hashlib.sha256();count=0;seen=set();by={r['carrier_ref']:r for r in rows};ends={}
   for r in rows:
    for t,x in enumerate(r['one_endpoint_reservations']):
     e={'input_interface_sha256':r['interface_sha256'],'reserved_type':t,**{k:x[k] for k in ['available','successor_boundary_resource_skin_sha256','successor_total_free_by_type']}}
     ends[r['carrier_ref'],t]=(h(e),e)
   with gzip.GzipFile(fileobj=z.open('science/evidence/G2_S1_PAIR_CONNECTION_RECORDS.jsonl.gz')) as f:
    for line in f:
     H.update(line);d=json.loads(line);left,right=d['left_carrier_ref'],d['right_carrier_ref'];assert left<=right
     a,b=map(int,d['connection_operator_ref'].rsplit(':',1)[1].split('>'));assert 0<=a<7 and 0<=b<7
     assert d['left_interface_sha256']==by[left]['interface_sha256'] and d['right_interface_sha256']==by[right]['interface_sha256']
     assert d['legality']=='LEGAL'
     ep=sorted([ends[left,a],ends[right,b]],key=lambda x:x[0]);ec=[x[1] for x in ep]
     payload={'schema_id':'IG_G_UPLIFT_S1_PAIR_OUTCOME_SEMANTICS_V1','legal':True,'reason':'PUBLIC_RESOURCES_AVAILABLE','endpoints':ec,'relation_kind':'G1_PUBLIC_TYPED_BRIDGE_RELATION','assembly_total_free_by_type':[ec[0]['successor_total_free_by_type'][t]+ec[1]['successor_total_free_by_type'][t] for t in range(7)]}
     assert h(payload)==d['outcome_science_sha256'] and d['outcome_ref']=='G2S1O:'+d['outcome_science_sha256']
     seen.add((left,right,a,b));count+=1
   expected=load('G2_S1_RECORD_STREAM_VERIFICATION.json');assert count==len(seen)==580351 and H.hexdigest()==expected['expected_stream_sha256']
   pairs=collections.Counter((a,b) for _,_,a,b in seen);assert len(pairs)==31 and set(pairs.values())=={18721}
   assert h([list(x) for x in sorted(pairs)])==load('G1_R100_COMPLETE_COHORT.json')['bridge_operator_catalog_sha256']
   return {'status':'PASS_PROJECTED_SEMANTICS_ONLY','rows_checked':count,'distinct_pair_operator_records':len(seen),'bridge_operators':31,'unordered_pairs_per_operator':18721,'stream_sha256':H.hexdigest(),'exact_assembly_realization_rerun':False}

if __name__=='__main__':
 out,recipes,rows=validate()
 if len(sys.argv)>1:out['stream_validation']=validate_stream(sys.argv[1],rows)
 print(json.dumps(out,indent=2))
