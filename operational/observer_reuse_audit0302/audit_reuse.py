"""Evidence-only observer recovery. Never imports or invokes scientific evaluators."""
import pathlib,json,hashlib,zipfile,ast,io,datetime
W=pathlib.Path('/workspace/scratch/2a87972b51af')
O=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def source_function(p,name):
 t=p.read_text();return ast.dump(next(n for n in ast.walk(ast.parse(t)) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
def digest_functions(z,trail=''):
 out=[]
 for n in z.namelist():
  if n.endswith('.zip'):out.extend(digest_functions(zipfile.ZipFile(io.BytesIO(z.read(n))),trail+'/'+n))
  elif n.endswith('.py'):
   t=z.read(n).decode()
   if 'def _digest_container' in t:
    f=next(n for n in ast.walk(ast.parse(t)) if isinstance(n,ast.FunctionDef) and n.name=='_digest_container')
    out.append(dict(path=trail+'/'+n,source=ast.get_source_segment(t,f),file_sha256=sha(z.read(n))))
 return out
def main():
 plan=read(W/'continuation0240/PLAN.json');preg=read(W/'continuation0240/PREREGISTRATION.json');reg=read(W/'continuation0285/PREREGISTRATION.json')
 assert sha((W/'continuation0240/PLAN.json').read_bytes())==preg['plan_sha256']
 old_ledger=read(W/'continuation0281/EARNED_LEDGER.json');old_credit={r['chunk_index']:r for r in old_ledger['completed_chunks']}
 closeout=read(W/'continuation0282/CLOSEOUT_AUDIT.json');assert sha((W/'continuation0281/EARNED_LEDGER.json').read_bytes())==closeout['earned_ledger_sha256']
 assert old_ledger['status']=='EARNED_NATIVE_COLD_AND_EXACT_SAVED' and len(old_credit)==114
 for rel,h in preg['source_sha256'].items():
  if rel.startswith('project/'):
   assert sha((W/'continuation0240'/rel).read_bytes())==h,rel
  elif rel.startswith('engine/'):
   assert sha((pathlib.Path('/tmp/ig_engine0237/infinity_grid')/rel.split('/',1)[1]).read_bytes())==h,rel
 old_tree=ast.parse((W/'continuation0240/project/worker.py').read_text());new_tree=ast.parse((W/'continuation0285/project/worker.py').read_text())
 old_check=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_case');new_check=next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_case')
 assert new_check.args.args[-1].arg=='primary_capture';new_check.args.args.pop();new_check.args.defaults.pop()
 new_check.body=[n for n in new_check.body if not (isinstance(n,ast.If) and ast.unparse(n.test)=='primary_capture is not None')]
 assert ast.dump(old_check,include_attributes=False)==ast.dump(new_check,include_attributes=False)
 # The observer worker adds primary_capture; exact source snapshots remain in evidence.
 pop_path=W/'continuation0231/HISTORICAL_V1_INTERFACE_POPULATION.json'
 assert sha(pop_path.read_bytes())=='6b7405f8e3374061f167cecd20225db4f54f22ad45ed8ac97cace64f1fe8f666'
 pop={r['carrier_ref']:r for r in read(pop_path)['interfaces']}
 resource=pathlib.Path('/tmp/ig_engine0237/infinity_grid/resources/decoder/O7_MATERIAL_ROOT_cb6f48641eb9.zip')
 assert sha(resource.read_bytes())=='cb6f48641eb99d374d19d5dc8d5bfada13e12cf1ecb20ed0e66958629ec08f1b'
 funcs=digest_functions(zipfile.ZipFile(resource));assert len(funcs)==1
 assert funcs[0]['source']=='''def _digest_container(tag, child_sigs):
    payload = (tag + "|" + "|".join(sorted(child_sigs))).encode()
    return hashlib.sha256(payload).hexdigest()'''
 rows=read(O/'STATE_OBJECTS.json');ledger=read(W/'continuation0302/EARNED_LEDGER.json');current={r['chunk_index']:r for r in ledger['completed_chunks']}
 counts=[];ordinals=set();classes={};cross=0;proofs=[]
 (O/'derived').mkdir(exist_ok=True)
 for row in rows:
  i=row['chunk'];d=pathlib.Path(row['proof_directory']);p=O/('old_state000.zip' if i==0 else f'state_{i:03d}.zip')
  raw=p.read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256'];z=zipfile.ZipFile(io.BytesIO(raw));assert z.testzip() is None
  native=read(d/'NATIVE_RESULT.json');cold=read(d/'COLD_AUDIT.json');assert native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED' and cold['public_projections_and_operational_witnesses_exact']
  assert sha((d/'NATIVE_RESULT.json').read_bytes())==old_credit[i]['native_result_sha256']
  assert sha((d/'COLD_AUDIT.json').read_bytes())==old_credit[i]['cold_audit_sha256']
  assert sha((d/'SAVE_RECEIPT.json').read_bytes())==old_credit[i]['save_receipt_sha256']
  path=next(n for n in z.namelist() if '/sealed/' in n and n.endswith(f'/artifacts/complete_class_Q2_{i:03d}.json'));payload=z.read(path)
  assert sha(payload)==cold['native_artifact_sha256']
  assert any(r['sha256']==sha(payload) for r in native['evidence'])
  artifact=read_json=json.loads(payload);out={r['ordinal']:r for r in artifact['cases']}
  cp=plan['chunks'][i];case_path=W/'continuation0240'/cp['cases_path'];assert sha(case_path.read_bytes())==cp['cases_sha256'];cases=read(case_path)
  expected_path=W/'continuation0285/expected'/f'{i:03d}.json';assert sha(expected_path.read_bytes())==reg['specifications'][i]['expectations_sha256'];exp={r['ordinal']:r for r in read(expected_path)['cases']}
  assert set(out)==set(exp)=={r['ordinal'] for r in cases['cases']};assert cold['cases_exact']==len(out)==cp['case_count']
  fresh=None
  if i in current:
   nd=pathlib.Path(current[i]['unit_directory']);nz=zipfile.ZipFile(nd/'NATIVE_CHECKPOINT_SLIM.zip')
   # Slim export stores the capture state as a dependency; use exact saved readbacks.
   nc=read(nd/'COLD_AUDIT.json');nn=read(nd/'NATIVE_RESULT.json');h=nc.get('native_artifact_sha256')
   if h is None:h=next(r['sha256'] for r in nn['evidence'] if r['path'].endswith(f'historical_public_observer_{i:03d}.json'))
   paths=[n for n in nz.namelist() if n.endswith(f'historical_public_observer_{i:03d}.json')]
   if paths:nb=nz.read(paths[0])
   else:
    ex=read(nd/'CHECKPOINT_EXPORT.json');state=next(r for r in ex['dependencies'] if any(v['role']=='checkpoint_state' for v in r['receipts']));m=read(nd/'READBACKS.json')[state['sha256']];sb=pathlib.Path(m['path']).read_bytes();assert sha(sb)==state['sha256'];sz=zipfile.ZipFile(io.BytesIO(sb));sn=next(n for n in sz.namelist() if '/sealed/' in n and n.endswith(f'historical_public_observer_{i:03d}.json'));nb=sz.read(sn)
   assert sha(nb)==h;fresh={r['ordinal']:r for r in json.loads(nb)['cases']}
  derived=[];chunkclasses=set()
  for c in cases['cases']:
   ordinal=c['ordinal'];assert ordinal not in ordinals;ordinals.add(ordinal);s=c['record'];v=out[ordinal];assert v['source_case_sha256']==sha(canon(c))==exp[ordinal]['source_case_sha256'];assert v['repaired_outcome_sha256']==s['outcome_science_sha256']==exp[ordinal]['outcome_science_sha256']
   a,b=map(int,s['connection_operator_ref'].rsplit(':',1)[1].split('>'));l,r=pop[s['left_carrier_ref']],pop[s['right_carrier_ref']];lr=l['one_endpoint_reservations'][a];rr=r['one_endpoint_reservations'][b];assert lr['available'] and rr['available']
   skin=sha(('R101|'+'|'.join(sorted([lr['successor_boundary_resource_skin_sha256'],rr['successor_boundary_resource_skin_sha256']]))).encode())
   caps=[x+y for x,y in zip(lr['successor_total_free_by_type'],rr['successor_total_free_by_type'])];assert len(caps)==7 and min(caps)>=0
   independently_accounted=[x+y for x,y in zip(l['total_free_by_type'],r['total_free_by_type'])];independently_accounted[a]-=1;independently_accounted[b]-=1;assert caps==independently_accounted
   edge=list(v['operational_witness']['bridge_edge'][:4]);assert edge==[0,1,a,b]
   reservations=[];q=v['public_projection'];assert q['science_sha256']==s['q2_one_reservation_successor_skins_sha256']
   assert sha(canon([dict(t=r['endpoint_type'],available=r['available'],successor_skin=r['successor_boundary_resource_skin_sha256']) for r in q['rows']]))==q['science_sha256']
   for qr in q['rows']:
    t=qr['endpoint_type'];assert qr['available']==(caps[t]>0);sc=caps.copy();sc[t]-=1
    reservations.append(dict(endpoint_type=t,available=qr['available'],successor_skin=qr['successor_boundary_resource_skin_sha256'],successor_total_caps=sc if qr['available'] else None))
   observer=dict(schema_id='IG_G2_S1_REALIZED_PUBLIC_ASSEMBLY_V1',assembly_skin=skin,total_caps=caps,typed_top_edge_multiset=[edge],one_endpoint_reservations=reservations);h=sha(canon(observer));assert h==exp[ordinal]['realized_public_sha256'],('historical',i,ordinal)
   key=s['outcome_science_sha256'];chunkclasses.add(key)
   if key in classes:assert classes[key]==observer
   else:classes[key]=observer
   if fresh is not None:
    assert observer==fresh[ordinal]['public_observer'] and h==fresh[ordinal]['public_observer_sha256'];assert v['operational_witness']==fresh[ordinal]['operational_witness'];cross+=1
   derived.append(dict(ordinal=ordinal,source_case_sha256=v['source_case_sha256'],outcome_science_sha256=key,public_observer=observer,public_observer_sha256=h,provenance='DERIVED_FROM_EARNED_Q2_ARTIFACT_AND_BOUND_PUBLIC_INPUTS_NO_NEW_REALIZATION'))
  assert len(chunkclasses)==cp['class_count'];write(O/'derived'/f'{i:03d}.json',dict(chunk=i,cases=derived))
  proofs.append(dict(chunk=i,state_object_sha256=row['sha256'],drive_file_id=row['drive_file_id'],native_artifact_sha256=sha(payload),native_result_sha256=sha((d/'NATIVE_RESULT.json').read_bytes()),cold_audit_sha256=sha((d/'COLD_AUDIT.json').read_bytes()),derived_observers_sha256=sha((O/'derived'/f'{i:03d}.json').read_bytes())))
  counts.append(dict(chunk=i,members=len(derived),classes=len(chunkclasses),historical_hashes_exact=True,fresh_comparison=i in current));print('PASS',i,flush=True)
 assert len(ordinals)==7058 and len(classes)==3523 and cross==2969
 result=dict(status='PASS_SAVED_EVIDENCE_REUSE_AUDIT',as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),registered_chunks=114,members=7058,classes=3523,exact_historical_observer_hashes=7058,fresh_native_observer_crosscheck_chunks=48,fresh_native_observer_crosscheck_members=cross,full_payload_and_operational_witness_crosschecks_exact=True,pending_full_replays_avoidable=66,new_scientific_realizations=0,new_native_runs=0,new_cold_runs=0,last_earned_native_observer_checkpoint=302,native_observer_earned_chunks=48,remaining_66_are_evidence_derivations_not_new_native_executions=True,full_580351_population_compared=False,canonicalization_impact_assessed=False,G2_promotion=False,new_admissions=0,master_slices=152,decision='Do not resume the 66 full replay chunks. Register an evidence-only amendment preserving old native/cold provenance and current 48 fresh observer proofs; exact save audit and derived payloads before additional credit.',digest_function=funcs[0],source_hashes={str(p):sha(p.read_bytes()) for p in [W/'continuation0240/project/worker.py',W/'continuation0285/project/worker.py',W/'continuation0240/project/historical/regime_scanner.py',W/'continuation0285/project/observer.py']},chunks=counts)
 write(O/'AUDIT_RESULT.json',result);write(O/'PROVENANCE.json',proofs);print(json.dumps({k:v for k,v in result.items() if k not in ['chunks','source_hashes','digest_function']}))
if __name__=='__main__':main()
