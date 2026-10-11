import pathlib,json,hashlib,shutil,sys,datetime,zipfile
W=pathlib.Path('/workspace/scratch/2a87972b51af');O=pathlib.Path('/workspace/scratch/54c3276ed25c');Q=W/'continuation0285';T=W/'continuation0288'
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def exact(rows):
 for r in rows:assert r['drive_exact_readback_verified'] and sha(r['path'])==r['sha256']
reg=read(Q/'PREREGISTRATION.json')
action=sys.argv[1]
if action=='setup':
 s=read(W/'CURRENT_STATUS.json');old=int(s['latest_checkpoint']);start=int(s['official_observer_next_chunk']);assert start<114 and not s.get('observer_batch_in_flight')
 P=W/f'continuation{old:04d}';L=read(P/'EARNED_LEDGER.json');assert L['status']=='EARNED_NATIVE_COLD_AND_EXACT_SAVED' and L['next_chunk_index']==start
 for r in L['save_receipts']:assert sha(r['path'])==r['sha256']
 cp=old+1;B=W/f'continuation{cp:04d}';assert not B.exists();B.mkdir();scope=list(range(start,min(start+3,114)));plan=dict(checkpoint=cp,start=start,end=scope[-1]+1,scope=scope,prior_ledger=str(P/'EARNED_LEDGER.json'));write(B/'BATCH_PLAN.json',plan)
 c=(T/'capture.py').read_text();c=c.replace('if idx>3:',f'if idx>{start}:').replace('continuation0287/EARNED_LEDGER.json',f'continuation{old:04d}/EARNED_LEDGER.json').replace("['next_chunk_index']==3",f"['next_chunk_index']=={start}").replace('ig_native0288_',f'ig_native{cp:04d}_')
 a=(T/'audit_cold.py').read_text().replace('ig_cold0288_',f'ig_cold{cp:04d}_').replace('ig_native0288_',f'ig_native{cp:04d}_')
 (B/'capture.py').write_text(c);(B/'audit_cold.py').write_text(a);shutil.copy2(__file__,B/'continuous_observer.py')
 prior=pathlib.Path(L['completed_chunks'][-1]['unit_directory'])
 for i in scope:
  D=B/f'{i:03d}';D.mkdir();shutil.copy2(Q/'specs'/f'{i:03d}.json',D/'SPEC.json')
  for n in ['capture.py','audit_cold.py']:shutil.copy2(B/n,D/n)
  for n in ['ack.py','run_native.py','prepare_drain.py','confirm_checkpoint.py','export_checkpoint.py']:shutil.copy2(T/'003'/n,D/n)
  shutil.copy2(prior/'READBACKS.json',D/'READBACKS.json')
 s.update(observer_batch_in_flight={'checkpoint':cp,'scope':scope,'directory':str(B),'credit_policy':'DO_NOT_ADVANCE_WITHOUT_FINAL_EXACT_SAVED_BATCH'},status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(W/'CURRENT_STATUS.json',s);print(json.dumps(plan))
else:
 cp=int(sys.argv[2]);B=W/f'continuation{cp:04d}';plan=read(B/'BATCH_PLAN.json');prior=read(plan['prior_ledger'])
 if action=='unit':
  i=int(sys.argv[3]);D=B/f'{i:03d}';r=read(B/f'UNIT_{i}_READBACKS.json');exact(r);n=read(D/'NATIVE_RESULT.json');c=read(D/'COLD_AUDIT.json');p=read(D/'CHECKPOINT_PRESERVED.json');row=reg['specifications'][i]
  assert n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED' and c['status']=='PASS_COLD_OFFICIAL_HISTORICAL_OBSERVER_CHUNK' and c['cases_exact']==row['case_count'] and c['complete_observer_classes_exact']==row['class_count'] and c['full_public_observer_payloads_hashes_Q2_and_operational_witnesses_exact'] and c['peak_RSS_bytes']<=reg['memory_budget_bytes'] and not p['pending_objects'] and p['pending_bytes']==0
  write(D/'UNIT_SAVE_RECEIPT.json',dict(chunk_index=i,status='PASS_NATIVE_COLD_AND_EXACT_UNIT_SAVED',artifacts=r,final_batch_save_credit_pending=True,master_slices=152,new_admissions=0,G2_promotion=False));print('UNIT_SAVED',i)
 elif action=='audit':
  rows=[];ordinals=set();classes=set();members=0;peak=0
  dirs={r['chunk_index']:pathlib.Path(r['unit_directory']) for r in prior['completed_chunks']}
  for i in plan['scope']:dirs[i]=B/f'{i:03d}'
  for i in range(plan['end']):
   D=dirs[i];s=read(D/'SPEC.json');n=read(D/'NATIVE_RESULT.json');c=read(D/'COLD_AUDIT.json');p=read(D/'CHECKPOINT_PRESERVED.json');r=read(D/'UNIT_SAVE_RECEIPT.json');row=reg['specifications'][i]
   assert sha(D/'SPEC.json')==row['spec_sha256'] and n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED' and c['status']=='PASS_COLD_OFFICIAL_HISTORICAL_OBSERVER_CHUNK' and n['result']['cases_checked']==c['cases_exact']==row['case_count'] and c['complete_observer_classes_exact']==row['class_count'] and c['full_public_observer_payloads_hashes_Q2_and_operational_witnesses_exact'] and c['peak_RSS_bytes']<=reg['memory_budget_bytes'] and not p['pending_objects'] and p['pending_bytes']==0 and r['status']=='PASS_NATIVE_COLD_AND_EXACT_UNIT_SAVED';exact(r['artifacts'])
   inp=next(x for x in s['inputs'] if x['logical_name']=='cases');cases=read(inp['path']);ids={x['ordinal'] for x in cases['cases']};keys=set(cases['class_keys']);assert not ids&ordinals and not keys&classes;ordinals|=ids;classes|=keys;members+=row['case_count'];peak=max(peak,c['peak_RSS_bytes'])
   rows.append(dict(chunk_index=i,case_count=row['case_count'],class_count=row['class_count'],unit_directory=str(D),specification_sha256=sha(D/'SPEC.json'),native_result_sha256=sha(D/'NATIVE_RESULT.json'),cold_audit_sha256=sha(D/'COLD_AUDIT.json'),checkpoint_preserved_sha256=sha(D/'CHECKPOINT_PRESERVED.json'),unit_save_receipt_sha256=sha(D/'UNIT_SAVE_RECEIPT.json')))
  assert members==len(ordinals);end=plan['end'];nextscope=f'OFFICIAL_OBSERVER_CHUNKS_{end:03d}_{min(end+2,113):03d}_NATIVE_SAVE_COLD_THEN_LEDGER' if end<114 else 'OBSERVER_SELECTED_CAMPAIGN_CLOSEOUT_FULL_POPULATION_AND_CANONICALIZATION_IMPACT_OPEN'
  out=dict(checkpoint=cp,status='PASS_OFFICIAL_NATIVE_COLD_UNITS_FINAL_BATCH_SAVE_PENDING',registration_sha256=sha(Q/'PREREGISTRATION.json'),completed_chunks=rows,pending_chunk_indices=list(range(end,114)),next_chunk_index=end,official_members_compared=members,official_complete_classes_compared=len(classes),added_members=members-prior['official_members_compared'],added_classes=len(classes)-prior['official_complete_classes_compared'],remaining_members=reg['selected_members']-members,remaining_classes=reg['selected_complete_classes']-len(classes),remaining_chunks=114-end,dry_coverage_credit=0,peak_cold_RSS_bytes=peak,master_slices=152,new_admissions=0,G2_promotion=False,full_historical_public_observer_compared=False,canonicalization_impact_assessed=False,next_scope=nextscope)
  write(B/'BATCH_AUDIT.json',out);write(B/'LEDGER_CANDIDATE.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['completed_chunks','pending_chunk_indices']}))
 elif action=='package':
  a=read(B/'BATCH_AUDIT.json');date=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).date().isoformat();prefix=f'IG_MASTER152_G2_OBSERVER_CONTINUATION{cp:04d}';report=O/f'{prefix}_{date}.txt';zpath=O/f'{prefix}_HANDOFF_{date}.zip'
  report.write_text(f"Checkpoint {cp:04d} — official historical public observer\n\nChunks {plan['start']:03d}–{plan['end']-1:03d} passed native execution and independent cold verification.\nAdded {a['added_members']} records / {a['added_classes']} complete classes.\nCumulative {a['official_members_compared']} / {a['official_complete_classes_compared']}. Remaining {a['remaining_chunks']} chunks / {a['remaining_members']} records / {a['remaining_classes']} classes.\nNext scope: {a['next_scope']}.\n\nMaster 152; no admissions or G2 promotion. Full 580,351-member historical observer comparison, canonicalization impact, full L0–G8 and RC remain open. Original v0292 controller identity is not claimed. Original Q2 campaign coverage remains unchanged. Dry credit zero.\nOne scientific worker; 4 GiB memory bound; no automatic overall runtime deadline. Exact final saves required before earned ledger activation.\n")
  files={str(p.relative_to(W)):p for p in B.rglob('*') if p.is_file() and p.suffix!='.txt' and '__pycache__' not in p.parts and p.name!='ACK_BATCH.json'}
  for p in [Q/'PREREGISTRATION.json',pathlib.Path(plan['prior_ledger']),W/'continuation0286/GATE_EARNED.json']:files[str(p.relative_to(W))]=p
  for p in (Q/'project').rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(W))]=p
  for i in range(plan['end'],min(plan['end']+3,114)):
   p=Q/'specs'/f'{i:03d}.json';files[str(p.relative_to(W))]=p
   for x in read(p)['inputs']:
    if x['logical_name'] in ['cases','observer_expected']:p=pathlib.Path(x['path']);files[str(p.relative_to(W))]=p
  files[report.name]=report;manifest={n:dict(sha256=sha(p),bytes=p.stat().st_size) for n,p in files.items()}
  with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
   for n,p in files.items():z.write(p,n)
   z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
  with zipfile.ZipFile(zpath) as z:
   assert z.testzip() is None
   for n,r in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==r['sha256']
  receipt=O/f'IG_MASTER152_G2_CHECKPOINT{cp:04d}_DELIVERY_RECEIPT_{date}.json';out=dict(report=str(report),zip=str(zpath),receipt=str(receipt));write(B/'DELIVERY_PATHS.json',out);print(json.dumps(out))
 elif action=='receipt':
  out=dict(checkpoint=cp,status='REPORT_AND_HANDOFF_EXACT_SAVED_RECEIPT_SAVE_PENDING',artifacts=read(B/'DRIVE_READBACKS.json'),audit=read(B/'BATCH_AUDIT.json'),ledger_activation_policy='Only activate after receipt exact Drive readback and all three Library saves succeed',master_slices=152,new_admissions=0,G2_promotion=False);write(read(B/'DELIVERY_PATHS.json')['receipt'],out)
 elif action=='activate':
  dr=read(B/'DRIVE_READBACKS.json');rr=read(B/'DELIVERY_RECEIPT_SAVED.json');lr=read(B/'LIBRARY_SAVE_RECEIPT.json');assert len(dr)==2 and len(rr)==1 and len(lr['results'])==3;exact(dr+rr)
  for r in lr['results']:assert r['status']=='succeeded' and r['local_metadata_applied']
  out=read(B/'LEDGER_CANDIDATE.json');out.update(status='EARNED_NATIVE_COLD_AND_EXACT_SAVED',save_receipts=[dict(path=str(B/n),sha256=sha(B/n)) for n in ['DRIVE_READBACKS.json','LIBRARY_SAVE_RECEIPT.json','DELIVERY_RECEIPT_SAVED.json']]);write(B/'EARNED_LEDGER.json',out)
  s=read(W/'CURRENT_STATUS.json');assert s['latest_checkpoint']==cp-1;s.update(latest_checkpoint=cp,official_observer_completed_chunks=list(range(plan['end'])),official_observer_next_chunk=plan['end'],official_observer_pending_chunks=out['remaining_chunks'],official_observer_members_compared=out['official_members_compared'],official_observer_complete_classes_compared=out['official_complete_classes_compared'],next_scope=out['next_scope'],observer_batch_in_flight=None,pending_bytes=0,current_phase='G2_OFFICIAL_OBSERVER_SELECTED_CAMPAIGN_COMPLETE' if plan['end']==114 else 'G2_OFFICIAL_OBSERVER_NATIVE_COLD_IN_PROGRESS',status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());s[f'checkpoint{cp:04d}_earned_ledger']=str(B/'EARNED_LEDGER.json');write(W/'CURRENT_STATUS.json',s);print('CHECKPOINT_EARNED',cp)
 elif action=='cleanup':
  assert read(B/'EARNED_LEDGER.json')['status']=='EARNED_NATIVE_COLD_AND_EXACT_SAVED';exact(read(B/'ACTIVATION_SAVED.json'))
  for i in plan['scope']:
   p=pathlib.Path(f'/tmp/ig_native{cp:04d}_{i:03d}');assert read(B/f'{i:03d}'/'UNIT_SAVE_RECEIPT.json')['status']=='PASS_NATIVE_COLD_AND_EXACT_UNIT_SAVED'
   if p.exists():shutil.rmtree(p)
  print('Preserved working stores released; exact readback objects retained')
