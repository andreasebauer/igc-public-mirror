import pathlib,json,hashlib,sys
B=pathlib.Path(__file__).parent;i=int(sys.argv[1]);D=B/f'{i:03d}';R=json.loads((B/f'UNIT_{i}_READBACKS.json').read_text())
for r in R:assert r['drive_exact_readback_verified'] and hashlib.sha256(pathlib.Path(r['path']).read_bytes()).hexdigest()==r['sha256']
n=json.loads((D/'NATIVE_RESULT.json').read_text());c=json.loads((D/'COLD_AUDIT.json').read_text());p=json.loads((D/'CHECKPOINT_PRESERVED.json').read_text());reg=json.loads((B.parent/'continuation0285/PREREGISTRATION.json').read_text());row=reg['specifications'][i]
assert n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED' and c['cases_exact']==row['case_count'] and c['complete_observer_classes_exact']==row['class_count'] and c['full_public_observer_payloads_hashes_Q2_and_operational_witnesses_exact'] and not p['pending_objects'] and p['pending_bytes']==0
out={'chunk_index':i,'status':'PASS_NATIVE_COLD_AND_EXACT_UNIT_SAVED','artifacts':R,'final_batch_save_credit_pending':True,'master_slices':152,'new_admissions':0,'G2_promotion':False};(D/'UNIT_SAVE_RECEIPT.json').write_text(json.dumps(out,indent=2)+'\n');print('Unit',i,'exact native/cold/checkpoint saves verified')
