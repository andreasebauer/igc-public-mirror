"""Package cold-audited historical27–32 native continuation."""
from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent;a=json.loads((B/'AUDIT.json').read_text());assert a['status']=='PASS_COLD_HISTORICAL_DEPTH26_CONTINUATION';assert json.loads((B/'CHECKPOINT_PRESERVED.json').read_text())['pending_bytes']==0
(B/'REPORT.txt').write_text('''CHECKPOINT0215 — HISTORICAL DEPTHS21–26 VERIFIED
2026-10-08

Fresh native capture continued from the independently audited exact historical depth20bootstrap. The isolated historical scientific helper closure is byte-identical to the qualified0213closure; current Decoder engine and global O7 identity are unchanged. No wrong-namespace earlier8–100states were reused.

Six native phases21–26 each committed193candidate builds and selected24roots, total1158candidate builds. Frozen recipe lists and31bridgepairs were checked. All24depth26construction digests, resource skins and seven-component capacity vectors exactly matched the hash-bound archived historical SOURCE_INPUT probes.

An independent cold native checkpoint restore verified every dependency SHA/length, all six exact canonical native identities and one committed task per phase. It reconstructed144roots across21–26, passed exact original DAG roundtrips, and repeated the historical depth26root/skin/capacity equality check. The audit generated no candidates and read no original-workspace scientific state. BOOTSTRAP26_HISTORICAL.json is compact exact scientific continuation state. Native preservation obligations are drained; raw Drive readbacks are verified.

Master151 remains unchanged, zero admissions, Q2 not generated, G2 not graduated, full L0–G8 incomplete. Historical reconstruction is now qualified through26; terminal100comparison remains pending. The original terminal pretty-JSON1GiB checkpoint failure is a separate future operational constraint.

NEXT: Fresh bounded historical27–32native capture from this exact bootstrap, same attested historical helpers and frozen selection rules. Qualify against archived depth32probes if available; otherwise bind the next available archived gate without inventing reference values. Never edit a frozen capture or regenerate committed phases. Complete save and cold DAG audit for every tranche before continuation.

RESTORE: Verify MANIFEST.json then CHECKPOINT_EXPORT dependencies; use restore_current.py with pinned python-fixed-host (CPython3.13.5/SQLite3.51.3/Decoder0.8.0.dev151+lib). READBACKS locates all saved raw dependency objects. NATIVE_CHECKPOINT_SLIM.zip restores actual0215native scope. Included0214handoff retains diagnosis, seed/depth14qualification and verified continuation through20. require_saved plus native environment admission precede any generation.193recipes/31pairs,24beam,193terminalroots,64MiB result transport,900seconds per task,2GiB workspace,1GiB state file. Frozen engine SHA60c729a8ee3ec81b0b367b44719e238271ad7d0ae9112cebe81253cf3f874c30.
''')
s=json.loads((W/'CURRENT_STATUS.json').read_text());s['checkpoint0214']=s['latest_audit'];s.pop('latest_handoff',None);s.update({'status_as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'latest_checkpoint':215,'current_phase':'G1_HISTORICAL_DEPTH26_VERIFIED','in_flight_active':'NONE','latest_capture_id':json.loads((B/'POINTER.json').read_text())['capture_id'],'active_native_runtime':json.loads((B/'POINTER.json').read_text())['workspace'],'completed_depth':26,'historical_verified_depth':26,'historical_reconstruction_acceptance':'THROUGH26_QUALIFIED_TERMINAL_PENDING','candidate_build_calls':1158,'generator_calls':6,'generator_calls_scope':'0215:six193candidate native generation phases21–26; cold audit generated none','pending_bytes':0,'master_slices':151,'new_admissions':0,'terminal_comparison':'NOT_RUN_AFTER_HISTORICAL_REPAIR','Q2_payload_available':False,'next_scope':'G1_HISTORICAL_BOUNDED27_32','next_scope_status':'FRESH_CAPTURE_REQUIRED_FROM_AUDITED_HISTORICAL_BOOTSTRAP26','latest_audit':a,'checkpoint0215':a,'code_mirror':json.loads((B/'CODE_MIRROR.json').read_text())});(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2));(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2))
files={str(p.relative_to(W)):p for p in sorted(B.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.txt','.zip','.log') and '__pycache__' not in p.parts and not p.name.startswith('private_') and p.name not in ('MIRROR_ELEMENTS.json','DELIVERABLES.json','SAVE_RECEIPT.json')}
boot=Path(json.loads((B/'BOOTSTRAP26_META.json').read_text())['path']);files['continuation0215/BOOTSTRAP26_HISTORICAL.json']=boot
pred=W/'IG_MASTER151_G1_HISTORICAL_CONTINUATION_0214_HANDOFF_2026-10-08.zip';files['predecessor/'+pred.name]=pred
manifest={k:{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for k,p in files.items()};out=W/'IG_MASTER151_G1_HISTORICAL_CONTINUATION_0215_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
r={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':215,'status':a['status'],'historical_verified_depth':26,'candidate_build_calls':1158,'master_slices':151,'new_admissions':0,'native_pending_bytes':0,'next_scope':s['next_scope']};(B/'DELIVERABLES.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
