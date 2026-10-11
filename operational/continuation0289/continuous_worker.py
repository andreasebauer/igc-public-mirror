"""One persistent orchestrator; native/cold/save gates; no scientific deadline."""
import pathlib,json,hashlib,urllib.request,concurrent.futures,subprocess,sys,time,datetime,os,fcntl,shutil,traceback
W=pathlib.Path('/workspace/scratch/2a87972b51af');O=pathlib.Path('/workspace/scratch/54c3276ed25c');P='/tmp/ig_admit0188/runtime_recovery/ig_runtime_v55_fresh_20260930/python-fixed-host';F='1ALFgcEwVXK7r4yzkP1tJQsOMUPilsAzZ'
sys.path.insert(0,str(O/'helpers_continuous'));from library_hosted_apps import HostedAppsClient
C=HostedAppsClient();IDS=json.loads((O/'continuous_connectors.json').read_text());LOCK=(O/'continuous_worker.lock').open('w');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2)+'\n')
def state(phase,**kwargs):
 s=dict(status='RUNNING',pid=os.getpid(),phase=phase,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs);write(O/'CONTINUOUS_PROGRESS.json',s);print(json.dumps(s),flush=True)
def call(provider,name,args):
 r=C.call_tool(IDS[provider],name,args)
 if r.get('isError'):raise RuntimeError(name+' failed')
 if r.get('structuredContent'):return r['structuredContent']
 for b in r.get('content',[]):
  if b.get('type')=='text':
   try:return json.loads(b['text'])
   except ValueError:pass
 raise RuntimeError('No structured response for '+name)
def run(args,log=None):
 if log:
  with pathlib.Path(log).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError('Execution failed; inspect '+str(log))
  return ''
 r=subprocess.run(args,capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr[-2000:]+r.stdout[-2000:])
 return r.stdout
def control(action,cp=None,idx=None):return run(['python3',str(O/'continuous_observer.py'),action]+([str(cp)] if cp is not None else [])+([str(idx)] if idx is not None else []))
def upload(path):
 u=call('drive','upload_file',{'file_uri':str(path),'parent_folder_id':F});f=call('drive','fetch',{'url':u['url'],'download_raw_file':True,'include_base64':False});return u['id'],f['file_uri']['download_url']
def saved_file(path):
 path=pathlib.Path(path);id,url=upload(path);q=path.parent/(path.name+'.raw_readback');urllib.request.urlretrieve(url,q);assert q.read_bytes()==path.read_bytes();q.unlink();return dict(path=str(path),drive_file_id=id,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,drive_exact_readback_verified=True)
def save_files(paths,target):
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(saved_file,paths))
 write(target,rows)
def transport(D,mode,cp,idx):
 S=read(D/('CAPTURE_PENDING.json' if mode=='capture' else 'DRAIN_PLAN.json'));key='pending_objects' if mode=='capture' else 'physical_objects';M=read(D/'READBACKS.json');needed=S[key];directory=pathlib.Path(f'/tmp/ig_verified{cp}_{idx}');directory.mkdir(exist_ok=True)
 def get(r):
  id,url=upload(r.get('local_object_path') or r['path']);p=directory/(r['sha256']+'.bin');urllib.request.urlretrieve(url,p);assert p.stat().st_size==r['size_bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];return r['sha256'],dict(id=id,path=str(p),bytes=p.stat().st_size,raw_readback_verified=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for k,v in pool.map(get,[r for r in needed if r['sha256'] not in M]):M[k]=v
 for r in needed:
  m=M[r['sha256']];p=pathlib.Path(m['path']);assert m['raw_readback_verified'] and p.stat().st_size==r['size_bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 write(D/'READBACKS.json',M)
 if mode=='checkpoint':write(D/'ACK_BATCH.json',dict(schema_id='IG_CHECKPOINT_ACK_BATCH_V1',capture_id=S['capture_id'],obligations=S['obligations'],readbacks=[dict(sha256=r['sha256'],kind='RAW',path=M[r['sha256']]['path'],drive_file_id=M[r['sha256']]['id']) for r in needed]))
def mirror(B):
 rows=[]
 for name in ['capture.py','audit_cold.py','continuous_observer.py','continuous_worker.py']:
  p=B/name;content=p.read_text();u=call('github','create_file',{'repository_full_name':'andreasebauer/igc-public-mirror','path':'operational/'+B.name+'/'+name,'content':content,'message':'Preserve continuous observer '+B.name+' operator '+name});f=call('github','fetch_file',{'repository_full_name':'andreasebauer/igc-public-mirror','path':'operational/'+B.name+'/'+name,'ref':u['commit_sha']});assert f['content']==content;rows.append(dict(name=name,commit_sha=u['commit_sha']))
 write(B/'CODE_MIRROR.json',dict(files=rows,commit=rows[-1]['commit_sha'],exact_readbacks_verified=True))
try:
 state('WAITING_FOR_SAVED_CONTROLLER_HANDOFF')
 while not (O/'HANDOFF_SEALED.json').exists():time.sleep(5)
 s=read(W/'CURRENT_STATUS.json');assert s['latest_checkpoint']==289 and s['observer_batch_in_flight'] is None
 assert read(W/'continuation0289/EARNED_LEDGER.json')['status']=='EARNED_NATIVE_COLD_AND_EXACT_SAVED'
 assert (W/'continuation0289/ACTIVATION_SAVED.json').exists()
 (O/'HANDOFF_REQUEST.json').unlink()
 while read(W/'CURRENT_STATUS.json')['official_observer_next_chunk']<114:
  if (O/'STOP_CONTINUOUS').exists():state('STOPPED_BY_USER_AT_SAVED_BOUNDARY');break
  plan=json.loads(control('setup'));cp=plan['checkpoint'];B=W/f'continuation{cp:04d}';shutil.copy2(__file__,B/'continuous_worker.py');state('MIRROR_OPERATORS',checkpoint=cp);mirror(B)
  for idx in plan['scope']:
   D=B/f'{idx:03d}';state('CAPTURE',checkpoint=cp,chunk=idx)
   if idx>plan['start']:shutil.copy2(B/f'{idx-1:03d}'/'READBACKS.json',D/'READBACKS.json')
   run([P,str(D/'capture.py')]);transport(D,'capture',cp,idx);run([P,str(D/'ack.py')]);state('NATIVE_EXECUTION',checkpoint=cp,chunk=idx);run([P,str(D/'run_native.py')],D/'NATIVE_LOG.txt')
   native=read(D/'NATIVE_RESULT.json');assert native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED'
   state('PRESERVE_CHECKPOINT',checkpoint=cp,chunk=idx);run([P,str(D/'prepare_drain.py')]);transport(D,'checkpoint',cp,idx);run([P,str(D/'confirm_checkpoint.py')]);run([P,str(D/'export_checkpoint.py')]);state('COLD_REPLAY',checkpoint=cp,chunk=idx);run([P,str(D/'audit_cold.py')],D/'COLD_LOG.txt')
   save_files([D/n for n in ['NATIVE_RESULT.json','COLD_AUDIT.json','NATIVE_CHECKPOINT_SLIM.zip']],B/f'UNIT_{idx}_READBACKS.json');control('unit',cp,idx);state('UNIT_NATIVE_COLD_EXACT_SAVED',checkpoint=cp,chunk=idx)
  state('CUMULATIVE_AUDIT_AND_DELIVERY',checkpoint=cp);control('audit',cp);paths=json.loads(control('package',cp));save_files([paths['report'],paths['zip']],B/'DRIVE_READBACKS.json');control('receipt',cp);save_files([paths['receipt']],B/'DELIVERY_RECEIPT_SAVED.json')
  state('SAVING_FILES_TO_LIBRARY',checkpoint=cp);request=dict(uploads=[dict(local_path=paths[n],purpose='create_library_file',library_artifact_type='report' if n=='report' else 'other') for n in ['report','zip','receipt']]);r=subprocess.run(['python3',str(O/'helpers_continuous/library_upload.py')],input=json.dumps(request),capture_output=True,text=True);assert r.returncode==0;receipt=json.loads(r.stdout.strip().splitlines()[-1]);assert len(receipt['results'])==3 and all(x['status']=='succeeded' and x['local_metadata_applied'] for x in receipt['results']);write(B/'LIBRARY_SAVE_RECEIPT.json',receipt)
  control('activate',cp);save_files([B/'EARNED_LEDGER.json',B/'LIBRARY_SAVE_RECEIPT.json',W/'CURRENT_STATUS.json'],B/'ACTIVATION_SAVED.json');control('cleanup',cp);state('CHECKPOINT_EARNED_AND_SAVED',checkpoint=cp,paths=paths)
 else:state('SELECTED_OBSERVER_CAMPAIGN_COMPLETE',next_chunk=114,master_slices=152,G2_promotion=False)
except BaseException as e:
 write(O/'CONTINUOUS_PROGRESS.json',dict(status='BLOCKED',phase='STOPPED_ON_FAILED_CHECK_OR_DEPENDENCY',error=str(e),time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid()));traceback.print_exc();raise
