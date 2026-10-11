"""One persistent orchestrator; native/cold/save gates; no scientific deadline."""


def _require_integrity(condition, message=None):
    """Acceptance invariant retained under optimized Python."""
    if not condition:
        if message is None:
            raise AssertionError()
        raise AssertionError(message())

import pathlib,json,hashlib,urllib.request,concurrent.futures,subprocess,sys,time,datetime,os,fcntl,shutil,traceback
from p0_control import write as durable_write, require_unoptimized, SaveJournal, recover_next_activation, ensure_bytes, stage
require_unoptimized()
W=pathlib.Path('/workspace/scratch/2a87972b51af');O=pathlib.Path('/workspace/scratch/54c3276ed25c');P='/tmp/ig_admit0188/runtime_recovery/ig_runtime_v55_fresh_20260930/python-fixed-host';F='1ALFgcEwVXK7r4yzkP1tJQsOMUPilsAzZ'
sys.path.insert(0,str(O/'helpers_continuous'));from library_hosted_apps import HostedAppsClient
C=HostedAppsClient();IDS=json.loads((O/'continuous_connectors.json').read_text());LOCK=(W/'OBSERVER_CONTROLLER.lock').open('a');fcntl.flock(LOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):durable_write(p,x)
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
 env=os.environ.copy();env.pop('PYTHONOPTIMIZE',None)
 if log:
  with pathlib.Path(log).open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env=env)
  if r.returncode:raise RuntimeError('Execution failed; inspect '+str(log))
  return ''
 r=subprocess.run(args,capture_output=True,text=True,env=env)
 if r.returncode:raise RuntimeError(r.stderr[-2000:]+r.stdout[-2000:])
 return r.stdout
def control(action,cp=None,idx=None):return run(['python3',str(O/'continuous_observer.py'),action]+([str(cp)] if cp is not None else [])+([str(idx)] if idx is not None else []))
def upload_object(path,name):
 u=call('drive','upload_file',{'file_uri':str(path),'file_name':name,'parent_folder_id':F});return u['id']
def download_object(id):
 f=call('drive','fetch',{'url':'https://drive.google.com/file/d/'+id+'/view','download_raw_file':True,'include_base64':False})
 with urllib.request.urlopen(f['file_uri']['download_url']) as response:return response.read()
def reconcile_object(name):
 quote=lambda x:x.replace('\\','\\\\').replace("'","\\'")
 query="name = '"+quote(name)+"' and '"+quote(F)+"' in parents and trashed = false"
 data=call('drive','search',{'special_filter_query_str':query,'best_effort_fetch':False,'topn':100})
 return [r['id'] for r in data.get('results',[]) if r.get('title')==name]
def saved_file(path):
 return SaveJournal(O/'SAVE_JOURNAL.json',upload_object,download_object,reconcile_object).save(path)
def save_files(paths,target):
 rows=[];old=read(target) if pathlib.Path(target).exists() else []
 for path in paths:
  path=pathlib.Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
  existing=next((r for r in old if r['path']==str(path) and r['sha256']==digest and r['drive_exact_readback_verified']),None)
  rows.append(existing or saved_file(path));write(target,rows)
def transport(D,mode,cp,idx):
 S=read(D/('CAPTURE_PENDING.json' if mode=='capture' else 'DRAIN_PLAN.json'));key='pending_objects' if mode=='capture' else 'physical_objects';M=read(D/'READBACKS.json');needed=S[key];directory=pathlib.Path(f'/tmp/ig_verified{cp}_{idx}');directory.mkdir(exist_ok=True)
 for r in needed:
  digest=r['sha256'];p=directory/(digest+'.bin')
  if digest not in M or not pathlib.Path(M[digest]['path']).is_file():
   saved=saved_file(r.get('local_object_path') or r['path']);data=download_object(saved['drive_file_id'])
   _require_integrity(len(data)==r['size_bytes'] and hashlib.sha256(data).hexdigest()==digest)
   from p0_control import atomic_bytes
   atomic_bytes(p,data);M[digest]=dict(id=saved['drive_file_id'],path=str(p),bytes=len(data),raw_readback_verified=True);write(D/'READBACKS.json',M)
  m=M[digest];p=pathlib.Path(m['path']);_require_integrity(m['raw_readback_verified'] and p.stat().st_size==r['size_bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==digest)
 if mode=='checkpoint':
  _require_integrity(bool(S['obligations'])==bool(needed))
  if needed:write(D/'ACK_BATCH.json',dict(schema_id='IG_CHECKPOINT_ACK_BATCH_V1',capture_id=S['capture_id'],obligations=S['obligations'],readbacks=[dict(sha256=r['sha256'],kind='RAW',path=M[r['sha256']]['path'],drive_file_id=M[r['sha256']]['id']) for r in needed]))
def mirror(B):
 rows=read(B/'CODE_MIRROR_PARTIAL.json')['files'] if (B/'CODE_MIRROR_PARTIAL.json').exists() else []
 for name in ['capture.py','audit_cold.py','continuous_observer.py','continuous_worker.py','p0_control.py']:
  p=B/name;content=p.read_text()
  previous=next((r for r in rows if r['name']==name),None)
  if previous:
   f=call('github','fetch_file',{'repository_full_name':'andreasebauer/igc-public-mirror','path':'operational/'+B.name+'/'+name,'ref':previous['commit_sha']});_require_integrity(f['content']==content);continue
  if (B/('MIRROR_INTENT_'+name+'.json')).exists():raise RuntimeError('MIRROR_UPLOAD_OUTCOME_UNKNOWN_RECONCILE_BEFORE_RETRY')
  write(B/('MIRROR_INTENT_'+name+'.json'),{'path':'operational/'+B.name+'/'+name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  u=call('github','create_file',{'repository_full_name':'andreasebauer/igc-public-mirror','path':'operational/'+B.name+'/'+name,'content':content,'message':'Preserve continuous observer '+B.name+' operator '+name});f=call('github','fetch_file',{'repository_full_name':'andreasebauer/igc-public-mirror','path':'operational/'+B.name+'/'+name,'ref':u['commit_sha']});_require_integrity(f['content']==content);rows.append(dict(name=name,commit_sha=u['commit_sha']));write(B/'CODE_MIRROR_PARTIAL.json',dict(files=rows))
 write(B/'CODE_MIRROR.json',dict(files=rows,commit=rows[-1]['commit_sha'],exact_readbacks_verified=True))
try:
 if not (O/'CONTROLLER_HANDOFF_ACCEPTED.json').exists():
  state('WAITING_FOR_SAVED_CONTROLLER_HANDOFF')
  while not (O/'HANDOFF_SEALED.json').exists():time.sleep(5)
  s=read(W/'CURRENT_STATUS.json');_require_integrity(not s.get('observer_batch_in_flight'))
  cp=s['latest_checkpoint'];_require_integrity(read(W/f'continuation{cp:04d}'/'EARNED_LEDGER.json')['status']=='EARNED_NATIVE_COLD_AND_EXACT_SAVED')
  _require_integrity((W/f'continuation{cp:04d}'/'ACTIVATION_SAVED.json').exists())
  write(O/'CONTROLLER_HANDOFF_ACCEPTED.json',{'checkpoint':cp,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
  (O/'HANDOFF_REQUEST.json').unlink(missing_ok=True)
 recover_next_activation(W)
 latest=read(W/'CURRENT_STATUS.json')['latest_checkpoint'];previous=W/f'continuation{latest:04d}'
 if (previous/'ACTIVATION_APPLIED.json').exists() and not (previous/'ACTIVATION_SAVED.json').exists():
  save_files([previous/'EARNED_LEDGER.json',previous/'LIBRARY_SAVE_RECEIPT.json',W/'CURRENT_STATUS.json'],previous/'ACTIVATION_SAVED.json')
 while read(W/'CURRENT_STATUS.json')['official_observer_next_chunk']<114:
  if (O/'STOP_CONTINUOUS').exists():state('STOPPED_BY_USER_AT_SAVED_BOUNDARY');break
  plan=json.loads(control('setup'));cp=plan['checkpoint'];B=W/f'continuation{cp:04d}';ensure_bytes(B/'continuous_worker.py',pathlib.Path(__file__).read_bytes());state('MIRROR_OPERATORS',checkpoint=cp)
  if not (B/'CODE_MIRROR.json').exists():mirror(B)
  for idx in plan['scope']:
   D=B/f'{idx:03d}';state('CAPTURE',checkpoint=cp,chunk=idx)
   if (D/'UNIT_SAVE_RECEIPT.json').exists():control('unit',cp,idx);continue
   binding=dict(spec_sha256=hashlib.sha256((D/'SPEC.json').read_bytes()).hexdigest(),operator_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(pathlib.Path(__file__).with_name('p0_control.py').read_bytes()).hexdigest())
   journal=D/'STAGES.json'
   if idx>plan['start'] and not (D/'STAGES.json').exists():shutil.copy2(B/f'{idx-1:03d}'/'READBACKS.json',D/'READBACKS.json')
   stage(journal,'capture',lambda:run([P,str(D/'capture.py')]),[D/'POINTER.json',D/'CAPTURE_PENDING.json'],binding);transport(D,'capture',cp,idx);run([P,str(D/'ack.py')]);state('NATIVE_EXECUTION',checkpoint=cp,chunk=idx);stage(journal,'native',lambda:run([P,str(D/'run_native.py')],D/'NATIVE_LOG.txt'),[D/'NATIVE_RESULT.json'],binding)
   native=read(D/'NATIVE_RESULT.json');_require_integrity(native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED')
   state('PRESERVE_CHECKPOINT',checkpoint=cp,chunk=idx);stage(journal,'prepare_drain',lambda:run([P,str(D/'prepare_drain.py')]),[D/'DRAIN_PLAN.json'],binding);transport(D,'checkpoint',cp,idx);run([P,str(D/'confirm_checkpoint.py')]);stage(journal,'export',lambda:run([P,str(D/'export_checkpoint.py')]),[D/'NATIVE_CHECKPOINT_SLIM.zip'],binding);state('COLD_REPLAY',checkpoint=cp,chunk=idx);stage(journal,'cold',lambda:run([P,str(D/'audit_cold.py')],D/'COLD_LOG.txt'),[D/'COLD_AUDIT.json'],binding)
   save_files([D/n for n in ['NATIVE_RESULT.json','COLD_AUDIT.json','NATIVE_CHECKPOINT_SLIM.zip']],B/f'UNIT_{idx}_READBACKS.json');control('unit',cp,idx);state('UNIT_NATIVE_COLD_EXACT_SAVED',checkpoint=cp,chunk=idx)
  state('CUMULATIVE_AUDIT_AND_DELIVERY',checkpoint=cp);control('audit',cp);paths=read(B/'DELIVERY_PATHS.json') if (B/'DELIVERY_PATHS.json').exists() else json.loads(control('package',cp));save_files([paths['report'],paths['zip']],B/'DRIVE_READBACKS.json');control('receipt',cp);save_files([paths['receipt']],B/'DELIVERY_RECEIPT_SAVED.json')
  state('SAVING_FILES_TO_LIBRARY',checkpoint=cp)
  if not (B/'LIBRARY_SAVE_RECEIPT.json').exists():
   if (B/'LIBRARY_UPLOAD_INTENT.json').exists():raise RuntimeError('LIBRARY_UPLOAD_OUTCOME_UNKNOWN_RECONCILE_BEFORE_RETRY')
   write(B/'LIBRARY_UPLOAD_INTENT.json',{'status':'INVOKED_OUTCOME_PENDING','paths':paths})
   library_env=os.environ.copy();library_env.pop('PYTHONOPTIMIZE',None)
   request=dict(uploads=[dict(local_path=paths[n],purpose='create_library_file',library_artifact_type='report' if n=='report' else 'other') for n in ['report','zip','receipt']]);r=subprocess.run(['python3',str(O/'helpers_continuous/library_upload.py')],input=json.dumps(request),capture_output=True,text=True,env=library_env);_require_integrity(r.returncode==0);receipt=json.loads(r.stdout.strip().splitlines()[-1]);_require_integrity(len(receipt['results'])==3 and all(x['status']=='succeeded' and x['local_metadata_applied'] for x in receipt['results']));write(B/'LIBRARY_SAVE_RECEIPT.json',receipt)
  control('activate',cp);save_files([B/'EARNED_LEDGER.json',B/'LIBRARY_SAVE_RECEIPT.json',W/'CURRENT_STATUS.json'],B/'ACTIVATION_SAVED.json');control('cleanup',cp);state('CHECKPOINT_EARNED_AND_SAVED',checkpoint=cp,paths=paths)
 else:state('SELECTED_OBSERVER_CAMPAIGN_COMPLETE',next_chunk=114,master_slices=152,G2_promotion=False)
except BaseException as e:
 write(O/'CONTINUOUS_PROGRESS.json',dict(status='BLOCKED',phase='STOPPED_ON_FAILED_CHECK_OR_DEPENDENCY',error=str(e),time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid()));traceback.print_exc();raise
