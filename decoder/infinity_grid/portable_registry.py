"""Portable project history for registered Decoder work.

Local writers use file locks. Independent chat copies exchange immutable
histories and report divergent heads. This is not a distributed lock service.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import uuid
import zipfile

from .canon import canonical_sha256, write_json_atomic
from . import submission as sub
from .invocation import InvocationRefused


def refuse(code, root=None, operation='project status'):
    path=Path(root) if root else None
    store=path.parent if path is not None and path.name=='coordination' else path
    args={'store':str(store) if store else None}
    if operation=='capture': args['specification']=None
    elif operation=='project adopt': args={'store':None,'workspace':str(path) if path else None}
    elif operation=='project import': args.update(archive=None,sha256=None)
    elif operation=='project resolve':
        args.update(key=code.partition(':')[2] or None,selected_head=None,reason=None)
    raise InvocationRefused(code, 'project', next_operation=operation,
                            required_arguments=args)


def read(path):
    return sub._read(path)


def put(root, raw):
    sha = sub._sha(raw); path = root/'objects'/sha
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw: refuse('PROJECT_OBJECT_MISMATCH', root)
    else:
        # Atomic immutable creation; readers never see partial content.
        fd, temporary = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            try: os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != raw: refuse('PROJECT_OBJECT_MISMATCH', root)
        finally: Path(temporary).unlink(missing_ok=True)
    return sha


def blob(root, sha):
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        refuse('PROJECT_OBJECT_IDENTIFIER', root)
    raw = (root/'objects'/sha).read_bytes()
    if sub._sha(raw) != sha: refuse('PROJECT_OBJECT_MISMATCH', root)
    return raw


@contextmanager
def lock(root):
    from .v05_controller_event_loop import _workspace_lock
    with _workspace_lock(root): yield


def events(root):
    meta = read(root/'PROJECT.json')
    if set(meta) != {'schema_id','project_id','name'} or meta['schema_id'] != 'IG_PORTABLE_PROJECT_V1':
        refuse('PROJECT_MANIFEST', root)
    rows = {}
    for p in sorted((root/'events').glob('*.json')):
        row = read(p)
        if set(row) != {'project_id','key','parents','value','reason','nonce'} or row['project_id'] != meta['project_id'] or canonical_sha256(row) != p.stem:
            refuse('PROJECT_HISTORY_MISMATCH', root)
        if not isinstance(row['key'], str) or not isinstance(row['parents'], list) or len(row['parents']) != len(set(row['parents'])):
            refuse('PROJECT_EVENT_FIELDS', root)
        rows[p.stem] = row
    for sha, row in rows.items():
        for parent in row['parents']:
            if parent not in rows or rows[parent]['key'] != row['key'] or parent == sha:
                refuse('PROJECT_HISTORY_PARENT_MISSING', root)
    # Detect cycles even in imported content with fabricated identifiers.
    visiting, visited = set(), set()
    def visit(sha):
        if sha in visiting: refuse('PROJECT_HISTORY_CYCLE', root)
        if sha in visited: return
        visiting.add(sha)
        for parent in rows[sha]['parents']: visit(parent)
        visiting.remove(sha); visited.add(sha)
    for sha in rows: visit(sha)
    return rows


def heads(rows, key):
    selected = {sha: r for sha,r in rows.items() if r['key'] == key}
    parents = {p for r in selected.values() for p in r['parents']}
    return sorted(set(selected)-parents)


def current(root, key, rows=None):
    rows = events(root) if rows is None else rows
    hs = heads(rows,key)
    if len(hs)>1: refuse('PROJECT_HISTORY_CONFLICT:'+key,root,'project resolve')
    return (hs[0], rows[hs[0]]['value']) if hs else (None,None)


def append(root, key, expected, value, reason):
    # Caller holds the project lock; all comparisons use the full observed head.
    rows=events(root)
    if heads(rows,key) != sorted(expected): refuse('PROJECT_STALE_PARENT:'+key,root)
    row={'project_id':read(root/'PROJECT.json')['project_id'], 'key':key,
         'parents':sorted(expected),'value':value,'reason':reason,'nonce':uuid.uuid4().hex}
    sha=canonical_sha256(row)
    write_json_atomic(root/'events'/(sha+'.json'),row)
    return sha


def engine_object(source):
    files={n:b for n,b in sub._tree(source).items() if not n.startswith('project/')}
    return sub._archive(files)


def initialize(store, name, source):
    root=Path(store).resolve()/'coordination'
    with lock(root):
        if (root/'PROJECT.json').exists(): refuse('PROJECT_ALREADY_EXISTS',root)
        raw=engine_object(source)
        write_json_atomic(root/'PROJECT.json',{'schema_id':'IG_PORTABLE_PROJECT_V1','project_id':uuid.uuid4().hex,'name':name})
        sha=put(root,raw)
        from .change_sessions import source_version
        append(root,'release',[],{'engine':sha,'version':source_version(Path(source)),
               'basis':'EXPLICIT_PROJECT_INITIALIZATION','validation':None},'Initial project source')
    return status(root)


def status(root):
    root=Path(root); rows=events(root)
    keys=sorted({r['key'] for r in rows.values()})
    conflicts={k:heads(rows,k) for k in keys if len(heads(rows,k))>1}
    return {'status':'CONFLICTS_REQUIRE_RECONCILIATION' if conflicts else 'READY',
            'project':read(root/'PROJECT.json'),'heads':{k:heads(rows,k) for k in keys},
            'conflicts':conflicts,'scope':'PORTABLE_HISTORY_LOCAL_LOCKS_NO_DISTRIBUTED_LOCK'}


def snapshot(root):
    with lock(root):
        events(root)
        files={'PROJECT.json':(root/'PROJECT.json').read_bytes()}
        for folder in ('events','objects'):
            for p in sorted((root/folder).glob('*')):
                if p.is_symlink() or not p.is_file(): refuse('PROJECT_FILE_TYPE',root)
                if folder=='objects': blob(root,p.name)
                files[p.relative_to(root).as_posix()]=p.read_bytes()
        return sub._archive(files)


def import_history(store, archive, expected_sha):
    raw=Path(archive).read_bytes()
    if sub._sha(raw)!=expected_sha: refuse('PROJECT_ARCHIVE_HASH')
    root=Path(store).resolve()/'coordination'
    with tempfile.TemporaryDirectory(prefix='ig-project-import-') as d:
        temp=Path(d)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if len(z.namelist())!=len(set(z.namelist())): refuse('PROJECT_DUPLICATE_PATH')
            for name in z.namelist():
                rel=sub._relative(name)
                if name!='PROJECT.json' and (len(rel.parts)!=2 or rel.parts[0] not in ('objects','events')):
                    refuse('PROJECT_ARCHIVE_MEMBER')
                target=temp/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
        events(temp)
        for p in (temp/'objects').glob('*'): blob(temp,p.name)
        with lock(root):
            if (root/'PROJECT.json').exists():
                if read(root/'PROJECT.json')!=read(temp/'PROJECT.json'): refuse('PROJECT_ID_MISMATCH',root)
            else: shutil.copy2(temp/'PROJECT.json',root/'PROJECT.json')
            for folder in ('objects','events'):
                for p in (temp/folder).glob('*'):
                    dest=root/folder/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists() and dest.read_bytes()!=p.read_bytes(): refuse('PROJECT_HISTORY_MISMATCH',root)
                    if not dest.exists(): shutil.copy2(p,dest)
            # All histories remain; conflicts do not choose a winner.
            result=status(root)
    return result


def resolve(root, key, selected, reason):
    if not reason.strip(): refuse('PROJECT_RECONCILIATION_REASON_REQUIRED',root)
    with lock(root):
        rows=events(root);hs=heads(rows,key)
        if len(hs)<2 or selected not in hs: refuse('PROJECT_CONFLICT_SELECTION_REQUIRED',root)
        value=rows[selected]['value']
        if key=='release':
            blob(root,value['engine'])
            if value.get('validation'): verify_capsule(root,value['validation'])
        if key.startswith('work:') and value.get('capsule'): verify_capsule(root,value['capsule'],value['identity'])
        sha=append(root,key,hs,value,'Reconcile: '+reason)
    return {'status':'RECONCILED','key':key,'head':sha,'preserved_heads':hs}


def actual_environment():
    return {'python':sys.version,'implementation':platform.python_implementation(),
            'system':platform.system(),'machine':platform.machine(),'release':platform.release(),
            'libc':list(platform.libc_ver())}


def capture_context(root, engine):
    rows=events(root)
    return {'project_id':read(root/'PROJECT.json')['project_id'],
            'release_head':current(root,'release',rows)[0],
            'provider_heads':{key:current(root,key,rows)[0] for key in sorted({r['key'] for r in rows.values()})
                              if key.startswith('provider:'+engine+':')}}


def identity(root, workspace, repeat, *, pinned_providers=None, pinned_actual_environment=None):
    rec=sub.capture_record(workspace)
    objects={o['role']:o['sha256'] for o in rec['objects']}
    engine=objects['engine_source'];rows=events(root)
    providers={}
    for key in sorted({r['key'] for r in rows.values()}):
        if pinned_providers is None and key.startswith('provider:'+engine+':'): providers[key]=current(root,key,rows)[1]
    if pinned_providers is not None: providers=pinned_providers
    job=rec['job']
    from .semantic_inputs import classify_capture_inputs
    classified=classify_capture_inputs(rec)
    question={k:v for k,v in job['question'].items() if k!='description'}
    value={'engine':engine,'implementation':objects.get('project_source',engine),
           'inputs':{o['logical_name']:o['sha256'] for o in classified['project_inputs']},
           'execution':job['execution'],'question':question,'output_contract':rec['output_contract'],
           'environment':rec['environment'],
           'actual_environment':actual_environment() if pinned_actual_environment is None else pinned_actual_environment,
           'resources':job['resources'],'providers':providers,'repeat_id':repeat}
    return value


def locate(workspace):
    root=Path(workspace).resolve()
    binding=read(root/'PROJECT_BINDING.json') if (root/'PROJECT_BINDING.json').exists() else None
    if binding is None: refuse('PROJECT_BINDING_REQUIRED',root,'project adopt')
    candidates=[root/'coordination',root.parent.parent/'coordination']
    hint=root/'PROJECT_LOCATION.json'
    if hint.exists(): candidates.append(Path(read(hint)['path']))
    for path in candidates:
        if (path/'PROJECT.json').is_file() and read(path/'PROJECT.json')['project_id']==binding['project_id']:
            return path,binding
    refuse('PROJECT_RESTORE_REQUIRED',root,'project import')


def adopt(store, workspace, repeat=None):
    root=Path(store).resolve()/'coordination';workspace=Path(workspace).resolve()
    if repeat is not None and (not isinstance(repeat,str) or not repeat.strip()): refuse('PROJECT_REPEAT_ID_REQUIRED',root)
    rec=sub.capture_record(workspace)
    if (workspace/'PROJECT_BINDING.json').exists():
        prior=read(workspace/'PROJECT_BINDING.json')
        if prior['project_id']!=read(root/'PROJECT.json')['project_id'] or prior['repeat_id']!=repeat:
            refuse('PROJECT_BINDING_IMMUTABLE',root)
        return sub.save_status(workspace)
    with lock(root):
        release_head,release=current(root,'release')
        value=identity(root,workspace,repeat)
        if rec.get('project_context',capture_context(root,value['engine']))!=capture_context(root,value['engine']):
            refuse('PROJECT_CONTEXT_CHANGED_RECAPTURE',root,'capture')
        if not release or release['engine']!=value['engine']: refuse('PROJECT_ENGINE_NOT_CURRENT',root)
    # The exact consulted history is saved as part of admission provenance.
    baseline=snapshot(root)
    object_store=workspace.parent.parent
    obj=sub._put_object(object_store,baseline,'.zip','project_baseline')
    binding={'schema_id':'IG_PROJECT_BINDING_V1','project_id':read(root/'PROJECT.json')['project_id'],
             'capture_id':rec['capture_id'],'release_head':release_head,'repeat_id':repeat,
             'identity':value,'work_id':canonical_sha256(value),'baseline_object':obj}
    binding['binding_sha256']=canonical_sha256(binding)
    write_json_atomic(workspace/'PROJECT_BINDING.json',binding)
    write_json_atomic(workspace/'PROJECT_LOCATION.json',{'path':str(root)})
    sub._put_object(object_store,sub._json_bytes(binding),'.json','project_binding')
    return sub.save_status(workspace)


def required_objects(workspace):
    p=Path(workspace)/'PROJECT_BINDING.json'
    if not p.exists(): return []
    row=read(p)
    if row.get('binding_sha256')!=canonical_sha256({k:v for k,v in row.items() if k!='binding_sha256'}):
        refuse('PROJECT_BINDING_MISMATCH',workspace)
    raw=p.read_bytes()
    return [row['baseline_object'],{'role':'project_binding','sha256':sub._sha(raw),
             'size_bytes':len(raw),'object_name':sub._sha(raw)+'.json'}]


def completion_capsule(root, workspace):
    """Preserve original completion and all bytes needed to reconstruct its proof."""
    from . import v05_controller_event_loop as loop
    workspace=Path(workspace);rec=sub.capture_record(workspace)
    admission=loop.validate_workspace_job(workspace,rec['job']['job_id'],check_loaded=False)
    done=loop.verified_completion(admission)
    if done is None: refuse('PROJECT_COMPLETION_REQUIRED',root)
    # One object per byte-content permits deduplication across source versions.
    files={}
    for path,rel in loop._snapshot_files(workspace):
        if rel.startswith('coordination/') or rel in {'PROJECT_LOCATION.json'}: continue
        files[rel]=put(root,path.read_bytes())
    capsule={'schema_id':'IG_PROJECT_COMPLETION_CAPSULE_V1','files':files,
             'completion_sha256':done['completion_sha256'],'job_id':rec['job']['job_id']}
    return put(root,sub._json_bytes(capsule))


def verify_capsule(root, sha, expected_identity=None, *, details=False):
    from . import v05_controller_event_loop as loop
    packet=json.loads(blob(root,sha))
    if packet.get('schema_id')!='IG_PROJECT_COMPLETION_CAPSULE_V1': refuse('PROJECT_CAPSULE_SCHEMA',root)
    with tempfile.TemporaryDirectory(prefix='ig-evidence-') as d:
        ws=Path(d)
        for name,digest in packet['files'].items():
            target=ws/sub._relative(name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob(root,digest))
        admission=loop.validate_workspace_job(ws,packet['job_id'],check_loaded=False)
        done=loop.verified_completion(admission)
        if done is None or done['completion_sha256']!=packet['completion_sha256']: refuse('PROJECT_COMPLETION_MISMATCH',root)
        if sub.save_status(ws)['status']!='SAVED': refuse('PROJECT_ORIGINAL_SAVE_EVIDENCE_REQUIRED',root)
        if expected_identity is not None:
            binding=read(ws/'PROJECT_BINDING.json')
            if binding['identity']!=expected_identity: refuse('PROJECT_REUSE_IDENTITY_MISMATCH',root)
        return {'completion':done,'capture':sub.capture_record(ws)} if details else done


class RunClaim:
    def __init__(self, admission, *, reuse_verification=False):
        self.admission=admission;self.workspace=admission['workspace']
        self.root,self.binding=locate(self.workspace);required_objects(self.workspace)
        b=self.binding;rec=sub.capture_record(self.workspace)
        if b['capture_id']!=rec['capture_id'] or b['work_id']!=canonical_sha256(b['identity']): refuse('PROJECT_BINDING_MISMATCH',self.root)
        self.key='work:'+b['work_id']
        _,prior=current(self.root,self.key)
        observed=identity(self.root,self.workspace,b['repeat_id'],
                          pinned_providers=b['identity']['providers'] if prior else None,
                          pinned_actual_environment=b['identity']['actual_environment'] if reuse_verification else None)
        if observed!=b['identity']: refuse('PROJECT_EXECUTION_IDENTITY_CHANGED',self.root,'capture')
        self.reused=None;self.head=None

    def _verified_reuse(self, value):
        if not value or value['state']!='COMPLETED': return None
        done=verify_capsule(self.root,value['capsule'],self.binding['identity'])
        return dict(done,reused=True,reuse_binding={'work_id':self.binding['work_id'],
            'requested_registration':self.admission['job']['registration_sha256'],
            'original_completion':done['completion_sha256']})

    def completed_reuse(self):
        """Return exact completed work without creating a run claim.

        This is the read-only compatibility path used before execution-only
        source and environment admission.  It still verifies the complete
        capsule, capture identity, producer evidence, and save receipts.
        """
        from .v05_controller_event_loop import _workspace_lock
        with _workspace_lock(self.root/'claims'/self.binding['work_id']):
            with lock(self.root):
                _,value=current(self.root,self.key)
                return self._verified_reuse(value)

    def __enter__(self):
        from .v05_controller_event_loop import _workspace_lock
        self.worklock=_workspace_lock(self.root/'claims'/self.binding['work_id'])
        self.worklock.__enter__()
        try:
            with lock(self.root):
                head,value=current(self.root,self.key)
                if value and value['state']=='COMPLETED':
                    self.reused=self._verified_reuse(value)
                    return self
                if value and value['capture_id']!=self.binding['capture_id']:
                    refuse('PROJECT_WORK_ALREADY_CLAIMED',self.root)
                if not value:
                    release_head,release=current(self.root,'release')
                    if release_head!=self.binding['release_head']: refuse('PROJECT_RELEASE_CHANGED_RECAPTURE',self.root,'capture')
                self.head=append(self.root,self.key,[head] if head else [],
                    {'state':'RUNNING','capture_id':self.binding['capture_id'],'identity':self.binding['identity']},
                    'Registered controller run or exact checkpoint resume')
            return self
        except BaseException:
            self.worklock.__exit__(*sys.exc_info());raise

    def complete(self, done):
        # Completion is verified and frozen from original files, never a status flag.
        with lock(self.root):
            capsule=completion_capsule(self.root,self.workspace)
            self.head=append(self.root,self.key,[self.head],{'state':'COMPLETED',
                'capture_id':self.binding['capture_id'],'identity':self.binding['identity'],
                'capsule':capsule,'completion_sha256':done['completion_sha256']},'Verified original completion')

    def __exit__(self, typ, value, tb):
        try:
            if typ is not None and self.head:
                with lock(self.root):
                    self.head=append(self.root,self.key,[self.head],{'state':'PAUSED',
                        'capture_id':self.binding['capture_id'],'identity':self.binding['identity'],
                        'reason':str(value)},'Interrupted or refused attempt; original claim retained')
        finally: self.worklock.__exit__(typ,value,tb)


def promote(root, expected_head, source, validation_workspace, reason):
    """Called by recorded change activation after its native verification/save gate."""
    from .v05_origin_guard import require_native_caller
    require_native_caller('infinity_grid.change_sessions', {'activate'}, 'project-release-activation')
    with lock(root):
        head,old=current(root,'release')
        if head!=expected_head: refuse('PROJECT_STALE_RELEASE_PARENT',root)
        capsule=completion_capsule(root,validation_workspace)
        done=verify_capsule(root,capsule)
        if done['status']!='COMPLETED': refuse('PROJECT_PASS_REQUIRED',root)
        from .change_sessions import source_version
        sha=put(root,engine_object(source))
        new=append(root,'release',[head],{'engine':sha,'version':source_version(Path(source)),
            'basis':'RECORDED_CHANGE','validation':capsule,'previous':head},reason)
        return new


def rollback_release(root, expected_engine, previous_engine, reason):
    from .v05_origin_guard import require_native_caller
    require_native_caller('infinity_grid.change_sessions', {'rollback'}, 'project-release-rollback')
    with lock(root):
        rows=events(root);head,value=current(root,'release',rows)
        if value['engine']!=expected_engine: refuse('PROJECT_ROLLBACK_CURRENT_MOVED',root)
        candidates=[r['value'] for r in rows.values() if r['key']=='release' and r['value']['engine']==previous_engine]
        if not candidates: refuse('PROJECT_ROLLBACK_HISTORY_REQUIRED',root)
        prior=candidates[-1];blob(root,previous_engine)
        if prior.get('validation'): verify_capsule(root,prior['validation'])
        return append(root,'release',[head],prior,'Rollback: '+reason)


def provider(root, request):
    fields={'engine','role','path','expected_head','correctness_capsule','performance_capsule','reason'}
    if set(request)!=fields or not request['role'].strip() or not request['reason'].strip(): refuse('PROJECT_PROVIDER_REQUEST',root)
    with lock(root):
        _,release=current(root,'release')
        if request['engine']!=release['engine']: refuse('PROJECT_PROVIDER_ENGINE',root)
        key='provider:'+request['engine']+':'+request['role'];head,_=current(root,key)
        if head!=request['expected_head']: refuse('PROJECT_PROVIDER_CONFLICT',root)
        with zipfile.ZipFile(io.BytesIO(blob(root,request['engine']))) as z:
            raw=z.read(sub._relative(request['path']).as_posix())
        provider_sha=sub._sha(raw)
        for kind in ('correctness','performance'):
            proof=verify_capsule(root,request[kind+'_capsule'],details=True)
            done,rec=proof['completion'],proof['capture']
            expected={'kind':kind,'engine':request['engine'],'role':request['role'],
                      'path':request['path'],'provider_sha256':provider_sha}
            if (done['status']!='COMPLETED' or rec['job']['execution']['kind']!='VALIDATION'
                or done['result'].get('status')!='PASS'
                or rec['output_contract'].get('optimization_check')!=expected):
                refuse('PROJECT_PROVIDER_CHECK_BINDING_REQUIRED',root)
        value={k:request[k] for k in ('path','correctness_capsule','performance_capsule')}
        value['provider_sha256']=provider_sha
        sha=append(root,key,[head] if head else [],value,request['reason'])
    return {'status':'PROVIDER_RECORDED','head':sha}


def main(argv):
    parser=argparse.ArgumentParser(description='Portable Decoder project history. No external host required.')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('init');p.add_argument('store');p.add_argument('name');p.add_argument('source')
    p=commands.add_parser('adopt');p.add_argument('store');p.add_argument('workspace');p.add_argument('--repeat')
    p=commands.add_parser('status');p.add_argument('store')
    p=commands.add_parser('export');p.add_argument('store');p.add_argument('output')
    p=commands.add_parser('import');p.add_argument('store');p.add_argument('archive');p.add_argument('sha256')
    p=commands.add_parser('resolve');p.add_argument('store');p.add_argument('key');p.add_argument('selected_head');p.add_argument('reason')
    p=commands.add_parser('provider');p.add_argument('store');p.add_argument('request')
    args=parser.parse_args(argv);root=Path(args.store).resolve()/'coordination'
    try:
        if args.command=='init': result=initialize(args.store,args.name,args.source)
        elif args.command=='adopt': result=adopt(args.store,args.workspace,args.repeat)
        elif args.command=='import': result=import_history(args.store,args.archive,args.sha256)
        elif args.command=='resolve': result=resolve(root,args.key,args.selected_head,args.reason)
        elif args.command=='provider': result=provider(root,read(args.request))
        elif args.command=='export':
            raw=snapshot(root);Path(args.output).write_bytes(raw)
            result={'status':'EXPORTED_LOCAL','sha256':sub._sha(raw),'path':str(Path(args.output).resolve()),'drive_save_confirmed':False}
        else: result=status(root)
        print(json.dumps(result,sort_keys=True,indent=2));return 0
    except Exception as exc:
        from .invocation import refusal_details
        print(json.dumps(refusal_details(exc,'project '+args.command)),file=sys.stderr);return 2
