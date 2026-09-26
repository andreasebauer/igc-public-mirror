from __future__ import annotations
import hashlib, json, os, shutil, zipfile
from pathlib import Path
from .canon import write_json_atomic, canonical_sha256
from .hashing import sha256_file
from .store import ArtifactStore
from .datasets import DatasetStore
from .safety import safe_archive_members, validate_identifier, validate_relative_path
from .verification import verify_run

FIXED_ZIP_TIME=(2020,1,1,0,0,0)

def _zip_tree(src:Path,dst:Path):
    dst.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dst,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(x for x in src.rglob('*') if x.is_file()):
            rel=validate_relative_path(p.relative_to(src).as_posix()); info=zipfile.ZipInfo(rel,FIXED_ZIP_TIME); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=(0o644&0xFFFF)<<16; z.writestr(info,p.read_bytes())
    return dst

def _manifest_files(root:Path,exclude=('INTERNAL_MANIFEST.json',)):
    out=[]
    for p in sorted(x for x in root.rglob('*') if x.is_file()):
        rel=validate_relative_path(p.relative_to(root).as_posix())
        if rel in exclude: continue
        out.append({'path':rel,'sha256':sha256_file(p),'size_bytes':p.stat().st_size})
    return out

INDEPENDENT_VERIFIER='''#!/usr/bin/env python3
import hashlib,json,sys,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
m=json.loads((root/'INTERNAL_MANIFEST.json').read_text()); fail=[]
expected={r['path'] for r in m['files']}|{'INTERNAL_MANIFEST.json'}
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
if actual!=expected: fail.append({'reason':'inventory','missing':sorted(expected-actual),'extra':sorted(actual-expected)})
for r in m['files']:
 p=root/r['path']
 if not p.is_file(): fail.append({'path':r['path'],'reason':'missing'}); continue
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=r['sha256'] or p.stat().st_size!=r['size_bytes']: fail.append({'path':r['path'],'reason':'mismatch','observed':h})
try:
 rr=json.loads((root/'run/run.json').read_text()); lock=json.loads((root/'ENVIRONMENT_LOCK.json').read_text()); wh=root/'wheelhouse'/lock['wheel']['filename']
 if hashlib.sha256(wh.read_bytes()).hexdigest()!=lock['wheel']['sha256']: fail.append({'reason':'wheel_lock_hash'})
 with zipfile.ZipFile(wh) as z: bm=json.loads(z.read('infinity_grid/_build_meta.json'))
 if bm.get('source_sha256')!=rr.get('code_identity',{}).get('source_sha256'): fail.append({'reason':'wheel_source_identity'})
except Exception as e: fail.append({'reason':'identity_check_exception','error':str(e)})
print(json.dumps({'status':'PASS' if not fail else 'FAIL','checked':len(m['files']),'failures':fail},sort_keys=True)); raise SystemExit(0 if not fail else 2)
'''

REPRODUCE_SCRIPT='''#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
from infinity_grid import build_meta
from infinity_grid.paths import resolve_root
from infinity_grid.store import ArtifactStore
from infinity_grid.datasets import DatasetStore
from infinity_grid.controller import Controller
import infinity_grid.adapters
original=json.loads((root/'run/run.json').read_text()); lock=json.loads((root/'ENVIRONMENT_LOCK.json').read_text()); bm=build_meta()
if bm.get('source_sha256')!=original.get('code_identity',{}).get('source_sha256'): raise SystemExit('installed runtime source identity mismatch')
wh=root/'wheelhouse'/lock['wheel']['filename']
if hashlib.sha256(wh.read_bytes()).hexdigest()!=lock['wheel']['sha256']: raise SystemExit('wheelhouse hash mismatch')
runtime=resolve_root(root/'runtime_root').ensure(); store=ArtifactStore(runtime.store); datasets=DatasetStore(store); plan=json.loads((root/'run/plan.json').read_text())
for idx,d in enumerate(plan['input_datasets']):
 manifest_path=root/'input_manifests'/f"{idx:02d}_{d['dataset_sha256']}.json"
 frozen_manifest=json.loads(manifest_path.read_text())
 observed=datasets.import_directory(
  root/'inputs'/f'{idx:02d}',
  logical_role=frozen_manifest['logical_role'],
  canonicalization_version=frozen_manifest.get('canonicalization_version','FILESET_V1'),
 )
 if observed['dataset_sha256']!=d['dataset_sha256']: raise SystemExit('dataset identity mismatch')
mode=json.loads((root/'EXECUTION_MODE.json').read_text())
rr=Controller(runtime).run(plan,execution_mode=mode)
expected=json.loads((root/'EXPECTED_RESULTS.json').read_text()); got={(a.get('logical_name'),a['sha256']) for a in rr['result_artifacts']}; want={(k,v) for k,v in expected.items()}
ok=(want==got and rr['lifecycle']=='COMPLETE_VALID')
print(json.dumps({'status':'PASS' if ok else 'FAIL','run_id':rr['run_id'],'expected':sorted(want),'observed':sorted(got),'source_sha256':bm.get('source_sha256')},sort_keys=True)); raise SystemExit(0 if ok else 3)
'''

class ReleaseGenerator:
    def __init__(self,paths): self.paths=paths; self.store=ArtifactStore(paths.store); self.datasets=DatasetStore(self.store)
    def _wheel(self):
        ws=sorted((self.paths.app/'wheelhouse').glob('*.whl'))
        if len(ws)!=1: raise RuntimeError(f'exactly one runtime wheel required, found {ws}')
        return ws[0]
    def create(self,run_id:str,mode:str):
        validate_identifier(run_id,field='run_id'); mode=mode.upper()
        if mode not in {'COMPACT','STANDALONE'}: raise ValueError(mode)
        vr=verify_run(self.paths,run_id)
        if vr.get('status')!='PASS': raise RuntimeError(f'release only from fully verified run: {vr}')
        run_dir=self.paths.runs/run_id; rr=json.loads((run_dir/'run.json').read_text()); plan=json.loads((run_dir/'plan.json').read_text()); wheel=self._wheel()
        content_key={'schema_id':'IG_RELEASE_CONTENT_KEY_V0_17_1','run_core_sha256':rr['run_core_sha256'],'mode':mode,'results':sorted((a.get('logical_name'),a['sha256'],a.get('size_bytes')) for a in rr.get('result_artifacts',[])),'input_datasets':sorted(d['dataset_sha256'] for d in plan.get('input_datasets',[])),'protocol_sha256':rr['protocol']['descriptor_sha256'],'wheel_sha256':sha256_file(wheel),'execution_mode':rr.get('execution_mode'),'run_record_sha256':canonical_sha256(rr)}
        content_sha=canonical_sha256(content_key); release_id=f'{run_id}-{mode.lower()}-{content_sha[:16]}'; validate_identifier(release_id,field='release_id'); outdir=self.paths.releases/release_id
        if outdir.exists():
            mp=outdir/'release_manifest.json'
            if not mp.is_file(): raise RuntimeError('pre-existing release directory without manifest')
            old=json.loads(mp.read_text())
            if old.get('content_sha256')!=content_sha: raise RuntimeError('conflicting pre-existing release identity')
            arc=outdir/old['archive']
            vv=verify_release_archive(arc)
            if vv.get('status')!='PASS' or sha256_file(arc)!=old.get('archive_sha256'): raise RuntimeError('pre-existing release failed verification')
            return {'status':'PASS','release_id':release_id,'archive':str(arc),'archive_sha256':old['archive_sha256'],'mode':mode,'manifest':old,'reused':True}
        stage=self.paths.workspace/f'.release-{release_id}'; shutil.rmtree(stage,ignore_errors=True); stage.mkdir(parents=True)
        shutil.copytree(run_dir,stage/'run',dirs_exist_ok=True); expected={}
        for a in rr.get('result_artifacts',[]):
            logical=a.get('logical_name') or a.get('source_name')
            if logical: validate_relative_path(logical); self.store.materialize(a['sha256'],stage/'results'/logical,expected_size=a['size_bytes']); expected[logical]=a['sha256']
        write_json_atomic(stage/'EXPECTED_RESULTS.json',expected); write_json_atomic(stage/'EXECUTION_MODE.json',rr.get('execution_mode') or {})
        descpath=self.paths.store/'protocols'/f"{rr['protocol']['descriptor_sha256']}.json"; (stage/'protocol').mkdir(); shutil.copy2(descpath,stage/'protocol/descriptor.json')
        with zipfile.ZipFile(wheel) as wz: wheel_meta=json.loads(wz.read('infinity_grid/_build_meta.json'))
        if wheel_meta.get('source_sha256')!=rr.get('code_identity',{}).get('source_sha256'): raise RuntimeError('runtime wheel source identity does not match run')
        (stage/'wheelhouse').mkdir(); shutil.copy2(wheel,stage/'wheelhouse'/wheel.name)
        lock={'schema_id':'IG_OFFLINE_ENVIRONMENT_LOCK_V0_17_1','python':rr['environment_identity']['descriptor']['python'],'runtime_sha256':rr['environment_identity']['runtime_sha256'],'source_sha256':wheel_meta.get('source_sha256'),'wheel':{'filename':wheel.name,'sha256':sha256_file(wheel),'size_bytes':wheel.stat().st_size},'network_downloads_required':False}; write_json_atomic(stage/'ENVIRONMENT_LOCK.json',lock)
        source_manifest=self.paths.app/'source'/'SOURCE_TREE_MANIFEST.json'
        if source_manifest.exists(): shutil.copy2(source_manifest,stage/'SOURCE_TREE_MANIFEST.json')
        for ref in rr.get('claim_refs',[]):
            cid=ref if isinstance(ref,str) else ref.get('claim_id'); ver=None if isinstance(ref,str) else ref.get('version'); matches=sorted((self.paths.store/'claims'/cid).glob('*.json')) if cid and ver is None else ([self.paths.store/'claims'/cid/f'v{int(ver):04d}.json'] if cid and ver else [])
            for cp in matches:
                if cp.is_file(): dst=stage/'claims'/cp.parent.name/cp.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(cp,dst)
        object_refs={'input_datasets':[],'result_artifacts':rr.get('result_artifacts',[])}
        for idx,d in enumerate(plan.get('input_datasets',[])):
            man=self.datasets.load(d['dataset_sha256']); md=stage/'input_manifests'; md.mkdir(exist_ok=True); write_json_atomic(md/f"{idx:02d}_{d['dataset_sha256']}.json",man); object_refs['input_datasets'].append({'dataset_sha256':d['dataset_sha256'],'logical_role':d.get('logical_role'),'shards':man['shards']})
            if mode=='STANDALONE': self.datasets.materialize(d['dataset_sha256'],stage/'inputs'/f'{idx:02d}')
        write_json_atomic(stage/'STORE_OBJECT_REFERENCES.json',object_refs)
        if mode=='COMPACT': (stage/'INPUT_RETRIEVAL.txt').write_text('Input objects are addressed by SHA-256 in STORE_OBJECT_REFERENCES.json and input_manifests/.\n')
        (stage/'verify_release.py').write_text(INDEPENDENT_VERIFIER)
        if mode=='STANDALONE': (stage/'reproduce.py').write_text(REPRODUCE_SCRIPT); (stage/'REPRODUCE_COMMAND.txt').write_text(f'python -m venv .venv\n.venv/bin/python -m pip install --no-index wheelhouse/{wheel.name}\n.venv/bin/python reproduce.py\n')
        else: (stage/'REPRODUCE_COMMAND.txt').write_text('Retrieve exact hashed input datasets, install pinned wheel, and execute run/plan.json.\n')
        internal={'schema_id':'IG_RELEASE_INTERNAL_MANIFEST_V0_17_1','release_id':release_id,'run_id':run_id,'mode':mode,'content_sha256':content_sha,'files':_manifest_files(stage)}; write_json_atomic(stage/'INTERNAL_MANIFEST.json',internal)
        outdir.mkdir(parents=True,exist_ok=False); archive=outdir/f'{release_id}.zip'; _zip_tree(stage,archive); archive_sha=sha256_file(archive); external={'schema_id':'IG_RELEASE_RECORD_V0_17_1','release_id':release_id,'run_id':run_id,'mode':mode,'content_sha256':content_sha,'archive':archive.name,'archive_sha256':archive_sha,'archive_size_bytes':archive.stat().st_size,'internal_manifest_sha256':sha256_file(stage/'INTERNAL_MANIFEST.json')}; write_json_atomic(outdir/'release_manifest.json',external); shutil.rmtree(stage,ignore_errors=True)
        return {'status':'PASS','release_id':release_id,'archive':str(archive),'archive_sha256':archive_sha,'mode':mode,'manifest':external,'reused':False}

def verify_release_archive(path:Path):
    path=Path(path); failures=[]
    try:
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if not n.endswith('/')]; dup,unsafe=safe_archive_members(names)
            if dup: failures.append({'reason':'duplicate_members','paths':dup})
            if unsafe: failures.append({'reason':'unsafe_members','paths':unsafe})
            bad=z.testzip()
            if bad: failures.append({'reason':'zip_crc','path':bad})
            m=json.loads(z.read('INTERNAL_MANIFEST.json')); expected={r['path'] for r in m.get('files',[])}|{'INTERNAL_MANIFEST.json'}; actual=set(names)
            if actual!=expected: failures.append({'reason':'inventory','missing':sorted(expected-actual),'extra':sorted(actual-expected)})
            for r in m.get('files',[]):
                try: data=z.read(r['path'])
                except KeyError: failures.append({'reason':'missing','path':r['path']}); continue
                if len(data)!=r['size_bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']: failures.append({'reason':'mismatch','path':r['path']})
            return {'status':'PASS' if not failures else 'FAIL','release_id':m.get('release_id'),'mode':m.get('mode'),'checked':len(m.get('files',[])),'failures':failures,'archive_sha256':sha256_file(path)}
    except Exception as e: return {'status':'FAIL','failures':[{'reason':'exception','error':str(e)}]}
