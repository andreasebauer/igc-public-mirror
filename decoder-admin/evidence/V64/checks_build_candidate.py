from pathlib import Path
import json,hashlib,ast,zipfile
D=Path(__file__).parent;R=Path('/tmp/ig_decoder_dev145_20261001');S=R/'decoder'
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
old=read(R/'decoder-import/source-manifest.json');assert old['version']=='0.8.0.dev144+lib'
write(D/'PARENT_SOURCE_IDENTITY.json',{k:v for k,v in old.items() if k!='files'})
p=read(S/'qualification/PROFILE.json');before={n:g['selectors'][:] for n,g in p['groups'].items()}
actual={str(f.relative_to(S)) for f in (S/'tests').glob('test_*.py')};selected=[x for g in before.values() for x in g]
missing=sorted(actual-set(selected));assert missing==['tests/test_paused_checkpoint_recovery.py','tests/test_validation_cancellation.py'] and not set(selected)-actual
p['groups']['functional']['selectors']=sorted(p['groups']['functional']['selectors']+missing)
selected=[x for g in p['groups'].values() for x in g['selectors']];assert len(selected)==len(set(selected))==len(actual)==170
p['status']='UNQUALIFIED_DEV145_PROFILE_COVERAGE'
p['support']['python_target']='CPython 3.13.5 pinned reviewed runtime; exact per-capture binding required'
p['blocking_conditions']=['Current-source native qualification and fixtures remain OPEN.','V62/V63 dev144 bounded results remain separately identified.','Full RC and independent-host recovery remain OPEN.']
p['source_reference'].update(revision='0.8.0.dev145+lib',parent_archive_sha256=old['archive_sha256'])
write(S/'qualification/PROFILE.json',p)
(S/'infinity_grid/_version.py').write_text('__version__ = "0.8.0.dev145+lib"\n')
b=read(S/'infinity_grid/_build_meta.json');b.update(engineering_attempt='DEV145_V64_PROFILE_COVERAGE',parent_release=old['version'],parent_source_archive_sha256=old['archive_sha256'],scope='Register two omitted active test suites; engine implementation unchanged',pending=['Current-source fixtures','Full RC','Independent-host recovery']);write(S/'infinity_grid/_build_meta.json',b)
rows=[]
for f in sorted(S.rglob('*')):
 if not f.is_file():continue
 n=f.relative_to(S).as_posix();assert not f.is_symlink() and '__pycache__' not in n and not n.endswith('.pyc')
 if f.suffix=='.py':ast.parse(f.read_bytes())
 rows.append([n,sha(f.read_bytes())])
changed=[n for n,h in rows if dict(old['files']).get(n)!=h];assert set(changed)=={'infinity_grid/_version.py','infinity_grid/_build_meta.json','qualification/PROFILE.json'}
archive=D/'DECODER_DEV145_SOURCE_2026-10-01.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for n,h in rows:
  i=zipfile.ZipInfo(n,date_time=(2026,10,1,0,0,0));i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,(S/n).read_bytes())
id={'version':'0.8.0.dev145+lib','source_sha256':digest({'schema_id':'IG_DECODER_ENGINEERING_SOURCE_TREE_V1','files':rows}),'package_sha256':digest({'schema_id':'IG_DECODER_EXECUTABLE_SOURCE_TREE_V1','files':[[n[len('infinity_grid/'):],h] for n,h in rows if n.startswith('infinity_grid/')]}),'archive_sha256':sha(archive.read_bytes()),'archive_size_bytes':archive.stat().st_size,'qualification':'UNQUALIFIED'}
write(D/'SOURCE_IDENTITY.json',id);write(R/'decoder-import/source-manifest.json',{**id,'files':rows});write(D/'STATIC_REVIEW.json',{'changed_files':changed,'added_functional_selectors':missing,'active_files':170,'registered_once':170,'engine_implementation_unchanged':True,'test_bodies_unchanged':True,'group_workers_unchanged':True})
print(json.dumps(id))
