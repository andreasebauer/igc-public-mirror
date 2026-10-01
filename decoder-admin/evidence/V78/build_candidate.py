from pathlib import Path
import json,hashlib,ast,zipfile
D=Path(__file__).parent;R=Path('/tmp/ig_decoder_dev147_20261001');S=R/'decoder'
read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
old=read(R/'decoder-import/source-manifest.json');assert old['version']=='0.8.0.dev146+lib'
write(D/'PARENT_SOURCE_IDENTITY.json',{k:v for k,v in old.items() if k!='files'})
p=read(S/'qualification/PROFILE.json');p['status']='UNQUALIFIED_DEV147_REVIEW_SUCCESSOR';p['source_reference'].update(revision='0.8.0.dev147+lib',parent_archive_sha256=old['archive_sha256']);write(S/'qualification/PROFILE.json',p)
(S/'infinity_grid/_version.py').write_text('__version__ = "0.8.0.dev147+lib"\n')
b=read(S/'infinity_grid/_build_meta.json');b.update(engineering_attempt='DEV147_V78_LOCK_REPAIR',parent_release=old['version'],parent_source_archive_sha256=old['archive_sha256'],scope='Wait for acknowledgment outbox lock during checkpoint publication; preserve fail-fast workload ownership',pending=['Fresh native pause validation','Exact-candidate fixtures','Full RC','Independent-host recovery']);write(S/'infinity_grid/_build_meta.json',b)
rows=[]
for f in sorted(S.rglob('*')):
 if not f.is_file():continue
 n=f.relative_to(S).as_posix();assert not f.is_symlink() and '__pycache__' not in n and not n.endswith('.pyc')
 if f.suffix=='.py':ast.parse(f.read_bytes())
 rows.append([n,sha(f.read_bytes())])
changed=[n for n,h in rows if dict(old['files']).get(n)!=h];assert set(changed)=={'infinity_grid/_version.py','infinity_grid/_build_meta.json','qualification/PROFILE.json','tests/test_change_preservation_rebind.py','tests/fixtures/preservation_rebind/DEV147_REVIEWED_CORE_CHANGES.json','infinity_grid/v05_controller_event_loop.py','infinity_grid/preservation.py','tests/test_preservation_batch.py'}
archive=D/'DECODER_DEV147_SOURCE_2026-10-01.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for n,h in rows:
  i=zipfile.ZipInfo(n,date_time=(2026,10,1,0,0,0));i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,(S/n).read_bytes())
id={'version':'0.8.0.dev147+lib','source_sha256':digest({'schema_id':'IG_DECODER_ENGINEERING_SOURCE_TREE_V1','files':rows}),'package_sha256':digest({'schema_id':'IG_DECODER_EXECUTABLE_SOURCE_TREE_V1','files':[[n[len('infinity_grid/'):],h] for n,h in rows if n.startswith('infinity_grid/')]}),'archive_sha256':sha(archive.read_bytes()),'archive_size_bytes':archive.stat().st_size,'qualification':'UNQUALIFIED'}
write(D/'SOURCE_IDENTITY.json',id);write(R/'decoder-import/source-manifest.json',{**id,'files':rows});write(D/'STATIC_REVIEW.json',{'changed_files':changed,'engine_implementation_unchanged':False,'reviewed_core_changes':2,'historical_pins_unchanged':True});print(json.dumps(id))
