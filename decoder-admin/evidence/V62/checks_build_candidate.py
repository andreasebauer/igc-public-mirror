from pathlib import Path
import ast,hashlib,json,zipfile
R=Path('/tmp/ig_decoder_dev144_20261001');S=R/'decoder';D=Path(__file__).parent
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
old=read(R/'decoder-import/source-manifest.json')
assert old['version']=='0.8.0.dev143+lib'
write(D/'PARENT_SOURCE_IDENTITY.json',{k:v for k,v in old.items() if k!='files'})
(S/'infinity_grid/_version.py').write_text('__version__ = "0.8.0.dev144+lib"\n')
b=read(S/'infinity_grid/_build_meta.json')
b.update(engineering_attempt='DEV144_V62_VALIDATION_CLEANUP',parent_release=old['version'],parent_source_archive_sha256=old['archive_sha256'],scope='Checkpoint state modes and paused refusal ordering; native qualification pending',pending=['Native cleanup regression checks','Fresh reliability case','Current candidate full RC'])
write(S/'infinity_grid/_build_meta.json',b)
p=read(S/'qualification/PROFILE.json');groups=p['groups'];order=p['group_order']
p['status']='UNQUALIFIED_DEV144_CLEANUP_REPAIR'
p['blocking_conditions']=['Native registered tests not yet executed for dev144.','V55 remains a failed lifecycle gate; no inherited qualification.','Full RC, fixtures and independent-host recovery remain OPEN.']
p['source_reference'].update(revision='0.8.0.dev144+lib',parent_archive_sha256=old['archive_sha256'])
write(S/'qualification/PROFILE.json',p)
rows=[];parsed=0
for f in sorted(S.rglob('*')):
 if not f.is_file():continue
 n=f.relative_to(S).as_posix();assert not f.is_symlink() and '__pycache__' not in n and not n.endswith('.pyc')
 if f.suffix=='.py':ast.parse(f.read_bytes());parsed+=1
 rows.append([n,sha(f.read_bytes())])
changed=[n for n,h in rows if dict(old['files']).get(n)!=h]
assert set(changed)=={'infinity_grid/_version.py','infinity_grid/_build_meta.json','qualification/PROFILE.json','infinity_grid/v05_controller_event_loop.py','infinity_grid/preservation.py','tests/test_paused_checkpoint_recovery.py'}
archive=D/'DECODER_DEV144_SOURCE_2026-10-01.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for n,h in rows:
  i=zipfile.ZipInfo(n,date_time=(2026,10,1,0,0,0));i.external_attr=0o100644<<16;i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,(S/n).read_bytes())
identity={'version':'0.8.0.dev144+lib','source_sha256':digest({'schema_id':'IG_DECODER_ENGINEERING_SOURCE_TREE_V1','files':rows}),'package_sha256':digest({'schema_id':'IG_DECODER_EXECUTABLE_SOURCE_TREE_V1','files':[[n[len('infinity_grid/'):],h] for n,h in rows if n.startswith('infinity_grid/')]}),'archive_sha256':sha(archive.read_bytes()),'archive_size_bytes':archive.stat().st_size,'qualification':'UNQUALIFIED'}
write(D/'SOURCE_IDENTITY.json',identity);write(R/'decoder-import/source-manifest.json',{**identity,'files':rows})
write(D/'STATIC_REVIEW.json',{'files':len(rows),'python_files_parsed':parsed,'changed_files':changed,'native_tests_executed':0,'new_regression_selectors':9,'group_definitions_unchanged':p['groups']==groups and p['group_order']==order})
(R/'decoder-import/VERIFICATION.txt').write_text('V62 dev144 static source build only; native execution pending.\n'+json.dumps(identity,indent=2)+'\n')
(R/'decoder-import/README.txt').write_text('CURRENT SOURCE dev144 UNQUALIFIED. Read ../DECODER_READ_FIRST.txt.\n')
(R/'DECODER_READ_FIRST.txt').write_text('DEV144 V62 CLEANUP REPAIR — UNQUALIFIED — 2026-10-01\nNative registered tests and fresh reliability gate pending.\nNo inherited dev144 qualification. V55 remains failed lifecycle cleanup.\nAll 23 RC rows OPEN. No science replay.\nVerify source: python3 -B decoder-admin/decoder.py verify-source\nRead decoder-admin/evidence/V62_REPORT.txt.\n')
ledger=read(R/'decoder-admin/RC_LEDGER.json')
ledger.setdefault('historical_candidates',[])
# Retain the old current-candidate record without assuming its history container type.
write(D/'PARENT_CANDIDATE_LEDGER.json',ledger.get('latest_candidate',{}))
ledger['latest_candidate']={**identity,'tests_executed':0,'status':'UNQUALIFIED','reason':'Native cancellation integration requires new-source testing'}
ledger['latest_source_repair']={'gate':'V62','version':identity['version'],'scope':'paused checkpoint modes/refusal ordering','native_tests_executed':0,'full_RC':'OPEN'}
write(R/'decoder-admin/RC_LEDGER.json',ledger)
print(json.dumps(identity));print(json.dumps(changed))
