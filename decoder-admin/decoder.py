"""Single administrative entry point; never runs Decoder jobs or installs packages."""
import argparse,hashlib,io,json,os,shutil,stat,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ADMIN=ROOT/'decoder-admin'
CAT=json.loads((ADMIN/'CATALOG.json').read_text())
CACHE=ROOT/'.decoder-cache'
def fail(message):raise RuntimeError(message)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def check(p,r):
 if p.is_symlink() or not p.is_file():fail('MISSING_OR_SYMLINK:'+str(p))
 if p.stat().st_size!=r['size_bytes'] or sha(p)!=r['sha256']:fail('OBJECT_MISMATCH:'+r['name'])
def item(name):return next(r for r in CAT['objects'] if r['name']==name)
def obj(name):
 r=item(name);p=CACHE/r['sha256'];check(p,r);return p
def unpack(z,dest):
 names=z.namelist()
 if len(names)!=len(set(names)):fail('DUPLICATE_ZIP_MEMBER')
 for i in z.infolist():
  n=Path(i.filename)
  if n.is_absolute() or '..' in n.parts or '\\' in i.filename or stat.S_ISLNK(i.external_attr>>16):fail('UNSAFE_ZIP_MEMBER')
  if (dest/n).exists() and not i.is_dir():fail('OVERLAPPING_ZIP_MEMBER:'+i.filename)
 z.extractall(dest)
def verify_runtime(dest):
 m=json.loads((ADMIN/'RUNTIME_MANIFEST.json').read_text());policy=json.loads((ADMIN/'RUNTIME_POLICY.json').read_text())
 if dest.is_symlink():fail('RUNTIME_ROOT_SYMLINK')
 if any(p.is_symlink() for p in dest.rglob('*')):fail('RUNTIME_SYMLINK')
 actual={p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
 if actual!=set(m['runtime_files']):fail('RUNTIME_FILE_SET')
 for n,r in m['runtime_files'].items():
  p=dest/n
  if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:fail('RUNTIME_BYTES:'+n)
  mode=0o755 if n in policy['executables'] else 0o644
  if stat.S_IMODE(p.stat().st_mode)!=mode:fail('RUNTIME_MODE:'+n)
 for n,r in m['host'].items():
  p=Path(n)
  if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:fail('HOST_BYTES:'+n)
 return {'status':'BYTES_AND_MODES_VERIFIED','runtime_files':len(actual),'host_files':len(m['host']),'qualification':'NOT_GRANTED','runtime_executed':False}
def restore(dest):
 dest=dest.absolute()
 if dest.exists() or dest.is_symlink():fail('DESTINATION_MUST_BE_NEW')
 if ROOT==dest or ROOT in dest.parents:fail('RUNTIME_MUST_BE_OUTSIDE_CHECKOUT')
 parts=[obj('runtime_base_1'),obj('runtime_base_2')];overlay=obj('runtime_overlay')
 dest.mkdir(parents=True)
 for p in parts:
  with zipfile.ZipFile(p) as z:unpack(z,dest)
 with zipfile.ZipFile(overlay) as z:
  # The outer archive is hash pinned; verify its manifest before using wheels.
  for row in json.loads(z.read('MANIFEST.json')):
   b=z.read(row['path'])
   if len(b)!=row['size_bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:fail('OVERLAY_MEMBER_MISMATCH')
  wheels=[n for n in z.namelist() if n.startswith('wheels/') and n.endswith('.whl')]
  if len(wheels)!=3:fail('OVERLAY_WHEEL_COUNT')
  for n in wheels:
   with zipfile.ZipFile(io.BytesIO(z.read(n))) as w:unpack(w,dest/'base/lib/python3.13/dist-packages')
 policy=json.loads((ADMIN/'RUNTIME_POLICY.json').read_text())
 for p in dest.rglob('*'):
  if p.is_file():p.chmod(0o755 if p.relative_to(dest).as_posix() in policy['executables'] else 0o644)
 result=verify_runtime(dest)
 print(json.dumps(result,indent=2))
def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
 sub.add_parser('status');sub.add_parser('verify-source');sub.add_parser('needs');sub.add_parser('verify-cache')
 x=sub.add_parser('import-object');x.add_argument('name',choices=[r['name'] for r in CAT['objects']]);x.add_argument('file',type=Path)
 for name in ['restore-runtime','verify-runtime']:
  x=sub.add_parser(name);x.add_argument('destination',type=Path)
 a=ap.parse_args()
 if a.command=='verify-source':subprocess.run([sys.executable,str(ROOT/'decoder-import/verify_source.py')],check=True)
 elif a.command=='status':print(json.dumps({'source':CAT['source'],'full_rc':'OPEN','runtime':'UNQUALIFIED; V16 launch failed before Python','missing_candidate_fixtures':CAT['unresolved_fixtures'],'science_replay_closure':CAT['science_replay_closure']},indent=2))
 elif a.command=='needs':print(json.dumps(CAT['objects'],indent=2))
 elif a.command=='import-object':
  r=item(a.name);check(a.file,r);CACHE.mkdir(exist_ok=True);target=CACHE/r['sha256']
  if target.exists():check(target,r)
  else:
   with target.open('xb') as f, a.file.open('rb') as source:shutil.copyfileobj(source,f)
   target.chmod(0o600);check(target,r)
  print(json.dumps({'imported':a.name,'sha256':r['sha256']}))
 elif a.command=='verify-cache':
  for r in CAT['objects']:check(CACHE/r['sha256'],r)
  print(json.dumps({'objects_verified':len(CAT['objects']),'qualification':'NOT_GRANTED'}))
 elif a.command=='restore-runtime':restore(a.destination)
 else:print(json.dumps(verify_runtime(a.destination.absolute()),indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:print('STOP: '+str(e),file=sys.stderr);sys.exit(1)
