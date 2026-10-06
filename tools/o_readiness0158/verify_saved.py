from pathlib import Path
import sys,json,zipfile,tempfile,hashlib,subprocess
sha=lambda b:hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory() as td:
 root=Path(td)
 with zipfile.ZipFile(sys.argv[1]) as z:
  for name,x in json.loads(z.read('MANIFEST.json')).items():assert sha(z.read(name))==x['sha256'] and len(z.read(name))==x['bytes']
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(root)
 original=json.load(open(root/'o_readiness0158/READINESS_VALIDATION.json'));subprocess.run([sys.executable,str(root/'o_readiness0158/validate.py')],cwd=root,check=True);new=json.load(open(root/'o_readiness0158/READINESS_VALIDATION.json'));assert original==new
 print(json.dumps({'status':'FRESH_UNPACK_FULL_READINESS_RECHECK_PASS','generation_calls':0,'O4_saved_counts':new['O4_saved_counts']}))
