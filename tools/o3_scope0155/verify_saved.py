from pathlib import Path
import sys,json,zipfile,hashlib,tempfile,importlib.util
sha=lambda b:hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory() as td:
 root=Path(td)
 with zipfile.ZipFile(sys.argv[1]) as z:
  for n,e in json.loads(z.read('MANIFEST.json')).items():assert sha(z.read(n))==e['sha256'] and len(z.read(n))==e['bytes']
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(root)
 primary=next((root/'o3_scope0155/sources').glob('*V1_3*/*'))
 for e in json.load(open(primary/'provenance/BUNDLE_MANIFEST.json'))['files']:assert sha((primary/e['path']).read_bytes())==e['sha256']
 code=primary/'code/o3_graduation_classification_audit_v1_3.py';sp=importlib.util.spec_from_file_location('saved',code);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);a=m.Audit();assert not a.source_checks();s,f,c,w,r=a.full_sweep();v=json.load(open(root/'o3_scope0155/READINESS_VALIDATION.json'));assert not f and s==v['summary'] and r==v['stored_key_roll_sha256'] and len(c)==v['component_rows']
 print(json.dumps({'status':'FRESH_UNPACK_PRIMARY_INTEGRITY_PASS','states':s['states'],'edges':s['edges'],'generator_calls':0}))
