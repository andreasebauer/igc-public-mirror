from pathlib import Path
import sys,json,zipfile,tempfile,hashlib,importlib
sha=lambda b:hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory() as td:
 root=Path(td)
 with zipfile.ZipFile(sys.argv[1]) as z:
  for name,x in json.loads(z.read('MANIFEST.json')).items():assert sha(z.read(name))==x['sha256'] and len(z.read(name))==x['bytes']
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(root)
 b=root/'o6_export0165';sys.path.insert(0,str(b));m=importlib.import_module('project.reader');p=json.load(open(b/'EXPORT_BINDINGS.json'));r=m.CarrierReader(b/'SCIENTIFIC_EXPORT.zip',p['archive_sha256'],p['root_sha256'],root/'o5_export0162/SCIENTIFIC_EXPORT.zip',root/'o4_export0159/SCIENTIFIC_EXPORT.zip',root/'o3_export0156/SCIENTIFIC_EXPORT.zip');n=json.load(open(b/'NATIVE_RESULT.json'))
 for k,v in r.report.items():assert n['result'][k]==v
 assert json.load(open(b/'COLD_REUSE_RESULT.json'))['status']=='PASS'
 for key,row in r.records.items():
  resources=r.resources(*key)
  for o5 in resources:
   for o4 in o5:
    for o3 in o4:
     for block in o3:
      for p,f in block:assert len(p)==len(f)==7 and all(0<=f[a]<=p[a] for a in range(7))
 print(json.dumps({'status':'FRESH_UNPACK_O6_AND_NESTED_O5_O4_O3_READER_PASS','report':r.report}));r.close()
