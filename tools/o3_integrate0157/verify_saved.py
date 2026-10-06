from pathlib import Path
import sys,json,zipfile,tempfile,hashlib,importlib
sha=lambda b:hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory() as td:
 root=Path(td)
 with zipfile.ZipFile(sys.argv[1]) as z:
  for name,x in json.loads(z.read('MANIFEST.json')).items():assert sha(z.read(name))==x['sha256'] and len(z.read(name))==x['bytes']
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(root)
 b=root/'o3_integrate0157';cat=json.load(open(b/'CATALOG_0144.json'));prev=json.load(open(root/'o2_integrate0152/CATALOG_0143.json'));ad=json.load(open(b/'SCOPED_ADMISSION.json'));v=json.load(open(b/'RELEASE_VERIFICATION.json'));assert len(cat['slices'])==144 and cat['slices'][:-1]==prev['slices'] and sha((b/'CATALOG_0144.json').read_bytes())==v['catalog_sha256'] and sha((b/'SCOPED_ADMISSION.json').read_bytes())==cat['slices'][-1]['scoped_admission_sha256']
 for name,h in json.load(open(b/'PREDECESSOR_SOURCE_HASHES.json')).items():assert sha((b/'project/previous_o2'/name).read_bytes())==h
 sys.path.insert(0,str(b));m=importlib.import_module('project.reader');s=cat['slices'][-1];r=m.CarrierReader(root/'o3_export0156/SCIENTIFIC_EXPORT.zip',s['archive']['sha256'],s['scientific_root_sha256'])
 for k in ['carriers','seed_carriers','parent_links','typed_edges','bank_entries','components','recomputed_keys','source_bound_fallback_keys']:assert r.report[k]==ad['counts'][k]
 assert json.load(open(b/'COLD_REUSE_RESULT.json'))['pending_bytes']==0
 print(json.dumps({'status':'FRESH_UNPACK_CATALOG_PREDECESSOR_AND_O3_READER_PASS','scientific_slices':144,'generation_calls':0}));r.close()
