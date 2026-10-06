from pathlib import Path
import sys,json,zipfile,tempfile,hashlib
sha=lambda b:hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory() as td:
 R=Path(td)
 with zipfile.ZipFile(sys.argv[1]) as z:
  for n,x in json.loads(z.read('MANIFEST.json')).items():assert sha(z.read(n))==x['sha256'] and len(z.read(n))==x['bytes']
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(R)
 B=R/'o7_additional_integrate0174';load=lambda n:json.loads((B/n).read_bytes())
 cat=load('CATALOG_0149.json');prev=json.loads((R/'o7_integrate0170/CATALOG_0148.json').read_bytes());ad=load('SCOPED_ADMISSION.json');v=load('RELEASE_VERIFICATION.json');s=cat['slices'][-1]
 assert cat['slices'][:-1]==prev['slices'] and len(cat['slices'])==149 and sha((B/'CATALOG_0149.json').read_bytes())==v['catalog_sha256'] and sha((B/'SCOPED_ADMISSION.json').read_bytes())==s['scoped_admission_sha256']
 for n,h in load('PREDECESSOR_SOURCE_HASHES.json').items():assert sha((B/'project/previous_o7'/n).read_bytes())==h
 sys.path.insert(0,str(B));from project.additional_export.reader import CarrierReader
 r=CarrierReader(R/'o7_additional_export0173/SCIENTIFIC_EXPORT.zip',s['archive']['sha256'],s['scientific_root_sha256'],*[R/x/'SCIENTIFIC_EXPORT.zip' for x in ['o7_export0169','o6_export0165','o5_export0162','o4_export0159','o3_export0156']])
 for k,x in ad['counts'].items():assert r.report[k]==x
 for k,row in r.records.items():
  assert r.lookup(*k)==row and len(r.resources(*k))==len(row['parent_ids'])
  for j,pid in enumerate(row['parent_ids']):assert r.owner(*k,j)['prototype_id']==pid
 for k,occ in r.occurrences.items():
  assert r.component(*k)==occ and r.object(occ['object_id'])==r.objects[occ['object_id']]
  assert len(r.component_resources(*k))==len(occ['record']['parent_ids'])
  for j,pid in enumerate(occ['record']['parent_ids']):assert r.component_owner(*k,j)['prototype_id']==pid
 assert load('COLD_REUSE_RESULT.json')['pending_bytes']==0;r.close();print(json.dumps({'status':'FRESH_UNPACK_CATALOG_PREDECESSOR_AND_ALL_ADDITIONAL_O7_ROUTES_PASS','scientific_slices':149,'generation_calls':0}))
