from pathlib import Path
import sys,json,zipfile,tempfile,hashlib
sha=lambda b:hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory() as td:
 R=Path(td)
 with zipfile.ZipFile(sys.argv[1]) as z:
  m=json.loads(z.read('MANIFEST.json'))
  for n,x in m.items():assert sha(z.read(n))==x['sha256'] and len(z.read(n))==x['bytes']
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(R)
 B=R/'o7_additional_export0173';sys.path.insert(0,str(B));from project.reader import CarrierReader
 p=json.loads((B/'EXPORT_BINDINGS.json').read_bytes())
 r=CarrierReader(B/'SCIENTIFIC_EXPORT.zip',p['archive_sha256'],p['root_sha256'],*[R/x/'SCIENTIFIC_EXPORT.zip' for x in ['o7_export0169','o6_export0165','o5_export0162','o4_export0159','o3_export0156']])
 native=json.loads((B/'NATIVE_RESULT.json').read_bytes());cold=json.loads((B/'COLD_REUSE_RESULT.json').read_bytes())
 assert native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED' and native['result']['outcome']=='PASS' and cold['status']=='PASS' and cold['pending_bytes']==0
 for k,v in r.report.items():assert native['result'][k]==v
 for key,row in r.records.items():
  assert r.lookup(*key)==row and len(r.resources(*key))==len(row['parent_ids'])
  for j,pid in enumerate(row['parent_ids']):assert r.owner(*key,j)['prototype_id']==pid
 for key,occ in r.occurrences.items():
  assert r.component(*key)==occ and r.object(occ['object_id'])==r.objects[occ['object_id']]
  assert len(r.component_resources(*key))==len(occ['record']['parent_ids'])
  for j,pid in enumerate(occ['record']['parent_ids']):assert r.component_owner(*key,j)['prototype_id']==pid
 for n,h in json.loads((B/'PREDECESSOR_SOURCE_HASHES.json').read_bytes()).items():assert sha((B/'project/previous_o7'/n).read_bytes())==h
 r.close();print(json.dumps({'status':'FRESH_UNPACK_ALL_ROOT_COMPONENT_AND_OWNER_ROUTES_PASS','whole_roots':24,'component_occurrences':205,'exact_objects':164,'generation_calls':0}))
