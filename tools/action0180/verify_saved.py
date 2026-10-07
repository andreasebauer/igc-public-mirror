"""Verify compact saved export and native completion bindings."""
from pathlib import Path
import hashlib,json,zipfile
B=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def validate():
 pins=json.loads((B/'EXPORT_BINDINGS.json').read_text());raw=(B/'SCIENTIFIC_EXPORT.zip').read_bytes();assert sha(raw)==pins['archive_sha256'] and len(raw)==pins['bytes']
 with zipfile.ZipFile(B/'SCIENTIFIC_EXPORT.zip') as z:
  b=z.read('ROOT.json');assert sha(b)==pins['root_sha256'];root=json.loads(b);data={};seen={'ROOT.json'}
  for x in root['content']:
   n='content/'+x['sha256']+'.blob';b=z.read(n);assert len(b)==x['bytes'] and sha(b)==x['sha256'];assert x['name'] not in data;data[x['name']]=json.loads(b);seen.add(n)
  assert len(z.namelist())==len(seen) and set(z.namelist())==seen
 assert set(data)=={'population','continuations','contract','gate','catalog','native_result','cold_reuse','registration'}
 assert root['authority']==data['contract']['authority']=='SAVED_PUBLIC_PROJECTION_ONLY'
 assert root['scientific_master_admission'] is False and root['exact_parent_DAG_available'] is False and root['Q2_payload_available'] is False
 n=data['native_result'];c=data['cold_reuse'];assert n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED'
 assert n['result_sha256']==pins['result_sha256']==root['result_sha256']==c['result_sha256']
 assert n['completion_sha256']==pins['completion_sha256']==root['completion_sha256']==c['completion_sha256']
 assert c['reused'] and c['pending_bytes']==0 and c['generation_reexecuted'] is False
 for field,want in {'interfaces':193,'interface_classes':192,'reservation_rows':1351,'continuation_hashes':1351,'generation_calls':0,'scientific_master_admission':False}.items():assert root[field]==n['result'][field]==want
 pop=data['population'];assert len(pop['interfaces'])==193 and len({r['interface_sha256'] for r in pop['interfaces']})==192
 assert sum(len(r['one_endpoint_reservations']) for r in pop['interfaces'])==1351 and sum(map(len,data['continuations']['rows'].values()))==1351
 assert root['master_catalog_sha256']=='84641ecb46ea253c9a8cbbeaa418c6cd14bda93512a763c583ce3ed3af559f7c'
 return {'status':'PASS_NATIVE_COMPLETION_AND_SAVED_EXPORT_BINDINGS','counts':{k:root[k] for k in ['interfaces','interface_classes','reservation_rows','continuation_hashes']},'completion_sha256':root['completion_sha256'],'result_sha256':root['result_sha256'],'cold_reuse':True,'pending_bytes':0,'generation_calls':0,'new_admissions':0,'master_release':'MASTER_DATA_V1_0149','scientific_slices':149,'authority':'SAVED_PUBLIC_PROJECTION_ONLY','next_scope':'WP6_SCOPED_G1_PUBLIC_PROJECTION_ADMISSION_AND_UNIFIED_READER'}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
