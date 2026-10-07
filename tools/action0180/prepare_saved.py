"""Package verified saved projections without constructing scientific objects."""
from pathlib import Path
import hashlib,json,zipfile
B=Path(__file__).resolve().parent;R=B.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_bytes())
result=load(B/'NATIVE_RESULT.json');reuse=load(B/'COLD_REUSE_RESULT.json')
assert result['status']=='COMPLETED' and result['evidence_status']=='VERIFIED'
assert reuse['reused'] and reuse['completion_sha256']==result['completion_sha256'] and reuse['pending_bytes']==0
spec=load(B/'SPEC.json');blobs={};items=[]
paths={x['logical_name']:Path(x['path']) for x in spec['inputs']}
paths.update(native_result=B/'NATIVE_RESULT.json',cold_reuse=B/'COLD_REUSE_RESULT.json',registration=R/'g_public_register0179/PROJECT_REGISTRATION.json')
for name,p in paths.items():
 b=p.read_bytes();h=sha(b);items.append({'name':name,'sha256':h,'bytes':len(b)});blobs[h]=b
root={'schema':'IG_SAVED_G1_PUBLIC_PROJECTION_EXPORT_V1','authority':'SAVED_PUBLIC_PROJECTION_ONLY','master_release':'MASTER_DATA_V1_0149','master_catalog_sha256':next(x['sha256'] for x in items if x['name']=='catalog'),'content':items,'completion_sha256':result['completion_sha256'],'result_sha256':result['result_sha256'],'interfaces':193,'interface_classes':192,'reservation_rows':1351,'continuation_hashes':1351,'exact_parent_DAG_available':False,'Q2_payload_available':False,'generation_calls':0,'scientific_master_admission':False}
raw=json.dumps(root,sort_keys=True,separators=(',',':')).encode()
with zipfile.ZipFile(B/'SCIENTIFIC_EXPORT.zip','w',zipfile.ZIP_DEFLATED) as z:
 def put(n,b):
  i=zipfile.ZipInfo(n,(2026,10,7,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b)
 put('ROOT.json',raw)
 for h,b in sorted(blobs.items()):put('content/'+h+'.blob',b)
meta={'archive_sha256':sha((B/'SCIENTIFIC_EXPORT.zip').read_bytes()),'root_sha256':sha(raw),'bytes':(B/'SCIENTIFIC_EXPORT.zip').stat().st_size,'completion_sha256':result['completion_sha256'],'result_sha256':result['result_sha256']}
(B/'EXPORT_BINDINGS.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
