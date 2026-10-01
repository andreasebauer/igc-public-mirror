from pathlib import Path
import json,hashlib,sys
D=Path(__file__).resolve().parent;prefix=sys.argv[1];stem='saved_stage_one_'+('prerun' if prefix=='PRERUN' else 'completed')
if sys.argv[2]=='split':
 raw=(D/(stem+'.zip')).read_bytes();rows=[]
 for i,start in enumerate(range(0,len(raw),25000000),1):
  data=raw[start:start+25000000];p=D/(prefix.lower()+f'_part_{i:02d}.bin');p.write_bytes(data)
  rows.append({'local_object_path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data),'role':'fixture_transport','logical_name':p.name})
 (D/(prefix+'_PARTS.json')).write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
else:
 cat={r['sha256']:r for r in json.loads((D/'CATALOG.json').read_text())};rows=json.loads((D/(prefix+'_PARTS.json')).read_text());p=D/(stem+'.readback.zip')
 with p.open('xb') as out:
  for r in rows:
   raw=Path(cat[r['sha256']]['readback']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'];out.write(raw)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads((D/(prefix+'_EXPORT.json')).read_text())['sha256']
 print('RECONSTRUCTED READBACK VERIFIED',prefix)
