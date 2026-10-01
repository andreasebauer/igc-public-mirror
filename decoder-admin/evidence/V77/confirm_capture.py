from pathlib import Path
import json,sys,hashlib
D=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());s=read(D/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import submission as sub
cat=read(D/'SAVE_CATALOG.json');by={r['sha256']:r for r in cat};bindings={r['sha256']:r for r in read(D/'INPUT_BINDINGS.json')};events=[]
for row in s['pending_objects']:
 if row['sha256'] in bindings and 'manifest' in bindings[row['sha256']]:
  b=bindings[row['sha256']];m=Path(b['manifest']);mr=by[hashlib.sha256(m.read_bytes()).hexdigest()];assert Path(mr['readback']).read_bytes()==m.read_bytes();sub.confirm_transport(w,row['sha256'],Path(mr['readback']),Path(b['parts_directory']),mr['drive_id']);events.append({'sha256':row['sha256'],'kind':'TRANSPORT','manifest_readback':mr['readback'],'parts_directory':b['parts_directory'],'drive_id':mr['drive_id']})
 else:
  r=by[row['sha256']];sub.confirm_save(w,row['sha256'],r['readback'],r['drive_id'],role=row['role'],logical_name=row['logical_name']);events.append({'sha256':row['sha256'],'kind':'RAW','readback':r['readback'],'drive_id':r['drive_id']})
state=sub.save_status(w);assert not state['pending_objects'];assert not list((w/'runtime/attempts').rglob('*.json'))
(D/'CAPTURE_ACK.json').write_text(json.dumps(state,indent=2)+'\n');(D/'CAPTURE_READBACK_BINDINGS.json').write_text(json.dumps(events,indent=2)+'\n');print('19 capture roles confirmed; zero attempts')
