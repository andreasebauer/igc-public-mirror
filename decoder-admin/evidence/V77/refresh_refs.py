from pathlib import Path
import json,hashlib,sys
D=Path(__file__).resolve().parent;G=D/sys.argv[1];read=lambda p:json.loads(p.read_text());cat=read(D/'SAVE_CATALOG.json');by={r['sha256']:r for r in cat};refs=[]
for r in cat:
 if not Path(r['readback']).is_file():continue
 refs.append({'sha256':r['sha256'],'kind':'RAW','drive_file_id':r['drive_id'],'path':r['readback']})
for b in read(D/'INPUT_BINDINGS.json'):
 if 'manifest' in b:
  m=Path(b['manifest']);r=by[hashlib.sha256(m.read_bytes()).hexdigest()];assert Path(r['readback']).read_bytes()==m.read_bytes()
  refs.append({'sha256':b['sha256'],'kind':'MULTIPART','drive_file_id':r['drive_id'],'manifest':r['readback'],'parts':b['parts_directory']})
(G/'SAVE_CATALOG.json').write_text(json.dumps(cat,indent=2));(G/'READBACKS.json').write_text(json.dumps(refs,indent=2))
