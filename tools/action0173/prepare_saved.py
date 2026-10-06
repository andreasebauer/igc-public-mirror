from pathlib import Path
import json,hashlib,sys
B=Path(__file__).resolve().parent;R=B.parent;mode=sys.argv[1];sha=lambda b:hashlib.sha256(b).hexdigest()
name='CHECKPOINT_READBACKS.json' if mode=='checkpoint' else 'READBACKS.json'
mapping={}
for pattern in ['*/READBACKS.json','*/CHECKPOINT_READBACKS.json']:
 for file in R.glob(pattern):
  for h,v in json.loads(file.read_bytes()).items():
   p=Path(v['path'])
   if h not in mapping and p.is_file() and sha(p.read_bytes())==h:mapping[h]=v
d=json.loads((B/('CHECKPOINT_PENDING.json' if mode=='checkpoint' else 'PENDING.json')).read_bytes())
rows=d['physical_objects'] if mode=='checkpoint' else d['pending_objects'];todo=[]
for row in rows:
 if row['sha256'] not in mapping:todo.append(row)
(B/name).write_text(json.dumps({row['sha256']:mapping[row['sha256']] for row in rows if row['sha256'] in mapping},indent=2)+'\n')
(B/('TO_SAVE_CHECKPOINT.json' if mode=='checkpoint' else 'TO_SAVE.json')).write_text(json.dumps(todo,indent=2)+'\n');print(json.dumps(todo))
