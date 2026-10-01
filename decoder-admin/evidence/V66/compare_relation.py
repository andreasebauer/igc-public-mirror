from pathlib import Path
import json,sqlite3,hashlib
D=Path(__file__).resolve().parent;B=D.parent
roots=[]
for p in [B/'v65_stage_one',D]:
 w=Path(json.loads((p/'CAPTURE_SAVE_STATUS.json').read_text())['workspace']);roots.append(w)
def relation(w):
 dbs=list((w/'runtime/sealed').rglob('partition.sqlite3'));assert len(dbs)==1
 with sqlite3.connect(dbs[0].as_uri()+'?mode=ro&immutable=1',uri=True) as c:rows=list(c.execute('SELECT t.task_id,c.representative_signature_bytes FROM task_results t JOIN classes c ON t.class_token=c.class_token ORDER BY t.task_id'))
 return rows
one,four=map(relation,roots);assert one==four and len(four)==48
meta=[json.loads(p.read_text())['execution'] for p in (roots[1]/'runtime/sealed').rglob('SUMMARY.json')]
assert any(m.get('workers')==4 and m.get('backend')=='LOCAL_PROCESS_POOL' for m in meta),meta
out={'status':'PASS','exact_relation_equal':True,'rows':48,'v65_rerun':False,'execution':meta,'comparison':'task_id and representative_signature_bytes exact equality'}
(D/'EXACT_RELATION_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
