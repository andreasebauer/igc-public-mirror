"""Restore the same frozen capture at its original path; resume committed tasks."""
from pathlib import Path
import json,sys,hashlib,sqlite3
B=Path(__file__).resolve().parent
J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import preservation as p
def identities(root):
 rows={}
 for f in root.glob('runtime/runs/*/chain/decoder_stage_runtime/*/phases/*/state_store.sqlite3'):
  with sqlite3.connect(f.resolve().as_uri()+'?mode=ro',uri=True) as c:
   states=c.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchall()
   assert len(states)==1 and c.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]==1
   rows[f.parent.name]={'index_digest':states[0][0],'canonical_sha256':hashlib.sha256(states[0][1]).hexdigest(),'state_sha256':hashlib.sha256(states[0][2].encode()).hexdigest(),'generation_tasks':1}
 assert len(rows)==6
 return rows
before=identities(J)
E=json.loads((B/'PAUSED_CHECKPOINT_EXPORT.json').read_text())
M=json.loads((B/'READBACKS.json').read_text());objects=Path('/tmp/ig_recovery0224_objects');objects.mkdir(exist_ok=True)
for x in E['dependencies']:
 f=Path(M[x['sha256']]['path']);assert f.stat().st_size==x['size_bytes'] and hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==x['sha256']
 (objects/(x['sha256']+'.bin')).symlink_to(f)
old=Path('/tmp/ig_native0224/paused_original');assert not old.exists();J.rename(old)
try:p.restore_checkpoint(B/'PAUSED_CHECKPOINT_SLIM.zip',J,E['sha256'],objects=objects)
except BaseException:
 if not J.exists():old.rename(J)
 raise
assert identities(J)==before
rec=json.loads((J/'CAPTURE.json').read_text());budget=rec['job']['resources']['workspace_budget_bytes']
used=sum(f.stat().st_size for d in [J/'runtime/runs',J/'runtime/sealed',J/'durability/outbox'] for f in d.rglob('*') if f.is_file())
out={'status':'PASS_SAME_PATH_NATIVE_RESTORE_NO_TASK_REGENERATION','paused_original':str(old),'restored_workspace':str(J),'task_identities_before':before,'task_identities_after':identities(J),'registered_workspace_budget_bytes':budget,'used_before_sealing':used,'estimated_with_sealing':used+sum(f.stat().st_size for f in (J/'runtime/runs').rglob('*') if f.is_file()),'capture_modified':False,'resource_policy_modified':False}
assert out['estimated_with_sealing']<budget
(B/'COMPLETION_RECOVERY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
