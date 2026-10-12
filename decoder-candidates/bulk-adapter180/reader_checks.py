from pathlib import Path
import sys,sqlite3,tempfile,json,hashlib
HERE=Path(__file__).resolve().parent
p=json.loads((HERE/'results/CAPTURE_PENDING.json').read_text());sys.path.insert(0,p['resume_workspace']+'/source')
from infinity_grid.v05_stage_runtime import StageScienceRuntime,StageRuntimeError
from infinity_grid.structural_encoding import structural_canonical_bytes
# Isolated read-only API unit check; synthetic capability, never dispatches science.
with tempfile.TemporaryDirectory(prefix='adapter180-reader-') as tmp:
 root=Path(tmp);db=root/'partition.sqlite3';c=sqlite3.connect(db)
 c.execute('create table task_results (task_id text,signature_sha256 text,class_token text)')
 c.execute('create table classes (class_token text,representative_signature_bytes blob)')
 raw=structural_canonical_bytes({'rows':[{'ordinal':1,'witness':{'full':[1,2,3]}}]});digest=hashlib.sha256(raw).hexdigest()
 c.executemany('insert into task_results values (?,?,?)',[('z',digest,'one'),('a',digest,'one')]);c.execute('insert into classes values (?,?)',('one',raw));c.commit()
 runtime=object.__new__(StageScienceRuntime);runtime._require_execution=lambda:None;runtime._phase_root=lambda phase:root
 assert [r['task_id'] for r in runtime.iter_partition_signatures('test')]==['a','z']
 for invalid,reason in [(None,'PARTITION_EXACT_BYTES_MISSING'),(b'changed','PARTITION_SIGNATURE_HASH_MISMATCH')]:
  c.execute('update classes set representative_signature_bytes=?',(invalid,));c.commit()
  try:list(runtime.iter_partition_signatures('test'))
  except StageRuntimeError as exc:assert str(exc)==reason
  else:raise AssertionError('invalid stored bytes admitted')
 c.close()
(HERE/'results/READER_CHECKS.json').write_text(json.dumps({'status':'PASS','checks':['ordered exact bytes','missing full bytes refused','corrupt full bytes refused'],'scope':'isolated read-only API unit checks; no scientific authority claim'},indent=2)+'\n')
print('3 reader checks PASS')
