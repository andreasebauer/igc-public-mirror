from pathlib import Path
from types import SimpleNamespace
import sys,json,time,resource,sqlite3
h=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(h/'source'))
from infinity_grid.bulk_mapping_store import MappingStore
from infinity_grid.canon import canonical_sha256 as sha
from infinity_grid.structural_encoding import structural_canonical_bytes as enc
reference=json.loads((h/'inputs/reference_full_witnesses.json').read_text())[0]
count=16384;tasks=[SimpleNamespace(task_id=f't{i:05d}',binding_sha256='a'*64,payload={'descriptors':[{'ordinal':j,'edge':reference['raw_edge'],'literal_descriptor_sha256':'d'*64} for j in range(i,i+4)]}) for i in range(0,count,4)]
p=h/'results/STORAGE_WINDOW.sqlite3';p.unlink(missing_ok=True);start=time.monotonic();s=MappingStore(p,{'scope':'SYNTHETIC_STORAGE_ONLY_REPEAT_ONE_REFERENCE','rows':count},tasks)
for t in reversed(tasks):
 rows=[dict(reference,ordinal=d['ordinal'],literal_descriptor_sha256=d['literal_descriptor_sha256']) for d in t.payload['descriptors']]
 signature={'schema':'IG_HET4_FULL_WITNESS_BATCH_V1','rows':rows};s.commit(t.task_id,{'signature_canonical_bytes':enc(signature),'signature_sha256':sha(signature),'outcome_count':4})
commit=time.monotonic()-start;start=time.monotonic();summary=s.summary();summary_seconds=time.monotonic()-start
query=s.conn.execute('explain query plan select ordinal from rows where task_id=? order by ordinal',(tasks[0].task_id,)).fetchall();s.conn.execute('pragma wal_checkpoint(TRUNCATE)');s.close();start=time.monotonic();restored=MappingStore(p,{'scope':'SYNTHETIC_STORAGE_ONLY_REPEAT_ONE_REFERENCE','rows':count},tasks);assert not restored.pending();restored.close()
out={'status':'PASS','scope':'Synthetic repeated reference rows; no new scientific canonicalization or census coverage','rows':count,'tasks':len(tasks),'commit_seconds':commit,'summary_seconds':summary_seconds,'restore_verify_seconds':time.monotonic()-start,'database_bytes':p.stat().st_size,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'task_query_plan':query,'summary':summary};(h/'results/STORAGE_WINDOW.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
