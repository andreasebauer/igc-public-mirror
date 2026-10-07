from pathlib import Path
import json,shutil,zipfile
from infinity_grid.v05_chain import ChainExecutionResult
from infinity_grid.v05_origin_guard import require_registered_output
from .reader import sha_file,SavedRecordReader
from .indexer import index_streams

def handler(stage,runtime):
 i=stage['input_artifacts'];p=stage['execution']['parameters'];d=Path(runtime._chain_dir).parent/'saved_s1_export'
 require_registered_output(d,'saved byte index export');d.mkdir()
 archive=d/'source.zip'
 for name,pin in p['input_pins'].items():
  if sha_file(i[name])!=pin['sha256'] or Path(i[name]).stat().st_size!=pin['bytes']:raise ValueError('INPUT_HASH:'+name)
 with archive.open('wb') as out:
  for j in range(5):
   with Path(i['part'+str(j)]).open('rb') as f:shutil.copyfileobj(f,out)
 if sha_file(archive)!=p['archive_sha256']:raise ValueError('ARCHIVE_HASH')
 with zipfile.ZipFile(archive) as z:
  for name,member in [('records.jsonl.gz','evidence/G2_S1_REPAIRED_D4_Q2_PAIR_CONNECTION_RECORDS.jsonl.gz'),('audits.jsonl.gz','evidence/G2_S1_REPAIRED_D4_Q2_AUDIT_SIGNATURES.jsonl.gz')]:
   with z.open(member) as src,(d/name).open('wb') as dest:shutil.copyfileobj(src,dest)
 archive.unlink()
 manifest=index_streams(d,json.loads(Path(i['population']).read_bytes()),json.loads(Path(i['continuations']).read_bytes()),json.loads(Path(i['summary']).read_bytes()),p['expected_counts'])
 reader=SavedRecordReader(d)
 try:
  first=reader.record_by_ordinal(0);last=reader.record_by_ordinal(580350)
  for row,n in [(first,0),(last,580350)]:
   if reader.record_by_pair_operator(row['left_carrier_ref'],row['right_carrier_ref'],row['connection_operator_ref'])!=row or reader.stored_audit_by_ordinal(n)['outcome_science_sha256']!=row['outcome_science_sha256']:raise ValueError('ROUTE_READBACK')
  for call in [lambda:reader.record_by_ordinal(True),lambda:reader.record_by_ordinal(-1),lambda:reader.record_by_ordinal(580351),lambda:reader.q2_payload(0),lambda:reader.realized_carrier(0)]:
   try:call()
   except ValueError:pass
   else:raise ValueError('EXCLUDED_ROUTE')
 finally:reader.close()
 return ChainExecutionResult(result={'outcome':'PASS','authority':manifest['authority'],'counts':manifest['counts'],'stored_observer_split_classes':0,'generation_calls':0,'scientific_master_admission':False,'exact_G1_parent_DAG_available':False,'Q2_payload_available':False,'index_manifest_sha256':sha_file(d/'INDEX_MANIFEST.json'),'export_relative_path':'saved_s1_export'})
