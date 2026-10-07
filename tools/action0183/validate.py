"""Bounded registration tests; full saved-stream execution is not started."""
from pathlib import Path
import gzip,hashlib,json,sqlite3,tempfile
from project.indexer import index_streams
from project.reader import SavedRecordReader
B=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def validate():
 spec=json.loads((B/'SPEC.json').read_text());reg=json.loads((B/'PROJECT_REGISTRATION.json').read_text())
 assert sha((B/'SPEC.json').read_bytes())==reg['spec_sha256'] and reg['native_execution_started'] is False
 for n,pin in reg['project_code'].items():assert sha((B/n).read_bytes())==pin['sha256']
 for n,pin in json.loads((B/'INPUT_PINS.json').read_text()).items():assert sha((B/'inputs'/n).read_bytes())==pin['sha256']
 preview=json.loads((B/'inputs/FIRST_ROW_BINDING_PREVIEW.json').read_text());pop=json.loads((B/'inputs/G2_S0_INTERFACE_POPULATION.json').read_text());cont=json.loads((B/'inputs/POST_RESERVATION_CONTINUATION_HASHES.json').read_text())
 rb=(json.dumps(preview['records']['payload'],sort_keys=True,separators=(',',':'))+'\n').encode();ab=(json.dumps(preview['audits']['payload'],sort_keys=True,separators=(',',':'))+'\n').encode()
 assert sha(rb)==preview['records']['raw_line_sha256'] and sha(ab)==preview['audits']['raw_line_sha256']
 def rejects(call):
  try:call()
  except (ValueError,KeyError,sqlite3.IntegrityError):return
  raise AssertionError('Invalid route/input accepted')
 def fixture(d,records,audits):
  d.mkdir()
  for n,b in [('records.jsonl.gz',records),('audits.jsonl.gz',audits)]:
   with gzip.GzipFile(filename=str(d/n),mode='wb',mtime=0) as f:f.write(b)
  return {'record_uncompressed_stream_sha256':sha(records),'audit_uncompressed_stream_sha256':sha(audits),'record_file_sha256':sha((d/'records.jsonl.gz').read_bytes()),'audit_file_sha256':sha((d/'audits.jsonl.gz').read_bytes())}
 with tempfile.TemporaryDirectory() as td:
  d=Path(td)/'single';summary=fixture(d,rb,ab);m=index_streams(d,pop,cont,summary,{'record_rows':1,'outcome_classes':1,'collision_classes':0,'maximum_class_size':1},strict_census=False)
  r=SavedRecordReader(d);row=preview['records']['payload'];ref=row['left_carrier_ref'];outcome=row['outcome_science_sha256']
  assert r.raw_record_by_ordinal(0)==rb and r.raw_audit_by_ordinal(0)==ab and r.record_by_pair_operator(ref,ref,row['connection_operator_ref'])==row
  assert r.outcome_members(outcome)==[0]
  for n in [True,-1,1,1.0,'0',None]:rejects(lambda n=n:r.record_by_ordinal(n))
  for call in [lambda:r.record_by_pair_operator(ref,ref,'missing'),lambda:r.outcome_members('missing'),lambda:r.q2_payload(0),lambda:r.realized_carrier(0)]:rejects(call)
  r.close();rejects(lambda:r.record_by_ordinal(0));rejects(lambda:r.outcome_members(outcome))
  duplicate=Path(td)/'duplicate';summary=fixture(duplicate,rb+rb,ab+ab);rejects(lambda:index_streams(duplicate,pop,cont,summary,{'record_rows':2,'outcome_classes':1,'collision_classes':1,'maximum_class_size':2},strict_census=False));assert not (duplicate/'INDEX_MANIFEST.json').exists()
  mismatch=Path(td)/'mismatch';bad=json.loads(ab);bad['outcome_science_sha256']='0'*64;summary=fixture(mismatch,rb,(json.dumps(bad)+'\n').encode());rejects(lambda:index_streams(mismatch,pop,cont,summary,{'record_rows':1,'outcome_classes':1,'collision_classes':0,'maximum_class_size':1},strict_census=False))
  alignment=Path(td)/'alignment';summary=fixture(alignment,rb,b'');rejects(lambda:index_streams(alignment,pop,cont,summary,{'record_rows':1,'outcome_classes':1,'collision_classes':0,'maximum_class_size':1},strict_census=False))
  # Synthetic SQL-only fixture tests nonunique class membership, never scientific data.
  db=sqlite3.connect(d/'index.sqlite');original=db.execute('SELECT * FROM records').fetchone()
  for n in [1,2]:
   x=list(original);x[0]=n;x[3]='SYNTHETIC_ROUTE_TEST:'+str(n);db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',x)
  db.commit();db.close();m['counts']['record_rows']=3;m['files']['index.sqlite']={'sha256':sha((d/'index.sqlite').read_bytes()),'bytes':(d/'index.sqlite').stat().st_size};(d/'INDEX_MANIFEST.json').write_text(json.dumps(m));r=SavedRecordReader(d);assert r.outcome_members(outcome)==[0,1,2];r.close()
 return {'status':'PASS_REGISTERED_READER_INDEX_PREFLIGHT','unit_fixture_records':1,'synthetic_class_membership_test':3,'negative_checks':['duplicate_pair_operator_key','audit_outcome_mismatch','unequal_stream_length','invalid_ordinal','unknown_pair_operator','missing_outcome','Q2_payload','exact_carrier','closed_reader'],'original_raw_line_roundtrip':True,'full_580351_row_index_built':False,'native_handler_called':False,'native_capture_completed':False,'generation_calls':0,'new_admissions':0,'master_release':'MASTER_DATA_V1_0150','scientific_slices':150,'next_scope':'WP6_NATIVE_SAVED_G2_S1_INDEX_EXPORT_CAPTURE_AND_EXECUTION'}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
