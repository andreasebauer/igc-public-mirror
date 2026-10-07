import json
from pathlib import Path
def census(reader):
 db=reader.saved.db
 if db.execute('PRAGMA quick_check').fetchone()!=('ok',):raise ValueError('SQLITE_INTEGRITY')
 count,lo,hi=db.execute('SELECT count(*),min(ordinal),max(ordinal) FROM records').fetchone()
 if (count,lo,hi)!=(580351,0,580350):raise ValueError('ORDINAL_CENSUS')
 if db.execute('SELECT count(*) FROM records WHERE left_ref>right_ref').fetchone()[0]:raise ValueError('PAIR_ORDER')
 if db.execute('SELECT count(*) FROM (SELECT left_ref,right_ref,operator FROM records GROUP BY left_ref,right_ref,operator)').fetchone()[0]!=count:raise ValueError('PAIR_OPERATOR_KEYS')
 classes=collisions=maxsize=splits=0
 for h,n,s in db.execute('SELECT outcome,count(*),count(DISTINCT observer) FROM records GROUP BY outcome'):
  members=reader.g2_s1_outcome_members(h.hex())
  if len(members)!=n or members!=sorted(set(members)):raise ValueError('CLASS_ROUTE')
  classes+=1;collisions+=n>1;maxsize=max(maxsize,n);splits+=s>1
 if (classes,collisions,maxsize,splits)!=(576785,3554,3,0):raise ValueError('COLLISION_CENSUS')
 refs=db.execute('SELECT count(*) FROM (SELECT left_ref FROM records UNION SELECT right_ref FROM records)').fetchone()[0]
 ops=db.execute('SELECT count(DISTINCT operator) FROM records').fetchone()[0]
 pairs=db.execute('SELECT count(*) FROM (SELECT left_ref,right_ref FROM records GROUP BY left_ref,right_ref)').fetchone()[0]
 if (refs,ops,pairs)!=(193,31,18721):raise ValueError('INPUT_CENSUS')
 if db.execute('SELECT count(*) FROM (SELECT left_ref,right_ref FROM records GROUP BY left_ref,right_ref HAVING count(*)!=31)').fetchone()[0]:raise ValueError('PAIR_OPERATOR_COMPLETENESS')
 return dict(record_rows_checked=count,outcome_classes_checked=classes,collision_classes_checked=collisions,maximum_class_size=maxsize,stored_observer_split_classes=splits)

def bounded_routes(r,i):
 row=r.g2_s1_record(0);audit=r.g2_s1_audit(0)
 if json.loads(r.g2_s1_raw_record(0))!=row or json.loads(r.g2_s1_raw_audit(0))!=audit:raise ValueError('RAW_ROUNDTRIP')
 if r.g2_s1_pair_operator(row['left_carrier_ref'],row['right_carrier_ref'],row['connection_operator_ref'])!=row:raise ValueError('PAIR_ROUTE')
 if 0 not in r.g2_s1_outcome_members(row['outcome_science_sha256']):raise ValueError('MEMBERSHIP')
 row['outcome_science_sha256']='mutated'
 if r.g2_s1_record(0)['outcome_science_sha256']=='mutated':raise ValueError('COPY_ISOLATION')
 def rejects(call):
  try:call()
  except (ValueError,KeyError):return
  raise ValueError('INVALID_ROUTE_ACCEPTED')
 for n in [True,-1,580351,1.0,'0',None]:rejects(lambda n=n:r.g2_s1_record(n))
 for call in [lambda:r.g2_s1_q2_payload(0),lambda:r.g2_s1_realized_carrier(0),lambda:r.g2_s1_outcome_members('missing'),lambda:r.lookup_g1_public_interface('missing')]:rejects(call)
 pop=json.loads(Path(i['g1_population']).read_bytes())['interfaces']
 for row in pop:
  if r.lookup_g1_public_interface(row['carrier_ref'])!=row:raise ValueError('G1_DELEGATION')
 old=0
 for key,row in r.previous.previous.additional.records.items():
  if r.lookup_o7(*key)!=r.previous.lookup_o7(*key):raise ValueError('O7_DELEGATION')
  old+=1
 if old!=24:raise ValueError('O7_CENSUS')
 return {'prior_g1_routes_checked':len(pop),'prior_o7_routes_checked':old,'negative_routes_checked':10,'raw_record_and_audit_roundtrip':True}
