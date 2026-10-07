"""Index existing saved JSONL bytes, never realize or construct carriers."""
from pathlib import Path
import collections,gzip,hashlib,json,sqlite3
from itertools import zip_longest
from .reader import sha_file

def semantic(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def index_streams(directory,population,continuations,summary,expected,*,strict_census=True):
 d=Path(directory);target=d/'index.sqlite'
 if target.exists() or (d/'INDEX_MANIFEST.json').exists():raise ValueError('EXISTING_OUTPUT')
 by={r['carrier_ref']:r for r in population['interfaces']};cont=continuations['rows'];R=hashlib.sha256();A=hashlib.sha256();ro=ao=0;operators=collections.Counter();pairs=set();count=0
 db=sqlite3.connect(target)
 try:
  db.execute('CREATE TABLE records(ordinal INTEGER PRIMARY KEY,left_ref BLOB NOT NULL,right_ref BLOB NOT NULL,operator TEXT NOT NULL,outcome BLOB NOT NULL,observer BLOB NOT NULL,roff INTEGER NOT NULL,aoff INTEGER NOT NULL,rlen INTEGER NOT NULL,alen INTEGER NOT NULL,rhash BLOB NOT NULL,ahash BLOB NOT NULL,UNIQUE(left_ref,right_ref,operator))')
  with gzip.open(d/'records.jsonl.gz','rb') as rf,gzip.open(d/'audits.jsonl.gz','rb') as af:
   for rb,ab in zip_longest(rf,af):
    if rb is None or ab is None:raise ValueError('AUDIT_ALIGNMENT_LENGTH')
    r=json.loads(rb);a=json.loads(ab);left,right=r['left_carrier_ref'],r['right_carrier_ref'];op=r['connection_operator_ref']
    prefix,types=op.rsplit(':',1);t,u=map(int,types.split('>'))
    if prefix!='G1_PUBLIC_BRIDGE_RELATION_V1' or not 0<=t<7 or not 0<=u<7 or left>right:raise ValueError('OPERATOR_ORDER')
    for side,ref,v in [('left',left,t),('right',right,u)]:
     if r[side+'_interface_sha256']!=by[ref]['interface_sha256'] or r['d4_'+side+'_post_reservation_public_continuation_sha256']!=cont[ref][str(v)]:raise ValueError('G1_INPUT_BINDING')
    if r['legality']!='LEGAL' or r['realization_status']!='PASS' or r['q2_projection_serialized'] is not False:raise ValueError('STORED_STATUS')
    payload={'schema_id':'IG_G_UPLIFT_S1_REPAIRED_PAIR_OUTCOME_SEMANTICS_V2','projected_outcome_science_sha256':r['projected_outcome_science_sha256'],'connection_operator_ref':op,'left_post_reservation_public_continuation_sha256':r['d4_left_post_reservation_public_continuation_sha256'],'right_post_reservation_public_continuation_sha256':r['d4_right_post_reservation_public_continuation_sha256'],'q2_one_reservation_successor_skins_sha256':r['q2_one_reservation_successor_skins_sha256'],'repair_basis':'D4_STRICT_PUBLIC_PRECURSOR_PLUS_Q2_STRICT_PUBLIC_RESIDUAL_WINNER','observer_scope':'ONE_STEP_REALIZED_PUBLIC_CONGRUENCE_CANDIDATE'}
    h=semantic(payload)
    if h!=r['outcome_science_sha256'] or h!=a['outcome_science_sha256'] or r['outcome_ref']!='G2S1O2:'+h:raise ValueError('OUTCOME_OR_AUDIT_HASH')
    db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(count,bytes.fromhex(left),bytes.fromhex(right),op,bytes.fromhex(h),bytes.fromhex(a['realized_public_sha256']),ro,ao,len(rb),len(ab),hashlib.sha256(rb).digest(),hashlib.sha256(ab).digest()))
    R.update(rb);A.update(ab);ro+=len(rb);ao+=len(ab);count+=1;operators[op]+=1;pairs.add((left,right))
    if count%10000==0:db.commit()
  db.commit();db.execute('CREATE INDEX outcome_members ON records(outcome,ordinal)');db.commit()
  classes=collision=maximum=splits=0
  for size,observers in db.execute('SELECT COUNT(*),COUNT(DISTINCT observer) FROM records GROUP BY outcome'):
   classes+=1;collision+=size>1;maximum=max(maximum,size);splits+=observers>1
  if count!=expected['record_rows'] or classes!=expected['outcome_classes'] or collision!=expected['collision_classes'] or maximum!=expected['maximum_class_size'] or splits:raise ValueError('CLASS_CENSUS')
  if R.hexdigest()!=summary['record_uncompressed_stream_sha256'] or A.hexdigest()!=summary['audit_uncompressed_stream_sha256']:raise ValueError('STREAM_HASH')
  if strict_census:
   if len(by)!=193 or len(pairs)!=18721 or len(operators)!=31 or set(operators.values())!={18721}:raise ValueError('PAIR_OPERATOR_CENSUS')
   refs=sorted(by)
   if pairs!={(a,b) for j,a in enumerate(refs) for b in refs[j:]}:raise ValueError('PAIR_COVERAGE')
   ops=sorted(tuple(map(int,k.rsplit(':',1)[1].split('>'))) for k in operators)
   if semantic([list(x) for x in ops])!='f10a566c1f8e8faf7419cca50e1ee84977c388beb9c5d2948e01290b68034e1a':raise ValueError('OPERATOR_BASIS')
  if db.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('SQLITE_INTEGRITY')
 finally:db.close()
 for name,key in [('records.jsonl.gz','record_file_sha256'),('audits.jsonl.gz','audit_file_sha256')]:
  if sha_file(d/name)!=summary[key]:raise ValueError('GZIP_HASH')
 manifest={'schema':'IG_SAVED_G2_S1_ORDINAL_INDEX_V1','authority':'SAVED_REPAIRED_PAIR_RECORDS_AND_PAIRED_STORED_OBSERVER_SIGNATURES_ONLY','counts':dict(expected,audit_signature_rows=count),'stored_observer_split_classes':splits,'record_uncompressed_sha256':R.hexdigest(),'audit_uncompressed_sha256':A.hexdigest(),'exact_G1_parent_DAG_available':False,'Q2_payload_available':False,'fresh_realization':False,'generation_calls':0,'files':{n:{'sha256':sha_file(d/n),'bytes':(d/n).stat().st_size} for n in ['index.sqlite','records.jsonl.gz','audits.jsonl.gz']}}
 (d/'INDEX_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
