"""Read original gzip bytes through a collision-preserving SQLite index."""
from pathlib import Path
import gzip,hashlib,json,sqlite3

def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

class SavedRecordReader:
 def __init__(self,directory):
  self.root=Path(directory).resolve();self.closed=False;self.db=None
  try:
   self.manifest=json.loads((self.root/'INDEX_MANIFEST.json').read_bytes())
   if self.manifest['authority']!='SAVED_REPAIRED_PAIR_RECORDS_AND_PAIRED_STORED_OBSERVER_SIGNATURES_ONLY':raise ValueError('AUTHORITY')
   for name in ['index.sqlite','records.jsonl.gz','audits.jsonl.gz']:
    pin=self.manifest['files'][name]
    if sha_file(self.root/name)!=pin['sha256'] or (self.root/name).stat().st_size!=pin['bytes']:raise ValueError('FILE_HASH')
   self.db=sqlite3.connect((self.root/'index.sqlite').as_uri()+'?mode=ro',uri=True)
   self.db.execute('PRAGMA query_only=ON')
  except BaseException:self.close();raise
 def _open(self):
  if self.closed:raise ValueError('CLOSED_READER')
 def _ordinal(self,n):
  self._open()
  if type(n)!=int or not 0<=n<self.manifest['counts']['record_rows']:raise ValueError('ORDINAL')
  row=self.db.execute('SELECT * FROM records WHERE ordinal=?',(n,)).fetchone()
  if row is None:raise KeyError(n)
  return row
 def _line(self,row,audit=False):
  offset,length,digest=(row[7],row[9],row[11]) if audit else (row[6],row[8],row[10])
  with gzip.open(self.root/('audits.jsonl.gz' if audit else 'records.jsonl.gz'),'rb') as f:f.seek(offset);b=f.read(length)
  if hashlib.sha256(b).digest()!=digest:raise ValueError('ROW_HASH')
  d=json.loads(b)
  if bytes.fromhex(d['outcome_science_sha256'])!=row[4]:raise ValueError('ROW_OUTCOME')
  return b
 def raw_record_by_ordinal(self,n):return self._line(self._ordinal(n))
 def record_by_ordinal(self,n):return json.loads(self.raw_record_by_ordinal(n))
 def raw_audit_by_ordinal(self,n):return self._line(self._ordinal(n),True)
 def stored_audit_by_ordinal(self,n):return json.loads(self.raw_audit_by_ordinal(n))
 def record_by_pair_operator(self,left,right,operator):
  self._open()
  if type(left)!=str or type(right)!=str or left>right:raise ValueError('CANONICAL_INPUT_ORDER')
  try:l,r=bytes.fromhex(left),bytes.fromhex(right)
  except ValueError:raise KeyError((left,right,operator))
  row=self.db.execute('SELECT ordinal FROM records WHERE left_ref=? AND right_ref=? AND operator=?',(l,r,operator)).fetchone()
  if row is None:raise KeyError((left,right,operator))
  return self.record_by_ordinal(row[0])
 def outcome_members(self,digest):
  self._open()
  try:h=bytes.fromhex(digest)
  except (ValueError,TypeError):raise KeyError(digest)
  rows=self.db.execute('SELECT ordinal FROM records WHERE outcome=? ORDER BY ordinal',(h,)).fetchall()
  if not rows:raise KeyError(digest)
  return [r[0] for r in rows]
 def coverage_and_provenance(self):self._open();return json.loads(json.dumps(self.manifest))
 def q2_payload(self,*a):self._open();raise ValueError('Q2_PAYLOAD_NOT_SERIALIZED')
 def realized_carrier(self,*a):self._open();raise ValueError('EXACT_CARRIER_UNAVAILABLE')
 def close(self):
  self.closed=True
  if self.db is not None:self.db.close();self.db=None
