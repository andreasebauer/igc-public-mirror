"""Pinned, read-only routing for the completed bounded J3 release.

No network, generation, repair, root relabelling or implicit target permutation.
Archives are checked on first access; coverage reports describe catalog scope.
"""
from pathlib import Path
from itertools import permutations
from contextlib import ExitStack
import hashlib, json, re, stat, zipfile
from .master_reader import MasterReader, _open_regular, _digest_file, _copy
from .support import canonical_bytes, strict_loads
from .prefix_reader import check, integer
from .validate_permutation import validate_row

CATALOG_SHA256='4753cee3dd1adb0b963f10ae11d98fb0c35640b43f3706f1d90b836150458102'
ORDERS=tuple(permutations(range(3)))
POPULATION=50116

class MissingDataError(ValueError): pass

class UnifiedMasterReader:
 def __init__(self, directory, *, canonical_directory, archive_directory=None):
  self._closed=False; self._stack=ExitStack(); self._cache=None
  self._directory=Path(directory)
  self._archive_directory=Path(archive_directory or self._directory/'archives')
  try:
   with _open_regular(self._directory/'CATALOG_0139.json') as f: raw=f.read(1048577)
   check(len(raw)<=1048576 and hashlib.sha256(raw).hexdigest()==CATALOG_SHA256,'ADMISSION_CATALOG_DIGEST')
   self._catalog=strict_loads(raw,max_bytes=1048576,max_nodes=100000)
   check(self._catalog['release_id']=='MASTER_DATA_V1_0139' and len(self._catalog['slices'])==139,'ADMISSION_RELEASE')
   self._canonical=self._stack.enter_context(MasterReader(canonical_directory))
   check(self._catalog['slices'][:14]==self._canonical._catalog['slices'],'CANONICAL_ADMISSION_PREFIX')
   self._routes={p:[None]*POPULATION for p in ORDERS[1:]}
   for n,s in enumerate(self._catalog['slices'][14:],14):
    check(s['format']=='IG_PERMUTATION_J3_SCIENTIFIC_EXPORT_V1','SLICE_FORMAT')
    p=self._order(s['scope']['ordered_roots']);check(p!=(0,1,2),'DUPLICATE_CANONICAL_SCOPE')
    lo,hi=s['scope']['exact_tids'];integer(lo,0,POPULATION);integer(hi,lo,POPULATION)
    check(s['counts']['carriers']==hi-lo+1,'SLICE_DECLARED_CENSUS')
    a=s['archive'];check(Path(a['file_name']).name==a['file_name'],'UNSAFE_ARCHIVE_NAME')
    for t in range(lo,hi+1):
     check(self._routes[p][t] is None,'DUPLICATE_MASTER_TID');self._routes[p][t]=n
   check(all(all(n is not None for n in r) for r in self._routes.values()),'MASTER_COVERAGE_GAP')
  except BaseException:
   self.close();raise
 def _ready(self):check(not self._closed,'READER_CLOSED')
 @staticmethod
 def _order(p):
  check(type(p) in (list,tuple) and len(p)==3 and all(type(v) is int for v in p) and tuple(p) in ORDERS,'ROOT_ORDER_UNAVAILABLE')
  return tuple(p)
 def close(self):self._closed=True;self._cache=None;self._stack.close()
 def __enter__(self):self._ready();return self
 def __exit__(self,*exc):self.close()
 def coverage_report(self):
  self._ready()
  return {'status':'PINNED_CATALOG_COVERAGE_PASS','release_id':'MASTER_DATA_V1_0139','catalog_sha256':CATALOG_SHA256,'scientific_slices':139,'carriers':300696,'root_orders':[list(p) for p in ORDERS],'carriers_per_order':50116,'missing_catalog_routes':0,'archive_integrity':'CANONICAL_14_CHECKED_ON_OPEN;_OTHER_SLICES_CHECKED_ON_FIRST_ACCESS','generator_calls':0,'full_L0_to_G8_complete':False}
 def _slice(self,n):
  self._ready()
  if self._cache and self._cache[0]==n:return self._cache[1]
  s=self._catalog['slices'][n];a=s['archive'];path=self._archive_directory/a['file_name']
  try:f=_open_regular(path)
  except FileNotFoundError:raise MissingDataError('MISSING_ADMITTED_ARCHIVE:'+a['file_name']) from None
  with f:
   check(_digest_file(f)==a['sha256'] and f.seek(0,2)==a['size_bytes'],'ADMITTED_ARCHIVE_DIGEST');f.seek(0)
   with zipfile.ZipFile(f) as z:
    infos=z.infolist();names=[i.filename for i in infos]
    check(len(names)==len(set(names))==s['counts']['scientific_files'],'ARCHIVE_MEMBER_CLOSURE')
    check(sum(i.file_size for i in infos)==s['counts']['scientific_bytes'],'ARCHIVE_SCIENTIFIC_SIZE')
    for i in infos:
     check(i.filename=='ROOT.json' or re.fullmatch(r'content/[0-9a-f]{64}\.blob',i.filename),'UNSAFE_ARCHIVE_MEMBER')
     check(not i.is_dir() and not stat.S_ISLNK(i.external_attr>>16) and not i.flag_bits&1 and 0<i.file_size<=4194304,'UNSAFE_ARCHIVE_MEMBER_TYPE_OR_BOUND')
    raw=z.read('ROOT.json');check(hashlib.sha256(raw).hexdigest()==s['scientific_root_sha256'],'SCIENTIFIC_ROOT_DIGEST')
    root=strict_loads(raw,max_bytes=65536,max_nodes=10000)
    check(root['schema_id']==s['format'] and root['dataset_id']==s['dataset_id'],'SCIENTIFIC_ROOT_IDENTITY')
    p=s['scope']['ordered_roots'];lo,hi=s['scope']['exact_tids']
    check(root['scope']=={'ordered_roots':p,'exact_tids':[lo,hi]},'SCIENTIFIC_SLICE_SCOPE')
    check(root['foundation_root_sha256']==self._catalog['slices'][0]['scientific_root_sha256'],'FOUNDATION_ROOT')
    check(root['foundation_catalog_sha256']==self._canonical.foundation_catalog_sha256() and re.fullmatch(r'[0-9a-f]{64}',root['accepted_catalog_sha256']) is not None,'FOUNDATION_CATALOG')
    check(root['counts']=={k:s['counts'][k] for k in ('carriers','events','option_realizations')},'ADMITTED_COUNTS')
    rows={};refs=set();events=witnesses=0
    for ref in root['shards']:
     sha=ref['sha256'];check(sha not in refs,'DUPLICATE_SCIENTIFIC_SHARD');refs.add(sha)
     raw=z.read('content/'+sha+'.blob');check(len(raw)==ref['size_bytes'] and hashlib.sha256(raw).hexdigest()==sha,'SCIENTIFIC_SHARD_DIGEST')
     data=strict_loads(raw,max_bytes=4194304,max_nodes=4194304)
     check(canonical_bytes(data)==raw,'NONCANONICAL_SHARD')
     check(set(data)=={'schema_id','parent_root','catalog_sha256','accepted_catalog_sha256','ordered_roots','rows'} and data['schema_id']=='IG_NATIVE_PERMUTATION_LABELLED_J3_ROWS_V1','SHARD_FIELDS')
     check(data['parent_root']==root['foundation_root_sha256'] and data['catalog_sha256']==root['foundation_catalog_sha256'] and data['accepted_catalog_sha256']==root['accepted_catalog_sha256'] and data['ordered_roots']==p,'SHARD_SCOPE')
     tids=[]
     for row in data['rows']:
      t=row['exact_tid'];integer(t,lo,hi+1);check(t not in rows and row['ordered_roots']==p,'ROW_CENSUS');rows[t]=row;tids.append(t)
      events+=len(row['events']);witnesses+=sum(len(e['realizations']) for e in row['events'])
     check(tids==sorted(tids),'ROW_ORDER')
    check(set(names)=={'ROOT.json'}|{'content/'+h+'.blob' for h in refs},'SCIENTIFIC_CONTENT_CLOSURE')
    check(set(rows)==set(range(lo,hi+1)) and events==s['counts']['events'] and witnesses==s['counts']['option_realizations'],'ACTUAL_SLICE_CENSUS')
  self._cache=(n,rows);return rows
 def _provenance(self,n):
  s=self._catalog['slices'][n]
  return {'release_id':'MASTER_DATA_V1_0139','admission_catalog_sha256':CATALOG_SHA256,'scientific_root_sha256':s['scientific_root_sha256'],'dataset_id':s['dataset_id'],'native_format':s['format'],'identity_profile':s['identity_profile'],'archive_sha256':s['archive']['sha256']}
 def lookup_j3(self,tid,*,ordered_roots=(0,1,2)):
  self._ready();integer(tid,0,POPULATION);p=self._order(ordered_roots)
  if p==(0,1,2):return self._canonical.lookup_j3(tid)
  n=self._routes[p][tid];row=self._slice(n)[tid]
  validate_row(self._canonical._parent,row,list(p))
  return _copy({'carrier':row,'provenance':self._provenance(n)})
 def iter_j3(self,start=0,stop=POPULATION,*,ordered_roots=(0,1,2)):
  self._ready();integer(start,0,POPULATION+1);integer(stop,start,POPULATION+1);p=self._order(ordered_roots)
  for t in range(start,stop):yield self.lookup_j3(t,ordered_roots=p)
 def primitive(self,sid):self._ready();return self._canonical.primitive(sid)
 def triple(self,tid):self._ready();return self._canonical.triple(tid)
 def lookup_abc(self,object_id,*,catalog_sha256):self._ready();return self._canonical.lookup_abc(object_id,catalog_sha256=catalog_sha256)
 def foundation_catalog_sha256(self):self._ready();return self._canonical.foundation_catalog_sha256()
 def lookup_target(self,tid,event_id,option_indices,*,ordered_roots=(0,1,2)):
  source=self.lookup_j3(tid,ordered_roots=ordered_roots);check(type(option_indices) in (list,tuple) and len(option_indices)==3,'REALIZATION_CHOICE_SHAPE')
  for v in option_indices:integer(v)
  matches=[w for e in source['carrier']['events'] if e['event_id']==event_id for w in e['realizations'] if w['option_indices']==list(option_indices)]
  check(len(matches)==1,'REALIZATION_NOT_FOUND');w=matches[0]
  return _copy({'source_j3_id':source['carrier']['j3_id'],'event_id':event_id,'realization':w,'target':self.lookup_j3(w['target_tid']),'target_root_order':[0,1,2],'target_order_semantics':'target_sids retains witness order; target_tid addresses the sorted canonical census; no permutation label inferred'})
 def verify(self,*,full_semantics=False,progress=None):
  """Exhaust stored content. Optional semantics exhausts existing validators."""
  self._ready();reports=[]
  canonical=self._canonical.verify(progress=progress) if full_semantics else self._canonical.integrity_report()
  for n,s in enumerate(self._catalog['slices'][14:],14):
   rows=self._slice(n)
   if full_semantics:
    for row in rows.values():validate_row(self._canonical._parent,row,s['scope']['ordered_roots'])
   report={'dataset_id':s['dataset_id'],'carriers':len(rows),'status':'FULL_SEMANTICS_PASS' if full_semantics else 'ALL_BYTES_AND_ROW_CENSUS_PASS'};reports.append(report)
   if progress:progress(_copy(report))
  return {'status':'UNIFIED_FULL_SEMANTICS_PASS' if full_semantics else 'UNIFIED_ARCHIVE_INTEGRITY_AND_CENSUS_PASS','coverage':self.coverage_report(),'canonical':canonical,'permutation_slices':reports,'full_semantic_validation':full_semantics}
