"""Unified read-only access to the fourteen immutable MASTER_DATA_V1_0014 roots.

Opening pins the admitted catalog, every archive and every root. Full scientific
validation is explicit verify(); opening reports integrity of admitted bytes.
No producer, native runtime, external dependency, network or repair path exists.
"""
from pathlib import Path
from bisect import bisect_left
from contextlib import ExitStack
import hashlib,json,os,re,stat,tempfile,zipfile
from .support import canonical_bytes,strict_loads,_ref
from .prefix_contract import parse
from .prefix_reader import PrefixReader,check,fields,integer
from .validate_rows import validate_rows

CATALOG_SHA256='3c7658dc26f4ea37240b19cad2b121908a918e395f47d3fc06420094d27f7ed9'
PARENT_ROOT='e1e2327070b2f9fc3bb548fc58a6961b70a0b885eb857d040ce0dcd5bfe4a351'
SEED_TIDS={4736,4758,4769}
POPULATION=50116

def _copy(value):return json.loads(canonical_bytes(value))
def _open_regular(path):
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 f=os.fdopen(fd,'rb')
 if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
  f.close();raise ValueError('UNSAFE_REGULAR_FILE')
 return f

def _digest_file(f):
 f.seek(0);h=hashlib.sha256()
 for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 f.seek(0);return h.hexdigest()

def _order(roots):
 check(type(roots) in (list,tuple) and len(roots)==3 and all(type(x) is int for x in roots) and list(roots)==[0,1,2],'ROOT_ORDER_UNAVAILABLE')

class MasterReader:
 def __init__(self,directory,*,catalog_sha256=CATALOG_SHA256):
  self._stack=ExitStack();self._closed=False;self._cache_key=None;self._cache=None;self._scientific_report=None
  self._directory=Path(directory)
  try:
   check(self._directory.is_dir() and not self._directory.is_symlink(),'UNSAFE_MASTER_DIRECTORY')
   check(catalog_sha256==CATALOG_SHA256,'UNSUPPORTED_ADMISSION_CATALOG')
   with _open_regular(self._directory/'CATALOG.json') as f:
    raw=f.read(65537)
   check(len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==CATALOG_SHA256,'ADMISSION_CATALOG_DIGEST')
   self._catalog=strict_loads(raw,max_bytes=65536,max_nodes=10000)
   check(self._catalog['release_id']=='MASTER_DATA_V1_0014' and len(self._catalog['slices'])==14,'ADMISSION_RELEASE')
   ad=self._directory/'archives';check(ad.is_dir() and not ad.is_symlink(),'UNSAFE_ARCHIVES_DIRECTORY')
   self._archives=[];self._roots=[];self._routes=[None]*POPULATION
   for n,s in enumerate(self._catalog['slices']):
    a=s['archive'];check(Path(a['file_name']).name==a['file_name'],'UNSAFE_ARCHIVE_NAME')
    f=self._stack.enter_context(_open_regular(ad/a['file_name']))
    check(os.fstat(f.fileno()).st_size==a['size_bytes'] and _digest_file(f)==a['sha256'],'ADMITTED_ARCHIVE_DIGEST:'+a['file_name'])
    z=self._stack.enter_context(zipfile.ZipFile(f));infos=z.infolist()
    names=[i.filename for i in infos]
    check(len(names)==len(set(names))==s['counts']['scientific_files'],'ARCHIVE_MEMBER_CLOSURE')
    for i in infos:
     check(i.filename=='ROOT.json' or re.fullmatch(r'content/[0-9a-f]{64}\.blob',i.filename) is not None,'UNSAFE_ARCHIVE_MEMBER')
     check(not i.is_dir() and not stat.S_ISLNK(i.external_attr>>16) and not i.flag_bits&1,'UNSAFE_ARCHIVE_MEMBER_TYPE')
     check(0<i.file_size<=(65536 if i.filename=='ROOT.json' else 4194304),'ARCHIVE_MEMBER_BOUND')
    check(sum(i.file_size for i in infos)==s['counts']['scientific_bytes'],'ARCHIVE_SCIENTIFIC_SIZE')
    root_raw=z.read('ROOT.json');check(hashlib.sha256(root_raw).hexdigest()==s['scientific_root_sha256'],'SCIENTIFIC_ROOT_DIGEST')
    root=parse(root_raw,65536)
    check(root['schema_id']==s['format'] and root['dataset_id']==s['dataset_id'] and root['identity_profile']==s['identity_profile'],'ADMITTED_ROOT_IDENTITY')
    self._archives.append(z);self._roots.append(root)
    if n==0:
     check(s['scientific_root_sha256']==PARENT_ROOT,'PARENT_IDENTITY')
     tmp=self._stack.enter_context(tempfile.TemporaryDirectory(prefix='ig_master_seed_'))
     # Members have exact safe names, no directory nodes and bounded lengths.
     for name in names:
      p=Path(tmp)/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
     self._parent=PrefixReader(tmp,PARENT_ROOT);report=self._parent.verify()
     for key,value in s['counts'].items():check(report[key]==value,'SEED_ADMISSION_COUNTS')
     for c in self._parent.carriers.values():
      tid=c['representative_tid'];check(tid in SEED_TIDS and self._routes[tid] is None,'SEED_ROUTING')
      self._routes[tid]=(0,None,None)
     continue
    fields(root,['schema_id','dataset_id','identity_profile','parent_root','catalog_sha256','scope','counts','shards'])
    check(root['parent_root']==PARENT_ROOT and root['catalog_sha256']==self._parent.root['catalog_sha256'],'PARENT_REFERENCE_CLOSURE')
    scope=s['scope'];lo,hi=scope['exact_tids'];integer(lo,0,POPULATION);integer(hi,lo,POPULATION)
    excluded=scope.get('excluded_already_admitted_tids',[])
    expected_scope={'exact_tids':[lo,hi],'ordered_roots':[0,1,2],'all_boundary_alternatives':True,'all_option_realizations':True,'candidate_classes_are_annotations':True,'unlisted_tids':'UNAVAILABLE'}
    if excluded:expected_scope['excluded_already_admitted_tids']=excluded
    check(canonical_bytes(root['scope'])==canonical_bytes(expected_scope),'SCIENTIFIC_SLICE_SCOPE')
    check(root['counts']=={k:v for k,v in s['counts'].items() if k not in ('scientific_files','scientific_bytes')},'SLICE_ADMISSION_COUNTS')
    check(type(root['shards']) is list and 0<len(root['shards'])<=64,'SLICE_SHARD_BOUND')
    refs=set();covered=set()
    for j,item in enumerate(root['shards']):
     fields(item,['content_ref','first_tid','last_tid','rows']);_ref(item['content_ref'])
     ref=item['content_ref'];check(ref['sha256'] not in refs,'DUPLICATE_SCIENTIFIC_SHARD');refs.add(ref['sha256'])
     integer(item['first_tid'],lo,hi+1);integer(item['last_tid'],item['first_tid'],hi+1);integer(item['rows'],1,8193)
     tids=[t for t in range(item['first_tid'],item['last_tid']+1) if t not in excluded]
     check(len(tids)==item['rows'],'SHARD_DECLARED_COVERAGE')
     info=z.getinfo('content/'+ref['sha256']+'.blob');check(info.file_size==int(ref['size_bytes']),'SHARD_LENGTH_REFERENCE')
     for k,tid in enumerate(tids):
      check(self._routes[tid] is None,'DUPLICATE_MASTER_TID');self._routes[tid]=(n,j,k);covered.add(tid)
    check(covered==set(range(lo,hi+1))-set(excluded) and len(covered)==root['counts']['carriers'],'SLICE_DECLARED_CENSUS')
    check(set(names)=={'ROOT.json'}|{'content/'+x+'.blob' for x in refs},'SCIENTIFIC_CONTENT_CLOSURE')
   check(all(x is not None for x in self._routes),'MASTER_COVERAGE_GAP')
   self._integrity_report={'status':'ADMITTED_BYTES_INTEGRITY_PASS','release_id':'MASTER_DATA_V1_0014','catalog_sha256':CATALOG_SHA256,'scientific_roots':14,'canonical_root_order':[0,1,2],'exact_tids':[0,50115],'carriers':50116,'events':6311684,'option_realizations':6311684,'seed_native_identities_retained':3,'generator_calls':0,'scientific_acceptance_granted_by_reader':False}
  except BaseException:
   self.close();raise
 def _ready(self):check(not self._closed,'READER_CLOSED')
 def close(self):
  self._closed=True;self._cache=None;self._stack.close()
 def __enter__(self):self._ready();return self
 def __exit__(self,*exc):self.close()
 def integrity_report(self):self._ready();return _copy(self._integrity_report)
 def _shard(self,n,j):
  self._ready();key=(n,j)
  if self._cache_key==key:return self._cache
  root=self._roots[n];item=root['shards'][j];ref=item['content_ref']
  raw=self._archives[n].read('content/'+ref['sha256']+'.blob')
  check(len(raw)==int(ref['size_bytes']) and hashlib.sha256(raw).hexdigest()==ref['sha256'],'SCIENTIFIC_SHARD_DIGEST')
  # Large admitted shards fit 4 MiB but can exceed the seed profile's
  # 500,000-node limit. Bound nodes by the maximum raw byte budget instead.
  data=strict_loads(raw,max_bytes=4194304,max_nodes=4194304)
  check(canonical_bytes(data)==raw,'NONCANONICAL_SCIENTIFIC_SHARD')
  fields(data,['schema_id','parent_root','catalog_sha256','rows'])
  check(data['schema_id']=='IG_NATIVE_EXACT_J3_ROWS_V1' and data['parent_root']==PARENT_ROOT and data['catalog_sha256']==self._parent.root['catalog_sha256'],'ROW_SHARD_IDENTITY')
  excluded=root['scope'].get('excluded_already_admitted_tids',[])
  tids=[t for t in range(item['first_tid'],item['last_tid']+1) if t not in excluded]
  check(type(data['rows']) is list and [r['exact_tid'] for r in data['rows']]==tids,'SHARD_ACTUAL_CENSUS')
  self._cache_key=key;self._cache=data;return data
 def _provenance(self,n):
  s=self._catalog['slices'][n]
  return {'release_id':'MASTER_DATA_V1_0014','admission_catalog_sha256':CATALOG_SHA256,'scientific_root_sha256':s['scientific_root_sha256'],'dataset_id':s['dataset_id'],'native_format':s['format'],'identity_profile':s['identity_profile'],'archive_sha256':s['archive']['sha256']}
 def lookup_j3(self,tid,*,ordered_roots=(0,1,2)):
  self._ready();integer(tid,0,POPULATION);_order(ordered_roots)
  n,j,k=self._routes[tid]
  row=self._parent.lookup_j3(tid) if n==0 else self._shard(n,j)['rows'][k]
  check(row.get('exact_tid',row.get('representative_tid'))==tid,'LOOKUP_ROUTE_IDENTITY')
  return _copy({'carrier':row,'provenance':self._provenance(n)})
 def iter_j3(self,start=0,stop=POPULATION,*,ordered_roots=(0,1,2)):
  self._ready();integer(start,0,POPULATION+1);integer(stop,start,POPULATION+1);_order(ordered_roots)
  for tid in range(start,stop):yield self.lookup_j3(tid,ordered_roots=ordered_roots)
 def primitive(self,sid):self._ready();return self._parent.primitive(sid)
 def triple(self,tid):self._ready();return self._parent.triple(tid)
 def lookup_abc(self,object_id,*,catalog_sha256):
  self._ready();return {'data':self._parent.lookup_abc(object_id,catalog_sha256=catalog_sha256),'provenance':self._provenance(0)}
 def foundation_catalog_sha256(self):self._ready();return self._parent.root['catalog_sha256']
 def lookup_target(self,tid,event_id,option_indices,*,ordered_roots=(0,1,2)):
  source=self.lookup_j3(tid,ordered_roots=ordered_roots)
  check(type(option_indices) in (list,tuple) and len(option_indices)==3,'REALIZATION_CHOICE_SHAPE')
  for v in option_indices:integer(v)
  matches=[o for e in source['carrier']['events'] if e['event_id']==event_id for o in e['realizations'] if o['option_indices']==list(option_indices)]
  check(len(matches)==1,'REALIZATION_NOT_FOUND')
  occ=matches[0]
  return _copy({'source_j3_id':source['carrier']['j3_id'],'event_id':event_id,'realization':occ,'target':self.lookup_j3(occ['target_tid'])})
 def verify(self,progress=None):
  """Exhaust every stored row and option witness using the existing verifier.

  Validation checks finite completeness against stored option arrays. It never
  generates or replaces a scientific carrier, event or formation.
  """
  self._ready()
  if self._scientific_report is not None:return _copy(self._scientific_report)
  prefix=self._parent.verify();reports=[{'scientific_root_sha256':PARENT_ROOT,**prefix}]
  seen=set(SEED_TIDS);ids={c['j3_id'] for c in self._parent.carriers.values()}
  events=prefix['j3_events'];realizations=prefix['j3_option_realizations']
  row_commitment=hashlib.sha256()
  for n,root in enumerate(self._roots[1:],1):
   lo,hi=root['scope']['exact_tids'];local_seen=set();e=r=0
   for j,item in enumerate(root['shards']):
    shard=self._shard(n,j);de,dr=validate_rows(self._parent,shard,local_seen,lo,hi+1);e+=de;r+=dr
    for row in shard['rows']:
     tid=row['exact_tid'];check(tid not in seen and row['j3_id'] not in ids,'DUPLICATE_MASTER_CARRIER')
     seen.add(tid);ids.add(row['j3_id'])
     row_commitment.update(canonical_bytes([tid,self._catalog['slices'][n]['scientific_root_sha256'],row['j3_id']])+b'\n')
   counts=root['counts'];check(len(local_seen)==counts['carriers'] and e==counts['events'] and r==counts['option_realizations'],'SCIENTIFIC_SLICE_CENSUS')
   for k in ('primitive_component_occurrences','external_port_occurrences','internal_incidence_edges'):check(counts[k]==len(local_seen)*3,'SCIENTIFIC_INCIDENCE_CENSUS')
   events+=e;realizations+=r
   report={'scientific_root_sha256':self._catalog['slices'][n]['scientific_root_sha256'],**counts,'status':'EXACT_J3_REFERENCE_CLOSURE_PASS'};reports.append(report)
   if progress:progress(_copy(report))
  check(seen==set(range(POPULATION)) and len(ids)==POPULATION and events==realizations==6311684,'MASTER_SCIENTIFIC_CLOSURE')
  self._scientific_report={**self._integrity_report,'status':'CANONICAL_J3_UNIFIED_SCIENTIFIC_REFERENCE_CLOSURE_PASS','full_semantic_validation':True,'missing_carriers':0,'duplicate_tids':0,'duplicate_carrier_ids':0,'target_carrier_coverage':'ALL_STORED_TARGET_TIDS_RESOLVE_IN_CANONICAL_POPULATION','prefix_ABC_objects':prefix['abc_objects'],'prefix_ABC_formations':prefix['abc_formations'],'extension_row_identity_commitment':row_commitment.hexdigest(),'slices':reports}
  return _copy(self._scientific_report)
