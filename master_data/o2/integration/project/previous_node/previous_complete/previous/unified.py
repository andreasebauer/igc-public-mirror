"""Immutable catalog routing across admitted J3 and recursive constructions."""
from pathlib import Path
import hashlib,json
from .bindings import CATALOG_SHA256,ADMISSION_SHA256,PREDECESSOR_SHA256
from .scientific_reader import read_archive
from .recursive_reader import check
from .ig_master import UnifiedMasterReader,MissingDataError

class IntegratedMasterReader:
 def __init__(self,catalog,admission,predecessor,recursive_archive,*,canonical_directory=None,archive_directory=None):
  self.closed=False;self.old=None;self.recursive=None
  try:
   raw=Path(catalog).read_bytes();check(hashlib.sha256(raw).hexdigest()==CATALOG_SHA256,'CATALOG_DIGEST');self.catalog=json.loads(raw)
   raw=Path(admission).read_bytes();check(hashlib.sha256(raw).hexdigest()==ADMISSION_SHA256,'ADMISSION_DIGEST');a=json.loads(raw)
   raw=Path(predecessor).read_bytes();check(hashlib.sha256(raw).hexdigest()==PREDECESSOR_SHA256,'PREDECESSOR_DIGEST');old=json.loads(raw)
   check(self.catalog['release_id']=='MASTER_DATA_V1_0140' and self.catalog['slices'][:-1]==old['slices'] and len(old['slices'])==139,'PREDECESSOR_ROUTES')
   s=self.catalog['slices'][-1];check(s['scoped_admission_sha256']==ADMISSION_SHA256 and a['decision']=='ACCEPTED_FOR_CANONICAL_REUSE_WITHIN_DECLARED_RECURSIVE_SCOPE','SCOPED_AUTHORITY')
   check(s['scientific_root_sha256']==a['scientific_root_sha256'] and s['archive']['sha256']==a['scientific_archive_sha256'],'SCIENTIFIC_AUTHORITY')
   check(s['counts']['constructions']==a['counts']['constructions']==199579,'ADMITTED_COUNT')
   self._canonical_directory=canonical_directory;self._archive_directory=archive_directory;self._predecessor=Path(predecessor).parent
   p=Path(recursive_archive);check(p.stat().st_size==s['archive']['size_bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==s['archive']['sha256'],'ARCHIVE_DIGEST')
   self.recursive,self.root=read_archive(p,s['scientific_root_sha256'],PREDECESSOR_SHA256,'3ad27f29023db1b39442d07383d08df0d2829819a014203c6cd10a79112bfc9b')
  except BaseException:self.close();raise
 def ready(self):check(not self.closed,'READER_CLOSED')
 def legacy(self):
  self.ready()
  if self.old is None:
   if self._canonical_directory is None:raise MissingDataError('MISSING_CANONICAL_DIRECTORY')
   self.old=UnifiedMasterReader(self._predecessor,canonical_directory=self._canonical_directory,archive_directory=self._archive_directory)
  return self.old
 def lookup_j3(self,*args,**kwargs):return self.legacy().lookup_j3(*args,**kwargs)
 def lookup_target(self,*args,**kwargs):return self.legacy().lookup_target(*args,**kwargs)
 def lookup_construction(self,object_id):self.ready();return self.recursive.lookup(object_id)
 def lookup_formation(self,formation_id):self.ready();return self.recursive.lookup_formation(formation_id)
 def projection_preimages(self,depth,boundary_sha256):self.ready();return self.recursive.projection_preimages(depth,boundary_sha256)
 def lineage(self,object_id):self.ready();return self.recursive.lineage(object_id)
 def primitive(self,sid):return self.legacy().primitive(sid)
 def iter_j3(self,*args,**kwargs):return self.legacy().iter_j3(*args,**kwargs)
 def lookup_abc(self,*args,**kwargs):return self.legacy().lookup_abc(*args,**kwargs)
 def foundation_catalog_sha256(self):return self.legacy().foundation_catalog_sha256()
 def triple(self,tid):return self.legacy().triple(tid)
 def coverage_report(self):
  self.ready();return {'release_id':self.catalog['release_id'],'j3_carriers':300696,'recursive_constructions':199579,'scientific_slices':140,'predecessor_routes_unchanged':True,'historical_weights_qualified':False,'generator_calls':0}
 def close(self):
  self.closed=True
  if self.old is not None:self.old.close()
  if self.recursive is not None:self.recursive.close()
 def __enter__(self):self.ready();return self
 def __exit__(self,*exc):self.close()
