from pathlib import Path
import json,hashlib
from .previous_o3.unified import O3MasterReader
from .reader import CarrierReader
from .bindings import CATALOG_SHA256,PREVIOUS_SHA256,ADMISSION_SHA256
class O4MasterReader:
 def __init__(self,catalog,o4_admission,o4_archive,*,previous_catalog,previous_arguments):
  self.closed=False;self.previous=None;self.o4=None
  try:
   def load(p,h):
    b=Path(p).read_bytes()
    if hashlib.sha256(b).hexdigest()!=h:raise ValueError('AUTHORITY_HASH')
    return json.loads(b)
   c=load(catalog,CATALOG_SHA256);old=load(previous_catalog,PREVIOUS_SHA256);ad=load(o4_admission,ADMISSION_SHA256);s=c['slices'][-1]
   if c['release_id']!='MASTER_DATA_V1_0145' or c['previous_catalog_sha256']!=PREVIOUS_SHA256 or c['slices'][:-1]!=old['slices'] or len(old['slices'])!=144:raise ValueError('PREDECESSOR_CATALOG')
   if ad['decision']!='ACCEPTED_FOR_REUSE_WITHIN_SAVED_O4_TYPED_INCIDENCE_AND_ADMITTED_O3_COMPONENT_SCOPE' or ad['scientific_root_sha256']!=s['scientific_root_sha256'] or ad['scientific_archive_sha256']!=s['archive']['sha256'] or s['scoped_admission_sha256']!=ADMISSION_SHA256 or ad['predecessor_catalog_sha256']!=PREVIOUS_SHA256 or s['scope']!=ad['scope'] or s['counts']!=ad['counts']:raise ValueError('SCOPED_ADMISSION')
   self.previous=O3MasterReader(previous_catalog,**previous_arguments);self.o4=CarrierReader(o4_archive,s['archive']['sha256'],s['scientific_root_sha256'],previous_arguments['o3_archive']);self.catalog=c
   for k in ['carriers','seed_carriers','typed_edges','components','O3_owner_occurrences','prototype_bindings']:
    if self.o4.report[k]!=ad['counts'][k]:raise ValueError('ADMISSION_COUNTS')
  except BaseException:self.close();raise
 def ready(self):
  if self.closed:raise ValueError('READER_CLOSED')
 def lookup_o4(self,lane,rank,digest):self.ready();return self.o4.lookup(lane,rank,digest)
 def o4_owner(self,lane,rank,digest,index):self.ready();return self.o4.owner(lane,rank,digest,index)
 def o4_components(self,lane,rank,digest):self.ready();return self.o4.components(lane,rank,digest)
 def o4_parents(self,*a):self.ready();return self.o4.parents(*a)
 def lookup_o3(self,*a,**k):self.ready();return self.previous.lookup_o3(*a,**k)
 def o3_parents(self,*a,**k):self.ready();return self.previous.o3_parents(*a,**k)
 def o3_components(self,*a,**k):self.ready();return self.previous.o3_components(*a,**k)
 def o3_qstate(self,*a,**k):self.ready();return self.previous.o3_qstate(*a,**k)
 def lookup_o2(self,*a,**k):self.ready();return self.previous.lookup_o2(*a,**k)
 def o2_parents(self,*a,**k):self.ready();return self.previous.o2_parents(*a,**k)
 def o2_observer(self,*a,**k):self.ready();return self.previous.o2_observer(*a,**k)
 def lookup_node_panel(self,*a,**k):self.ready();return self.previous.lookup_node_panel(*a,**k)
 def node_parents(self,*a,**k):self.ready();return self.previous.node_parents(*a,**k)
 def lookup_pair(self,*a,**k):self.ready();return self.previous.lookup_pair(*a,**k)
 def lookup_pair_formation(self,*a,**k):self.ready();return self.previous.lookup_pair_formation(*a,**k)
 def lookup_construction(self,*a,**k):self.ready();return self.previous.lookup_construction(*a,**k)
 def lookup_formation(self,*a,**k):self.ready();return self.previous.lookup_formation(*a,**k)
 def lineage(self,*a,**k):self.ready();return self.previous.lineage(*a,**k)
 def projection_preimages(self,*a,**k):self.ready();return self.previous.projection_preimages(*a,**k)
 def lookup_j3(self,*a,**k):self.ready();return self.previous.lookup_j3(*a,**k)
 def lookup_target(self,*a,**k):self.ready();return self.previous.lookup_target(*a,**k)
 def iter_j3(self,*a,**k):self.ready();return self.previous.iter_j3(*a,**k)
 def primitive(self,*a,**k):self.ready();return self.previous.primitive(*a,**k)
 def triple(self,*a,**k):self.ready();return self.previous.triple(*a,**k)
 def lookup_abc(self,*a,**k):self.ready();return self.previous.lookup_abc(*a,**k)
 def foundation_catalog_sha256(self):self.ready();return self.previous.foundation_catalog_sha256()
 def coverage_report(self):
  self.ready();r=self.previous.coverage_report();r.update(release_id='MASTER_DATA_V1_0145',scientific_slices=145,o4_saved_typed_carriers=dict(self.o4.report),o4_complete_derivation_lineage=False);return r
 def close(self):
  self.closed=True
  if self.o4:self.o4.close()
  if self.previous:self.previous.close()
 def __enter__(self):self.ready();return self
 def __exit__(self,*exc):self.close()
