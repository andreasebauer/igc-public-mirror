from pathlib import Path
import json,hashlib
from .previous_complete.unified import CompleteMasterReader
from .panel_reader import PanelReader
from .bindings import CATALOG_SHA256,PREVIOUS_SHA256,ADMISSION_SHA256

class NodeMasterReader:
 def __init__(self,catalog,node_admission,node_archive,*,previous_catalog,previous_arguments):
  self.closed=False;self.previous=None;self.nodes=None
  try:
   def load(p,h):
    b=Path(p).read_bytes()
    if hashlib.sha256(b).hexdigest()!=h:raise ValueError('AUTHORITY_HASH')
    return json.loads(b)
   c=load(catalog,CATALOG_SHA256);old=load(previous_catalog,PREVIOUS_SHA256);ad=load(node_admission,ADMISSION_SHA256);s=c['slices'][-1]
   if c['release_id']!='MASTER_DATA_V1_0142' or c['slices'][:-1]!=old['slices'] or len(old['slices'])!=141:raise ValueError('PREDECESSOR_CATALOG')
   if ad['decision']!='ACCEPTED_FOR_CANONICAL_REUSE_WITHIN_DECLARED_NODE_SELECTED_PANEL_SCOPE' or ad['scientific_root_sha256']!=s['scientific_root_sha256'] or ad['scientific_archive_sha256']!=s['archive']['sha256'] or s['scoped_admission_sha256']!=ADMISSION_SHA256 or ad['predecessor_catalog_sha256']!=PREVIOUS_SHA256:raise ValueError('SCOPED_ADMISSION')
   self.previous=CompleteMasterReader(previous_catalog,**previous_arguments);self.nodes=PanelReader(node_archive,s['archive']['sha256'],s['scientific_root_sha256']);self.catalog=c
   if self.nodes.report['edges']!=ad['counts']['edges'] or self.nodes.report['panel_occurrences']!=ad['counts']['panel_occurrences']:raise ValueError('ADMISSION_COUNTS')
  except BaseException:self.close();raise
 def ready(self):
  if self.closed:raise ValueError('READER_CLOSED')
 def lookup_node_panel(self,level,boundary_sha256):self.ready();return self.nodes.lookup(level,boundary_sha256)
 def node_parents(self,level,boundary_sha256):
  x=self.lookup_node_panel(level,boundary_sha256);return [self.lookup_node_panel(level-1,h) for h in x['parents']]
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
  self.ready();r=self.previous.coverage_report();r.update(release_id='MASTER_DATA_V1_0142',scientific_slices=142,node_selected_panels=dict(self.nodes.report));return r
 def close(self):
  self.closed=True
  if self.nodes:self.nodes.close()
  if self.previous:self.previous.close()
 def __enter__(self):self.ready();return self
 def __exit__(self,*exc):self.close()
