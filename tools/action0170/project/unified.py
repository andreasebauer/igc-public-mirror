from pathlib import Path
import json,hashlib
from .previous_o6.unified import O6MasterReader
from .reader import CarrierReader
from .bindings import CATALOG_SHA256,PREVIOUS_SHA256,ADMISSION_SHA256
class O7MasterReader:
 def __init__(self,catalog,o7_admission,o7_archive,*,previous_catalog,previous_arguments):
  self.closed=False;self.previous=None;self.o7=None
  try:
   def load(p,h):
    b=Path(p).read_bytes()
    if hashlib.sha256(b).hexdigest()!=h:raise ValueError('AUTHORITY_HASH')
    return json.loads(b)
   c=load(catalog,CATALOG_SHA256);old=load(previous_catalog,PREVIOUS_SHA256);ad=load(o7_admission,ADMISSION_SHA256);s=c['slices'][-1]
   if c['release_id']!='MASTER_DATA_V1_0148' or c['previous_catalog_sha256']!=PREVIOUS_SHA256 or c['slices'][:-1]!=old['slices'] or len(old['slices'])!=147:raise ValueError('PREDECESSOR_CATALOG')
   if ad['decision']!='ACCEPTED_FOR_REUSE_WITHIN_RECOVERED_SAVED_O7_TYPED_INCIDENCE_AND_ADMITTED_O6_COMPONENT_SCOPE' or ad['scientific_root_sha256']!=s['scientific_root_sha256'] or ad['scientific_archive_sha256']!=s['archive']['sha256'] or s['scoped_admission_sha256']!=ADMISSION_SHA256 or ad['predecessor_catalog_sha256']!=PREVIOUS_SHA256 or s['scope']!=ad['scope'] or s['counts']!=ad['counts']:raise ValueError('SCOPED_ADMISSION')
   self.previous=O6MasterReader(previous_catalog,**previous_arguments);a=previous_arguments['previous_arguments'];a4=a['previous_arguments'];a3=a4['previous_arguments'];self.o7=CarrierReader(o7_archive,s['archive']['sha256'],s['scientific_root_sha256'],previous_arguments['o6_archive'],a['o5_archive'],a4['o4_archive'],a3['o3_archive']);self.catalog=c
   for k in ['carriers','seed_carriers','typed_edges','components','O6_owner_occurrences','selected_O6_bindings','external_twin_roots']:
    if self.o7.report[k]!=ad['counts'][k]:raise ValueError('ADMISSION_COUNTS')
  except BaseException:self.close();raise
 def ready(self):
  if self.closed:raise ValueError('READER_CLOSED')
 def lookup_o7(self,*a,**k):self.ready();return self.o7.lookup(*a,**k)
 def o7_owner(self,*a,**k):self.ready();return self.o7.owner(*a,**k)
 def o7_components(self,*a,**k):self.ready();return self.o7.components(*a,**k)
 def o7_resources(self,*a,**k):self.ready();return self.o7.resources(*a,**k)
 def o7_parents(self,*a,**k):self.ready();return self.o7.parents(*a,**k)
 def coverage_report(self):
  self.ready();r=self.previous.coverage_report();r.update(release_id='MASTER_DATA_V1_0148',scientific_slices=148,o7_recovered_saved_typed_carriers=dict(self.o7.report),o7_complete_derivation_lineage=False,o7_graduated=False,automatic_o8_authorized=False);return r
 def close(self):
  self.closed=True
  if self.o7:self.o7.close()
  if self.previous:self.previous.close()
 def __enter__(self):self.ready();return self
 def __exit__(self,*exc):self.close()
 def lookup_o6(self,*a,**k):self.ready();return self.previous.lookup_o6(*a,**k)
 def o6_owner(self,*a,**k):self.ready();return self.previous.o6_owner(*a,**k)
 def o6_components(self,*a,**k):self.ready();return self.previous.o6_components(*a,**k)
 def o6_parents(self,*a,**k):self.ready();return self.previous.o6_parents(*a,**k)
 def lookup_o5(self,*a,**k):self.ready();return self.previous.lookup_o5(*a,**k)
 def o5_owner(self,*a,**k):self.ready();return self.previous.o5_owner(*a,**k)
 def o5_components(self,*a,**k):self.ready();return self.previous.o5_components(*a,**k)
 def o5_parents(self,*a,**k):self.ready();return self.previous.o5_parents(*a,**k)
 def o5_resources(self,*a,**k):self.ready();return self.previous.o5_resources(*a,**k)
 def lookup_o4(self,*a,**k):self.ready();return self.previous.lookup_o4(*a,**k)
 def o4_owner(self,*a,**k):self.ready();return self.previous.o4_owner(*a,**k)
 def o4_components(self,*a,**k):self.ready();return self.previous.o4_components(*a,**k)
 def o4_parents(self,*a,**k):self.ready();return self.previous.o4_parents(*a,**k)
 def o6_resources(self,*a,**k):self.ready();return self.previous.o6_resources(*a,**k)
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
 def foundation_catalog_sha256(self,*a,**k):self.ready();return self.previous.foundation_catalog_sha256(*a,**k)
