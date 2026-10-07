"""Master150 adds observer-scoped G1 projections to unchanged prior readers."""
from pathlib import Path
import hashlib,json,tempfile,zipfile
from .previous.unified import O7ExtendedMasterReader
from .projection.reader import ProjectionReader
from .bindings import CATALOG_SHA256,PREVIOUS_SHA256,ADMISSION_SHA256

class G1ProjectionMasterReader:
 def __init__(self,catalog,admission,archive,*,previous_catalog,previous_arguments):
  self.closed=False;self.previous=None;self.projection=None;self._temp=None
  try:
   def load(p,h):
    b=Path(p).read_bytes()
    if hashlib.sha256(b).hexdigest()!=h:raise ValueError('AUTHORITY_HASH')
    return json.loads(b)
   cat=load(catalog,CATALOG_SHA256);old=load(previous_catalog,PREVIOUS_SHA256);ad=load(admission,ADMISSION_SHA256);s=cat['slices'][-1]
   if cat['release_id']!='MASTER_DATA_V1_0150' or cat['slices'][:-1]!=old['slices'] or len(old['slices'])!=149 or cat['previous_catalog_sha256']!=PREVIOUS_SHA256:raise ValueError('PREDECESSOR_CATALOG')
   if ad['decision']!='ACCEPTED_FOR_REUSE_WITHIN_SAVED_G1_PUBLIC_PROJECTION_SCOPE' or s['scoped_admission_sha256']!=ADMISSION_SHA256 or s['counts']!=ad['counts'] or s['scope']!=ad['scope'] or ad['predecessor_catalog_sha256']!=PREVIOUS_SHA256:raise ValueError('SCOPED_ADMISSION')
   raw=Path(archive).read_bytes()
   if hashlib.sha256(raw).hexdigest()!=s['archive']['sha256'] or s['archive']['sha256']!=ad['scientific_archive_sha256']:raise ValueError('ARCHIVE_HASH')
   with zipfile.ZipFile(archive) as z:
    raw=z.read('ROOT.json')
    if hashlib.sha256(raw).hexdigest()!=s['scientific_root_sha256'] or s['scientific_root_sha256']!=ad['scientific_root_sha256']:raise ValueError('ROOT_HASH')
    root=json.loads(raw);data={};seen={'ROOT.json'}
    for x in root['content']:
     name='content/'+x['sha256']+'.blob';b=z.read(name)
     if hashlib.sha256(b).hexdigest()!=x['sha256'] or len(b)!=x['bytes'] or x['name'] in data:raise ValueError('CONTENT_HASH')
     data[x['name']]=b;seen.add(name)
    if set(z.namelist())!=seen or len(z.namelist())!=len(seen):raise ValueError('CONTENT_CLOSURE')
   if root['authority']!='SAVED_PUBLIC_PROJECTION_ONLY' or root['master_catalog_sha256']!=PREVIOUS_SHA256 or root['exact_parent_DAG_available'] is not False or root['Q2_payload_available'] is not False:raise ValueError('PROJECTION_SCOPE')
   self._temp=tempfile.TemporaryDirectory(prefix='ig_g1_saved_');d=Path(self._temp.name)
   pins={}
   for name in ['population','continuations']:
    (d/name).write_bytes(data[name]);pins[name]={'sha256':hashlib.sha256(data[name]).hexdigest(),'bytes':len(data[name])}
   self.projection=ProjectionReader(d/'population',d/'continuations',pins)
   self.previous=O7ExtendedMasterReader(previous_catalog,**previous_arguments);self.catalog=cat
  except BaseException:self.close();raise
 def ready(self):
  if self.closed:raise ValueError('CLOSED_READER')
 def g1_public_refs(self):self.ready();return self.projection.refs()
 def lookup_g1_public_interface(self,ref):self.ready();return self.projection.interface(ref)
 def g1_public_reservation(self,ref,t):self.ready();return self.projection.reservation(ref,t)
 def g1_public_continuation_hash(self,ref,t):self.ready();return self.projection.continuation_hash(ref,t)
 def g1_public_interface_class(self,h):self.ready();return self.projection.interface_class(h)
 def g1_public_provenance(self):self.ready();return self.projection.provenance()
 def g1_exact_state(self,ref):self.ready();return self.projection.exact_state(ref)
 def g1_q2_payload(self,ref):self.ready();return self.projection.q2_payload(ref)
 def coverage_report(self):
  self.ready();r=self.previous.coverage_report();r.update(release_id='MASTER_DATA_V1_0150',scientific_slices=150,g1_saved_public_projection={'interfaces':193,'interface_classes':192,'reservation_rows':1351,'continuation_hashes':1351,'authority':'SAVED_PUBLIC_PROJECTION_ONLY'},g1_exact_parent_DAG_available=False,g1_Q2_payload_available=False);return r
 def __getattr__(self,name):
  if name.startswith('_'):raise AttributeError(name)
  self.ready();return getattr(self.previous,name)
 def close(self):
  self.closed=True
  if self.previous:self.previous.close()
  if self.projection:self.projection.close()
  if self._temp:self._temp.cleanup()
 def __enter__(self):self.ready();return self
 def __exit__(self,*exc):self.close()
