"""Master151: admitted saved S1 records, with unchanged predecessor readers."""
from pathlib import Path
import hashlib
from .bindings import CATALOG_SHA256, ADMISSION_SHA256, CANDIDATE_SOURCE_PINS
from .verified_candidate.unified import G2SavedRecordCandidateReader, pinned

class G2SavedRecordMasterReader(G2SavedRecordCandidateReader):
 def __init__(self,catalog,admission,saved_directory,*,candidate,proposal,previous_catalog,previous_arguments):
  self.closed=False;self.previous=None;self.saved=None
  try:
   cat=pinned(catalog,CATALOG_SHA256);ad=pinned(admission,ADMISSION_SHA256)
   if ad['decision']!='ACCEPTED_FOR_REUSE_WITHIN_SAVED_G2_S1_RECORD_AND_STORED_AUDIT_SCOPE':raise ValueError('ADMISSION_DECISION')
   if cat['release_id']!='MASTER_DATA_V1_0151' or cat['publication_status']!='ADMITTED_WITHIN_DECLARED_SAVED_SCOPE':raise ValueError('RELEASE_STATE')
   base=Path(__file__).parent/'verified_candidate'
   actual={str(p.relative_to(base)) for p in base.rglob('*.py')}
   if actual!=set(CANDIDATE_SOURCE_PINS):raise ValueError('CANDIDATE_SOURCE_CLOSURE')
   for n,h in CANDIDATE_SOURCE_PINS.items():
    if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h:raise ValueError('CANDIDATE_SOURCE_PIN')
   super().__init__(candidate,proposal,saved_directory,previous_catalog=previous_catalog,previous_arguments=previous_arguments)
   old=pinned(previous_catalog,ad['predecessor_catalog_sha256']);s=cat['slices'][-1]
   if len(cat['slices'])!=151 or cat['slices'][:-1]!=old['slices'] or cat['previous_catalog_sha256']!=ad['predecessor_catalog_sha256']:raise ValueError('PREDECESSOR_SLICES')
   if s['scoped_admission_sha256']!=ADMISSION_SHA256 or s['scientific_root_sha256']!=ad['index_manifest_sha256']:raise ValueError('ADMISSION_BINDING')
   for k in ('counts','files','scope'):
    if s[k]!=ad[k] or s[k]!=self.catalog['slices'][-1][k]:raise ValueError('SAVED_SCOPE_BINDING')
   if ad['authority']!=self.proposal['authority'] or ad['generation_calls']!=0:raise ValueError('SCOPE_AUTHORITY')
   self.catalog=cat;self.admission=ad
  except BaseException:self.close();raise
 def coverage_report(self):
  self.ready();r=self.previous.coverage_report()
  r.update(release_id='MASTER_DATA_V1_0151',scientific_slices=151,
           publication_status='ADMITTED_WITHIN_DECLARED_SAVED_SCOPE',
           g2_saved_s1=self.saved.coverage_and_provenance(),
           g2_exact_G1_parent_DAG_available=False,g2_Q2_payload_available=False)
  return r
