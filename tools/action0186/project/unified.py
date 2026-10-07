"""Candidate reader. Opening it does not publish or admit a scientific slice."""
from pathlib import Path
import json,hashlib
from .bindings import CANDIDATE_SHA256,PROPOSAL_SHA256,PREVIOUS_SHA256,SOURCE_PINS,MANIFEST_SHA256
from .saved_reader import SavedRecordReader
from .previous.unified import G1ProjectionMasterReader

def pinned(path,digest):
 b=Path(path).read_bytes()
 if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('BOUND_INPUT_HASH')
 return json.loads(b)

class G2SavedRecordCandidateReader:
 def __init__(self,candidate,proposal,saved_directory,*,previous_catalog,previous_arguments):
  self.closed=False;self.previous=None;self.saved=None
  try:
   cat=pinned(candidate,CANDIDATE_SHA256);ad=pinned(proposal,PROPOSAL_SHA256);old=pinned(previous_catalog,PREVIOUS_SHA256)
   if ad['decision']!='PROPOSED_PENDING_NATIVE_INTEGRATION_VERIFICATION' or cat['publication_status']!='CANDIDATE_PENDING_NATIVE_VERIFICATION':raise ValueError('PROPOSAL_STATE')
   if len(old['slices'])!=150 or len(cat['slices'])!=151 or cat['slices'][:-1]!=old['slices']:raise ValueError('PREDECESSOR_SLICES')
   if cat['previous_catalog_sha256']!=PREVIOUS_SHA256 or cat['slices'][-1]['scoped_admission_proposal_sha256']!=PROPOSAL_SHA256:raise ValueError('CATALOG_BINDING')
   base=Path(__file__).parent/'previous'
   if set(SOURCE_PINS)!={str(p.relative_to(base)) for p in base.rglob('*.py')}:raise ValueError('SOURCE_CLOSURE')
   for n,h in SOURCE_PINS.items():
    if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h:raise ValueError('PREVIOUS_SOURCE')
   manifest=pinned(Path(saved_directory)/'INDEX_MANIFEST.json',MANIFEST_SHA256)
   if manifest['authority']!=ad['authority'] or manifest['counts']!=ad['counts'] or manifest['files']!=ad['files']:raise ValueError('INDEX_SCOPE')
   self.saved=SavedRecordReader(saved_directory)
   self.previous=G1ProjectionMasterReader(previous_catalog,**previous_arguments)
   self.catalog=cat;self.proposal=ad
  except BaseException:self.close();raise
 def ready(self):
  if self.closed:raise ValueError('CLOSED_READER')
 def g2_s1_record(self,ordinal):self.ready();return self.saved.record_by_ordinal(ordinal)
 def g2_s1_raw_record(self,ordinal):self.ready();return self.saved.raw_record_by_ordinal(ordinal)
 def g2_s1_audit(self,ordinal):self.ready();return self.saved.stored_audit_by_ordinal(ordinal)
 def g2_s1_raw_audit(self,ordinal):self.ready();return self.saved.raw_audit_by_ordinal(ordinal)
 def g2_s1_pair_operator(self,left,right,operator):self.ready();return self.saved.record_by_pair_operator(left,right,operator)
 def g2_s1_outcome_members(self,digest):self.ready();return self.saved.outcome_members(digest)
 def g2_s1_provenance(self):self.ready();return self.saved.coverage_and_provenance()
 def g2_s1_q2_payload(self,*a):self.ready();return self.saved.q2_payload(*a)
 def g2_s1_realized_carrier(self,*a):self.ready();return self.saved.realized_carrier(*a)
 def coverage_report(self):
  self.ready();r=self.previous.coverage_report();r.update(proposed_release='MASTER_DATA_V1_0151',proposed_scientific_slices=151,publication_status='CANDIDATE_PENDING_NATIVE_VERIFICATION',g2_saved_s1=self.saved.coverage_and_provenance());return r
 def __getattr__(self,name):
  if name.startswith('_'):raise AttributeError(name)
  self.ready();return getattr(self.previous,name)
 def close(self):
  self.closed=True
  if self.saved:self.saved.close()
  if self.previous:self.previous.close()
 def __enter__(self):self.ready();return self
 def __exit__(self,*a):self.close()
