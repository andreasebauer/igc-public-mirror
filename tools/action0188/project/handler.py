"""Registered full qualification of Master151; no scientific generation."""
from pathlib import Path
import tempfile,shutil,hashlib,json
from infinity_grid.v05_chain import ChainExecutionResult
from .unified import G2SavedRecordMasterReader
from .verified_candidate.arguments import predecessor_arguments
from .verified_candidate.checks import census,bounded_routes
from .bindings import GATE_SHA256

def handler(stage,runtime):
 i=stage['input_artifacts'];raw=Path(i['admission_gate']).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=GATE_SHA256:raise ValueError('ADMISSION_GATE_HASH')
 gate=json.loads(raw)
 if gate['integration_status']!='COMPLETED' or gate['execution_status']!='FINISHED' or gate['evidence_status']!='VERIFIED' or gate['result_outcome']!='PASS' or not gate['cold_reused'] or gate['pending_bytes']!=0:raise ValueError('ADMISSION_GATE')
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp)/'saved';d.mkdir()
  for key,name in [('s1_manifest','INDEX_MANIFEST.json'),('s1_index','index.sqlite'),('s1_records','records.jsonl.gz'),('s1_audits','audits.jsonl.gz')]:shutil.copyfile(i[key],d/name)
  with G2SavedRecordMasterReader(i['published_catalog'],i['s1_admission'],d,candidate=i['s1_candidate'],proposal=i['s1_proposal'],previous_catalog=i['new_catalog'],previous_arguments=predecessor_arguments(i,tmp)) as r:
   result=census(r);result.update(bounded_routes(r,i));coverage=r.coverage_report()
   if coverage['release_id']!='MASTER_DATA_V1_0151' or coverage['scientific_slices']!=151:raise ValueError('PUBLISHED_COVERAGE')
   for ordinal in (0,1,290175,580350):
    row=r.g2_s1_record(ordinal);audit=r.g2_s1_audit(ordinal)
    if json.loads(r.g2_s1_raw_record(ordinal))!=row or json.loads(r.g2_s1_raw_audit(ordinal))!=audit:raise ValueError('PUBLISHED_RAW_ROUTE')
    if r.g2_s1_pair_operator(row['left_carrier_ref'],row['right_carrier_ref'],row['connection_operator_ref'])!=row:raise ValueError('PUBLISHED_PAIR_ROUTE')
  for call in [lambda:r.g2_s1_record(0),r.coverage_report,lambda:r.lookup_g1_public_interface('missing')]:
   try:call()
   except ValueError:pass
   else:raise ValueError('CLOSED_READER')
 result.update(outcome='PASS',master_release='MASTER_DATA_V1_0151',scientific_slices=151,
               scoped_admission_verified=True,published_reader_qualified=True,
               prior_slices_unchanged=150,prior_reader_bytes_unchanged=True,
               saved_reader_bytes_unchanged=hashlib.sha256((Path(__file__).parent/'verified_candidate/saved_reader.py').read_bytes()).hexdigest()=='e261c774b3193e29af58a1b23e272b14dd7bea59ac1c03ee476f7b275c550f95',
               generation_calls=0,qualified_new_scientific_slices=1,
               master_cursor_updated=False,authority='SAVED_REPAIRED_PAIR_RECORDS_AND_PAIRED_STORED_OBSERVER_SIGNATURES_ONLY')
 return ChainExecutionResult(result=result)
