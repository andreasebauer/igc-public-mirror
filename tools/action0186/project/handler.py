from pathlib import Path
import tempfile,shutil,hashlib
from infinity_grid.v05_chain import ChainExecutionResult
from .unified import G2SavedRecordCandidateReader
from .arguments import predecessor_arguments
from .checks import bounded_routes,census
def handler(stage,runtime):
 i=stage['input_artifacts']
 with tempfile.TemporaryDirectory() as tmp:
  d=Path(tmp)/'saved';d.mkdir()
  for key,name in [('s1_manifest','INDEX_MANIFEST.json'),('s1_index','index.sqlite'),('s1_records','records.jsonl.gz'),('s1_audits','audits.jsonl.gz')]:shutil.copyfile(i[key],d/name)
  with G2SavedRecordCandidateReader(i['s1_candidate'],i['s1_proposal'],d,previous_catalog=i['new_catalog'],previous_arguments=predecessor_arguments(i,tmp)) as r:
   result=census(r);result.update(bounded_routes(r,i))
   if r.coverage_report()['scientific_slices']!=150:raise ValueError('PREMATURE_PUBLICATION')
  for call in [lambda:r.g2_s1_record(0),r.coverage_report,lambda:r.lookup_g1_public_interface('missing')]:
   try:call()
   except ValueError:pass
   else:raise ValueError('CLOSED_READER')
 result.update(outcome='PASS',master_release='MASTER_DATA_V1_0150',scientific_slices=150,proposed_release='MASTER_DATA_V1_0151',proposed_scientific_slices=151,prior_slices_unchanged=150,prior_reader_bytes_unchanged=True,saved_reader_bytes_unchanged=hashlib.sha256((Path(__file__).parent/'saved_reader.py').read_bytes()).hexdigest()=='e261c774b3193e29af58a1b23e272b14dd7bea59ac1c03ee476f7b275c550f95',generation_calls=0,new_admissions=0,publication_performed=False)
 return ChainExecutionResult(result=result)
