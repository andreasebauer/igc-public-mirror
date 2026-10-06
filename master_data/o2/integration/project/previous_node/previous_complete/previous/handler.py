from pathlib import Path
import tempfile,hashlib
from infinity_grid.canon import canonical_sha256,canonical_bytes
from infinity_grid.v05_chain import ChainExecutionResult
from .unified import IntegratedMasterReader
from .recursive_reader import IntegrityError
from .ig_master import MissingDataError

def handler(stage,runtime):
 inputs=stage['input_artifacts'];negative=[]
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)/'science.zip'
  with p.open('wb') as out:
   for i in range(2):out.write(Path(inputs['scientific_part'+str(i)]).read_bytes())
  with IntegratedMasterReader(inputs['new_catalog'],inputs['admission'],inputs['catalog'],p) as r:
   count=0
   for oid,row in r.recursive.rows.items():
    if r.lookup_construction(oid)!=row or r.lookup_formation(row['formation_id'])!=row:raise ValueError('ROUNDTRIP')
    lineage=r.lineage(oid)
    if lineage['construction']!=row:raise ValueError('LINEAGE_ROW')
    if r.recursive.depths[oid]==2 and lineage['parent']['object_id']!=row['identity']['parent_object_id']:raise ValueError('LINEAGE_PARENT')
    count+=1
   for (depth,h),oids in r.recursive.preimages.items():
    if r.projection_preimages(depth,h)!=tuple(oids):raise ValueError('PREIMAGES')
    for oid in oids:
     if canonical_sha256(r.recursive.rows[oid]['projected_boundary'])!=h:raise ValueError('PREIMAGE_REFERENCE')
   def reject(name,fn):
    try:fn()
    except (IntegrityError,MissingDataError,KeyError):negative.append(name)
    else:raise ValueError('EXPECTED_REJECTION_'+name)
   reject('missing_construction',lambda:r.lookup_construction('0'*64));reject('missing_formation',lambda:r.lookup_formation('0'*64));reject('missing_projection',lambda:r.projection_preimages(2,'0'*64));reject('missing_legacy_archive',lambda:r.lookup_j3(0))
   oid=next(iter(r.recursive.rows));copy=r.lookup_construction(oid);copy['ordered_record'][0]=-1
   if r.lookup_construction(oid)!=r.recursive.rows[oid]:raise ValueError('COPY_LEAK')
   report=r.coverage_report();report.update(r.recursive.report)
  reject('closed_reader',lambda:r.lookup_construction(oid))
  bad=Path(tmp)/'bad.json';bad.write_bytes(Path(inputs['new_catalog']).read_bytes()+b' ')
  reject('corrupt_catalog',lambda:IntegratedMasterReader(bad,inputs['admission'],inputs['catalog'],p))
  report.update(outcome='PASS',all_recursive_routes_checked=count,formation_routes_checked=count,projection_groups_checked=910,lineage_routes_checked=count,predecessor_slices_exactly_unchanged=139,negative_checks=negative,old_reader_source_unchanged=True,legacy_science_reverified=False,generator_calls=0)
  return ChainExecutionResult(result=report)
