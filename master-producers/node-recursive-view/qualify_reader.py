"""Full positive closure and negative reader checks, over pinned saved bytes."""
from pathlib import Path
import json,copy,hashlib
from reader import RecursiveReader,IntegrityError,read
R=Path(__file__).resolve().parent;profile=json.loads((R/'EXPORT_PROFILE.json').read_text());locations=json.loads((R/'LOCATIONS.json').read_text())
r=RecursiveReader(R/'ROOT.json',profile['root_sha256'],locations);report=dict(r.report);rejected=[]
for depth in [1,2]:
 oid=next(oid for oid,d in r.depths.items() if d==depth);row=r.lookup(oid);assert r.lookup_formation(row['formation_id'])==row;assert r.lineage(oid)['construction']==row;assert oid in r.projection_preimages(depth,__import__('canonical_kernel').canonical_sha256(row['projected_boundary']))
 for field,value in [('formation_id','bad'),('bridge',[]),('external_port_origins',[]),('projected_port_to_ordered_port',[]),('microscopic_realization_product_count',0)]:
  bad=copy.deepcopy(row);bad[field]=value
  try:r.validate(bad,depth)
  except (IntegrityError,KeyError,ValueError,IndexError):rejected.append(f'depth{depth}:{field}')
  else:raise AssertionError('CORRUPTION_ACCEPTED')
for action,label in [(lambda:r.lookup('missing'),'missing_object'),(lambda:r.lookup_formation('missing'),'missing_formation'),(lambda:r.projection_preimages(2,'missing'),'missing_boundary'),(lambda:read(R/'ROOT.json','0'*64,limit=1048576),'wrong_root_hash')]:
 try:action()
 except IntegrityError:rejected.append(label)
 else:raise AssertionError('INVALID_LOOKUP_ACCEPTED')
shard=r.root['sources'][0]['shards'][0];old=locations[shard['sha256']];locations[shard['sha256']]=str(R/'absent.bin')
try:RecursiveReader(R/'ROOT.json',profile['root_sha256'],locations)
except IntegrityError:rejected.append('missing_saved_shard')
else:raise AssertionError('MISSING_SHARD_ACCEPTED')
r.close()
try:r.lookup(oid)
except IntegrityError:rejected.append('closed_reader')
else:raise AssertionError('CLOSED_READER_ACCEPTED')
report.update(root_sha256=profile['root_sha256'],negative_checks_rejected=rejected,admitted_primitive_closure=json.loads((R/'ADMITTED_PRIMITIVE_CLOSURE.json').read_text())['status'],candidate_only=True,native_export_registered=False)
(R/'READER_QUALIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
