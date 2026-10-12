from pathlib import Path
from types import SimpleNamespace
import sys,json,zipfile
h=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(h/'source'));sys.path.insert(0,str(h))
from project.bulk_adapter import evaluate
from project.historical_alias import old_edge_alias,canonicalize_edges,context as alias_context
from infinity_grid.structural_encoding import structural_canonical_bytes as encode
from infinity_grid.bulk_descriptor_plan import build_plan
from infinity_grid import submission as sub
z=zipfile.ZipFile(h/'inputs/preserved_prefix_archive.zip');saved={}
for n in z.namelist():
 if n.startswith('data/official_rows/') and n.endswith('.jsonl'):
  for line in z.read(n).splitlines():
   r=json.loads(line);saved[r['ordinal']]=r
reference={r['ordinal']:r for r in json.loads((h/'inputs/reference_full_witnesses.json').read_bytes())}
sample=json.loads((h/'inputs/qualification_sample.json').read_bytes());prefix_compares=0
for d in sample:
 r=evaluate({'descriptors':[d]})['signature']['rows'][0];expected=reference[d['ordinal']]
 for key in expected:assert encode(r[key])==encode(expected[key]),key
 if d['ordinal'] in saved:
  expected=saved[d['ordinal']];assert r['old_edge_alias']==expected['old_edge_alias'];prefix_compares+=1
plan=build_plan(h/'inputs/raw_descriptor_archive.zip',h/'inputs/preserved_prefix_archive.zip',window_rows=16384)
assert plan['preserved_rows']==56832 and sum(x['rows'] for x in plan['windows'])==595534
sub._project_closure(sub._tree(h/'project'),{'requirements':[]})
(h/'inputs/full_domain_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
out={'status':'PASS','rows':len(sample),'full_reference_witness_exact':True,'historical_alias_prefix_compares':prefix_compares,'frozen_prefix_rows':56832,'remaining_descriptors':595534,'windows':len(plan['windows']),'project_capture_dependencies_valid':True,'scientific_dispatch':False};(h/'results/ADAPTER_AND_DOMAIN.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
