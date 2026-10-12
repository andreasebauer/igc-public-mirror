from pathlib import Path
from types import SimpleNamespace
import sys,json,time
h=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(h/'source'))
from infinity_grid import o7_edge_canonicalization as k
from infinity_grid.structural_encoding import structural_canonical_bytes as encode
obj=json.loads((h/'project/context.json').read_text());ctx=SimpleNamespace(n=len(obj['parents']),parents=tuple(SimpleNamespace(**p) for p in obj['parents']))
sample=json.loads((h/'inputs/qualification_sample.json').read_text());ref=json.loads((h/'inputs/reference_full_witnesses.json').read_text());by={r['ordinal']:r for r in ref}
timing={}
for cached in (False,True):
 k.reset_hierarchy_cache(enabled=cached);start=time.perf_counter()
 for d in sample:
  edge=(tuple(d['edge']),);r=k.state_identity(ctx,edge);assert k.verify_relabeling_witness(ctx,edge,r['edges'],r['witness'])
  expected=by[d['ordinal']]
  assert r['object_identity']==expected['V3_identity'] and r['legacy_context_identity']==expected['V2_context_alias']
  assert encode(r['edges'])==encode(expected['corrected_edges']) and encode(r['witness'])==encode(expected['relabeling_witness'])
 timing[str(cached)]={'wall_seconds':time.perf_counter()-start,'cache':k.hierarchy_cache_statistics()}
out={'status':'PASS','rows_each':len(sample),'full_reference_witness_exact':True,'timings':timing,'speedup':timing['False']['wall_seconds']/timing['True']['wall_seconds'],'scope':'Local kernel checks, no native or scientific census dispatch'}
(h/'results/KERNEL_PARITY_AND_TIMING.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
