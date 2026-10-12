#!/usr/bin/env python3
"""Classify all 21 affected retained C1 cases using independent typed profiles.

Each case runs in a separate OS process. Both historical and corrected results
are retained; equivalence is checked against the pre-canonicalization input.
"""


def _require_integrity(condition, message=None):
    """Acceptance invariant retained under optimized Python."""
    if not condition:
        if message is None:
            raise AssertionError()
        raise AssertionError(message())

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infinity_grid.o7_c1_typed_baseline import reconstruct_original_case
from infinity_grid.o7_edge_canonicalization import canonicalize_edges
from infinity_grid.o7_c1_outer_action_profile import independent_materialization_profile


def case(task):
    kind,index,root=task
    # Memoize only byte-identical immutable parent resources within this case.
    # Raw/new resources with a different nested order are recomputed separately.
    from infinity_grid import o6_resource_observer as observer
    from infinity_grid.canon import canonical_sha256
    original_profile=observer.resource_profile
    cache={}
    def freeze(value):
        return tuple(freeze(x) for x in value) if isinstance(value,(tuple,list)) else value
    def cached_profile(resource,templates,pairs):
        key=(freeze(resource),canonical_sha256(templates),freeze(pairs))
        if key not in cache:
            if len(cache)>=32:cache.clear()
            cache[key]=original_profile(resource,templates,pairs)
        return cache[key].copy()
    observer.resource_profile=cached_profile
    result=reconstruct_original_case(Path(root)/f'{kind}_{index}',kind=kind,index=index,
        canonicalizer=canonicalize_edges,profile_calculator=independent_materialization_profile)
    dest=Path(root)/f'{kind}_{index}.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true');a=p.parse_args()
    evidence=json.loads((a.output/'NATIVE_IDENTITY_VALIDATION.json').read_text())
    tasks=[(r['kind'],r['index'],str(a.output/'affected_profiles')) for r in evidence['records'] if r['kind'] in ['O7xO6','O7xO7'] and not r['legacy_materialization_preserved']]
    _require_integrity(len(tasks)==21)
    start=time.monotonic();results=[];remaining=[]
    for task in tasks:
        kind,index,root=task;checkpoint=Path(root)/f'{kind}_{index}.json'
        if a.resume and checkpoint.is_file():
            row=json.loads(checkpoint.read_text())
            _require_integrity(row['kind']==kind and row['index']==index)
            _require_integrity(row['canonicalization_version']=='IG_E7_NESTED_BLISS_V2')
            _require_integrity(row['status']=='PASS_WITH_EXPLAINED_HISTORICAL_DIFFERENCE')
            _require_integrity(all(row['checks'].values()))
            results.append(row)
        else:
            remaining.append(task)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(case,t) for t in remaining]
        for future in as_completed(futures):
            row=future.result();results.append(row);print(row['kind'],row['index'],row['status'],flush=True)
    result={'schema_id':'IG_STEP1_AFFECTED_C1_PROFILES_V1','status':'PASS' if all(r['status']=='PASS_WITH_EXPLAINED_HISTORICAL_DIFFERENCE' for r in results) else 'FAIL',
        'cases':sorted(results,key=lambda r:(r['kind'],r['index'])),
        'affected_cases':len(results),'explained_typed_profile_differences':sum(not r['historical_checks']['typed_profile'] for r in results),
        'raw_projection_preservation_checks':sum(r['checks'].get('raw_profile_preserved',False) for r in results),
        'legacy_baseline_reproductions':sum(r['checks'].get('historical_baseline_reproduced',False) for r in results),
        'wall_seconds':time.monotonic()-start,'reused_completed_validation_cases':len(tasks)-len(remaining),
        'resume_scope':'Engineering validation checkpoints at unchanged mathematical kernels; not a fresh corpus claim',
        'scope':'All affected cases in the retained 192-case C1 panel; no full census regeneration'}
    (a.output/'AFFECTED_C1_PROFILES.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2),flush=True)
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
