"""Preregistered one-case feasibility check; never an official or admission result."""
from pathlib import Path
import json,sys,time,resource,traceback
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204');sys.path.insert(0,str(B))
from project.worker import prepare,check_case
if __name__=='__main__':
    try:
        spec=json.loads((B/'SPEC.json').read_text());h=spec['execution']['parameters']['bindings'];i={x['logical_name']:x['path'] for x in spec['inputs']}
        payload={k+'_path':i[k] for k in h};payload.update({k+'_sha256':v for k,v in h.items()});t=time.monotonic()
        engine,by_ref,pop,cases=prepare(payload);print('LOAD_AND_DAG_RESTORE_COMPLETE',flush=True)
        data=check_case(engine,by_ref,pop,cases['cases'][0],[tuple(x) for x in cases['bridge_pairs']],spec['execution']['parameters']['source_binding'])
        (B/'DRY_CASE.json').write_text(json.dumps(data,indent=2)+'\n')
        out=dict(status='PASS_ONE_CASE_DRY_PREFLIGHT',official_cases_executed=0,dry_cases_checked=1,G1_candidate_generation=0,dry_G2_primary_realizations=1,dry_G2_swap_checks=1,elapsed_seconds=time.monotonic()-t,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,master_slices=152,new_admissions=0,G2_promotion=False)
        if out['peak_RSS_bytes']>spec['resources']['memory_budget_bytes']:raise ValueError('DRY_MEMORY_BUDGET')
        (B/'DRY_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
    except BaseException:
        (B/'STOPPED.txt').write_text(traceback.format_exc());raise
