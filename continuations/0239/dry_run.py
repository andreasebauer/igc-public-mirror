"""Declared first complete collision-class dry gate; no official case credit."""
from pathlib import Path
import sys,json,time,resource,hashlib,traceback
B=Path(__file__).resolve().parent;Q=B.parent/'continuation0238';sys.path.insert(0,'/tmp/ig_engine0237');sys.path.insert(0,str(Q))
from project.worker import prepare,check_case
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pre=json.loads((Q/'NEXT_PREREGISTRATION.json').read_text());s=json.loads((Q/'NEXT_SPEC.json').read_text());audit=json.loads((Q/'NEXT_AUDIT.json').read_text());assert sha(Q/'NEXT_SPEC.json')==audit['specification_sha256'] and sha(Q/'NEXT_PREREGISTRATION.json')==audit['preregistration_sha256']
    for n,h in pre['source_sha256'].items():assert sha(Path('/tmp/ig_engine0237/infinity_grid')/n.split('/',1)[1] if n.startswith('engine/') else Q/n)==h
    h=s['execution']['parameters']['bindings'];i={x['logical_name']:x['path'] for x in s['inputs']};payload={k+'_path':i[k] for k in h};payload.update({k+'_sha256':v for k,v in h.items()});t=time.monotonic()
    engine,refs,pop,cases=prepare(payload);print('LOAD_AND_DAG_RESTORE_COMPLETE',flush=True)
    selected=[x for x in cases['cases'] if x['ordinal'] in pre['dry_case_ordinals']];assert len(selected)==2
    data=[check_case(engine,refs,pop,x,[tuple(y) for y in cases['bridge_pairs']],pre['source_binding']) for x in selected]
    assert data[0]['public_projection']==data[1]['public_projection']
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert peak<=pre['memory_budget_bytes']
    (B/'DRY_CASES.json').write_text(json.dumps(data,indent=2)+'\n')
    out=dict(status='PASS_FIRST_COMPLETE_CLASS_RESOURCE_DRY',dry_cases_checked=2,complete_Q2_classes_checked=1,official_cases_executed=0,G1_candidate_generation=0,dry_G2_primary_realizations=2,dry_G2_swap_checks=2,elapsed_seconds=time.monotonic()-t,peak_RSS_bytes=peak,memory_budget_bytes=pre['memory_budget_bytes'],diverse62_resource_feasibility='Not established by two cases; native runtime resource stopping rule remains active',master_slices=152,new_admissions=0,G2_promotion=False,full_historical_public_observer_compared=False)
    (B/'DRY_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (B/'STOPPED.txt').write_text(traceback.format_exc());raise
