"""Freeze bounded native specs; no scientific execution."""
from pathlib import Path
import json,hashlib,copy
B=Path(__file__).resolve().parent;W=B.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,o):p.write_text(json.dumps(o,sort_keys=True,indent=2)+'\n')
def main():
    plan=json.loads((B/'PLAN.json').read_text());base=json.loads((W/'continuation0238/NEXT_SPEC.json').read_text());(B/'specs').mkdir(exist_ok=True);specs=[]
    code={str(p.relative_to(B)):sha(p) for p in (B/'project').rglob('*.py')}
    for n in ('uplift_structural.py','workflow_guard.py'):code['engine/'+n]=sha(Path(base['engine_source'])/'infinity_grid'/n)
    binding=hashlib.sha256(json.dumps(code,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    for row in plan['chunks']:
        s=copy.deepcopy(base);job=f"MASTER.G2.COMPLETE.CLASS.Q2.REMAINING.0240.{row['chunk_index']:03d}"
        s['job_id']=job;s['project_source']=str(B/'project');s['question'].update(stage_id=job,description=f"Frozen remaining chunk {row['chunk_index']}: {row['case_count']} members / {row['class_count']} whole classes; historical deterministic Q2 only",outcomes=['PASS_COMPLETE_CLASS_Q2_CHUNK_V1'])
        p=s['execution']['parameters'];p['bindings']['cases']=row['cases_sha256'];p.update(source_binding=binding,chunk_id=row['chunk_id'],case_count=row['case_count'],class_count=row['class_count'])
        s['inputs'][2].update(path=str(B/row['cases_path']),sha256=row['cases_sha256'])
        replacements={'/outcome':'PASS_COMPLETE_CLASS_Q2_CHUNK_V1','/cases_checked':row['case_count'],'/complete_Q2_classes_checked':row['class_count'],'/G2_primary_realizations':row['case_count'],'/G2_swap_checks':row['case_count']}
        for check in s['output_contract']['result_checks']:
            if check['pointer'] in replacements:check['equals']=replacements[check['pointer']]
        s['output_contract']['result_checks'].append({'pointer':'/chunk_id','equals':row['chunk_id']})
        path=B/'specs'/f"{row['chunk_index']:03d}.json";write(path,s);specs.append({'chunk_index':row['chunk_index'],'path':str(path.relative_to(B)),'sha256':sha(path)})
    registration={'schema_id':'IG_REMAINING_COMPLETE_CLASS_Q2_PREREG_V1','plan_sha256':sha(B/'PLAN.json'),'source_sha256':code,'specifications':specs,'remaining_classes':3523,'remaining_members':7058,'chunk_count':len(specs),'max_members_per_chunk':62,'execution_model':'One bounded native capture per chunk. Shared runtime task transactions resume only missing work within that capture. Cross-capture completion requires exact registered hashes, native VERIFIED result, independent cold equality for ALL public and operational fields, zero pending objects and exact saved readbacks. Never skip on native result alone.','resource_gate':'Native adapter/resume controls and first next-chunk dry resource gate REQUIRED before official execution. Existing 32-carrier pilot evidence does not establish all 178-carrier resource feasibility.','workers':1,'memory_budget_bytes':4294967296,'workspace_budget_bytes_per_capture':2147483648,'max_result_bytes':16777216,'max_work_seconds_per_task':1800,'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1','prior_earned_classes':31,'prior_earned_members':62,'new_cases_executed':0,'historical_full_observer_gate':'BLOCKED_UNTIL_EXACT_SERIALIZER_SOURCE_BOUND','full_historical_public_observer_compared':False,'G2_promotion':False,'master_slices':152,'new_admissions':0,'stopping_rule':base['question']['stopping_rule'],'dry_first_class_ordinals':sorted(c['ordinal'] for c in json.loads((B/'chunks/000.json').read_text())['cases'] if c['record']['outcome_science_sha256']==plan['chunks'][0]['class_keys'][0]),'dry_official_overlap_declared':True,'blind_holdout_claim':False}
    write(B/'PREREGISTRATION.json',registration)
    write(B/'LEDGER_INITIAL.json',{'schema_id':'IG_REMAINING_Q2_EARNED_LEDGER_V1','plan_sha256':sha(B/'PLAN.json'),'preregistration_sha256':sha(B/'PREREGISTRATION.json'),'completed_chunks':[],'pending_chunk_indices':[r['chunk_index'] for r in plan['chunks']],'next_chunk_index':0,'remaining_classes':3523,'remaining_members':7058,'new_cases_executed':0})
    print(json.dumps({'specs_registered':len(specs),'source_files':len(code),'next_spec':str(B/'specs/000.json')}))
if __name__=='__main__':main()
