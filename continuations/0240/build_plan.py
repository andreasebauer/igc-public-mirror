"""Register whole-class chunks from saved membership; no scientific execution."""
from pathlib import Path
import json, gzip, hashlib, shutil, datetime
B=Path(__file__).resolve().parent; W=B.parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj): p.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
def main():
    source=W/'continuation0238/COLLISION_MEMBERS.json.gz'
    assert sha(source)=='6ebea91f46e042d75229aa98b25e00b5be932e4f70fcfa0353ed88c26a393e89'
    prior=json.loads((W/'continuation0238/PILOT_CASES.json').read_text())
    cold=json.loads((W/'continuation0239/COLD_AUDIT.json').read_text())
    preserved=json.loads((W/'continuation0239/CHECKPOINT_PRESERVED.json').read_text())
    saved=json.loads((W/'continuation0239/SAVE_RECEIPT.json').read_text())
    assert cold['cases_exact']==62 and cold['complete_Q2_classes_exact']==31 and cold['public_projections_and_operational_witnesses_exact']
    assert not preserved['pending_objects'] and preserved['pending_bytes']==0
    assert all(v['drive_exact_readback_verified'] and v['local_metadata_applied'] for v in saved['artifacts'].values())
    frozen={'schema_id':'IG_REMAINING_COMPLETE_CLASS_Q2_SELECTION_RULE_V1','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_sha256':sha(source),'prior_case_sha256':sha(W/'continuation0238/PILOT_CASES.json'),'prior_cold_sha256':sha(W/'continuation0239/COLD_AUDIT.json'),'prior_preservation_sha256':sha(W/'continuation0239/CHECKPOINT_PRESERVED.json'),'prior_save_receipt_sha256':sha(W/'continuation0239/SAVE_RECEIPT.json'),'selection_rule':'Exclude only the 31 earned complete class keys. Sort remaining keys lexicographically, greedily append whole classes to each chunk up to 62 members, start next chunk before overflow. Retain every source ordinal and full saved record. No fresh scientific screening.','max_members_per_chunk':62,'new_scientific_cases_executed':0}
    rule=B/'SELECTION_PREREGISTRATION.json'
    if rule.exists():
        old=json.loads(rule.read_text()); frozen['frozen_utc']=old['frozen_utc']; assert old==frozen
    else: write(rule,frozen)
    all_data=json.loads(gzip.decompress(source.read_bytes())); rows={r['ordinal']:r for r in all_data['rows']}
    earned=set(prior['class_keys']); assert len(earned)==31
    classes=sorted((c for c in all_data['classes'] if c['outcome_science_sha256'] not in earned),key=lambda c:c['outcome_science_sha256'])
    chunks=[]; group=[]; size=0
    for c in classes:
        n=len(c['member_ordinals']); assert n in (2,3)
        if size+n>62: chunks.append(group); group=[]; size=0
        group.append(c); size+=n
    if group: chunks.append(group)
    (B/'chunks').mkdir(exist_ok=True)
    manifest=[]
    for index,group in enumerate(chunks):
        ordinals=sorted(o for c in group for o in c['member_ordinals'])
        data={'schema_id':'IG_REMAINING_COMPLETE_CLASS_Q2_CASES_V1','chunk_index':index,'cases':[rows[o] for o in ordinals],'class_keys':[c['outcome_science_sha256'] for c in group],'bridge_pairs':prior['bridge_pairs']}
        p=B/'chunks'/f'{index:03d}.json'; write(p,data)
        manifest.append({'chunk_index':index,'chunk_id':f'complete_class_Q2_{index:03d}','cases_path':str(p.relative_to(B)),'cases_sha256':sha(p),'case_count':len(ordinals),'class_count':len(group),'class_keys':data['class_keys'],'member_ordinals':ordinals})
    assert len(classes)==3523 and sum(c['case_count'] for c in manifest)==7058
    plan={'schema_id':'IG_REMAINING_COMPLETE_CLASS_Q2_PLAN_V1','selection_preregistration_sha256':sha(rule),'membership_sha256':sha(source),'earned_class_keys':sorted(earned),'earned_member_ordinals':sorted(c['ordinal'] for c in prior['cases']),'chunks':manifest,'remaining_class_count':3523,'remaining_member_count':7058,'full_historical_public_observer_compared':False,'G2_promotion':False,'master_slices':152,'new_admissions':0}
    write(B/'PLAN.json',plan)
    if not (B/'project').exists():
        shutil.copytree(W/'continuation0238/project',B/'project',ignore=shutil.ignore_patterns('__pycache__'))
    # Keep the scientific evaluator byte-for-byte frozen; only the handler is adapted.
    assert sha(B/'project/worker.py')=='fa39f23632baa6512f6db391eeb9d3ecf3968f333a5016ab4cc52cb9a2f6aa00'
    print(json.dumps({'chunks':len(manifest),'classes':len(classes),'members':7058,'size_histogram':{str(n):sum(len(c['member_ordinals'])==n for c in classes) for n in (2,3)},'last_chunk_members':manifest[-1]['case_count']}))
if __name__=='__main__':main()
