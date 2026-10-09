"""Independent membership, partition, native-contract and closure audit."""
from pathlib import Path
import json,gzip,hashlib,collections,sys,shutil,tempfile
B=Path(__file__).resolve().parent;W=B.parent;E=Path('/tmp/ig_engine0237');sys.path.insert(0,str(E))
from infinity_grid.canon import canonical_sha256
from infinity_grid.result_contracts import normalize
from infinity_grid.workflow_guard import preflight
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=json.loads((B/'PLAN.json').read_text());r=json.loads((B/'PREREGISTRATION.json').read_text());s=json.loads(gzip.decompress((W/'continuation0238/COLLISION_MEMBERS.json.gz').read_bytes()))
    assert sha(W/'continuation0238/COLLISION_MEMBERS.json.gz')==p['membership_sha256']
    prior=json.loads((W/'continuation0238/PILOT_CASES.json').read_text());earned=set(prior['class_keys']);assert p['earned_class_keys']==sorted(earned)
    rows={x['ordinal']:x for x in s['rows']};grouped=collections.defaultdict(set)
    for x in s['rows']:grouped[x['record']['outcome_science_sha256']].add(x['ordinal'])
    expected=sorted(set(grouped)-earned);seen=[];ordinals=[]
    # Independently recover the greedy boundaries from original grouped membership.
    boundaries=[];current=[];count=0
    for key in expected:
        if count+len(grouped[key])>62:boundaries.append(current);current=[];count=0
        current.append(key);count+=len(grouped[key])
    if current:boundaries.append(current)
    assert len(boundaries)==len(p['chunks'])==len(r['specifications'])==114
    binding=canonical_sha256(r['source_sha256'])
    for name,digest in r['source_sha256'].items():assert sha(E/'infinity_grid'/name.split('/',1)[1] if name.startswith('engine/') else B/name)==digest
    for i,(chunk,specrow) in enumerate(zip(p['chunks'],r['specifications'])):
        assert chunk['chunk_index']==specrow['chunk_index']==i
        path=B/chunk['cases_path'];assert sha(path)==chunk['cases_sha256']
        c=json.loads(path.read_text());assert c['class_keys']==chunk['class_keys']==boundaries[i]
        target=sorted(o for k in c['class_keys'] for o in grouped[k]);assert target==chunk['member_ordinals']==[x['ordinal'] for x in c['cases']]
        assert len(target)==chunk['case_count']<=62 and len(c['class_keys'])==chunk['class_count']
        assert all(rows[x['ordinal']]==x for x in c['cases']);assert c['bridge_pairs']==prior['bridge_pairs']
        seen+=c['class_keys'];ordinals+=target
        specpath=B/specrow['path'];assert sha(specpath)==specrow['sha256'];spec=json.loads(specpath.read_text());params=spec['execution']['parameters']
        assert params['source_binding']==binding and params['chunk_id']==chunk['chunk_id'] and params['case_count']==len(target)
        for x in spec['inputs']:assert sha(Path(x['path']))==x['sha256']==params['bindings'][x['logical_name']]
        normalized=normalize(spec['output_contract'],spec['execution'],spec['question']);assert normalized['claim']=='EXECUTION_ONLY' and normalized['declared_outcomes']==['PASS_COMPLETE_CLASS_Q2_CHUNK_V1']
        checks={x['pointer']:x['equals'] for x in spec['output_contract']['result_checks']};assert checks['/cases_checked']==len(target) and checks['/complete_Q2_classes_checked']==len(c['class_keys']) and checks['/chunk_id']==chunk['chunk_id'] and checks['/new_admissions']==0 and checks['/G2_promotion'] is False and checks['/full_historical_public_observer_compared'] is False
    assert seen==expected and len(seen)==len(set(seen))==3523 and len(ordinals)==len(set(ordinals))==7058
    assert not set(ordinals)&set(p['earned_member_ordinals'])
    for f in (B/'project').rglob('*.py'):
        if f.name!='handler.py':assert sha(f)==sha(W/'continuation0238'/f.relative_to(B))
    with tempfile.TemporaryDirectory(prefix='ig_preflight0240_') as t:
        root=Path(t);shutil.copytree(E/'infinity_grid',root/'infinity_grid');shutil.copytree(B/'project',root/'project');gate=preflight(root,[root/'project/handler.py',root/'project/worker.py'])
    assert gate['status']=='PASS' and len(gate['modules'])==22
    (B/'PREFLIGHT.json').write_text(json.dumps(gate,indent=2)+'\n')
    hist=dict(collections.Counter(len(grouped[k]) for k in expected));assert hist=={2:3511,3:12}
    out=dict(status='PASS_INDEPENDENT_REMAINING_Q2_REGISTRATION_AUDIT',chunks=114,complete_classes=3523,member_records=7058,class_size_histogram=hist,prior_overlap=0,all_members_retained_exact=True,greedy_partition_exact=True,native_contracts_normalized=114,closure_modules_checked=22,scientific_helpers_unchanged=True,new_cases_executed=0,plan_sha256=sha(B/'PLAN.json'),preregistration_sha256=sha(B/'PREREGISTRATION.json'),resource_gate='PENDING_NATIVE_ADAPTER_AND_DRY',full_historical_observer_gate='BLOCKED_EXACT_SERIALIZER_BINDING',master_slices=152,new_admissions=0,G2_promotion=False)
    (B/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
