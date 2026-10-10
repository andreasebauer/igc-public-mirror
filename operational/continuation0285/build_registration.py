"""Saved-evidence alignment and new observer capture specifications; no scientific execution."""
import ast,copy,gzip,hashlib,json,pathlib,shutil,zipfile
B=pathlib.Path(__file__).resolve().parent;W=B.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(o):return json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def write(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,sort_keys=True,indent=2)+'\n')
def main():
    plan=json.loads((W/'continuation0240/PLAN.json').read_text())
    prereg=json.loads((W/'continuation0240/PREREGISTRATION.json').read_text())
    assert sha(W/'continuation0240/PLAN.json')==prereg['plan_sha256']
    selected={};chunks={}
    for row in plan['chunks']:
        p=W/'continuation0240'/row['cases_path'];assert sha(p)==row['cases_sha256']
        chunks[row['chunk_index']]=json.loads(p.read_text())
        for case in chunks[row['chunk_index']]['cases']:
            assert case['ordinal'] not in selected;selected[case['ordinal']]=case
    assert len(selected)==7058
    z=zipfile.ZipFile(W/'continuation0283/Infinity_Grid_Decoder_G2_S1_D4_Q2_FULL_REAUDIT_COMPLETE_2026-09-02.zip')
    assert sha(pathlib.Path(z.filename))=='8bb9cc6e53e77425fb75985f3681635cea75375d2aa227b97774ae740af9b97f'
    sigs={};rh=hashlib.sha256();sh=hashlib.sha256();rows=0
    with gzip.GzipFile(fileobj=z.open('evidence/G2_S1_REPAIRED_D4_Q2_PAIR_CONNECTION_RECORDS.jsonl.gz')) as rf,gzip.GzipFile(fileobj=z.open('evidence/G2_S1_REPAIRED_D4_Q2_AUDIT_SIGNATURES.jsonl.gz')) as sf:
        while True:
            rb=rf.readline();sb=sf.readline()
            if not rb and not sb:break
            assert rb and sb;rh.update(rb);sh.update(sb);r=json.loads(rb);s=json.loads(sb)
            assert r['outcome_science_sha256']==s['outcome_science_sha256']
            # Ordinals in the frozen selection are zero based.
            if rows in selected:
                case=selected[rows];assert case['record']==r
                sigs[rows]={'ordinal':rows,'source_case_sha256':hashlib.sha256(canonical(case)).hexdigest(),**s}
            rows+=1
    assert rows==580351 and len(sigs)==7058
    assert rh.hexdigest()=='9023c8d4a53ff6bba87a5b6954bac8d37008257aaf5214c67c0c6ba8ddf28049'
    assert sh.hexdigest()=='7ff1a9ce2a63f99928e07d82f42c29feebda27122fd5629e4bbf2b796d52283d'
    candidate=W/'continuation0283/OBSERVER_CANDIDATE_SOURCE.txt'
    def fn(p):return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='public_assembly')
    assert ast.dump(fn(candidate),include_attributes=False)==ast.dump(fn(B/'project/observer.py'),include_attributes=False)
    code={str(p.relative_to(B)):sha(p) for p in (B/'project').rglob('*.py')}
    prior_engine={k:v for k,v in prereg['source_sha256'].items() if k.startswith('engine/')}
    binding=hashlib.sha256(canonical({**code,**prior_engine})).hexdigest();specs=[]
    for row in plan['chunks']:
        idx=row['chunk_index'];expected=B/'expected'/f'{idx:03d}.json'
        write(expected,{'schema_id':'IG_HISTORICAL_PUBLIC_OBSERVER_EXPECTATIONS_V1','chunk_index':idx,'cases':[sigs[c['ordinal']] for c in chunks[idx]['cases']]})
        s=json.loads((W/'continuation0240/specs'/f'{idx:03d}.json').read_text());job=f'MASTER.G2.PUBLIC.OBSERVER.0285.{idx:03d}';chunk=f'historical_public_observer_{idx:03d}'
        s.update(job_id=job,project_source=str(B/'project'));s['question'].update(stage_id=job,description=f'Historical full public observer comparison for frozen Q2 chunk {idx}; {row["case_count"]} members, not full population',outcomes=['PASS_BOUND_HISTORICAL_OBSERVER_CHUNK_V1'],stopping_rule='Stop first prior Q2 check, public observer hash/payload split, source/input/runtime/resource/cold-restore mismatch; no admission or G2 promotion')
        params=s['execution']['parameters'];params.update(chunk_id=chunk,source_binding=binding);params['bindings']['observer_expected']=sha(expected)
        s['inputs'].append({'logical_name':'observer_expected','path':str(expected),'sha256':sha(expected)})
        s['output_contract']['result_checks']=[{'pointer':'/'+k,'equals':v} for k,v in {'outcome':'PASS_BOUND_HISTORICAL_OBSERVER_CHUNK_V1','chunk_id':chunk,'cases_checked':row['case_count'],'complete_observer_classes_checked':row['class_count'],'historical_observer_hashes_compared':row['case_count'],'full_historical_public_observer_compared':False,'master_slices':152,'new_admissions':0,'G2_promotion':False}.items()]
        p=B/'specs'/f'{idx:03d}.json';write(p,s);specs.append({'chunk_index':idx,'spec_path':str(p.relative_to(B)),'spec_sha256':sha(p),'expectations_sha256':sha(expected),'case_count':row['case_count'],'class_count':row['class_count']})
    preflight={'engine_present':pathlib.Path(s['engine_source']).is_dir(),'missing_bound_inputs':sorted({i['path'] for i in s['inputs'] if not pathlib.Path(i['path']).is_file()}),'free_disk_bytes':shutil.disk_usage(B).free,'native_execution_permitted':False,'reason':'Pinned engine/runtime/input recovery and native dry gate required; specifications not yet native-preflight qualified.'}
    registration={'checkpoint':285,'status':'FROZEN_OBSERVER_SPECIFICATIONS_NATIVE_PREFLIGHT_PENDING','specifications':specs,'selected_members':7058,'selected_complete_classes':3523,'historical_rows_aligned':rows,'historical_record_stream_sha256':rh.hexdigest(),'historical_signature_stream_sha256':sh.hexdigest(),'original_v0292_controller_identity_claimed':False,'binding_basis':'Earlier S1R public_assembly exact AST plus recovered v0301 reconstruction payload and saved downstream mapping certificate; native hash equality remains a test, not a premise.','source_sha256':code,'required_engine_files':prior_engine,'source_binding':binding,'dry_first_chunk':0,'dry_official_overlap_declared':True,'blind_holdout_claim':False,'workers':1,'memory_budget_bytes':4294967296,'minimum_free_disk_bytes':10737418240,'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1','cold_verification':'Every public observer payload/hash, prior Q2 output and operational witness must equal independent cold execution; zero pending saved objects; exact external readbacks before earned ledger advancement.','stopping_rule':s['question']['stopping_rule'],'full_580351_population_compared':False,'earlier_31_class_pilot_included':False,'new_scientific_executions':0,'master_slices':152,'new_admissions':0,'G2_promotion':False,'next_scope':'RESTORE_PINNED_ENGINE_RUNTIME_BOOTSTRAP_AND_ENVIRONMENT_BINDINGS_THEN_NATIVE_OBSERVER_DRY_GATE'}
    write(B/'PREREGISTRATION.json',registration);write(B/'PREFLIGHT.json',preflight)
    print(json.dumps({'specifications':len(specs),'members':len(sigs),'preflight':preflight}))
if __name__=='__main__':main()
