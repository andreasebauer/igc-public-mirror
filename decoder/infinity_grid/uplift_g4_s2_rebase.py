from __future__ import annotations
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import gzip, json
from .canon import canonical_sha256

class G4S2RebaseError(RuntimeError): pass

def _resource(name:str)->Path:
    return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def spec()->dict[str,Any]:
    p=_resource('G4_S2_REBASE_MINIMAL_SUFFICIENT_READ_SPEC_V1.json')
    o=json.loads(p.read_text())
    if o.get('schema_id')!='IG_G4_S2_REBASE_MINIMAL_SUFFICIENT_READ_SPEC_V1': raise G4S2RebaseError('bad spec schema')
    if canonical_sha256({k:v for k,v in o.items() if k!='science_sha256'})!=o.get('science_sha256'): raise G4S2RebaseError('spec hash mismatch')
    return o

def load_signatures(path:str|Path)->dict[str,Any]:
    with gzip.open(Path(path),'rt',encoding='utf-8') as fh: o=json.load(fh)
    rows=o.get('rows',[])
    if len(rows)!=spec()['input_evidence']['expected_s1_observations']: raise G4S2RebaseError(f'expected 992 rows, got {len(rows)}')
    return o

def verify_authority(*,s0:Mapping[str,Any],s1:Mapping[str,Any],s1_replay:Mapping[str,Any],s1_closeout:Mapping[str,Any])->dict[str,Any]:
    f=[]
    if s0.get('schema_id')!='IG_G4_S0_REBASE_INTERFACE_RESULT_V1' or s0.get('status')!='PASS': f.append('S0')
    if s1.get('schema_id')!='IG_G4_S1_REBASE_CONTEXT_READ_RESULT_V1' or s1.get('status')!='PASS': f.append('S1')
    if s1.get('science_sha256')!=spec()['input_evidence']['g4_s1_rebase_science_sha256']: f.append('S1_SCIENCE_BINDING')
    if s1.get('old_tier1_falsified_by_rebase') is not True: f.append('OLD_TIER1_NOT_FALSIFIED')
    if s1.get('next_authorized_stage')!='G4:S2.REBASE': f.append('S1_UNLOCK')
    if s1_replay.get('certification')!='CERTIFIED_PASS' or s1_replay.get('primary_science_sha256')!=s1.get('science_sha256'): f.append('S1_REPLAY')
    if s1_closeout.get('status')!='CERTIFIED_PASS' or s1_closeout.get('next_authorized_stage')!='G4:S2.REBASE': f.append('S1_CLOSEOUT')
    if f: raise G4S2RebaseError('authority failed: '+','.join(f))
    out={'schema_id':'IG_G4_S2_REBASE_AUTHORITY_V1','status':'PASS','s0_science_sha256':s0.get('science_sha256'),'s1_science_sha256':s1.get('science_sha256'),'s1_replay_science_sha256':s1_replay.get('science_sha256'),'s1_closeout_science_sha256':s1_closeout.get('science_sha256'),'spec_science_sha256':spec()['science_sha256'],'lower_layer_rematerialization':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES'}
    out['science_sha256']=canonical_sha256(out); return out

def _tier_values(s0:Mapping[str,Any])->tuple[dict[str,dict[int,Any]],dict[int,str],dict[str,str]]:
    c=s0['certified_term_corpus']; d=c['hidden_diagnostics']; values={}; key_to_ref={}
    for k,t in c['terms'].items():
        ref=str(t['term']['term_ref']); key_to_ref[k]=ref
        diag=d[k]
        values[ref]={
            0:list(t['term']['total_caps']),
            1:diag['tier1_scalar_tree_summaries'],
            2:diag['repaired_hidden_state']['shell_profile_multiset'],
            3:diag['repaired_hidden_state']['paired_support_load_distance_signature_multiset'],
            4:{'shell_profile_multiset':diag['repaired_hidden_state']['shell_profile_multiset'],'paired_support_load_distance_signature_multiset':diag['repaired_hidden_state']['paired_support_load_distance_signature_multiset']},
            5:diag['topology_canon'],
        }
    names={int(x['tier']):x['name'] for x in spec()['ordered_tiers']}
    return values,names,key_to_ref

def _vh(v:Any)->str: return canonical_sha256({'descriptor_value':v})

def audit(*,s0:Mapping[str,Any],s1:Mapping[str,Any],s1_replay:Mapping[str,Any],s1_closeout:Mapping[str,Any],signature_index:Mapping[str,Any])->dict[str,Any]:
    auth=verify_authority(s0=s0,s1=s1,s1_replay=s1_replay,s1_closeout=s1_closeout)
    rows=list(signature_index['rows']); values,names,key_to_ref=_tier_values(s0); refs=set(values)
    if set(s1['term_load']['term_key_to_ref'].values())!=refs: raise G4S2RebaseError('term refs mismatch')
    tiers=[]; first=None
    eligibility={int(x['tier']):bool(x['eligible']) for x in spec()['ordered_tiers']}
    reasons={int(x['tier']):x['reason'] for x in spec()['ordered_tiers']}
    for tier in range(6):
        grouped=defaultdict(list)
        for r in rows:
            tr,cr=str(r['target_ref']),str(r['context_ref'])
            if tr not in refs or cr not in refs: raise G4S2RebaseError('row outside corpus')
            op=tuple(map(int,r['operator'])); ori=str(r['orientation']); sh=str(r['operational_signature']['science_sha256'])
            grouped[(_vh(values[tr][tier]),_vh(values[cr][tier]),op,ori)].append(sh)
        conflicts=[]; max_mult=0
        for k,vals in grouped.items():
            max_mult=max(max_mult,len(vals)); uniq=sorted(set(vals))
            if len(uniq)>1: conflicts.append({'target_descriptor_sha256':k[0],'context_descriptor_sha256':k[1],'operator':list(k[2]),'orientation':k[3],'representative_count':len(vals),'observer_value_count':len(uniq),'observer_hashes':uniq})
        rr={'tier':tier,'name':names[tier],'eligible':eligibility[tier],'eligibility_reason':reasons[tier],'descriptor_class_count':len({_vh(values[r][tier]) for r in refs}),'quotient_pair_context_key_count':len(grouped),'max_representatives_per_key':max_mult,'conflict_count':len(conflicts),'complete_observer_single_valued':len(conflicts)==0,'conflicts':conflicts[:12]}
        rr['science_sha256']=canonical_sha256(rr); tiers.append(rr)
        if first is None and rr['eligible'] and rr['complete_observer_single_valued']: first=rr
    # Fail closed on expected scientific facts from S1.
    if tiers[0]['complete_observer_single_valued']: raise G4S2RebaseError('CAPS7 unexpectedly sufficient')
    if tiers[1]['complete_observer_single_valued']: raise G4S2RebaseError('old Tier1 unexpectedly sufficient despite certified falsification')
    if first is None:
        status='REVIEW_REQUIRED'; cls='G4_S2_REBASE_NO_ELIGIBLE_REPAIRED_READ_SUFFICIENT_S3_REBASE_LOCKED'; nxt=None; cand=None
    else:
        status='PASS'; cls=f"G4_S2_REBASE_TIER{first['tier']}_{first['name']}_BOUNDED_PAIR_OBSERVER_CANDIDATE_EARNED_S3_REBASE_UNLOCKED"; nxt='G4:S3.REBASE'; cand={'tier':first['tier'],'name':first['name']}
    out={'schema_id':'IG_G4_S2_REBASE_MINIMAL_SUFFICIENT_READ_RESULT_V1','status':status,'stage_ref':'G4:S2.REBASE','classification':cls,'authority':auth,'cost_control':{'new_science_kernels':0,'reused_s1_observer_rows':len(rows),'lower_layer_rematerialization':False},'frozen_domain':{'term_count':4,'operator_count':31,'orientation_count':2,'s1_observation_count':len(rows)},'tier_audit':tiers,'minimal_sufficient_eligible_tier':cand,'candidate_read_earned_for_s3_rebase':cand is not None,'public_descriptor_promoted':False,'hidden_state_promoted':False,'g4_s3_rebase_unlocked':cand is not None,'g4_rebase_complete':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','g4_graduated':False,'next_authorized_stage':nxt,'nonclaims':spec()['nonclaims']}
    out['science_sha256']=canonical_sha256(out); return out

def compare(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    checks={'primary_pass':primary.get('status')=='PASS','cold_pass':cold.get('status')=='PASS','science_sha_equal':primary.get('science_sha256')==cold.get('science_sha256'),'classification_equal':primary.get('classification')==cold.get('classification'),'tier_audit_equal':primary.get('tier_audit')==cold.get('tier_audit'),'candidate_equal':primary.get('minimal_sufficient_eligible_tier')==cold.get('minimal_sufficient_eligible_tier')}
    ok=all(checks.values()); out={'schema_id':'IG_G4_S2_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256')}; out['science_sha256']=canonical_sha256(out); return out

def closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
    ok=replay.get('certification')=='CERTIFIED_PASS' and primary.get('status')=='PASS' and cold.get('status')=='PASS'
    out={'schema_id':'IG_G4_S2_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':primary.get('classification'),'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'replay_science_sha256':replay.get('science_sha256'),'minimal_sufficient_eligible_tier':primary.get('minimal_sufficient_eligible_tier'),'candidate_read_earned_for_s3_rebase':primary.get('candidate_read_earned_for_s3_rebase') is True,'public_descriptor_promoted':False,'g4_s3_rebase_unlocked':ok,'g4_rebase_complete':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','next_authorized_stage':'G4:S3.REBASE' if ok else None}; out['science_sha256']=canonical_sha256(out); return out
