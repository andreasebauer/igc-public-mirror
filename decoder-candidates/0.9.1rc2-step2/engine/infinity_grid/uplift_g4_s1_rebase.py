from __future__ import annotations
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
from .canon import canonical_sha256
from .g4_term_state import G3TermState
from .uplift_g4_s1 import pair_context_kernel_measurement, expand_pair_context_kernel_signature

class G4S1RebaseError(RuntimeError): pass

def _resource(name:str)->Path: return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))
def _checked(name:str,schema:str)->dict[str,Any]:
    obj=json.loads(_resource(name).read_text())
    if obj.get('schema_id')!=schema: raise G4S1RebaseError(f'bad {name} schema')
    field='registry_sha256' if name=='G_UPLIFT_EXPERIMENT_REGISTRY_V1.json' else 'science_sha256'
    if canonical_sha256({k:v for k,v in obj.items() if k!=field})!=obj.get(field): raise G4S1RebaseError(f'{name} hash mismatch')
    return obj

def s1_rebase_spec(): return _checked('G4_S1_REBASE_CONTEXT_READ_SPEC_V1.json','IG_G4_S1_REBASE_CONTEXT_READ_SPEC_V1')
def challenge_tiers(): return _checked('G4_REBASE_CHALLENGE_READ_TIERS_V1.json','IG_G4_REBASE_CHALLENGE_READ_TIERS_V1')

def verify_authority(s0:Mapping[str,Any], replay:Mapping[str,Any], closeout:Mapping[str,Any])->dict[str,Any]:
    f=[]
    if s0.get('schema_id')!='IG_G4_S0_REBASE_INTERFACE_RESULT_V1' or s0.get('status')!='PASS': f.append('S0_SCHEMA_STATUS')
    if s0.get('next_authorized_stage')!='G4:S1.REBASE' or s0.get('g4_s1_rebase_unlocked') is not True: f.append('S0_UNLOCK')
    if s0.get('historical_g4_forward_use')!='BLOCKED_UNTIL_REBASE_COMPLETES': f.append('S0_HISTORICAL_BLOCK')
    c=s0.get('certified_term_corpus',{})
    if c.get('schema_id')!='IG_G4_S0_REBASE_G3_TERM_CORPUS_V1' or set(c.get('terms',{}))!={'A','B','C','D'}: f.append('S0_CORPUS')
    cps={x['pair_id']:x for x in c.get('challenge_pairs',[])}
    if set(cps)!={'N4_PATH_STAR','N9_TIER1_COLLISION_REPAIRED_H_SPLIT'}: f.append('CHALLENGE_PAIRS')
    q=cps.get('N9_TIER1_COLLISION_REPAIRED_H_SPLIT',{})
    if not(q.get('same_caps7') is True and q.get('same_old_tier1') is True and q.get('different_repaired_H') is True): f.append('N9_REBASE_CHALLENGE')
    if replay.get('schema_id')!='IG_G4_S0_REBASE_REPLAY_COMPARISON_V1' or replay.get('certification')!='CERTIFIED_PASS': f.append('S0_REPLAY')
    if replay.get('primary_science_sha256')!=s0.get('science_sha256') or replay.get('cold_science_sha256')!=s0.get('science_sha256'): f.append('S0_REPLAY_BINDING')
    if closeout.get('schema_id')!='IG_G4_S0_REBASE_CERTIFIED_CLOSEOUT_V1' or closeout.get('status')!='CERTIFIED_PASS' or closeout.get('next_authorized_stage')!='G4:S1.REBASE': f.append('S0_CLOSEOUT')
    if closeout.get('primary_science_sha256')!=s0.get('science_sha256') or closeout.get('certified_term_corpus_sha256')!=c.get('science_sha256'): f.append('S0_CLOSEOUT_BINDING')
    out={'schema_id':'IG_G4_S1_REBASE_AUTHORITY_V1','status':'PASS' if not f else 'FAIL','failures':f,'s0_science_sha256':s0.get('science_sha256'),'s0_term_corpus_sha256':c.get('science_sha256'),'s1_rebase_spec_sha256':s1_rebase_spec()['science_sha256'],'tier_resource_sha256':challenge_tiers()['science_sha256'],'g3_caps7_graduation_preserved':True,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES'}; out['science_sha256']=canonical_sha256(out)
    if f: raise G4S1RebaseError('authority failed: '+','.join(f))
    return out

def load_states(s0:Mapping[str,Any]):
    c=s0['certified_term_corpus']; states={k:G3TermState.from_wire(v['term']) for k,v in c['terms'].items()}
    if set(states)!={'A','B','C','D'}: raise G4S1RebaseError('four-term corpus lost')
    # exact public collisions per frozen pairs
    for p in c['challenge_pairs']:
        if states[p['left']].total_caps!=states[p['right']].total_caps: raise G4S1RebaseError('public collision lost')
    refs={k:states[k].construction_digest for k in states}
    if len(set(refs.values()))!=4: raise G4S1RebaseError('term refs not unique')
    meta={'schema_id':'IG_G4_S1_REBASE_TERM_LOAD_V1','status':'PASS','term_key_to_ref':refs,'exact_carrier_count':4,'challenge_pairs':c['challenge_pairs'],'lower_layer_rematerialization':False,'hidden_structure_used_as_context_input':False,'producer_term_corpus_sha256':c['science_sha256']}; meta['science_sha256']=canonical_sha256(meta)
    return {refs[k]:states[k] for k in states}, refs, meta

def _diag_value(diag:Mapping[str,Any],tier:int):
    if tier==0: return None
    if tier==1: return diag['tier1_scalar_tree_summaries']
    if tier==2: return diag['repaired_hidden_state']['shell_profile_multiset']
    if tier==3: return diag['repaired_hidden_state']['paired_support_load_distance_signature_multiset']
    if tier==4: return {'shell_profile_multiset':diag['repaired_hidden_state']['shell_profile_multiset'],'paired_support_load_distance_signature_multiset':diag['repaired_hidden_state']['paired_support_load_distance_signature_multiset']}
    if tier==5: return diag['topology_canon']
    raise G4S1RebaseError('bad tier')

def _pair_tier_analysis(s0:Mapping[str,Any],p:Mapping[str,Any]):
    d=s0['certified_term_corpus']['hidden_diagnostics']; L,R=p['left'],p['right']; rows=[]
    for t in challenge_tiers()['ordered_tiers']:
        n=int(t['tier'])
        if n==0: va=vb=list(G3TermState.from_wire(s0['certified_term_corpus']['terms'][L]['term']).total_caps)
        else: va,vb=_diag_value(d[L],n),_diag_value(d[R],n)
        rows.append({'tier':n,'name':t['name'],'pair_equal':va==vb,'left_value':va,'right_value':vb})
    return rows

def finalize(*,s0:Mapping[str,Any],authority:Mapping[str,Any],load_meta:Mapping[str,Any],rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    refs=load_meta['term_key_to_ref']; expected=4*4*31*2
    if len(rows)!=expected: raise G4S1RebaseError(f'expected {expected} rows, got {len(rows)}')
    bykey={}
    for r in rows:
        k=(str(r['target_ref']),str(r['context_ref']),str(r['orientation']),int(r['operator'][0]),int(r['operator'][1]))
        if k in bykey: raise G4S1RebaseError('duplicate row')
        bykey[k]=r
    pair_results=[]; any_sep=False; old_tier1_falsified=False
    for p in s0['certified_term_corpus']['challenge_pairs']:
        lk,rk=p['left'],p['right']; lr,rr=refs[lk],refs[rk]; witness=None
        # compare the two targets in every same frozen context/orientation/operator
        for ctxk,ctxr in sorted(refs.items()):
            for ori in ('TARGET_LEFT_CONTEXT_RIGHT','CONTEXT_LEFT_TARGET_RIGHT'):
                for a,b in sorted({(k[3],k[4]) for k in bykey}):
                    ka=(lr,ctxr,ori,a,b); kb=(rr,ctxr,ori,a,b)
                    if ka in bykey and kb in bykey and bykey[ka]['operational_signature']['science_sha256']!=bykey[kb]['operational_signature']['science_sha256']:
                        witness={'context_key':ctxk,'context_ref':ctxr,'orientation':ori,'operator':[a,b],'left_signature':bykey[ka]['operational_signature'],'right_signature':bykey[kb]['operational_signature']}; break
                if witness: break
            if witness: break
        tiers=_pair_tier_analysis(s0,p); lowest=next((x for x in tiers if int(x['tier'])>0 and not x['pair_equal']),None)
        sep=witness is not None; any_sep|=sep
        if p['pair_id']=='N9_TIER1_COLLISION_REPAIRED_H_SPLIT' and sep and p.get('same_old_tier1') is True: old_tier1_falsified=True
        pair_results.append({'pair_id':p['pair_id'],'left_key':lk,'right_key':rk,'same_caps7':p['same_caps7'],'same_old_tier1':p['same_old_tier1'],'different_repaired_H':p['different_repaired_H'],'operationally_separated':sep,'separation_witness':witness,'tier_analysis':tiers,'lowest_pair_separating_tier':None if lowest is None else {'tier':lowest['tier'],'name':lowest['name']}})
    if not any_sep: cls=s1_rebase_spec()['outcomes']['no_structure_read']; outcome='STRUCTURE_NOT_READ'
    elif old_tier1_falsified: cls=s1_rebase_spec()['outcomes']['structure_read_old_tier1_falsified']; outcome='STRUCTURE_READ_OLD_TIER1_FALSIFIED'
    else: cls=s1_rebase_spec()['outcomes']['structure_read_old_tier1_survives']; outcome='STRUCTURE_READ_OLD_TIER1_NOT_FALSIFIED'
    out={'schema_id':'IG_G4_S1_REBASE_CONTEXT_READ_RESULT_V1','status':'PASS','stage_ref':'G4:S1.REBASE','classification':cls,'outcome':outcome,'authority':dict(authority),'term_load':dict(load_meta),'context_basis':{'target_count':4,'context_partner_count':4,'operator_count':31,'orientation_count':2,'kernel_count':496,'scientific_observation_count':len(rows),'lower_layer_rematerialization':False},'challenge_pair_results':pair_results,'old_tier1_falsified_by_rebase':old_tier1_falsified,'promotion':False,'hidden_state_promoted':False,'g4_s2_rebase_unlocked':True,'g4_rebase_complete':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','g4_graduated':False,'next_authorized_stage':'G4:S2.REBASE','nonclaims':list(s1_rebase_spec()['nonclaims'])}; out['science_sha256']=canonical_sha256(out); return out

_WORKER_STATES=None
def init_worker(payload:Mapping[str,Any]):
    global _WORKER_STATES; _WORKER_STATES={str(r):G3TermState.from_wire(w) for r,w in payload['states'].items()}
def kernel_worker(payload:Mapping[str,Any]):
    if _WORKER_STATES is None: raise G4S1RebaseError('worker not initialized')
    l,r=str(payload['left_ref']),str(payload['right_ref']); op=[int(x) for x in payload['operator']]
    return {'left_ref':l,'right_ref':r,'operator':op,'kernel_measurement':pair_context_kernel_measurement(left=_WORKER_STATES[l],right=_WORKER_STATES[r],operator=op)}

def compare_cold_replay(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    checks={'primary_pass':primary.get('status')=='PASS','cold_pass':cold.get('status')=='PASS','science_sha_equal':primary.get('science_sha256')==cold.get('science_sha256'),'source_sha_equal':primary.get('source_sha256')==cold.get('source_sha256'),'registry_sha_equal':primary.get('registry_sha256')==cold.get('registry_sha256'),'classification_equal':primary.get('classification')==cold.get('classification'),'outcome_equal':primary.get('outcome')==cold.get('outcome'),'challenge_pair_results_equal':primary.get('challenge_pair_results')==cold.get('challenge_pair_results')}
    ok=all(checks.values()); out={'schema_id':'IG_G4_S1_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'primary_source_sha256':primary.get('source_sha256'),'cold_source_sha256':cold.get('source_sha256'),'primary_registry_sha256':primary.get('registry_sha256'),'cold_registry_sha256':cold.get('registry_sha256')}; out['science_sha256']=canonical_sha256(out); return out

def certified_closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
    ok=replay.get('certification')=='CERTIFIED_PASS' and primary.get('status')=='PASS' and cold.get('status')=='PASS'
    out={'schema_id':'IG_G4_S1_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':primary.get('classification'),'outcome':primary.get('outcome'),'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'replay_science_sha256':replay.get('science_sha256'),'source_sha256':primary.get('source_sha256'),'registry_sha256':primary.get('registry_sha256'),'old_tier1_falsified_by_rebase':primary.get('old_tier1_falsified_by_rebase'),'g4_s2_rebase_unlocked':ok,'g4_rebase_complete':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','next_authorized_stage':'G4:S2.REBASE' if ok else None}; out['science_sha256']=canonical_sha256(out); return out
