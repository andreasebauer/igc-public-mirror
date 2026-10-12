from __future__ import annotations

"""Thin G6:S6R evaluators using declared KernelView services only."""
from typing import Any, Mapping
from .v05_kernel_services import current_kernel_view

def recursive_closure_holdout_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    view=current_kernel_view(); probe_ref=str(payload.get('observer_probe_ref','D2_PATH')); observer_op=tuple(payload.get('observer_operator',(0,0)))
    left,right=payload['left_tree'],payload['right_tree']; op=tuple(payload['operator'])
    ql=view.call('OBSERVER_Q',left,probe_ref,observer_op); qr=view.call('OBSERVER_Q',right,probe_ref,observer_op)
    parent_decode_fail=0; write_diagnostics={}
    try:
        wr=view.call('OBSERVER_WRITE',ql,qr,op,probe_ref,observer_op); abstract=wr['states']; write_diagnostics=wr['diagnostics']
        if write_diagnostics['left_decoded_unrooted_canon']!=view.call('EXACT_IDENTITY',left): parent_decode_fail+=1
        if write_diagnostics['right_decoded_unrooted_canon']!=view.call('EXACT_IDENTITY',right): parent_decode_fail+=1
    except Exception:
        parent_decode_fail+=1; abstract=tuple()
    raw=view.call('EXACT_RELATION',left,right,op)
    raw_child_q=[(c,view.call('OBSERVER_Q',c,probe_ref,observer_op)) for c in raw['children']]
    raw_projected=tuple(sorted({qc for _c,qc in raw_child_q},key=repr)); child_decode_fail=0
    for c,qc in raw_child_q:
        dec=view.call('OBSERVER_DECODE',qc,probe_ref,observer_op)
        if dec['status']!='PASS' or dec['unrooted_canon']!=view.call('EXACT_IDENTITY',c): child_decode_fail+=1
    match=(parent_decode_fail==0 and child_decode_fail==0 and abstract==raw_projected)
    return {'signature':('PASS',len(raw['canons']),len(raw_projected)) if match else ('FAIL',parent_decode_fail,child_decode_fail,len(raw_projected),len(abstract)),
      'metrics':{'write_match':1 if match else 0,'write_mismatch':0 if match else 1,'parent_decode_failure':parent_decode_fail,'child_decode_failure':child_decode_fail,
      'raw_exact_child_count':len(raw['canons']),'projected_child_state_count':len(raw_projected),'left_q_outcomes':len(ql),'right_q_outcomes':len(qr),
      'q_only_parent_decodes_reused':2 if parent_decode_fail==0 and write_diagnostics else 0,'raw_child_observer_states_reused_for_decode':len(raw_child_q)}}

def recursive_closure_independent_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    view=current_kernel_view(); probe_ref=str(payload.get('observer_probe_ref','D2_PATH')); observer_op=tuple(payload.get('observer_operator',(0,0)))
    left,right=payload['left_tree'],payload['right_tree']; op=tuple(payload['operator'])
    ql=view.call('OBSERVER_Q',left,probe_ref,observer_op); qr=view.call('OBSERVER_Q',right,probe_ref,observer_op)
    dl=view.call('LEGACY_OBSERVER_DECODE_ORACLE',ql,probe_ref,observer_op); dr=view.call('LEGACY_OBSERVER_DECODE_ORACLE',qr,probe_ref,observer_op)
    decode_ok=(dl['candidate_count']==1 and dr['candidate_count']==1 and dl['candidates'][0]==view.call('EXACT_IDENTITY',left) and dr['candidates'][0]==view.call('EXACT_IDENTITY',right))
    if decode_ok:
        rel_abs=view.call('EXACT_RELATION',dl['tree'],dr['tree'],op)
        abstract=tuple(sorted({view.call('OBSERVER_Q',c,probe_ref,observer_op) for c in rel_abs['children']},key=repr))
    else: abstract=tuple()
    raw=view.call('EXACT_RELATION',left,right,op); raw_projected=tuple(sorted({view.call('OBSERVER_Q',c,probe_ref,observer_op) for c in raw['children']},key=repr))
    match=decode_ok and abstract==raw_projected
    return {'signature':('PASS',len(raw['canons'])) if match else ('FAIL',dl['candidate_count'],dr['candidate_count'],len(raw_projected),len(abstract)),
      'metrics':{'independent_match':1 if match else 0,'independent_mismatch':0 if match else 1,'legacy_left_candidates':dl['candidate_count'],'legacy_right_candidates':dr['candidate_count']}}


def l2_common_parent_uniqueness_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    view=current_kernel_view(); state=payload['state_tree']; probe_ref=str(payload.get('observer_probe_ref','D2_PATH')); observer_op=tuple(payload.get('observer_operator',(0,0)))
    ident=view.call('EXACT_IDENTITY',state); q=view.call('OBSERVER_Q',state,probe_ref,observer_op); dec=view.call('OBSERVER_DECODE',q,probe_ref,observer_op)
    cands=tuple(dec.get('candidates') or ()); ok=(int(dec.get('candidate_count',-1))==1 and len(cands)==1 and cands[0]==ident)
    return {'signature':('PASS',int(payload['leaf_count'])) if ok else ('FAIL',int(payload['leaf_count']),int(dec.get('candidate_count',-1)),ident,cands),
      'metrics':{'l2_good_candidate_state':1 if ok else 0,'l2_bad_candidate_state':0 if ok else 1,'l2_candidate_count':int(dec.get('candidate_count',-1)),'l2_q_outcome_count':len(q)}}

def l2_sibling_q_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    """Partition distinct sibling children by their exact q-state within one core."""
    view=current_kernel_view(); child=payload['child_tree']; core=payload['core_identity']
    q=view.call('OBSERVER_Q',child,str(payload.get('observer_probe_ref','D2_PATH')),tuple(payload.get('observer_operator',(0,0))))
    return {'signature':(core,q),'metrics':{'l2p_sibling_child':1,'l2p_q_outcome_count':len(q)}}
