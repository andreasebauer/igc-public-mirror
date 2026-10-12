from __future__ import annotations

"""Thin G6:S5R evaluators using only declared Decoder KernelView services."""
from typing import Any, Mapping
from .v05_kernel_services import current_kernel_view

def observer_inversion_descriptor_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    view=current_kernel_view(); state=payload['state_tree']; probe_ref=str(payload.get('probe_ref','D2_PATH')); op=tuple(payload.get('operator',(0,0)))
    q=view.call('OBSERVER_Q',state,probe_ref,op); actual=view.call('EXACT_IDENTITY',state)
    decoded=view.call('OBSERVER_DECODE',q,probe_ref,op); stats=decoded['stats']; ok=(decoded['status']=='PASS' and decoded['unrooted_canon']==actual)
    if ok:
        signature=('PASS',view.call('ATTACHMENT_RESPONSE_DESCRIPTOR',decoded['tree']))
    else:
        signature=('DECODE_FAILURE',decoded['candidate_count'],tuple(decoded['candidates']))
    return {'signature':signature,'outcome_count':len(q),'metrics':{
      'decode_ok':1 if ok else 0,'decode_failure':0 if ok else 1,'decode_candidate_count':decoded['candidate_count'],
      'observer_outcome_count':len(q),'parent_vertices':int(state['n']),
      'inversion_edges_scanned':stats['edges_scanned'],'inversion_operator_edges':stats['operator_edges'],
      'inversion_size_compatible_edges':stats['size_compatible_edges'],'inversion_component_splits':stats['component_splits'],
      'inversion_component_canon_calls':stats['component_canon_calls'],'inversion_fast_verified_decode':int(stats.get('fast_verified_decode',0)),
      'inversion_outcomes_examined':int(stats.get('fast_verified_outcomes_examined',stats.get('observer_outcomes',0))),
      'inversion_outcomes_short_circuited':int(stats.get('observer_outcomes_short_circuited',0)),
    }}

def observer_state_write_law_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    view=current_kernel_view(); probe_ref=str(payload.get('observer_probe_ref','D2_PATH')); observer_op=tuple(payload.get('observer_operator',(0,0)))
    left,right=payload['left_tree'],payload['right_tree']; op=tuple(payload['operator'])
    ql=view.call('OBSERVER_Q',left,probe_ref,observer_op); qr=view.call('OBSERVER_Q',right,probe_ref,observer_op)
    decode_failure=0; write_diagnostics={}
    try:
        wr=view.call('OBSERVER_WRITE',ql,qr,op,probe_ref,observer_op); abstract=wr['states']; write_diagnostics=wr['diagnostics']
        l_ok=write_diagnostics['left_decoded_unrooted_canon']==view.call('EXACT_IDENTITY',left)
        r_ok=write_diagnostics['right_decoded_unrooted_canon']==view.call('EXACT_IDENTITY',right)
        if not (l_ok and r_ok): decode_failure=1
    except Exception:
        decode_failure=1; abstract=tuple()
    raw=view.call('EXACT_RELATION',left,right,op)
    raw_projected=tuple(sorted({view.call('OBSERVER_Q',c,probe_ref,observer_op) for c in raw['children']},key=repr))
    match=(decode_failure==0 and abstract==raw_projected); expected_child_n=payload.get('expected_child_n'); size_ok=True
    if expected_child_n is not None and raw['children']:
        size_ok=all(int(c['n'])==int(expected_child_n) for c in raw['children']); match=match and size_ok
    signature=('PASS',len(raw['canons']),len(raw_projected)) if match else ('WRITE_MISMATCH',len(raw_projected),len(abstract),raw_projected,abstract)
    return {'signature':signature,'metrics':{
      'write_match':1 if match else 0,'write_mismatch':0 if match else 1,'parent_decode_failure':decode_failure,
      'raw_exact_child_count':len(raw['canons']),'projected_child_state_count':len(raw_projected),'left_q_outcomes':len(ql),'right_q_outcomes':len(qr),
      'out_of_u_size_check':1 if expected_child_n is not None and size_ok else 0,'q_only_parent_decodes_reused':2 if decode_failure==0 and write_diagnostics else 0,
    }}
