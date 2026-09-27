from __future__ import annotations

"""Thin G6 fresh-marker evaluators using only registered KernelView services."""
from typing import Any, Mapping
from .v05_kernel_services import current_kernel_view

def marker_write_holdout_evaluator(payload:Mapping[str,Any])->dict[str,Any]:
    import time
    view=current_kernel_view(); left,right=payload['left_tree'],payload['right_tree']; op=tuple(payload['operator'])
    t0=time.perf_counter(); ql=view.call('MARKER_Q',left); qr=view.call('MARKER_Q',right); tq=time.perf_counter()-t0
    parent_fail=0; child_fail=0; diagnostics={}
    try:
        t1=time.perf_counter(); wr=view.call('MARKER_WRITE',ql,qr,op); tw=time.perf_counter()-t1
        abstract=wr['states']; diagnostics=wr['diagnostics']
        if diagnostics.get('left_decoded_unrooted_canon')!=view.call('EXACT_IDENTITY',left): parent_fail+=1
        if diagnostics.get('right_decoded_unrooted_canon')!=view.call('EXACT_IDENTITY',right): parent_fail+=1
    except Exception:
        parent_fail+=1; abstract=tuple(); tw=0.0
    rows=tuple(diagnostics.get('raw_child_rows',tuple()))
    raw_projected=tuple(diagnostics.get('projected_marker_states',tuple()))
    if not diagnostics.get('fused_raw_projection') or len(rows)!=int(diagnostics.get('raw_child_count',-1)):
        parent_fail+=1
    t2=time.perf_counter()
    for record,q,expected_can in rows:
        try:
            dec=view.call('MARKER_DECODE',q)
            if dec['status']!='PASS' or dec['unrooted_canon']!=expected_can or view.call('EXACT_IDENTITY',record)!=expected_can: child_fail+=1
        except Exception: child_fail+=1
    td=time.perf_counter()-t2
    match=(parent_fail==0 and child_fail==0 and abstract==raw_projected)
    return {'signature':('PASS',len(rows),len(raw_projected)) if match else ('FAIL',parent_fail,child_fail,len(raw_projected),len(abstract)),
      'metrics':{'marker_write_match':1 if match else 0,'marker_write_mismatch':0 if match else 1,'marker_parent_decode_failure':parent_fail,'marker_child_decode_failure':child_fail,'raw_exact_child_count':len(rows),'projected_marker_state_count':len(raw_projected),'left_marker_outcomes':len(ql),'right_marker_outcomes':len(qr),'marker_parent_q_wall_seconds':tq,'marker_write_wall_seconds':tw,'marker_child_decode_wall_seconds':td,'marker_fused_raw_projection_used':1 if diagnostics.get('fused_raw_projection') else 0}}

def marker_write_independent_evaluator(payload:Mapping[str,Any])->dict[str,Any]:
    view=current_kernel_view(); left,right=payload['left_tree'],payload['right_tree']; op=tuple(payload['operator'])
    ql=view.call('MARKER_Q',left); qr=view.call('MARKER_Q',right)
    dl=view.call('LEGACY_MARKER_DECODE_ORACLE',ql); dr=view.call('LEGACY_MARKER_DECODE_ORACLE',qr)
    decode_ok=(dl['candidate_count']==1 and dr['candidate_count']==1 and dl['candidates'][0]==view.call('EXACT_IDENTITY',left) and dr['candidates'][0]==view.call('EXACT_IDENTITY',right))
    if decode_ok:
        rel=view.call('EXACT_RELATION',dl['tree'],dr['tree'],op)
        abstract=tuple(sorted({view.call('MARKER_Q',c) for c in rel['children']},key=repr))
    else: abstract=tuple()
    raw=view.call('EXACT_RELATION',left,right,op); raw_projected=tuple(sorted({view.call('MARKER_Q',c) for c in raw['children']},key=repr))
    match=decode_ok and abstract==raw_projected
    return {'signature':('PASS',len(raw['canons'])) if match else ('FAIL',dl['candidate_count'],dr['candidate_count'],len(raw_projected),len(abstract)),
      'metrics':{'marker_independent_match':1 if match else 0,'marker_independent_mismatch':0 if match else 1,'marker_legacy_left_candidates':dl['candidate_count'],'marker_legacy_right_candidates':dr['candidate_count']}}
