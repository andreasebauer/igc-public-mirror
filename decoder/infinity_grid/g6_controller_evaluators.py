from __future__ import annotations

"""Thin G6 evaluators. Reusable exact semantics belong to Decoder KernelView."""
from typing import Any, Mapping
from .v05_kernel_services import current_kernel_view

def _relation_states(rel):
    return [{'identity':can,'state':tree} for can,tree in zip(rel['canons'],rel['children'])]

def exact_one_step_relation_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    view=current_kernel_view(); position=str(payload.get('position','LEFT')).upper()
    left,right=payload['state_tree'],payload['probe_tree']
    if position=='RIGHT': left,right=right,left
    elif position!='LEFT': raise ValueError(f'unsupported relation position {position}')
    rel=view.call('EXACT_RELATION',left,right,payload['operator'])
    return {'signature':rel['canons'],'outcome_count':len(rel['canons']),'metrics':dict(rel['metrics'])}

def g6_s1_universe_generation_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    view=current_kernel_view(); basis=view.call('AUTHORITY_BASIS')
    rel=view.call('EXACT_RELATION',basis[str(payload['left_ref'])],basis[str(payload['right_ref'])],payload['operator'])
    return {'states':_relation_states(rel),'metrics':{k:rel['metrics'][k] for k in ('attempted_owner_pairs','legal_owner_pairs','rooted_owner_pair_candidates','child_canon_constructions','exact_outcomes_retained')}}

def g6_s3_axis_a_generation_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    view=current_kernel_view(); basis=view.call('AUTHORITY_BASIS'); op=(0,0)
    a,b,c=(str(x) for x in payload['triple']); by_identity={}
    metrics={'relation_evaluations':0,'attempted_owner_pairs':0,'legal_owner_pairs':0,'rooted_owner_pair_candidates':0,'child_canon_constructions':0}
    def absorb(rel):
        metrics['relation_evaluations']+=1
        for key in ('attempted_owner_pairs','legal_owner_pairs','rooted_owner_pair_candidates','child_canon_constructions'): metrics[key]+=int(rel['metrics'][key])
        for can,tree in zip(rel['canons'],rel['children']): by_identity.setdefault(can,tree)
    mids=view.call('EXACT_RELATION',basis[a],basis[b],op); metrics['relation_evaluations']+=1
    for key in ('attempted_owner_pairs','legal_owner_pairs','rooted_owner_pair_candidates','child_canon_constructions'): metrics[key]+=int(mids['metrics'][key])
    for mid in mids['children']: absorb(view.call('EXACT_RELATION',mid,basis[c],op))
    mids=view.call('EXACT_RELATION',basis[b],basis[c],op); metrics['relation_evaluations']+=1
    for key in ('attempted_owner_pairs','legal_owner_pairs','rooted_owner_pair_candidates','child_canon_constructions'): metrics[key]+=int(mids['metrics'][key])
    for mid in mids['children']: absorb(view.call('EXACT_RELATION',basis[a],mid,op))
    states=[{'identity':can,'state':by_identity[can]} for can in sorted(by_identity,key=repr)]; metrics['exact_outcomes_retained']=len(states)
    return {'states':states,'metrics':metrics}

def g6_s3_axis_b_generation_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    view=current_kernel_view(); basis=view.call('AUTHORITY_BASIS'); parent=payload['state_tree']
    contexts=(('D2_PATH',(0,0),'LEFT'),('D2_PATH',(0,1),'LEFT'),('D2_PATH',(1,0),'RIGHT'),('D4_PATH',(2,4),'LEFT'))
    by_identity={}; metrics={'relation_evaluations':0,'attempted_owner_pairs':0,'legal_owner_pairs':0,'rooted_owner_pair_candidates':0,'child_canon_constructions':0}
    for bref,op,pos in contexts:
        left,right=(parent,basis[bref]) if pos=='LEFT' else (basis[bref],parent)
        rel=view.call('EXACT_RELATION',left,right,op); metrics['relation_evaluations']+=1
        for key in ('attempted_owner_pairs','legal_owner_pairs','rooted_owner_pair_candidates','child_canon_constructions'): metrics[key]+=int(rel['metrics'][key])
        for can,tree in zip(rel['canons'],rel['children']): by_identity.setdefault(can,tree)
    states=[{'identity':can,'state':by_identity[can]} for can in sorted(by_identity,key=repr)]; metrics['exact_outcomes_retained']=len(states)
    return {'states':states,'metrics':metrics}

def g6_s3_higher_generation_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    axis=str(payload['axis']).upper()
    if axis=='A': return g6_s3_axis_a_generation_evaluator(payload)
    if axis=='B': return g6_s3_axis_b_generation_evaluator(payload)
    raise ValueError(f'unknown G6:S3R generation axis {axis}')

def g6_s3_l1_legacy_signature_evaluator(payload: Mapping[str,Any]) -> dict[str,Any]:
    view=current_kernel_view(); basis=view.call('AUTHORITY_BASIS'); probe_ref=str(payload.get('probe_ref','D2_PATH')); operator=tuple(payload.get('operator',(0,0))); position=str(payload.get('position','LEFT')).upper()
    if probe_ref not in basis: raise ValueError(f'unknown G6 L1 probe_ref {probe_ref}')
    if position not in {'LEFT','RIGHT'}: raise ValueError(f'unsupported G6 L1 position {position}')
    left,right=(payload['state_tree'],basis[probe_ref]) if position=='LEFT' else (basis[probe_ref],payload['state_tree'])
    rel=view.call('EXACT_RELATION',left,right,operator)
    signature=[{'context':[probe_ref,list(operator),position],'outcomes':[repr(x) for x in rel['canons']]}]
    return {'signature':signature,'outcome_count':len(rel['canons']),'metrics':dict(rel['metrics'])}
