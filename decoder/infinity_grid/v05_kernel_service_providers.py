from __future__ import annotations

"""O3B implementations for the restricted Decoder KernelView.

Scientific evaluators must not import this module directly.  Runtime binds only
services declared by EvaluatorSpec.  Stage construction may request a restricted
PUBLIC view through ``build_stage_kernel_view``.
"""
from typing import Any, Mapping
from collections import Counter
from .v05_kernel_services import EvaluatorSpec, KernelView, PRIMARY, KERNEL_SERVICE_CATALOG, PUBLIC
from .exact_tree_relation_kernel import (
    configure_relation_kernel, get_relation_kernel, tree_from_record, tree_to_record,
)
from .g5_capabilities import _g5_s1_carriers
from .g6_s5r_crw_kernel import (
    exact_observer_state, decode_exact_observer_state,
    parent_candidates_from_observer_canons_profiled,
    write_exact_observer_states, attachment_response_descriptor,
)
from .g6_reference_oracles import (
    legacy_parent_candidates_from_observer_canons, legacy_tree_from_rooted_canon,
)
from .g6_fresh_marker_kernel import marker_observer_state, decode_marker_observer_state, write_marker_states
from .g6_marker_reference_oracle import legacy_marker_decode_oracle

class KernelServiceProviderError(RuntimeError): pass

def _basis_trees():
    c=_g5_s1_carriers()
    return {k:c[k] for k in sorted(c)}

def _basis_records():
    return {k:tree_to_record(v) for k,v in _basis_trees().items()}

def _probe(ref: str):
    b=_basis_trees(); key=str(ref)
    if key not in b: raise KernelServiceProviderError('UNKNOWN_AUTHORITY_BASIS:'+key)
    return b[key]

def _authority_basis_service():
    return _basis_records()

def _operator_basis_service():
    return tuple(get_relation_kernel().authority.operators)

def _public_read_service(tree_record: Mapping[str,Any]):
    k=get_relation_kernel(); tree=tree_from_record(tree_record); prepared=k.prepare(tree)
    if not prepared.legal:
        return {'schema_id':'IG_DECODER_V05_G4_PUBLIC_READ_V1','legal':False,'caps7':None,'H_class_bag':dict(sorted(Counter(tree.H_classes).items())),'failure':'RESOURCE_UNAVAILABLE'}
    total=[sum(int(row[t]) for row in prepared.local_remaining) for t in range(7)]
    return {'schema_id':'IG_DECODER_V05_G4_PUBLIC_READ_V1','legal':True,'descriptor':'CAPS7_PLUS_H_CLASS_BAG','caps7':total,'H_class_bag':dict(sorted(Counter(tree.H_classes).items()))}

def _exact_identity_service(tree_record: Mapping[str,Any]):
    k=get_relation_kernel(); return k.prepare(tree_from_record(tree_record)).unrooted_canon

def _relation_metrics(rel, before, after):
    m={
      'attempted_owner_pairs':rel.attempted_owner_pairs,
      'legal_owner_pairs':rel.legal_owner_pairs,
      'rooted_owner_pair_candidates':rel.rooted_owner_pair_candidates,
      'child_canon_constructions':rel.child_canon_constructions,
      'reused_cavity_branches':rel.reused_cavity_branches,
      'reroot_vertices_visited':rel.reroot_vertices_visited,
      'exact_outcomes_retained':int(getattr(rel,'exact_outcome_count',len(getattr(rel,'canons',())))),
    }
    for key,value in after.items():
        if key.startswith(('preparation_','relation_cache_','relation_profile_','observer_q_cache_','observer_decode_cache_')) or key=='prepared_parent_constructions':
            m[key]=value-before.get(key,0)
    return m

def _exact_relation_service(left_record: Mapping[str,Any], right_record: Mapping[str,Any], operator):
    k=get_relation_kernel(); before=k.metrics()
    rel=k.relation(tree_from_record(left_record),tree_from_record(right_record),tuple(operator)); after=k.metrics()
    return {
      'canons':rel.canons,
      'children':tuple(tree_to_record(c) for c in rel.children),
      'metrics':_relation_metrics(rel,before,after),
    }

def _exact_relation_profile_service(left_record: Mapping[str,Any], right_record: Mapping[str,Any], operator):
    k=get_relation_kernel(); before=k.metrics()
    prof=k.relation_profile(tree_from_record(left_record),tree_from_record(right_record),tuple(operator)); after=k.metrics()
    return {
      'exact_outcome_count':int(prof.exact_outcome_count),
      'metrics':_relation_metrics(prof,before,after),
    }

def _exact_relation_profile_batch_service(left_record: Mapping[str,Any], right_record: Mapping[str,Any], operators):
    k=get_relation_kernel(); before=k.metrics()
    ops=tuple(tuple(op) for op in operators)
    profiles=k.relation_profile_batch(tree_from_record(left_record),tree_from_record(right_record),ops)
    after=k.metrics()
    metrics={
      'attempted_owner_pairs':sum(int(p.attempted_owner_pairs) for p in profiles),
      'legal_owner_pairs':sum(int(p.legal_owner_pairs) for p in profiles),
      'rooted_owner_pair_candidates':sum(int(p.rooted_owner_pair_candidates) for p in profiles),
      'child_canon_constructions':sum(int(p.child_canon_constructions) for p in profiles),
      'reused_cavity_branches':sum(int(p.reused_cavity_branches) for p in profiles),
      'reroot_vertices_visited':sum(int(p.reroot_vertices_visited) for p in profiles),
      'exact_outcomes_retained':sum(int(p.exact_outcome_count) for p in profiles),
    }
    for key,value in after.items():
        if key.startswith(('preparation_','relation_cache_','relation_profile_','observer_q_cache_','observer_decode_cache_')) or key=='prepared_parent_constructions':
            metrics[key]=int(value)-int(before.get(key,0))
    return {
      'exact_outcome_counts':tuple(int(p.exact_outcome_count) for p in profiles),
      'metrics':metrics,
    }

def _exact_relation_profile_family_service(left_record: Mapping[str,Any], right_records, operators):
    """One exact right-major profile family for a common left carrier."""
    k=get_relation_kernel(); before=k.metrics()
    ops=tuple(tuple(op) for op in operators)
    rights=tuple(tree_from_record(record) for record in right_records)
    profiles=k.relation_profile_family(tree_from_record(left_record),rights,ops)
    after=k.metrics()
    metrics={
      'attempted_owner_pairs':sum(int(p.attempted_owner_pairs) for p in profiles),
      'legal_owner_pairs':sum(int(p.legal_owner_pairs) for p in profiles),
      'rooted_owner_pair_candidates':sum(int(p.rooted_owner_pair_candidates) for p in profiles),
      'child_canon_constructions':sum(int(p.child_canon_constructions) for p in profiles),
      'reused_cavity_branches':sum(int(p.reused_cavity_branches) for p in profiles),
      'reroot_vertices_visited':sum(int(p.reroot_vertices_visited) for p in profiles),
      'exact_outcomes_retained':sum(int(p.exact_outcome_count) for p in profiles),
    }
    for key,value in after.items():
        if key.startswith(('preparation_','relation_cache_','relation_profile_','observer_q_cache_','observer_decode_cache_')) or key=='prepared_parent_constructions':
            metrics[key]=int(value)-int(before.get(key,0))
    return {
      'exact_outcome_counts':tuple(int(p.exact_outcome_count) for p in profiles),
      'metrics':metrics,
    }

def _relation_enabled_service(left_record: Mapping[str,Any], right_record: Mapping[str,Any], operator):
    k=get_relation_kernel(); return k.enabled(tree_from_record(left_record),tree_from_record(right_record),tuple(operator))

def _first_exact_child_service(left_record: Mapping[str,Any], right_record: Mapping[str,Any], operator):
    k=get_relation_kernel(); rel=k.relation(tree_from_record(left_record),tree_from_record(right_record),tuple(operator))
    if not rel.children: return None
    return {'canon':rel.canons[0],'tree':tree_to_record(rel.children[0])}

def _observer_q_service(tree_record: Mapping[str,Any], probe_ref='D2_PATH', observer_operator=(0,0)):
    k=get_relation_kernel(); return exact_observer_state(tree_from_record(tree_record),_probe(str(probe_ref)),tuple(observer_operator),kernel=k)

def _observer_decode_service(observer_state, probe_ref='D2_PATH', observer_operator=(0,0)):
    k=get_relation_kernel(); probe=_probe(str(probe_ref)); state=tuple(observer_state)
    try:
        decoded,stats=decode_exact_observer_state(state,probe,tuple(observer_operator),kernel=k)
        can=k.prepare(decoded).unrooted_canon
        return {'status':'PASS','candidate_count':1,'candidates':(can,), 'tree':tree_to_record(decoded), 'unrooted_canon':can, 'stats':dict(stats)}
    except Exception:
        candidates,stats=parent_candidates_from_observer_canons_profiled(state,probe,tuple(observer_operator),kernel=k)
        return {'status':'DECODE_FAILURE','candidate_count':len(candidates),'candidates':candidates,'tree':None,'unrooted_canon':None,'stats':dict(stats)}

def _observer_write_service(left_state, right_state, operator, probe_ref='D2_PATH', observer_operator=(0,0)):
    k=get_relation_kernel(); diagnostics={}
    states=write_exact_observer_states(tuple(left_state),tuple(right_state),tuple(operator),probe=_probe(str(probe_ref)),observer_operator=tuple(observer_operator),kernel=k,diagnostics=diagnostics)
    return {'states':states,'diagnostics':diagnostics}

def _attachment_response_descriptor_service(tree_record: Mapping[str,Any]):
    k=get_relation_kernel(); return attachment_response_descriptor(tree_from_record(tree_record),kernel=k)

def _marker_q_service(tree_record: Mapping[str,Any]):
    k=get_relation_kernel(); return marker_observer_state(tree_from_record(tree_record),kernel=k)

def _marker_decode_service(observer_state):
    k=get_relation_kernel()
    try:
        decoded,stats=decode_marker_observer_state(tuple(observer_state),kernel=k); can=k.prepare(decoded).unrooted_canon
        return {'status':'PASS','candidate_count':1,'candidates':(can,),'tree':tree_to_record(decoded),'unrooted_canon':can,'stats':dict(stats)}
    except Exception:
        return {'status':'DECODE_FAILURE','candidate_count':0,'candidates':tuple(),'tree':None,'unrooted_canon':None,'stats':{}}

def _marker_write_service(left_state,right_state,operator):
    k=get_relation_kernel(); diagnostics={}; states=write_marker_states(tuple(left_state),tuple(right_state),tuple(operator),kernel=k,diagnostics=diagnostics)
    return {'states':states,'diagnostics':diagnostics}

def _legacy_marker_decode_oracle_service(observer_state):
    return legacy_marker_decode_oracle(tuple(observer_state))

def _legacy_observer_decode_oracle_service(observer_state, probe_ref='D2_PATH', observer_operator=(0,0)):
    probe=_probe(str(probe_ref)); candidates=legacy_parent_candidates_from_observer_canons(tuple(observer_state),probe,tuple(observer_operator))
    decoded=None
    if len(candidates)==1:
        decoded=tree_to_record(legacy_tree_from_rooted_canon(candidates[0]))
    return {'candidate_count':len(candidates),'candidates':candidates,'tree':decoded}

_ALL_PROVIDERS={
  'AUTHORITY_BASIS':_authority_basis_service,
  'OPERATOR_BASIS':_operator_basis_service,
  'PUBLIC_READ':_public_read_service,
  'EXACT_IDENTITY':_exact_identity_service,
  'EXACT_RELATION':_exact_relation_service,
  'EXACT_RELATION_PROFILE':_exact_relation_profile_service,
  'EXACT_RELATION_PROFILE_BATCH':_exact_relation_profile_batch_service,
  'EXACT_RELATION_PROFILE_FAMILY':_exact_relation_profile_family_service,
  'RELATION_ENABLED':_relation_enabled_service,
  'FIRST_EXACT_CHILD':_first_exact_child_service,
  'OBSERVER_Q':_observer_q_service,
  'OBSERVER_DECODE':_observer_decode_service,
  'OBSERVER_WRITE':_observer_write_service,
  'MARKER_Q':_marker_q_service,
  'MARKER_DECODE':_marker_decode_service,
  'MARKER_WRITE':_marker_write_service,
  'ATTACHMENT_RESPONSE_DESCRIPTOR':_attachment_response_descriptor_service,
  'LEGACY_OBSERVER_DECODE_ORACLE':_legacy_observer_decode_oracle_service,
  'LEGACY_MARKER_DECODE_ORACLE':_legacy_marker_decode_oracle_service,
}

def build_kernel_service_providers(spec: EvaluatorSpec):
    out={}
    for service in spec.allowed_kernel_services:
        provider=_ALL_PROVIDERS.get(service)
        if provider is None: raise KernelServiceProviderError('KERNEL_SERVICE_PROVIDER_MISSING:'+service)
        out[service]=provider
    return out

def build_stage_kernel_view(stage_id: str, allowed_services, *, scope_identity: str):
    services=tuple(str(x) for x in allowed_services)
    for service in services:
        row=KERNEL_SERVICE_CATALOG.get(service)
        if row is None or row['visibility']!=PUBLIC:
            raise KernelServiceProviderError('STAGE_KERNEL_SERVICE_NONPUBLIC:'+service)
    configure_relation_kernel(scope_identity=str(scope_identity))
    spec=EvaluatorSpec('infinity_grid.stage:'+str(stage_id).replace(':','_'),PRIMARY,services)
    return KernelView(spec,build_kernel_service_providers(spec))
