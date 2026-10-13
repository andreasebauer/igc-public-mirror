from __future__ import annotations

"""G4:R0 post-graduation structural reconnaissance.

Non-promoting exact incidence read behind the graduated
D=(CAPS7, Tier-1 class bag) quotient. Previous-layer G3 units stay atomic for
G4 action semantics; R0 may inspect their certified S0 G2-unit incidence only to
form a one-level flattened structural diagnostic.
"""
from dataclasses import dataclass
from functools import cached_property
from importlib.resources import files
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .g4_term_state import G3TermState
from .uplift_g4_s1 import load_s0_term_states
from .uplift_g4_s5 import descriptor_from_public_parts
from .uplift_g4_s6 import tier1_index
from .uplift_g3_r0 import _graph_metrics, _tree_canon, unlabeled_tree_shape_census, hierarchical_tree_theorem as g3_hierarchical_tree_theorem

class G4R0Error(RuntimeError): pass
_SPEC='G4_R0_STRUCTURAL_RECON_SPEC_V1.json'

def _resource(name:str)->Path:
    return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def r0_spec()->dict[str,Any]:
    obj=json.loads(_resource(_SPEC).read_text(encoding='utf-8'))
    if obj.get('schema_id')!='IG_G4_R0_STRUCTURAL_RECON_SPEC_V1': raise G4R0Error('bad G4:R0 spec schema')
    if canonical_sha256({k:v for k,v in obj.items() if k!='science_sha256'})!=obj.get('science_sha256'): raise G4R0Error('G4:R0 spec hash mismatch')
    return obj

def _reserve_leaf_with_owner(st:G3TermState,t:int):
    t=int(t); out=[]
    for i,c in enumerate(st.node_caps):
        if c[t]<=0: continue
        nc=[list(x) for x in st.node_caps]; nc[i][t]-=1
        out.append((i,G3TermState(tuple(tuple(x) for x in nc),st.typed_edges)))
    # G3TermState.reserve_external_relation dedupes by exact term digest. The owner-index
    # form above must have the same exact successor set.
    if {x.construction_digest for _,x in out}!={x.construction_digest for x in st.reserve_external_relation(t)}:
        raise G4R0Error('owner-explicit reserve disagrees with certified G3 leaf reserve relation')
    return tuple(out)

def _canon_state(leaves:Sequence[G3TermState], edges:Sequence[Sequence[int]])->tuple[Any,...]:
    n=len(leaves); ed=[tuple(map(int,e)) for e in edges]
    best=None
    for p in permutations(range(n)):
        cols=[None]*n
        for old,new in enumerate(p): cols[new]=leaves[old].construction_digest
        rr=[]
        for u,v,a,b,iu,iv in ed:
            nu,nv=p[u],p[v]
            if nu<nv: rr.append((nu,nv,a,b,iu,iv))
            else: rr.append((nv,nu,b,a,iv,iu))
        cand=(tuple(cols),tuple(sorted(rr)))
        if best is None or cand<best: best=cand
    return best

@dataclass(frozen=True)
class G4R0IncidenceState:
    leaves:tuple[G3TermState,...]
    edges:tuple[tuple[int,int,int,int,int,int],...]
    leaf_class_labels:tuple[str,...]
    def __post_init__(self):
        if len(self.leaves)!=len(self.leaf_class_labels): raise G4R0Error('leaf/class mismatch')
    @cached_property
    def total_caps(self)->tuple[int,...]:
        return tuple(sum(x.total_caps[t] for x in self.leaves) for t in range(7))
    @cached_property
    def construction_digest(self)->str:
        return canonical_sha256({'schema_id':'IG_G4_R0_INCIDENCE_STATE_CANON_V1','canon':_canon_state(self.leaves,self.edges),'classes':sorted(self.leaf_class_labels)})
    def reserve_with_owner(self,t:int):
        out=[]
        for li,leaf in enumerate(self.leaves):
            for internal_owner,succ in _reserve_leaf_with_owner(leaf,t):
                ll=list(self.leaves); ll[li]=succ
                out.append((li,internal_owner,G4R0IncidenceState(tuple(ll),self.edges,self.leaf_class_labels)))
        return tuple(out)

def compose_seed(state:G4R0IncidenceState, seed:G3TermState, seed_label:str, a:int,b:int):
    out={}
    right=_reserve_leaf_with_owner(seed,b)
    new_idx=len(state.leaves)
    for owner,inner_left,ls in state.reserve_with_owner(a):
        for inner_right,rs in right:
            leaves=ls.leaves+(rs,); labels=ls.leaf_class_labels+(seed_label,)
            e=(owner,new_idx,int(a),int(b),inner_left,inner_right)
            st=G4R0IncidenceState(leaves,ls.edges+(e,),labels)
            out.setdefault(st.construction_digest,st)
    return tuple(out[k] for k in sorted(out))

def coarse_graph(st:G4R0IncidenceState):
    return len(st.leaves),[(u,v) for u,v,_a,_b,_iu,_iv in st.edges]

def flattened_g2_graph(st:G4R0IncidenceState):
    offsets=[]; z=0
    for leaf in st.leaves: offsets.append(z); z+=len(leaf.node_caps)
    edges=[]
    for li,leaf in enumerate(st.leaves):
        off=offsets[li]
        for u,v,a,b in leaf.typed_edges: edges.append({'u':off+u,'v':off+v,'grade':'G3_INTERNAL','types':[a,b]})
    for u,v,a,b,iu,iv in st.edges:
        edges.append({'u':offsets[u]+iu,'v':offsets[v]+iv,'grade':'G4_CROSS','types':[a,b]})
    return z,edges

def _descriptor(st:G4R0IncidenceState, alphabet:Sequence[str]):
    return descriptor_from_public_parts(caps7=st.total_caps,tier1_labels=st.leaf_class_labels,alphabet=alphabet)

def hierarchical_tree_theorem()->dict[str,Any]:
    g3=g3_hierarchical_tree_theorem()
    expected=r0_spec()['authority']['g3_r0_hierarchical_tree_theorem_science_sha256']
    if g3.get('status')!='PASS' or g3.get('science_sha256')!=expected:
        raise G4R0Error('certified G3 hierarchical tree theorem identity mismatch')
    out={
      'schema_id':'IG_G4_R0_HIERARCHICAL_TREE_THEOREM_V1',
      'status':'PASS',
      'proof_method':'THREE_LEVEL_STRUCTURAL_INDUCTION_ON_GRADUATED_G4_BINARY_RELATION_GRAMMAR',
      'coarse_base':'One graduated G3 unit contracts to one vertex and has no G4 cross edge.',
      'coarse_step':'Every admitted G4 binary constructor joins two disjoint connected complete G4 terms with exactly one typed G4 cross relation between one eligible G3 owner unit on the left and one on the right. Relation-valued owner choice changes the endpoint, not the fact that exactly one cross edge is added.',
      'coarse_conclusion':'After contracting each graduated G3 base unit, every finite connected G4 construction has a finite tree incidence graph: m=n-1, beta=0, every G4 cross relation is a bridge.',
      'mid_scale_authority':g3['science_sha256'],
      'mid_scale_step':'Each coarse G3 vertex expands to its certified finite G2-unit tree. Replacing vertices by disjoint G2-unit trees and joining those trees along one G4 cross edge for every coarse edge preserves connectedness and acyclicity.',
      'mid_scale_conclusion':'The flattened G2-unit graph is a finite tree naturally edge-graded into G3_INTERNAL and G4_CROSS relations. Contracting each G3 internal component recovers the coarse G3-unit tree.',
      'fine_scale_conclusion':'By the certified G3 theorem, each G2 unit itself expands to a finite G1-unit tree; therefore full recursive expansion also remains a finite tree with G2_INTERNAL/G3_CROSS/G4_CROSS grades. R0 does not materialize that lower layer.',
      'scope':'all finite complete G4 relation terms in the certified frozen G4 grammar; incidence reads remain non-promoting reconnaissance only',
      'nonclaims':['NO_PHYSICAL_SPACE','NO_DIMENSION','NO_METRIC_EMBEDDING','NO_TIME','NO_TOPOLOGY_PROMOTION','NO_SHELL_PROFILE_PROMOTION'],
    }
    out['science_sha256']=canonical_sha256(out); return out

def verify_authority(graduation_certificate:Mapping[str,Any],s0_result:Mapping[str,Any],s3_result:Mapping[str,Any]):
    a=r0_spec()['authority']; fail=[]
    if graduation_certificate.get('schema_id')!='IG_G4_GRADUATION_CERTIFICATE_V1' or graduation_certificate.get('status')!='PASS': fail.append('G4_GRADUATION_CERTIFICATE')
    if graduation_certificate.get('science_sha256')!=a['g4_graduation_certificate_science_sha256']: fail.append('G4_GRADUATION_HASH')
    if graduation_certificate.get('g4_graduated') is not True or graduation_certificate.get('r0_unlocked') is not True: fail.append('R0_NOT_UNLOCKED')
    if graduation_certificate.get('authorizes')!='G4:R0_POST_GRADUATION_STRUCTURAL_RECONNAISSANCE': fail.append('R0_AUTHORIZATION')
    if graduation_certificate.get('graduated_descriptor')!=a['graduated_descriptor']: fail.append('DESCRIPTOR')
    grammar=graduation_certificate.get('graduated_grammar',{})
    if int(grammar.get('binary_operator_count',-1))!=31 or int(grammar.get('reservation_actions',-1))!=7: fail.append('G4_GRAMMAR_BASIS')
    if grammar.get('routing_semantics')!='RELATION_VALUED_ALL_ELIGIBLE_OWNER_CHOICES_V1': fail.append('G4_ROUTING')
    if graduation_certificate.get('primary_s6_science_sha256')!=a['g4_s6_science_sha256']: fail.append('G4_S6_SCIENCE')
    if graduation_certificate.get('stable_s6_science_payload_sha256')!=a['g4_s6_stable_payload_sha256']: fail.append('G4_S6_STABLE_PAYLOAD')
    if s0_result.get('schema_id')!='IG_G4_S0_INTERFACE_EXTRACTION_RESULT_V2' or s0_result.get('status')!='PASS': fail.append('S0')
    if s0_result.get('certified_term_corpus',{}).get('science_sha256')!=a['g4_s0_term_corpus_sha256']: fail.append('S0_CORPUS')
    if s3_result.get('schema_id')!='IG_G4_S3_HIGHER_ORDER_COMPATIBILITY_IRREDUCIBILITY_RESULT_V1' or s3_result.get('status')!='PASS': fail.append('S3')
    if s3_result.get('candidate_interface_index',{}).get('science_sha256')!=a['g4_s3_candidate_interface_index_sha256']: fail.append('S3_TIER1_INDEX')
    try:
        states,_=load_s0_term_states(s0_result)
        if len(states)!=2 or any(len(st.node_caps)!=int(a['g2_units_per_frozen_g3_seed']) for st in states.values()): fail.append('S0_G3_TERM_ARITY')
        for st in states.values():
            gm=_graph_metrics(len(st.node_caps),[(int(e[0]),int(e[1])) for e in st.typed_edges])
            if not (gm['connected'] and gm['beta']==0 and len(st.typed_edges)==len(st.node_caps)-1): fail.append('S0_G3_INTERNAL_TREE'); break
    except Exception as exc:
        fail.append('S0_TERM_LOAD:'+str(exc))
    try:
        g3=g3_hierarchical_tree_theorem()
        if g3.get('status')!='PASS' or g3.get('science_sha256')!=a['g3_r0_hierarchical_tree_theorem_science_sha256']: fail.append('G3_R0_TREE_THEOREM')
    except Exception as exc:
        fail.append('G3_R0_TREE_THEOREM:'+str(exc))
    out={'schema_id':'IG_G4_R0_AUTHORITY_AUDIT_V1','status':'PASS' if not fail else 'FAIL','failures':fail,'graduation_science_sha256':graduation_certificate.get('science_sha256'),'s6_science_sha256':graduation_certificate.get('primary_s6_science_sha256'),'s6_stable_payload_sha256':graduation_certificate.get('stable_s6_science_payload_sha256'),'s0_term_corpus_sha256':s0_result.get('certified_term_corpus',{}).get('science_sha256'),'s3_candidate_interface_index_sha256':s3_result.get('candidate_interface_index',{}).get('science_sha256'),'g3_r0_hierarchical_tree_theorem_science_sha256':a['g3_r0_hierarchical_tree_theorem_science_sha256']}
    out['science_sha256']=canonical_sha256(out)
    if fail: raise G4R0Error('G4:R0 authority failed: '+','.join(fail))
    return out

def _choose_seed(states,ref_labels,ops):
    ref=sorted(states)[0]; seed=states[ref]; caps=tuple(seed.total_caps)
    best=None
    for a,b in ops:
        score=(min(caps[a],caps[b]),caps[a]+caps[b],-a,-b)
        if best is None or score>best[0]: best=(score,(a,b))
    return ref,seed,ref_labels[ref],best[1]

def run_g4_r0_recon(*,engine:Any,graduation_certificate:Mapping[str,Any],s0_result:Mapping[str,Any],s3_result:Mapping[str,Any])->dict[str,Any]:
    spec=r0_spec(); auth=verify_authority(graduation_certificate,s0_result,s3_result); theorem=hierarchical_tree_theorem()
    states,repro=load_s0_term_states(s0_result); alpha,ref_labels,tier1vals=tier1_index(s3_result)
    ops=sorted({tuple(map(int,x)) for x in engine.session.bridge_pairs})
    if len(ops)!=31: raise G4R0Error('G4:R0 requires complete frozen 31-operator basis')
    seed_ref,seed,seed_label,(a,b)=_choose_seed(states,ref_labels,ops)
    need=spec['shape_census']['max_g3_units']-1
    if min(seed.total_caps[a],seed.total_caps[b])<need+1: raise G4R0Error('chosen G4:R0 seed/operator lacks conservative shape headroom')
    current=(G4R0IncidenceState((seed,),tuple(),(seed_label,)),)
    rows=[]; witness=None
    for n in range(2,spec['exact_growth_probe']['max_g3_units']+1):
        nxt={}
        for st in current:
            for x in compose_seed(st,seed,seed_label,a,b):
                nxt.setdefault(x.construction_digest,x)
                if len(nxt)>spec['exact_growth_probe']['hard_exact_state_cap_per_rank']: raise G4R0Error(f'exact state cap exceeded at n={n}')
        if not nxt: raise G4R0Error(f'exact growth exhausted at n={n}')
        current=tuple(nxt[k] for k in sorted(nxt))
        coarse={}; flat_canons=set(); descs={}; grade_profiles=set()
        for st in current:
            cn,cp=coarse_graph(st); cm=_graph_metrics(cn,cp)
            if not (cn==n and len(cp)==n-1 and cm['connected'] and cm['beta']==0 and cm['bridges']==len(cp)): raise G4R0Error('coarse tree invariant')
            cc=_tree_canon(cn,cp)
            fn,fe=flattened_g2_graph(st); fp=[(e['u'],e['v']) for e in fe]; fm=_graph_metrics(fn,fp)
            gh={'G3_INTERNAL':sum(e['grade']=='G3_INTERNAL' for e in fe),'G4_CROSS':sum(e['grade']=='G4_CROSS' for e in fe)}
            exp_internal=sum(len(x.typed_edges) for x in st.leaves)
            if not (fm['connected'] and fm['beta']==0 and fm['bridges']==len(fp) and len(fp)==fn-1 and gh['G3_INTERNAL']==exp_internal and gh['G4_CROSS']==n-1): raise G4R0Error('flattened G2 tree invariant')
            fc=_tree_canon(fn,fp); flat_canons.add(fc); grade_profiles.add((gh['G3_INTERNAL'],gh['G4_CROSS']))
            d=_descriptor(st,alpha); descs[d['science_sha256']]=d
            coarse.setdefault(cc,{'topology_canon':cc,'metrics':cm,'example_state_digest':st.construction_digest,'flattened_g2_metrics':fm,'flattened_g2_topology_canon':fc,'edge_grade_histogram':gh})
        if len(descs)!=1: raise G4R0Error('graduated descriptor not uniform at homogeneous fixed rank')
        row={'g3_unit_count':n,'owner_explicit_incidence_branch_count_after_outer_canon':len(current),'unique_coarse_unlabelled_tree_topologies':len(coarse),'unique_flattened_g2_tree_topologies':len(flat_canons),'unique_graduated_descriptor_states':1,'graduated_descriptor':next(iter(descs.values())),'flattened_grade_profiles':[list(x) for x in sorted(grade_profiles)],'coarse_topologies':[coarse[k] for k in sorted(coarse)]}
        rows.append(row)
        if witness is None and len(coarse)>1:
            ex=[coarse[k] for k in sorted(coarse)[:2]]
            witness={'g3_unit_count':n,'shared_graduated_descriptor':row['graduated_descriptor'],'shared_g4_bridge_operator':[a,b],'shared_g4_bridge_multiset_count':n-1,'topology_A':ex[0],'topology_B':ex[1],'meaning':'Same graduated G4 descriptor and same typed G4 bridge-consumption multiset, but nonisomorphic intrinsic G3-unit incidence trees.'}
    if witness is None: raise G4R0Error('no same-descriptor different-coarse-topology witness')
    shape_rows=unlabeled_tree_shape_census(spec['shape_census']['max_g3_units'])
    obs=[int(r['unlabelled_tree_topology_count']) for r in shape_rows]
    if obs!=spec['shape_census']['expected_unlabelled_tree_counts_n1_to_nmax']: raise G4R0Error('unlabelled tree census mismatch')
    shape={'schema_id':'IG_G4_R0_BOUNDED_TREE_SHAPE_REALISABILITY_CERTIFICATE_V1','status':'PASS','method':'EXHAUSTIVE_PRUEFER_ENUMERATION_PLUS_SEQUENTIAL_LEAF_ATTACHMENT_REALISABILITY_UNDER_CONSERVATIVE_ENDPOINT_HEADROOM','max_g3_units':spec['shape_census']['max_g3_units'],'chosen_seed_ref':seed_ref,'chosen_operator':[a,b],'seed_endpoint_headroom':[seed.total_caps[a],seed.total_caps[b]],'required_max_tree_degree':spec['shape_census']['max_g3_units']-1,'all_shapes_realisable_by_sequential_leaf_attachment':True,'rows':shape_rows,'exact_materialisation_boundary':f"Exact owner-explicit G4 incidence branches materialised only through n={spec['exact_growth_probe']['max_g3_units']}; higher shape coverage is a structural certificate, not an exact branch census."}
    shape['science_sha256']=canonical_sha256(shape)
    result={'schema_id':'IG_G4_R0_STRUCTURAL_RECON_RESULT_V1','status':'PASS','classification':spec['outcomes']['pass'],'promotion':False,'g4_graduated_before_r0':True,'g4_graduation_preserved':True,'g5_started':False,'authority':auth,'hierarchical_tree_theorem':theorem,'frozen_question_sha256':spec['science_sha256'],'s0_term_load':repro,'tier1_alphabet':alpha,'tier1_values':tier1vals,'exact_growth_probe':{'seed_g3_term_ref':seed_ref,'seed_caps7':list(seed.total_caps),'seed_tier1_class':seed_label,'repeated_g4_bridge_operator':[a,b],'rows':rows},'public_descriptor_topology_separation_witness':witness,'bounded_tree_shape_realisability':shape,'intrinsic_structure_status':'NONTRIVIAL_HIERARCHICAL_EDGE_GRADED_TREE_STRUCTURE_PRESENT','coarse_distance_status':'INTRINSIC_G3_UNIT_GRAPH_DISTANCE_PRESENT','fine_distance_status':'INTRINSIC_FLATTENED_G2_GRAPH_DISTANCE_PRESENT','contraction_status':'CONTRACTING_G3_INTERNAL_G2_TREE_COMPONENTS_RECOVERS_COARSE_G4_TREE','cycle_status':'NO_COARSE_G4_OR_FLATTENED_G2_INCIDENCE_CYCLES_UNDER_FROZEN_ONE_CROSS_BINARY_GRAMMAR','topology_visibility':'GRADUATED_CAPS7_PLUS_TIER1_CLASS_BAG_DOES_NOT_DETERMINE_NEW_G3_UNIT_CONNECTION_TOPOLOGY','topology_promoted':False,'shell_profile_promoted':False,'dimension_status':'NOT_EARNED','geometry_status':'PHYSICAL_GEOMETRY_NOT_EARNED','next_recommendation':'AFTER_COLD_CERTIFICATION_REVIEW_G4_R0_AND_DESIGN_G4_R1; DO_NOT_PROMOTE_INCIDENCE_TOPOLOGY_WITHOUT_A_NEW_REGISTERED_OBSERVER','nonclaims':spec['nonclaims']}
    result['science_sha256']=canonical_sha256(result)
    return result

def stable_payload(result:Mapping[str,Any])->dict[str,Any]:
    return {k:v for k,v in result.items() if k not in {'source_sha256','source_version','registry_sha256','execution_metadata'}}

def compare_cold_replay(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    ps=canonical_sha256(stable_payload(primary)); cs=canonical_sha256(stable_payload(cold))
    fail=[]
    if primary.get('science_sha256')!=cold.get('science_sha256'): fail.append('SCIENCE_SHA')
    if primary.get('source_sha256')!=cold.get('source_sha256'): fail.append('SOURCE_SHA')
    if primary.get('registry_sha256')!=cold.get('registry_sha256'): fail.append('REGISTRY_SHA')
    if primary.get('classification')!=cold.get('classification'): fail.append('CLASSIFICATION')
    if ps!=cs: fail.append('STABLE_PAYLOAD')
    out={'schema_id':'IG_G4_R0_COLD_REPLAY_COMPARISON_V1','status':'PASS' if not fail else 'FAIL','certification':'CERTIFIED_PASS' if not fail else 'CERTIFICATION_FAILED','failures':fail,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'primary_source_sha256':primary.get('source_sha256'),'cold_source_sha256':cold.get('source_sha256'),'primary_registry_sha256':primary.get('registry_sha256'),'cold_registry_sha256':cold.get('registry_sha256'),'science_sha_equal':primary.get('science_sha256')==cold.get('science_sha256'),'source_sha_equal':primary.get('source_sha256')==cold.get('source_sha256'),'registry_sha_equal':primary.get('registry_sha256')==cold.get('registry_sha256'),'classification_equal':primary.get('classification')==cold.get('classification'),'stable_scientific_payload_exact_equal':ps==cs,'stable_scientific_payload_sha256':ps,'same_registered_decoder_native_experiment':True}
    out['comparison_sha256']=canonical_sha256(out); return out

def certified_closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],comparison:Mapping[str,Any])->dict[str,Any]:
    ok=primary.get('status')=='PASS' and cold.get('status')=='PASS' and comparison.get('certification')=='CERTIFIED_PASS'
    out={'schema_id':'IG_G4_R0_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAILED','experiment_id':'G4:R0.STRUCTURAL_AUDIT','classification':primary.get('classification') if ok else 'G4_R0_REVIEW_REQUIRED_NO_PROMOTION','science_sha256':primary.get('science_sha256'),'stable_science_payload_sha256':comparison.get('stable_scientific_payload_sha256'),'comparison_sha256':comparison.get('comparison_sha256'),'source_sha256':primary.get('source_sha256'),'registry_sha256':primary.get('registry_sha256'),'promotion':False,'g4_graduation_preserved':True,'topology_promoted':False,'shell_profile_promoted':False,'g5_started':False,'next_authorized_stage':'G4:R1_DESIGN_AFTER_HUMAN_REVIEW','failures':comparison.get('failures',[])}
    out['closeout_sha256']=canonical_sha256(out); return out
