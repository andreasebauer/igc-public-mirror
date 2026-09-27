from __future__ import annotations
from typing import Any, Mapping
from importlib.resources import files
from pathlib import Path
import json
from .canon import canonical_sha256
from .uplift_g4_r0 import G4R0IncidenceState, coarse_graph, flattened_g2_graph, _choose_seed, _reserve_leaf_with_owner
from .uplift_g4_s1_rebase import load_states
from .uplift_g4_s5_rebase import h_index, descriptor
from .uplift_g3_r0 import _graph_metrics, _tree_canon, unlabeled_tree_shape_census, hierarchical_tree_theorem as g3_tree_theorem

class G4R0RebaseError(RuntimeError): pass

def _resource(name:str)->Path:
    return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def spec()->dict[str,Any]:
    o=json.loads(_resource('G4_R0_REBASE_STRUCTURAL_RECON_SPEC_V1.json').read_text())
    if o.get('schema_id')!='IG_G4_R0_REBASE_STRUCTURAL_RECON_SPEC_V1': raise G4R0RebaseError('bad spec schema')
    if canonical_sha256({k:v for k,v in o.items() if k!='science_sha256'})!=o.get('science_sha256'): raise G4R0RebaseError('spec hash mismatch')
    return o

def verify_authority(*,s6:Mapping[str,Any],s6_replay:Mapping[str,Any],s6_closeout:Mapping[str,Any],s0:Mapping[str,Any],s3:Mapping[str,Any])->dict[str,Any]:
    f=[]
    if s6.get('schema_id')!='IG_G4_S6_REBASE_RECURSIVE_CLOSURE_RESULT_V1' or s6.get('status')!='PASS': f.append('S6')
    if s6.get('g4_graduated') is not True or s6.get('g4_r0_rebase_unlocked') is not True: f.append('R0_UNLOCK')
    if s6.get('public_descriptor')!='CAPS7_PLUS_H_CLASS_BAG': f.append('DESCRIPTOR')
    if s6.get('classification')!='G4_REGRADUATED_CAPS7_PLUS_H_CLASS_BAG_RECURSIVE_RELATION_GRAMMAR_EARNED_R0_REBASE_UNLOCKED': f.append('CLASSIFICATION')
    if s6_replay.get('certification')!='CERTIFIED_PASS' or s6_replay.get('primary_science_sha256')!=s6.get('science_sha256'): f.append('S6_REPLAY')
    if s6_closeout.get('status')!='CERTIFIED_PASS' or s6_closeout.get('next_authorized_stage')!='G4:R0.REBASE': f.append('S6_CLOSEOUT')
    if s0.get('schema_id')!='IG_G4_S0_REBASE_INTERFACE_RESULT_V1' or s0.get('status')!='PASS': f.append('S0_REBASE')
    ci=s3.get('candidate_interface',{})
    if ci.get('schema_id')!='IG_G4_S3_REBASE_CANDIDATE_INDEX_V1' or ci.get('candidate_class_count')!=4: f.append('S3_H_INDEX')
    g3=g3_tree_theorem()
    if g3.get('status')!='PASS': f.append('G3_TREE_THEOREM')
    if f: raise G4R0RebaseError('authority failed: '+','.join(f))
    o={'schema_id':'IG_G4_R0_REBASE_AUTHORITY_V1','status':'PASS','s6_science_sha256':s6.get('science_sha256'),'s6_replay_science_sha256':s6_replay.get('science_sha256'),'s6_closeout_science_sha256':s6_closeout.get('science_sha256'),'s0_science_sha256':s0.get('science_sha256'),'s3_candidate_interface_science_sha256':ci.get('science_sha256'),'g3_hierarchical_tree_theorem_science_sha256':g3.get('science_sha256'),'graduated_descriptor':'CAPS7_PLUS_H_CLASS_BAG','historical_g4_forward_use':'SUPERSEDED_BY_CERTIFIED_REBASE'}
    o['science_sha256']=canonical_sha256(o); return o

def hierarchical_tree_theorem()->dict[str,Any]:
    g3=g3_tree_theorem()
    o={'schema_id':'IG_G4_R0_REBASE_HIERARCHICAL_TREE_THEOREM_V1','status':'PASS','proof_method':'STRUCTURAL_INDUCTION_ON_REGRADUATED_G4_ONE_CROSS_BINARY_GRAMMAR','coarse_base':'One G3 leaf is one coarse vertex and has no G4-cross edge.','coarse_step':'Every frozen binary G4 constructor joins two disjoint connected G4 terms by exactly one new typed G4-cross relation between one eligible owner leaf on each side. Relation-valued owner choice changes only the endpoint, never the one-edge count.','coarse_conclusion':'Every finite connected G4 term contracts to a finite tree on its G3 leaves: m=n-1, beta_1=0, and every G4-cross relation is a bridge.','mid_scale_authority':g3.get('science_sha256'),'mid_scale_conclusion':'Each G3 leaf expands to its certified G2-unit tree. Replacing each coarse vertex by that tree and each coarse edge by one G4-cross edge preserves connectedness and acyclicity. The flattened G2-unit graph is therefore a tree graded by G3_INTERNAL and G4_CROSS edges.','contraction':'Contracting each connected G3-internal G2 component recovers the coarse G3-leaf tree.','scope':'ALL_FINITE_COMPLETE_TERMS_OF_REGRADUATED_G4_REBASE_GRAMMAR','promotion':False,'nonclaims':['NO_PHYSICAL_GEOMETRY','NO_DIMENSION','NO_METRIC_EMBEDDING','NO_TIME','NO_TOPOLOGY_PROMOTION','NO_RAW_H_PROMOTION']}
    o['science_sha256']=canonical_sha256(o); return o

def _desc(st:G4R0IncidenceState, alphabet):
    return descriptor(caps7=st.total_caps,h_labels=st.leaf_class_labels,alphabet=alphabet)

def _deterministic_graft_seed(state:G4R0IncidenceState, seed, seed_label:str, *, outer_owner:int, a:int, b:int)->G4R0IncidenceState:
    """One exact lawful G4 graft with a prescribed coarse owner and deterministic internal witnesses.

    This is a witness lane, not a frontier enumerator.  It still checks the chosen
    internal reservation successors against the certified G3 relation through
    _reserve_leaf_with_owner.
    """
    if outer_owner < 0 or outer_owner >= len(state.leaves):
        raise G4R0RebaseError('prescribed outer owner out of range')
    left = _reserve_leaf_with_owner(state.leaves[outer_owner], int(a))
    right = _reserve_leaf_with_owner(seed, int(b))
    if not left or not right:
        raise G4R0RebaseError('prescribed witness graft has no lawful reservation witness')
    li, lsucc = min(left, key=lambda x:(x[1].construction_digest,int(x[0])))
    ri, rsucc = min(right, key=lambda x:(x[1].construction_digest,int(x[0])))
    leaves=list(state.leaves); leaves[outer_owner]=lsucc
    new_idx=len(leaves); leaves.append(rsucc)
    labels=state.leaf_class_labels+(str(seed_label),)
    edge=(int(outer_owner),int(new_idx),int(a),int(b),int(li),int(ri))
    return G4R0IncidenceState(tuple(leaves),state.edges+(edge,),labels)

def _audit_exact_state(st:G4R0IncidenceState, *, expected_n:int, alphabet):
    cn,cp=coarse_graph(st); cm=_graph_metrics(cn,cp)
    if not (cn==expected_n and len(cp)==expected_n-1 and cm['connected'] and cm['beta']==0 and cm['bridges']==len(cp)):
        raise G4R0RebaseError('targeted coarse tree invariant')
    cc=_tree_canon(cn,cp)
    fn,fe=flattened_g2_graph(st); fp=[(e['u'],e['v']) for e in fe]; fm=_graph_metrics(fn,fp)
    gh={'G3_INTERNAL':sum(e['grade']=='G3_INTERNAL' for e in fe),'G4_CROSS':sum(e['grade']=='G4_CROSS' for e in fe)}
    if not (fm['connected'] and fm['beta']==0 and fm['bridges']==len(fp) and len(fp)==fn-1 and gh['G4_CROSS']==expected_n-1):
        raise G4R0RebaseError('targeted flattened tree invariant')
    return {'topology_canon':cc,'metrics':cm,'example_state_digest':st.construction_digest,
            'flattened_g2_metrics':fm,'flattened_g2_topology_canon':_tree_canon(fn,fp),
            'edge_grade_histogram':gh,'graduated_descriptor':_desc(st,alphabet)}

def finalize(*,s6:Mapping[str,Any],s6_replay:Mapping[str,Any],s6_closeout:Mapping[str,Any],s0:Mapping[str,Any],s3:Mapping[str,Any],operator_basis:list[list[int]])->dict[str,Any]:
    auth=verify_authority(s6=s6,s6_replay=s6_replay,s6_closeout=s6_closeout,s0=s0,s3=s3)
    theorem=hierarchical_tree_theorem(); sp=spec()
    states,refs,_meta=load_states(s0)
    alpha,refmap,hvals=h_index(s3)
    ops=sorted({tuple(map(int,x)) for x in operator_basis})
    if len(ops)!=31: raise G4R0RebaseError('31-operator basis required')
    seed_ref,seed,seed_label,(a,b)=_choose_seed(states,refmap,ops)
    need=sp['shape_census']['max_g3_units']-1
    if min(seed.total_caps[a],seed.total_caps[b])<need+1: raise G4R0RebaseError('seed/operator lacks shape headroom')

    # Targeted exact lane only: construct one path and one star witness.  No
    # owner-explicit frontier is ever enumerated.
    base=G4R0IncidenceState((seed,),tuple(),(seed_label,))
    path=base; star=base; rows=[]
    for n in range(2,sp['exact_growth_probe']['max_g3_units']+1):
        path=_deterministic_graft_seed(path,seed,seed_label,outer_owner=n-2,a=a,b=b)
        star_owner=0
        star=_deterministic_graft_seed(star,seed,seed_label,outer_owner=star_owner,a=a,b=b)
        pa=_audit_exact_state(path,expected_n=n,alphabet=alpha)
        sa=_audit_exact_state(star,expected_n=n,alphabet=alpha)
        if pa['graduated_descriptor']['science_sha256']!=sa['graduated_descriptor']['science_sha256']:
            raise G4R0RebaseError('targeted path/star descriptors differ')
        unique_topos=len({pa['topology_canon'],sa['topology_canon']})
        rows.append({'g3_unit_count':n,'targeted_exact_branch_count':1 if path.construction_digest==star.construction_digest else 2,
                     'unique_coarse_unlabelled_tree_topologies':unique_topos,
                     'unique_graduated_descriptor_states':1,
                     'graduated_descriptor':pa['graduated_descriptor'],
                     'path_topology':pa,'star_topology':sa})

    pa=rows[-1]['path_topology']; sa=rows[-1]['star_topology']
    if pa['topology_canon']==sa['topology_canon']:
        raise G4R0RebaseError('targeted n=4 path/star witness unexpectedly isomorphic')
    witness={'g3_unit_count':4,'shared_graduated_descriptor':rows[-1]['graduated_descriptor'],
             'shared_g4_bridge_operator':[a,b],'shared_g4_bridge_multiset_count':3,
             'topology_A':pa,'topology_B':sa,
             'meaning':'Exact targeted path/star constructions have the same re-graduated CAPS7+H-class-bag descriptor and same typed G4 bridge-consumption multiset, but nonisomorphic intrinsic G3-unit incidence trees.'}

    shape_rows=unlabeled_tree_shape_census(sp['shape_census']['max_g3_units'])
    observed=[int(r['unlabelled_tree_topology_count']) for r in shape_rows]
    if observed!=sp['shape_census']['expected_unlabelled_tree_counts_n1_to_nmax']: raise G4R0RebaseError('shape census mismatch')
    shape={'schema_id':'IG_G4_R0_REBASE_BOUNDED_TREE_SHAPE_REALISABILITY_V2','status':'PASS',
           'method':'EXHAUSTIVE_PRUEFER_ENUMERATION_PLUS_STRUCTURAL_SEQUENTIAL_LEAF_ATTACHMENT_CERTIFICATE_NO_OWNER_FRONTIER_ENUMERATION',
           'max_g3_units':sp['shape_census']['max_g3_units'],'chosen_seed_ref':seed_ref,'chosen_h_class':seed_label,
           'chosen_operator':[a,b],'all_shapes_realisable_by_sequential_leaf_attachment':True,'rows':shape_rows,
           'exact_materialisation_boundary':'Only two targeted exact n=4 witness constructions (path and star) are materialised; all broader topology coverage is theorem/certificate based.'}
    shape['science_sha256']=canonical_sha256(shape)
    o={'schema_id':'IG_G4_R0_REBASE_STRUCTURAL_RECON_RESULT_V2','status':'PASS','stage_ref':'G4:R0.REBASE',
       'classification':sp['outcomes']['pass'],'promotion':False,'g4_graduation_preserved':True,
       'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','public_descriptor_coordinate_count':11,'authority':auth,
       'hierarchical_tree_theorem':theorem,
       'targeted_exact_witness_lane':{'mode':sp['exact_growth_probe']['mode'],'seed_g3_term_ref':seed_ref,
          'seed_caps7':list(seed.total_caps),'seed_h_class':seed_label,'repeated_g4_bridge_operator':[a,b],'rows':rows},
       'public_descriptor_topology_separation_witness':witness,'bounded_tree_shape_realisability':shape,
       'intrinsic_structure_status':'NONTRIVIAL_HIERARCHICAL_EDGE_GRADED_TREE_STRUCTURE_PERSISTS_AFTER_REBASE',
       'coarse_distance_status':'INTRINSIC_G3_UNIT_GRAPH_DISTANCE_PRESENT','fine_distance_status':'INTRINSIC_FLATTENED_G2_GRAPH_DISTANCE_PRESENT',
       'contraction_status':'CONTRACTING_G3_INTERNAL_G2_TREE_COMPONENTS_RECOVERS_COARSE_G4_TREE',
       'cycle_status':'NO_COARSE_G4_OR_FLATTENED_G2_INCIDENCE_CYCLES_UNDER_FROZEN_ONE_CROSS_BINARY_GRAMMAR',
       'topology_visibility':'CAPS7_PLUS_H_CLASS_BAG_DOES_NOT_DETERMINE_G3_UNIT_CONNECTION_TOPOLOGY',
       'topology_promoted':False,'raw_h_promoted':False,'g5_started':False,
       'next_authorized_stage':'G4:R1.REBASE_DESIGN_AFTER_HUMAN_REVIEW',
       'cost_control':{'new_parallel_science_kernels':0,'lower_layer_rematerialization':False,
          'exact_owner_frontier_enumeration':False,'targeted_exact_witness_constructions':2},
       'nonclaims':sp['nonclaims']}
    o['science_sha256']=canonical_sha256(o); return o

def compare(p:Mapping[str,Any],c:Mapping[str,Any])->dict[str,Any]:
    checks={'primary_pass':p.get('status')=='PASS','cold_pass':c.get('status')=='PASS','science_sha_equal':p.get('science_sha256')==c.get('science_sha256'),'classification_equal':p.get('classification')==c.get('classification'),'tree_theorem_equal':p.get('hierarchical_tree_theorem')==c.get('hierarchical_tree_theorem'),'witness_equal':p.get('public_descriptor_topology_separation_witness')==c.get('public_descriptor_topology_separation_witness')}; ok=all(checks.values())
    o={'schema_id':'IG_G4_R0_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':p.get('science_sha256'),'cold_science_sha256':c.get('science_sha256')}; o['science_sha256']=canonical_sha256(o); return o

def closeout(p:Mapping[str,Any],c:Mapping[str,Any],r:Mapping[str,Any])->dict[str,Any]:
    ok=r.get('certification')=='CERTIFIED_PASS' and p.get('status')=='PASS' and c.get('status')=='PASS'
    o={'schema_id':'IG_G4_R0_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':p.get('classification'),'primary_science_sha256':p.get('science_sha256'),'cold_science_sha256':c.get('science_sha256'),'replay_science_sha256':r.get('science_sha256'),'g4_graduation_preserved':ok,'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG' if ok else None,'topology_promoted':False,'g5_started':False,'next_authorized_stage':'G4:R1.REBASE_DESIGN_AFTER_HUMAN_REVIEW' if ok else None}; o['science_sha256']=canonical_sha256(o); return o
