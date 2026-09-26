from __future__ import annotations

"""G5:R1 finite hidden-fiber and relation-valued grafting audit.

Non-promoting reconnaissance over the already-graduated G5 public algebra.  The
stage defines the exact hidden combinatorial fiber and its endpoint-typed tree
grafting law; it does not add topology to the public state.
"""
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import json

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .uplift_g3_r0 import _tree_canon, _graph_metrics
from .uplift_g3_r1 import _generate_tree_shapes

class G5R1Error(RuntimeError): pass
_SPEC='G5_R1_FIBER_GRAFT_AUDIT_SPEC_V1.json'

def _resource(name:str)->Path:
    return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def r1_spec()->dict[str,Any]:
    obj=json.loads(_resource(_SPEC).read_text(encoding='utf-8'))
    if obj.get('schema_id')!='IG_G5_R1_FIBER_GRAFT_AUDIT_SPEC_V1': raise G5R1Error('bad G5:R1 spec schema')
    if canonical_sha256({k:v for k,v in obj.items() if k!='science_sha256'})!=obj.get('science_sha256'): raise G5R1Error('G5:R1 spec hash mismatch')
    return obj

def _pub(q:Mapping[str,Any])->dict[str,Any]:
    return {k:q.get(k) for k in ('legal','descriptor','caps7','H_class_bag')}

def verify_authority(r0_primary:Mapping[str,Any],r0_verification:Mapping[str,Any],r0_closeout:Mapping[str,Any])->dict[str,Any]:
    s=r1_spec(); a=s['authority']; f=[]
    if r0_primary.get('schema_id')!='IG_G5_R0_POST_GRADUATION_STRUCTURAL_RECON_RESULT_V1' or r0_primary.get('status')!='PASS': f.append('R0_PRIMARY_STATUS')
    if r0_primary.get('science_sha256')!=a['g5_r0_primary_science_sha256']: f.append('R0_PRIMARY_IDENTITY')
    if r0_primary.get('classification')!=a['required_r0_classification']: f.append('R0_CLASSIFICATION')
    if r0_primary.get('g5_graduation_preserved') is not True or r0_primary.get('topology_promoted') is not False: f.append('R0_FIREWALL')
    if r0_verification.get('schema_id')!='IG_G5_R0_INDEPENDENT_VERIFICATION_V1' or r0_verification.get('status')!='PASS': f.append('R0_VERIFICATION_STATUS')
    if r0_verification.get('verification_sha256')!=a['g5_r0_independent_verification_sha256']: f.append('R0_VERIFICATION_IDENTITY')
    if r0_closeout.get('schema_id')!='IG_G5_R0_CERTIFIED_CLOSEOUT_V1' or r0_closeout.get('status')!='CERTIFIED_PASS': f.append('R0_CLOSEOUT_STATUS')
    if r0_closeout.get('closeout_sha256')!=a['g5_r0_closeout_sha256']: f.append('R0_CLOSEOUT_IDENTITY')
    if r0_primary.get('first_same_public_nonisomorphic_tree_witness',{}).get('same_public') is not True: f.append('R0_FIBER_WITNESS')
    out={'schema_id':'IG_G5_R1_AUTHORITY_CHECK_V1','status':'PASS' if not f else 'FAIL','failures':f,'r0_primary_science_sha256':r0_primary.get('science_sha256'),'r0_verification_sha256':r0_verification.get('verification_sha256'),'r0_closeout_sha256':r0_closeout.get('closeout_sha256'),'public_descriptor':a['required_public_descriptor'],'g5_graduation_preserved':True,'promotion':False}
    out['science_sha256']=canonical_sha256(out)
    if f: raise G5R1Error('authority mismatch: '+','.join(f))
    return out

def _shell_multiset(m:Mapping[str,Any])->list[list[int]]:
    return [list(x) for x in sorted(tuple(int(y) for y in r) for r in m.get('shell_profiles',[]))]

def shape_census(max_n:int)->dict[str,Any]:
    s=r1_spec(); shapes=_generate_tree_shapes(max_n); rows=[]; first_shell_collision=None
    observed=[]
    for n in range(1,max_n+1):
        observed.append(len(shapes[n])); groups=defaultdict(list); transition_edges=0; branch_counts=[]
        for canon,edges in sorted(shapes[n].items()):
            met=_graph_metrics(n,edges); groups[json.dumps(_shell_multiset(met),separators=(',',':'))].append(canon)
            if n<max_n:
                ch={_tree_canon(n+1,list(edges)+[(v,n)]) for v in range(n)}; transition_edges+=len(ch); branch_counts.append(len(ch))
        collisions=[v for v in groups.values() if len(v)>1]
        if first_shell_collision is None and collisions:
            first_shell_collision={'hidden_vertex_count':n,'collision_group_size':len(collisions[0]),'topology_canons':collisions[0][:4]}
        rows.append({'hidden_vertex_count':n,'unlabelled_tree_topology_count':len(shapes[n]),'distinct_leaf_attachment_transition_edges':transition_edges if n<max_n else None,'min_distinct_children_per_topology':min(branch_counts) if branch_counts else None,'max_distinct_children_per_topology':max(branch_counts) if branch_counts else None,'shell_profile_class_count':len(groups),'shell_profile_collision_group_count':len(collisions)})
    exp=list(s['bounded_shape_census']['expected_unlabelled_tree_counts_n1_to_nmax'])[:max_n]
    if observed!=exp: raise G5R1Error(f'tree census mismatch observed={observed} expected={exp}')
    out={'schema_id':'IG_G5_R1_HOMOGENEOUS_FIBER_SHAPE_CENSUS_V1','status':'PASS','max_hidden_vertices':max_n,'rows':rows,'first_shell_profile_collision_within_scope':first_shell_collision,'expected_counts_match':True,'interpretation':'Homogeneous H-class C / operator (0,0) structural slice only; no compact-read sufficiency claim.'}
    out['science_sha256']=canonical_sha256(out); return out

def exact_fiber_witness(r0_primary:Mapping[str,Any])->dict[str,Any]:
    w=dict(r0_primary['first_same_public_nonisomorphic_tree_witness'])
    if not w.get('same_public') or not w.get('nonisomorphic'): raise G5R1Error('R0 topology-blind fiber witness failed')
    out={'schema_id':'IG_G5_R1_EXACT_FIBER_WITNESS_V1','status':'PASS','fiber_base':w['shared_public_state'],'fiber_type':'FINITE_COMBINATORIAL_SET_FIBER','known_distinct_exact_strata_lower_bound':2,'hidden_vertex_count':w['hidden_vertex_count'],'strata':[{'name':'A','edges':w['tree_A_edges'],'exact_canon':w['tree_A_canon']},{'name':'B','edges':w['tree_B_edges'],'exact_canon':w['tree_B_canon']}],'public_descriptor_topology_blind':True,'meaning':'At least two nonisomorphic endpoint-decorated hidden trees lie over one exact graduated public state.'}
    out['science_sha256']=canonical_sha256(out); return out

def relation_valued_graft_witness()->dict[str,Any]:
    ad=G4AcceptedAdapter(); ops=set(ad.operator_basis())
    if len(ops)!=31 or (0,0) not in ops: raise G5R1Error('frozen operator basis mismatch')
    parent=DecoratedG4Tree(4,((0,1),(1,2),(2,3)),('C','C','C','C'),((0,0),(0,0),(0,0)))
    ppub=_pub(ad.public_read(parent)); children=[]
    for owner in (0,1):
        rel=ad.graft_relation(parent,owner,new_H_class='C',operator=(0,0))
        if len(rel)!=1: raise G5R1Error('expected exactly one explicit-owner graft result')
        t=rel[0]; children.append({'owner_vertex':owner,'edges':[list(x) for x in t.edges],'exact_canon':repr(ad.unrooted_canon(t)),'public':_pub(ad.public_read(t)),'metrics':_graph_metrics(t.n,t.edges)})
    same_public=children[0]['public']==children[1]['public']; noniso=children[0]['exact_canon']!=children[1]['exact_canon']
    if not same_public or not noniso: raise G5R1Error('relation-valued graft witness failed')
    out={'schema_id':'IG_G5_R1_RELATION_VALUED_GRAFT_WITNESS_V1','status':'PASS','parent_public':ppub,'parent_shape':'4_VERTEX_PATH','operator':[0,0],'new_H_class':'C','lawful_owner_choices':[0,1],'children':children,'same_graduated_public_output':same_public,'nonisomorphic_exact_children':noniso,'distinct_exact_child_count':len({c['exact_canon'] for c in children}),'conclusion':'One fixed exact parent plus one fixed public graft operation can have multiple hidden exact child isomorphism classes solely from lawful owner choice.'}
    out['science_sha256']=canonical_sha256(out); return out

def hidden_graft_law()->dict[str,Any]:
    s=r1_spec(); out={'schema_id':'IG_G5_R1_ENDPOINT_TYPED_DECORATED_TREE_GRAFTING_LAW_V1','status':'PASS','hidden_carrier':'FINITE_H_CLASS_LABELED_TREES_WITH_ENDPOINT_TYPED_G5_EDGE_OPERATORS','operation':s['hidden_composition_law']['rule'],'relation_valued_reason':'The graduated public state forgets attachment-owner incidence; inequivalent lawful owner pairs can produce nonisomorphic exact hidden children with one public shadow.','public_shadow':s['hidden_composition_law']['public_shadow'],'tree_preservation':'Disjoint union of two trees plus exactly one cross edge is a tree.','endpoint_semantics':'Directed operator entries remain attached to their respective endpoints under storage reversal/isomorphism.','scope':s['hidden_composition_law']['scope'],'associativity_scope':'Graduated public composition is closed by G5:S6; raw hidden relation-valued outcome equality under arbitrary rebracketing is not newly claimed here.','promotion':False}
    out['science_sha256']=canonical_sha256(out); return out

def run_g5_r1_fiber_graft_audit(*,engine:Any,r0_primary:Mapping[str,Any],r0_verification:Mapping[str,Any],r0_closeout:Mapping[str,Any])->dict[str,Any]:
    s=r1_spec(); auth=verify_authority(r0_primary,r0_verification,r0_closeout); fiber=exact_fiber_witness(r0_primary); witness=relation_valued_graft_witness(); law=hidden_graft_law(); census=shape_census(int(s['bounded_shape_census']['max_hidden_vertices']))
    # Certified R0 1..7 prefix must remain exact.
    prefix=[int(x['unlabelled_tree_topology_count']) for x in census['rows'][:7]]
    r0prefix=[int(x['unlabelled_tree_topology_count']) for x in r0_primary['bounded_homogeneous_tree_shape_census']]
    if prefix!=r0prefix: raise G5R1Error('R0 tree prefix mismatch')
    out={'schema_id':'IG_G5_R1_FIBER_GRAFT_RESULT_V1','status':'PASS','stage_ref':'G5:R1','classification':s['outcomes']['pass'],'promotion':False,'g5_graduation_preserved':True,'g6_started':False,'authority':auth,'fiber_definition':s['fiber_definition'],'exact_fiber_witness':fiber,'hidden_composition_law':law,'relation_valued_graft_witness':witness,'bounded_homogeneous_shape_census':census,'exact_reference_hidden_read':s['candidate_read_policy']['exact_reference_read'],'compact_read_search':s['candidate_read_policy']['compact_read_search'],'minimality_claim':False,'topology_promoted':False,'descriptor_changed':False,'next_authorized_stage':'G5:R2_DESIGN_AFTER_HUMAN_REVIEW','cost_control':s['cost_control'],'nonclaims':s['nonclaims']}
    out['science_sha256']=canonical_sha256(out); return out

def stable_payload(result:Mapping[str,Any])->dict[str,Any]:
    return {k:v for k,v in result.items() if k not in {'source_sha256','source_version','registry_sha256','execution_metadata'}}

def compare_cold_replay(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    ps=canonical_sha256(stable_payload(primary)); cs=canonical_sha256(stable_payload(cold)); f=[]
    if primary.get('science_sha256')!=cold.get('science_sha256'): f.append('SCIENCE_SHA')
    if primary.get('classification')!=cold.get('classification'): f.append('CLASSIFICATION')
    if ps!=cs: f.append('STABLE_PAYLOAD')
    out={'schema_id':'IG_G5_R1_COLD_REPLAY_COMPARISON_V1','status':'PASS' if not f else 'FAIL','certification':'CERTIFIED_PASS' if not f else 'CERTIFICATION_FAILED','failures':f,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'stable_scientific_payload_exact_equal':ps==cs,'stable_scientific_payload_sha256':ps}
    out['comparison_sha256']=canonical_sha256(out); return out

def certified_closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],comparison:Mapping[str,Any],independent_shape_check:Mapping[str,Any],independent_graft_check:Mapping[str,Any])->dict[str,Any]:
    ok=primary.get('status')=='PASS' and cold.get('status')=='PASS' and comparison.get('certification')=='CERTIFIED_PASS' and independent_shape_check.get('status')=='PASS' and independent_graft_check.get('status')=='PASS'
    out={'schema_id':'IG_G5_R1_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAILED','experiment_id':'G5:R1.FIBER_GRAFT_AUDIT','classification':primary.get('classification') if ok else 'G5_R1_REVIEW_REQUIRED_NO_PROMOTION','science_sha256':primary.get('science_sha256'),'stable_science_payload_sha256':comparison.get('stable_scientific_payload_sha256'),'comparison_sha256':comparison.get('comparison_sha256'),'independent_shape_check_sha256':independent_shape_check.get('science_sha256'),'independent_graft_check_sha256':independent_graft_check.get('science_sha256'),'source_sha256':primary.get('source_sha256'),'registry_sha256':primary.get('registry_sha256'),'promotion':False,'g5_graduation_preserved':True,'topology_promoted':False,'g6_started':False,'next_authorized_stage':'G5:R2_DESIGN_AFTER_HUMAN_REVIEW' if ok else None,'failures':comparison.get('failures',[])}
    out['closeout_sha256']=canonical_sha256(out); return out
