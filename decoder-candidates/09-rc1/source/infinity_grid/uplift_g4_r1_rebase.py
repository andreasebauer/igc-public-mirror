from __future__ import annotations

"""Registered non-promoting G4:R1.REBASE finite-fiber / decorated-tree grafting audit."""

from importlib.resources import files
from typing import Any, Mapping
from collections import defaultdict
import json

from .canon import canonical_sha256
from .uplift_g3_r0 import _graph_metrics, _tree_canon
from .uplift_g3_r1 import _generate_tree_shapes

class G4R1RebaseError(RuntimeError):
    pass

_SPEC='resources/uplift/G4_R1_REBASE_FIBER_GRAFT_AUDIT_SPEC_V1.json'

def spec()->dict[str,Any]:
    o=json.loads(files('infinity_grid').joinpath(_SPEC).read_text())
    if o.get('schema_id')!='IG_G4_R1_REBASE_FIBER_GRAFT_AUDIT_SPEC_V1':
        raise G4R1RebaseError('bad spec schema')
    expected=o.get('science_sha256'); payload={k:v for k,v in o.items() if k!='science_sha256'}
    if canonical_sha256(payload)!=expected: raise G4R1RebaseError('spec hash mismatch')
    return o

def verify_authority(r0:Mapping[str,Any], replay:Mapping[str,Any], closeout:Mapping[str,Any])->dict[str,Any]:
    s=spec(); f=[]
    if r0.get('status')!='PASS' or r0.get('schema_id')!='IG_G4_R0_REBASE_STRUCTURAL_RECON_RESULT_V2': f.append('R0_RESULT')
    if str(r0.get('science_sha256'))!=s['authority']['g4_r0_primary_science_sha256']: f.append('R0_IDENTITY')
    if r0.get('classification')!='G4_R0_REBASE_HIERARCHICAL_EDGE_GRADED_TREE_STRUCTURE_PERSISTS_TARGETED_SAME_DESCRIPTOR_PATH_STAR_WITNESS_EARNED_R1_REBASE_DESIGN_AUTHORIZED': f.append('R0_CLASSIFICATION')
    if r0.get('public_descriptor')!=s['authority']['required_public_descriptor'] or r0.get('topology_promoted') is not False: f.append('R0_FIREWALL')
    if replay.get('status')!='PASS' or str(replay.get('science_sha256'))!=s['authority']['g4_r0_replay_science_sha256']: f.append('R0_REPLAY')
    if closeout.get('status')!='CERTIFIED_PASS' or str(closeout.get('science_sha256'))!=s['authority']['g4_r0_closeout_science_sha256']: f.append('R0_CLOSEOUT')
    o={'schema_id':'IG_G4_R1_REBASE_AUTHORITY_V1','status':'PASS' if not f else 'FAIL','failures':f,'r0_science_sha256':r0.get('science_sha256'),'r0_replay_science_sha256':replay.get('science_sha256'),'r0_closeout_science_sha256':closeout.get('science_sha256'),'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','promotion':False}
    o['science_sha256']=canonical_sha256(o)
    if f: raise G4R1RebaseError('authority failed: '+','.join(f))
    return o

def _shell_multiset(m:Mapping[str,Any])->list[list[int]]:
    return [list(x) for x in sorted(tuple(int(y) for y in r) for r in m.get('shell_profiles',[]))]

def _shape_census(max_n:int)->dict[str,Any]:
    shapes=_generate_tree_shapes(max_n)
    rows=[]; first_shell_collision=None
    for n in range(1,max_n+1):
        groups=defaultdict(list)
        child_edges=0
        for c,edges in sorted(shapes[n].items()):
            met=_graph_metrics(n,edges)
            groups[json.dumps(_shell_multiset(met),separators=(',',':'))].append(c)
            if n<max_n:
                child_edges += len({_tree_canon(n+1,list(edges)+[(v,n)]) for v in range(n)})
        collisions=[v for v in groups.values() if len(v)>1]
        if first_shell_collision is None and collisions:
            first_shell_collision={'g3_unit_count':n,'collision_group_size':len(collisions[0]),'topology_canons':collisions[0][:4]}
        rows.append({'g3_unit_count':n,'unlabelled_tree_topology_count':len(shapes[n]),'distinct_leaf_attachment_transition_edges':child_edges if n<max_n else None,'shell_profile_class_count':len(groups),'shell_profile_collision_group_count':len(collisions)})
    return {'schema_id':'IG_G4_R1_REBASE_HOMOGENEOUS_FIBER_SHAPE_CENSUS_V1','status':'PASS','max_g3_units':max_n,'rows':rows,'first_shell_profile_collision_within_scope':first_shell_collision,'interpretation':'Homogeneous H-class slice only; diagnostic shape census, not a global shell-profile sufficiency claim.'}

def _fiber_from_r0(r0:Mapping[str,Any])->dict[str,Any]:
    w=r0['public_descriptor_topology_separation_witness']
    A=w['topology_A']; B=w['topology_B']
    if A['topology_canon']==B['topology_canon']: raise G4R1RebaseError('R0 witness lost topology separation')
    if A['graduated_descriptor']['science_sha256']!=B['graduated_descriptor']['science_sha256']: raise G4R1RebaseError('R0 witness public descriptor mismatch')
    return {
      'schema_id':'IG_G4_R1_REBASE_EXACT_FIBER_WITNESS_V1','status':'PASS',
      'fiber_base':w['shared_graduated_descriptor'],
      'exact_known_coarse_strata_count_lower_bound':2,
      'coarse_strata':[
        {'name':'PATH','topology_canon':A['topology_canon'],'degree_sequence':A['metrics']['degree_sequence'],'diameter':A['metrics']['diameter'],'flattened_g2_topology_canon':A['flattened_g2_topology_canon']},
        {'name':'STAR','topology_canon':B['topology_canon'],'degree_sequence':B['metrics']['degree_sequence'],'diameter':B['metrics']['diameter'],'flattened_g2_topology_canon':B['flattened_g2_topology_canon']},
      ],
      'fiber_type':'FINITE_COMBINATORIAL_SET_FIBER',
      'public_descriptor_topology_blind':True,
      'meaning':'At least two nonisomorphic intrinsic G3-unit tree strata lie over one exact regraduated CAPS7+H-class-bag point.'
    }

def _graft_law()->dict[str,Any]:
    return {
      'schema_id':'IG_G4_R1_REBASE_DECORATED_TREE_GRAFTING_LAW_V1','status':'PASS',
      'hidden_carrier':'FINITE_G3_UNIT_TREES_WITH_VERTEX_H_CLASS_LABELS_AND_TYPED_G4_CROSS_EDGES',
      'operation':'For hidden representatives T,U and directed bridge type (a,b), enumerate every lawful owner pair (u,v) and add exactly one typed G4_CROSS edge u--v; inherit all vertex H-class labels and all pre-existing edge grades.',
      'relation_valued_reason':'Different eligible owner pairs may yield distinct hidden representatives while sharing the same public shadow.',
      'public_shadow':'C_ab((f,m),(g,n))=(f+g-e_a-e_b,m+n)',
      'tree_preservation':'Joining two disjoint trees by one cross edge yields a tree for every lawful owner choice.',
      'associativity_scope':'Public descriptor composition is recursively closed by certified G4:S6.REBASE. Raw hidden topology associativity is not claimed; relation-valued graft outcomes are organized modulo isomorphism.',
      'promotion':False
    }

def finalize(*,r0:Mapping[str,Any],r0_replay:Mapping[str,Any],r0_closeout:Mapping[str,Any])->dict[str,Any]:
    s=spec(); auth=verify_authority(r0,r0_replay,r0_closeout); fiber=_fiber_from_r0(r0); law=_graft_law(); census=_shape_census(int(s['bounded_shape_census']['max_g3_units']))
    out={
      'schema_id':'IG_G4_R1_REBASE_FIBER_GRAFT_RESULT_V1','status':'PASS','stage_ref':'G4:R1.REBASE',
      'classification':s['pass_classification'],'authority':auth,'fiber_definition':s['fiber_definition'],'exact_fiber_witness':fiber,
      'hidden_composition_law':law,'bounded_homogeneous_shape_census':census,
      'exact_reference_hidden_read':s['candidate_read_policy']['exact_reference_read'],
      'shell_only_status':s['candidate_read_policy']['shell_only_status'],
      'minimality_claim':False,'promotion':False,'topology_promoted':False,'raw_h_promoted':False,'g4_graduation_preserved':True,'g5_started':False,
      'next_authorized_stage':'G4:R2.REBASE_DESIGN','cost_control':s['cost_control'],'nonclaims':s['nonclaims']
    }
    out['science_sha256']=canonical_sha256(out); return out

def compare(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    checks={
      'primary_pass':primary.get('status')=='PASS','cold_pass':cold.get('status')=='PASS',
      'classification_equal':primary.get('classification')==cold.get('classification'),
      'science_sha_equal':primary.get('science_sha256')==cold.get('science_sha256'),
      'fiber_equal':primary.get('exact_fiber_witness')==cold.get('exact_fiber_witness'),
      'graft_law_equal':primary.get('hidden_composition_law')==cold.get('hidden_composition_law'),
      'shape_census_equal':primary.get('bounded_homogeneous_shape_census')==cold.get('bounded_homogeneous_shape_census')}
    ok=all(checks.values()); o={'schema_id':'IG_G4_R1_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256')}; o['science_sha256']=canonical_sha256(o); return o

def closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
    ok=replay.get('status')=='PASS'; o={'schema_id':'IG_G4_R1_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':primary.get('classification') if ok else None,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'replay_science_sha256':replay.get('science_sha256'),'g4_graduation_preserved':ok,'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG' if ok else None,'topology_promoted':False,'raw_h_promoted':False,'g5_started':False,'next_authorized_stage':'G4:R2.REBASE_DESIGN' if ok else None}; o['science_sha256']=canonical_sha256(o); return o
