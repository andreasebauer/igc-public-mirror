from __future__ import annotations

"""G5:R0 post-graduation structural reconnaissance.

Non-promoting exact hidden-tree read behind the graduated
D=(CAPS7,H-class bag) finite composition algebra.  This stage asks what exact
finite incidence is hidden by the graduated quotient; it does not reopen or
change the G5 public descriptor/law.
"""
from itertools import product
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import json

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .g5_capabilities import _g5_s1_carriers
from .uplift_g3_r0 import _tree_canon, _graph_metrics

class G5R0Error(RuntimeError): pass
_SPEC='G5_R0_POST_GRADUATION_STRUCTURAL_RECON_SPEC_V1.json'

def _resource(name:str)->Path:
    return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def r0_spec()->dict[str,Any]:
    obj=json.loads(_resource(_SPEC).read_text(encoding='utf-8'))
    if obj.get('schema_id')!='IG_G5_R0_POST_GRADUATION_STRUCTURAL_RECON_SPEC_V1':
        raise G5R0Error('bad G5:R0 spec schema')
    if canonical_sha256({k:v for k,v in obj.items() if k!='science_sha256'})!=obj.get('science_sha256'):
        raise G5R0Error('G5:R0 spec hash mismatch')
    return obj

def _prufer_edges(n:int, seq:tuple[int,...])->tuple[tuple[int,int],...]:
    if n==1: return tuple()
    if n==2: return ((0,1),)
    deg=[1]*n
    for x in seq: deg[x]+=1
    out=[]
    for x in seq:
        leaf=next(i for i,d in enumerate(deg) if d==1)
        out.append((leaf,x)); deg[leaf]-=1; deg[x]-=1
    rem=[i for i,d in enumerate(deg) if d==1]
    out.append((rem[0],rem[1]))
    return tuple(out)

def unlabeled_tree_shape_census(max_n:int)->list[dict[str,Any]]:
    rows=[]
    for n in range(1,max_n+1):
        reps={}
        seqs=[tuple()] if n<=2 else product(range(n),repeat=n-2)
        labelled=0
        for seq in seqs:
            e=_prufer_edges(n,tuple(seq)); labelled+=1
            c=_tree_canon(n,e)
            reps.setdefault(c,e)
        rows.append({'n':n,'labelled_pruefer_tree_count':labelled,'unlabelled_tree_topology_count':len(reps),'representatives':[list(map(list,reps[k])) for k in sorted(reps)]})
    return rows

def _homogeneous_tree(n:int, edges:tuple[tuple[int,int],...], *, klass:str='C', op:tuple[int,int]=(0,0))->DecoratedG4Tree:
    return DecoratedG4Tree(n,tuple(edges),tuple([klass]*n),tuple([op]*len(edges)))

def _public_projection(read:Mapping[str,Any])->dict[str,Any]:
    return {k:read.get(k) for k in ('legal','descriptor','caps7','H_class_bag')}

def verify_authority(graduation_decision:Mapping[str,Any], independent_verification:Mapping[str,Any], primary_s6:Mapping[str,Any])->dict[str,Any]:
    a=r0_spec()['authority']; fail=[]
    if graduation_decision.get('status')!='PASS' or graduation_decision.get('decision')!='G5_GRADUATED' or graduation_decision.get('g5_graduated') is not True: fail.append('G5_GRADUATION_DECISION')
    if graduation_decision.get('science_sha256')!=a['g5_graduation_decision_science_sha256']: fail.append('G5_GRADUATION_DECISION_HASH')
    if graduation_decision.get('g5_public_descriptor')!=a['graduated_descriptor']: fail.append('G5_DESCRIPTOR')
    law=graduation_decision.get('composition_law') or {}
    if law.get('scope')!=a['graduated_scope'] or law.get('write')!=a['graduated_write_law'] or law.get('enabled')!=a['graduated_enabled_rule']: fail.append('G5_LAW')
    if independent_verification.get('status')!='PASS' or independent_verification.get('verification_sha256')!=a['g5_s6_independent_verification_sha256']: fail.append('G5_S6_VERIFICATION')
    if primary_s6.get('status')!='PASS' or primary_s6.get('s6_stage_science_sha256')!=a['g5_s6_stage_science_sha256']: fail.append('G5_S6_PRIMARY')
    out={'schema_id':'IG_G5_R0_AUTHORITY_CHECK_V1','status':'PASS' if not fail else 'FAIL','failures':fail,'graduated_descriptor':graduation_decision.get('g5_public_descriptor'),'graduated_scope':law.get('scope'),'graduation_decision_science_sha256':graduation_decision.get('science_sha256'),'s6_verification_sha256':independent_verification.get('verification_sha256'),'s6_stage_science_sha256':primary_s6.get('s6_stage_science_sha256')}
    out['science_sha256']=canonical_sha256(out)
    if fail: raise G5R0Error('authority mismatch: '+','.join(fail))
    return out

def hierarchical_tree_theorem()->dict[str,Any]:
    out={
      'schema_id':'IG_G5_R0_HIDDEN_TREE_THEOREM_V1','status':'PASS',
      'proof_method':'STRUCTURAL_INDUCTION_ON_GRADUATED_G5_FINITE_BINARY_GRAMMAR',
      'base':'Each H-class atom is one exact hidden G3-unit vertex with no edge.',
      'step':'Every admitted G5 binary constructor joins two disjoint connected exact finite decorated trees by exactly one typed edge between an eligible endpoint owner on the left and one on the right.',
      'conclusion':'Every finite exact generated G5 carrier is a connected finite tree: for n hidden H-class/G3-unit vertices it has n-1 exact relation edges, beta=0, and every relation edge is a bridge.',
      'public_factorisation':'The graduated public state D=(CAPS7,H-class bag) and its write law remain exact for the frozen observer, but D does not encode the hidden tree incidence.',
      'scope':'ALL_FINITE_GENERATED_G5_TERMS_UNDER_FROZEN_31_OPERATOR_GRAMMAR',
      'nonclaims':['NO_TOPOLOGY_PROMOTION','NO_DESCRIPTOR_MINIMALITY','NO_PHYSICAL_SPACE','NO_DIMENSION','NO_METRIC','NO_TIME','NO_SPACETIME','NO_G6_LAUNCH']
    }
    out['science_sha256']=canonical_sha256(out); return out

def run_g5_r0_recon(*, engine:Any, graduation_decision:Mapping[str,Any], independent_verification:Mapping[str,Any], primary_s6:Mapping[str,Any])->dict[str,Any]:
    spec=r0_spec(); auth=verify_authority(graduation_decision,independent_verification,primary_s6); theorem=hierarchical_tree_theorem(); ad=G4AcceptedAdapter()
    if len(ad.operator_basis())!=31 or (0,0) not in set(ad.operator_basis()): raise G5R0Error('frozen operator basis mismatch')
    if ad.descriptor().get('public_descriptor')!='CAPS7_PLUS_H_CLASS_BAG': raise G5R0Error('adapter descriptor mismatch')
    # Reconfirm the two exact S1 topology witnesses that share a public state.
    c=_g5_s1_carriers(); witness_rows=[]
    for a,b in [('D2_PATH','D2_BROOM'),('D4_PATH','D4_BROOM')]:
        qa,qb=ad.public_read(c[a]),ad.public_read(c[b])
        witness_rows.append({'pair':[a,b],'public_equal':_public_projection(qa)==_public_projection(qb),'exact_hidden_canon_equal':ad.unrooted_canon(c[a])==ad.unrooted_canon(c[b]),'public':_public_projection(qa),'n_vertices':[c[a].n,c[b].n]})
    if not all(x['public_equal'] and not x['exact_hidden_canon_equal'] for x in witness_rows): raise G5R0Error('stored S1 topology separation witness failed')

    # Exact homogeneous tree-shape census. All vertices are H-class C and all edges use (0,0).
    shape_rows=unlabeled_tree_shape_census(int(spec['shape_census']['max_hidden_vertices']))
    observed=[int(r['unlabelled_tree_topology_count']) for r in shape_rows]
    if observed!=list(spec['shape_census']['expected_unlabelled_tree_counts_n1_to_nmax']): raise G5R0Error('unlabelled tree census mismatch')
    bounded=[]; first_multi=None
    for row in shape_rows:
        n=int(row['n']); pubs={}; exact=set(); metrics=[]
        for eraw in row['representatives']:
            e=tuple(tuple(map(int,x)) for x in eraw); t=_homogeneous_tree(n,e); q=ad.public_read(t)
            if not q.get('legal'): raise G5R0Error(f'homogeneous shape unexpectedly illegal n={n}')
            pubs[canonical_sha256(_public_projection(q))]=_public_projection(q); exact.add(repr(ad.unrooted_canon(t)))
            gm=_graph_metrics(n,e); metrics.append(gm)
            if not (gm['connected'] and gm['beta']==0 and gm['bridges']==len(e) and len(e)==max(0,n-1)): raise G5R0Error('tree metrics failed')
        if len(pubs)!=1 or len(exact)!=int(row['unlabelled_tree_topology_count']): raise G5R0Error(f'public/topology census mismatch n={n}')
        rec={'hidden_vertex_count':n,'unlabelled_tree_topology_count':int(row['unlabelled_tree_topology_count']),'exact_decorated_hidden_canon_count':len(exact),'unique_graduated_public_states':len(pubs),'shared_public_state':next(iter(pubs.values())),'all_shapes_legal':True,'all_shapes_are_trees':True}
        bounded.append(rec)
        if first_multi is None and rec['unlabelled_tree_topology_count']>1:
            reps=row['representatives'][:2]; ta=_homogeneous_tree(n,tuple(tuple(x) for x in reps[0])); tb=_homogeneous_tree(n,tuple(tuple(x) for x in reps[1])); first_multi={'hidden_vertex_count':n,'shared_public_state':rec['shared_public_state'],'tree_A_edges':reps[0],'tree_B_edges':reps[1],'tree_A_canon':repr(ad.unrooted_canon(ta)),'tree_B_canon':repr(ad.unrooted_canon(tb)),'nonisomorphic':ad.unrooted_canon(ta)!=ad.unrooted_canon(tb),'same_public':_public_projection(ad.public_read(ta))==_public_projection(ad.public_read(tb))}
    if first_multi is None or not first_multi['nonisomorphic'] or not first_multi['same_public']: raise G5R0Error('no topology-blindness witness')

    result={'schema_id':'IG_G5_R0_POST_GRADUATION_STRUCTURAL_RECON_RESULT_V1','status':'PASS','classification':spec['outcomes']['pass'],'promotion':False,'g5_graduated_before_r0':True,'g5_graduation_preserved':True,'g6_started':False,'authority':auth,'hierarchical_tree_theorem':theorem,'frozen_question_sha256':spec['science_sha256'],'s1_topology_separation_reconfirmation':witness_rows,'bounded_homogeneous_tree_shape_census':bounded,'first_same_public_nonisomorphic_tree_witness':first_multi,'intrinsic_structure_status':'GLOBAL_FINITE_HIDDEN_DECORATED_TREE_STRUCTURE_PRESENT','intrinsic_distance_status':'HIDDEN_G3_UNIT_GRAPH_DISTANCE_PRESENT','cycle_status':'NO_HIDDEN_INCIDENCE_CYCLES_UNDER_FROZEN_ONE_CROSS_BINARY_GRAMMAR','topology_visibility':'GRADUATED_CAPS7_PLUS_H_CLASS_BAG_DOES_NOT_DETERMINE_HIDDEN_TREE_TOPOLOGY','topology_promoted':False,'descriptor_changed':False,'dimension_status':'NOT_EARNED','geometry_status':'PHYSICAL_GEOMETRY_NOT_EARNED','next_recommendation':'AFTER_COLD_CERTIFICATION_REVIEW_G5_R0_AND_DESIGN_G5_R1; DO_NOT_PROMOTE_HIDDEN_TOPOLOGY_WITHOUT_A_NEW_REGISTERED_OBSERVER','nonclaims':spec['nonclaims']}
    result['science_sha256']=canonical_sha256(result); return result

def stable_payload(result:Mapping[str,Any])->dict[str,Any]:
    return {k:v for k,v in result.items() if k not in {'source_sha256','source_version','registry_sha256','execution_metadata'}}

def compare_cold_replay(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    ps=canonical_sha256(stable_payload(primary)); cs=canonical_sha256(stable_payload(cold)); fail=[]
    if primary.get('science_sha256')!=cold.get('science_sha256'): fail.append('SCIENCE_SHA')
    if primary.get('classification')!=cold.get('classification'): fail.append('CLASSIFICATION')
    if ps!=cs: fail.append('STABLE_PAYLOAD')
    out={'schema_id':'IG_G5_R0_COLD_REPLAY_COMPARISON_V1','status':'PASS' if not fail else 'FAIL','certification':'CERTIFIED_PASS' if not fail else 'CERTIFICATION_FAILED','failures':fail,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'stable_scientific_payload_exact_equal':ps==cs,'stable_scientific_payload_sha256':ps,'same_registered_decoder_native_experiment':True}
    out['comparison_sha256']=canonical_sha256(out); return out

def certified_closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],comparison:Mapping[str,Any],independent_shape_check:Mapping[str,Any])->dict[str,Any]:
    ok=primary.get('status')=='PASS' and cold.get('status')=='PASS' and comparison.get('certification')=='CERTIFIED_PASS' and independent_shape_check.get('status')=='PASS'
    out={'schema_id':'IG_G5_R0_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAILED','experiment_id':'G5:R0.STRUCTURAL_AUDIT','classification':primary.get('classification') if ok else 'G5_R0_REVIEW_REQUIRED_NO_PROMOTION','science_sha256':primary.get('science_sha256'),'stable_science_payload_sha256':comparison.get('stable_scientific_payload_sha256'),'comparison_sha256':comparison.get('comparison_sha256'),'independent_shape_check_sha256':independent_shape_check.get('science_sha256'),'source_sha256':primary.get('source_sha256'),'registry_sha256':primary.get('registry_sha256'),'promotion':False,'g5_graduation_preserved':True,'topology_promoted':False,'g6_started':False,'next_authorized_stage':'G5:R1_DESIGN_AFTER_HUMAN_REVIEW','failures':comparison.get('failures',[])}
    out['closeout_sha256']=canonical_sha256(out); return out
