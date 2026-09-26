from __future__ import annotations
"""G4:R3.REBASE: exact scope-boundary challenge of the R2 topology-only predictive read."""
from importlib.resources import files
from collections import defaultdict
from itertools import product, permutations
from typing import Any, Mapping, Sequence
import json
from .canon import canonical_sha256
from .uplift_g3_r1 import _generate_tree_shapes
from .uplift_g4_r2_rebase import action_signature as r2_action_signature, state_read as r2_state_read

class G4R3RebaseError(RuntimeError): pass
_SPEC='resources/uplift/G4_R3_REBASE_DECORATED_SCOPE_CHALLENGE_SPEC_V1.json'

def spec()->dict[str,Any]:
    o=json.loads(files('infinity_grid').joinpath(_SPEC).read_text())
    if o.get('schema_id')!='IG_G4_R3_REBASE_DECORATED_SCOPE_CHALLENGE_SPEC_V1': raise G4R3RebaseError('bad spec schema')
    e=o.get('science_sha256'); p={k:v for k,v in o.items() if k!='science_sha256'}
    if canonical_sha256(p)!=e: raise G4R3RebaseError('spec hash mismatch')
    return o

def verify_authority(r2:Mapping[str,Any], replay:Mapping[str,Any], closeout:Mapping[str,Any])->dict[str,Any]:
    s=spec(); f=[]
    if r2.get('schema_id')!='IG_G4_R2_REBASE_PREDICTIVE_HIDDEN_READ_RESULT_V1' or r2.get('status')!='PASS': f.append('R2_RESULT')
    if str(r2.get('science_sha256'))!=s['authority']['g4_r2_primary_science_sha256']: f.append('R2_IDENTITY')
    if r2.get('selected_read_definition',{}).get('name')!=s['authority']['required_r2_read']: f.append('R2_READ')
    if r2.get('all_ranks_claim') is not False or r2.get('topology_promoted') is not False: f.append('R2_FIREWALL')
    if replay.get('status')!='PASS' or str(replay.get('science_sha256'))!=s['authority']['g4_r2_replay_science_sha256']: f.append('R2_REPLAY')
    if closeout.get('status')!='CERTIFIED_PASS' or str(closeout.get('science_sha256'))!=s['authority']['g4_r2_closeout_science_sha256']: f.append('R2_CLOSEOUT')
    o={'schema_id':'IG_G4_R3_REBASE_AUTHORITY_V1','status':'PASS' if not f else 'FAIL','failures':f,'r2_science_sha256':r2.get('science_sha256'),'r2_replay_science_sha256':replay.get('science_sha256'),'r2_closeout_science_sha256':closeout.get('science_sha256'),'r2_scope':r2.get('bounded_scope'),'promotion':False}
    o['science_sha256']=canonical_sha256(o)
    if f: raise G4R3RebaseError('authority failed: '+','.join(f))
    return o

def decorated_canon(n:int, edges:Sequence[Sequence[int]], colors:Sequence[int], grades:Sequence[int])->tuple[Any,...]:
    if len(colors)!=n or len(grades)!=len(edges): raise G4R3RebaseError('decoration arity')
    best=None
    for perm in permutations(range(n)):
        # perm old -> new
        c=[None]*n
        for old,new in enumerate(perm): c[new]=int(colors[old])
        ee=[]
        for (a,b),g in zip(edges,grades):
            x,y=perm[int(a)],perm[int(b)]
            if x>y: x,y=y,x
            ee.append((x,y,int(g)))
        cand=(tuple(c),tuple(sorted(ee)))
        if best is None or cand<best: best=cand
    return best if best is not None else (tuple(),tuple())

def child_canon(n:int,edges:Sequence[Sequence[int]],colors:Sequence[int],grades:Sequence[int],v:int,new_color:int,new_grade:int)->tuple[Any,...]:
    return decorated_canon(n+1,list(edges)+[(int(v),n)],list(colors)+[int(new_color)],list(grades)+[int(new_grade)])

def _minimal_witnesses()->dict[str,Any]:
    # H-colour witness: same one-vertex topology/R2 action read, different inherited H colour.
    e1=[]; sig1=r2_action_signature(1,e1,0)
    h0=child_canon(1,e1,[0],[],0,0,0); h1=child_canon(1,e1,[1],[],0,0,0)
    if h0==h1: raise G4R3RebaseError('H witness collapsed')
    # Edge-grade witness: same two-vertex topology and H colours, but existing hidden edge grade differs.
    e2=[(0,1)]; sig2=r2_action_signature(2,e2,0)
    g0=child_canon(2,e2,[0,0],[0],0,0,0); g1=child_canon(2,e2,[0,0],[1],0,0,0)
    if g0==g1: raise G4R3RebaseError('grade witness collapsed')
    return {'schema_id':'IG_G4_R3_REBASE_MINIMAL_COUNTEREXAMPLES_V1','status':'PASS',
      'h_color':{'g3_unit_count':1,'same_r2_action_signature':True,'r2_action_signature':sig1,'parent_h_colors':[[0],[1]],'fixed_new_leaf_h_class':0,'fixed_new_edge_grade':0,'distinct_exact_decorated_child_canons':2},
      'edge_grade':{'g3_unit_count':2,'same_r2_action_signature':True,'r2_action_signature':sig2,'parent_h_colors':[0,0],'parent_edge_grades':[[0],[1]],'fixed_new_leaf_h_class':0,'fixed_new_edge_grade':0,'distinct_exact_decorated_child_canons':2}}

def _sweep(max_n:int)->dict[str,Any]:
    shapes=_generate_tree_shapes(max_n); rows=[]; any_nonpredictive=False
    for n in range(1,max_n+1):
        parent_states=set(); r2_state_classes=set(); action_groups=defaultdict(set); decorated_instances=0
        for _,edges in sorted(shapes[n].items()):
            for colors in product(range(2),repeat=n):
                for grades in product(range(2),repeat=max(0,n-1)):
                    pc=decorated_canon(n,edges,colors,grades)
                    if pc in parent_states: continue
                    parent_states.add(pc); decorated_instances+=1
                    r2_state_classes.add(r2_state_read(n,edges))
                    for v in range(n):
                        key=(r2_action_signature(n,edges,v),0,0)
                        action_groups[key].add(child_canon(n,edges,colors,grades,v,0,0))
        bad=[len(x) for x in action_groups.values() if len(x)>1]
        if bad: any_nonpredictive=True
        rows.append({'g3_unit_count':n,'underlying_topology_count':len(shapes[n]),'exact_decorated_parent_state_count':len(parent_states),'r2_topology_state_read_class_count':len(r2_state_classes),'r2_action_class_count':len(action_groups),'nonpredictive_r2_action_class_count':len(bad),'max_distinct_exact_children_within_one_r2_action_class':max(bad) if bad else 1})
    if not any_nonpredictive: raise G4R3RebaseError('decorated sweep found no R2 nonpredictivity')
    o={'schema_id':'IG_G4_R3_REBASE_DECORATED_SCOPE_SWEEP_V1','status':'PASS','max_g3_units':max_n,'palette':{'h_classes':2,'edge_grades':2},'rows':rows,'r2_read_predictive_on_decorated_scope':False,'interpretation':'This falsifies only extension of the R2 topology-only read beyond its certified homogeneous untyped scope; it does not retract R2 inside that scope.'}
    o['science_sha256']=canonical_sha256(o); return o

def finalize(*,r2:Mapping[str,Any],r2_replay:Mapping[str,Any],r2_closeout:Mapping[str,Any])->dict[str,Any]:
    s=spec(); auth=verify_authority(r2,r2_replay,r2_closeout); wit=_minimal_witnesses(); sw=_sweep(int(s['challenge_scope']['max_g3_units']))
    out={'schema_id':'IG_G4_R3_REBASE_DECORATED_SCOPE_CHALLENGE_RESULT_V1','status':'PASS','stage_ref':'G4:R3.REBASE','classification':s['pass_classification'],'authority':auth,'challenge_scope':s['challenge_scope'],'minimal_counterexamples':wit,'bounded_decorated_sweep':sw,'r2_certified_scope_preserved':True,'r2_generalization_to_full_r1_hidden_carrier':False,'decoration_aware_compact_read_earned':False,'promotion':False,'topology_promoted':False,'public_descriptor_changed':False,'raw_h_promoted':False,'g4_graduation_preserved':True,'g5_started':False,'next_authorized_stage':'G4:R4.REBASE_DESIGN','cost_control':s['cost_control'],'nonclaims':s['nonclaims']}
    out['science_sha256']=canonical_sha256(out); return out

def compare(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    checks={'primary_pass':primary.get('status')=='PASS','cold_pass':cold.get('status')=='PASS','classification_equal':primary.get('classification')==cold.get('classification'),'science_sha_equal':primary.get('science_sha256')==cold.get('science_sha256'),'witnesses_equal':primary.get('minimal_counterexamples')==cold.get('minimal_counterexamples'),'sweep_equal':primary.get('bounded_decorated_sweep')==cold.get('bounded_decorated_sweep')}
    ok=all(checks.values()); o={'schema_id':'IG_G4_R3_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256')}; o['science_sha256']=canonical_sha256(o); return o

def closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
    ok=replay.get('status')=='PASS'; o={'schema_id':'IG_G4_R3_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':primary.get('classification') if ok else None,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'replay_science_sha256':replay.get('science_sha256'),'g4_graduation_preserved':ok,'r2_certified_scope_preserved':ok,'r2_generalization_to_full_r1_hidden_carrier':False if ok else None,'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG' if ok else None,'topology_promoted':False,'public_descriptor_changed':False,'g5_started':False,'next_authorized_stage':'G4:R4.REBASE_DESIGN' if ok else None}; o['science_sha256']=canonical_sha256(o); return o
