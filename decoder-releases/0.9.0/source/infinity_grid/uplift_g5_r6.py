from __future__ import annotations

"""G5:R6 single-payload marker-free hidden separation audit.

R5 established bounded injectivity for the complete 62-payload one-step action
signature. R6 compresses the action observer to one ordinary certified payload,
C/(0,0), and challenges it on fresh n=9 and multi-endpoint-typed n=5 panels.
The action label is deliberately non-fresh on the admitted parents.
"""
from collections import defaultdict
from importlib.resources import files
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .uplift_g3_r1 import _generate_tree_shapes
from .uplift_g5_r5 import _q_key, _relation_child_canons

class G5R6Error(RuntimeError): pass
_SPEC='G5_R6_SINGLE_PAYLOAD_MARKER_FREE_SEPARATION_SPEC_V1.json'

def _resource(name:str)->Path: return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def r6_spec()->dict[str,Any]:
    obj=json.loads(_resource(_SPEC).read_text(encoding='utf-8'))
    if obj.get('schema_id')!='IG_G5_R6_SINGLE_PAYLOAD_MARKER_FREE_SEPARATION_SPEC_V1': raise G5R6Error('bad G5:R6 spec schema')
    payload={k:v for k,v in obj.items() if k!='science_sha256'}
    if canonical_sha256(payload)!=obj.get('science_sha256'): raise G5R6Error('G5:R6 spec hash mismatch')
    return obj

def verify_authority(r5_primary:Mapping[str,Any],r5_verification:Mapping[str,Any],r5_closeout:Mapping[str,Any])->dict[str,Any]:
    s=r6_spec();a=s['authority'];f=[]
    if r5_primary.get('schema_id')!='IG_G5_R5_MARKER_FREE_ONE_STEP_SEPARATION_RESULT_V1' or r5_primary.get('status')!='PASS':f.append('R5_PRIMARY_SCHEMA_OR_STATUS')
    if str(r5_primary.get('science_sha256'))!=a['g5_r5_primary_science_sha256']:f.append('R5_PRIMARY_IDENTITY')
    if r5_primary.get('marker_free_one_step_separation_earned_on_frozen_scope') is not True:f.append('R5_SEPARATION_AUTHORITY')
    if r5_primary.get('promotion') is not False or r5_primary.get('g5_graduation_preserved') is not True:f.append('R5_FIREWALL')
    if r5_verification.get('status')!='PASS' or str(r5_verification.get('verification_sha256'))!=a['g5_r5_independent_verification_sha256']:f.append('R5_VERIFICATION')
    if r5_closeout.get('status')!='CERTIFIED_PASS' or str(r5_closeout.get('closeout_sha256'))!=a['g5_r5_closeout_sha256']:f.append('R5_CLOSEOUT')
    out={'schema_id':'IG_G5_R6_AUTHORITY_CHECK_V1','status':'PASS' if not f else 'FAIL','failures':f,'r5_primary_science_sha256':r5_primary.get('science_sha256'),'r5_verification_sha256':r5_verification.get('verification_sha256'),'r5_closeout_sha256':r5_closeout.get('closeout_sha256')};out['science_sha256']=canonical_sha256(out)
    if f:raise G5R6Error('R5 authority mismatch: '+','.join(f))
    return out

def _tree_record(t:DecoratedG4Tree)->dict[str,Any]:
    return {'n':int(t.n),'edges':[list(map(int,e)) for e in t.edges],'H_classes':list(t.H_classes),'edge_operators':[list(map(int,o)) for o in t.edge_operators]}

def _single_relation(ad:G4AcceptedAdapter,t:DecoratedG4Tree)->tuple[tuple[Any,...],...]:
    return _relation_child_canons(ad,t,'C',(0,0))

def _audit(exact:Mapping[tuple[Any,...],DecoratedG4Tree],*,panel_id:str,raw:int,illegal:int,domain_meta:Mapping[str,Any])->dict[str,Any]:
    ad=G4AcceptedAdapter();byq=defaultdict(list);fail=[]
    for p,t in exact.items():
        pub=ad.public_read(t)
        if not pub.get('legal'): fail.append('ILLEGAL_PARENT_AFTER_FILTER');continue
        byq[_q_key(pub)].append((p,t))
    qrows=[];first=None;empty=0;maxkids=0
    for q in sorted(byq):
        owners={};coll=0
        for p,t in sorted(byq[q],key=lambda x:x[0]):
            rel=_single_relation(ad,t);empty+=int(not rel);maxkids=max(maxkids,len(rel))
            old=owners.get(rel)
            if old is None: owners[rel]=(p,t)
            elif old[0]!=p:
                coll+=1
                if first is None:
                    first={'Q':{'caps7':list(q[0]),'H_class_bag':[list(x) for x in q[1]]},'parent_a':_tree_record(old[1]),'parent_b':_tree_record(t),'parent_a_exact_canon':old[0],'parent_b_exact_canon':p,'common_single_payload_child_relation':rel}
        qrows.append({'Q_caps7':list(q[0]),'Q_H_class_bag':[list(x) for x in q[1]],'exact_parent_count':len(byq[q]),'distinct_single_payload_relation_count':len(owners),'collision_count':coll})
    out={'schema_id':'IG_G5_R6_SINGLE_PAYLOAD_PANEL_RESULT_V1','panel_id':panel_id,'status':'PASS' if not fail else 'FAIL','raw_parent_count':int(raw),'rejected_illegal_parent_count':int(illegal),'exact_parent_canon_count':len(exact),'public_Q_fiber_count':len(byq),'single_payload':{'new_H_class':'C','operator':[0,0]},'q_rows':qrows,'evaluated_exact_parent_relations':len(exact),'empty_child_relation_count':empty,'max_exact_child_canons_in_relation':maxkids,'within_Q_single_payload_relation_injective':first is None,'collision_witness':first,'structural_equality_used':True,'digest_only_equality_used':False,'domain_meta':dict(domain_meta),'failures':fail};out['science_sha256']=canonical_sha256(out)
    if fail:raise G5R6Error(panel_id+' failed: '+','.join(fail))
    return out

def fresh_n9_panel()->dict[str,Any]:
    ad=G4AcceptedAdapter();sh=_generate_tree_shapes(9)[9];exact={};raw=illegal=0
    for edges in sh.values():
        for colors in product(('C','D'),repeat=9):
            if 'C' not in colors or 'D' not in colors: continue
            raw+=1;t=DecoratedG4Tree(9,tuple(edges),tuple(colors),tuple((0,0) for _ in range(8)));pub=ad.public_read(t)
            if not pub.get('legal'):illegal+=1;continue
            exact.setdefault(ad.unrooted_canon(t),t)
    return _audit(exact,panel_id='FRESH_N9_MIXED_CD_FIXED_00_EDGES_SINGLE_C00_ACTION',raw=raw,illegal=illegal,domain_meta={'unlabelled_shape_count':len(sh),'mixed_colorings_per_shape':510,'action_label_nonfresh_by_construction':True})

def endpoint_typed_n5_panel()->dict[str,Any]:
    ad=G4AcceptedAdapter();sh=_generate_tree_shapes(5)[5];ops=((0,0),(0,1),(1,0),(2,4),(4,2));exact={};raw=illegal=0;assign=0
    for edge_ops in product(ops,repeat=4):
        if (0,0) not in edge_ops or all(x==(0,0) for x in edge_ops): continue
        assign+=1
        for edges in sh.values():
            for colors in product(('C','D'),repeat=5):
                if 'C' not in colors or 'D' not in colors:continue
                raw+=1;t=DecoratedG4Tree(5,tuple(edges),tuple(colors),tuple(edge_ops));pub=ad.public_read(t)
                if not pub.get('legal'):illegal+=1;continue
                exact.setdefault(ad.unrooted_canon(t),t)
    return _audit(exact,panel_id='FRESH_N5_MIXED_CD_FIVE_OPERATOR_PARENT_PANEL_SINGLE_C00_ACTION',raw=raw,illegal=illegal,domain_meta={'unlabelled_shape_count':len(sh),'mixed_colorings_per_shape':30,'parent_operator_subset':[list(x) for x in ops],'admitted_edge_assignment_count':assign,'action_label_nonfresh_by_construction':True})

def run_g5_r6_single_payload_separation(*,engine:Any,r5_primary:Mapping[str,Any],r5_verification:Mapping[str,Any],r5_closeout:Mapping[str,Any])->dict[str,Any]:
    s=r6_spec();auth=verify_authority(r5_primary,r5_verification,r5_closeout);p=fresh_n9_panel();e=endpoint_typed_n5_panel();inj=bool(p['within_Q_single_payload_relation_injective'] and e['within_Q_single_payload_relation_injective']);w=p.get('collision_witness') or e.get('collision_witness');classification=s['pass_classifications']['injective' if inj else 'collision']
    out={'schema_id':'IG_G5_R6_SINGLE_PAYLOAD_MARKER_FREE_SEPARATION_RESULT_V1','status':'PASS','stage_ref':'G5:R6','classification':classification,'promotion':False,'g5_graduation_preserved':True,'g6_started':False,'authority':auth,'fresh_n9_panel':p,'endpoint_typed_n5_panel':e,'single_payload_marker_free_separation_earned_on_frozen_scope':inj,'single_payload_collision_earned_on_frozen_scope':not inj,'first_collision_witness':w,'global_single_payload_minimality_claim':False,'all_finite_carrier_single_payload_separation_claim':False,'exact_hidden_canon_promoted_to_public':False,'topology_promoted':False,'public_descriptor_changed':False,'design_history':s['design_history'],'next_authorized_stage':s['next_authorized_stage'],'cost_control':s['cost_control'],'nonclaims':s['nonclaims']};out['science_sha256']=canonical_sha256(out);return out

def stable_payload(r:Mapping[str,Any])->dict[str,Any]:return {k:v for k,v in r.items() if k not in {'source_sha256','source_version','registry_sha256','execution_metadata'}}

def compare_cold_replay(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    ps=canonical_sha256(stable_payload(primary));cs=canonical_sha256(stable_payload(cold));f=[]
    if primary.get('science_sha256')!=cold.get('science_sha256'):f.append('SCIENCE_SHA')
    if primary.get('classification')!=cold.get('classification'):f.append('CLASSIFICATION')
    if ps!=cs:f.append('STABLE_PAYLOAD')
    out={'schema_id':'IG_G5_R6_COLD_REPLAY_COMPARISON_V1','status':'PASS' if not f else 'FAIL','certification':'CERTIFIED_PASS' if not f else 'CERTIFICATION_FAILED','failures':f,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'stable_scientific_payload_exact_equal':ps==cs,'stable_scientific_payload_sha256':ps};out['comparison_sha256']=canonical_sha256(out);return out

def certified_closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],comparison:Mapping[str,Any],independent:Mapping[str,Any])->dict[str,Any]:
    ok=primary.get('status')=='PASS' and cold.get('status')=='PASS' and comparison.get('certification')=='CERTIFIED_PASS' and independent.get('status')=='PASS'
    out={'schema_id':'IG_G5_R6_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAILED','experiment_id':'G5:R6.SINGLE_PAYLOAD_MARKER_FREE_SEPARATION','classification':primary.get('classification') if ok else 'G5_R6_REVIEW_REQUIRED_NO_PROMOTION','science_sha256':primary.get('science_sha256'),'comparison_sha256':comparison.get('comparison_sha256'),'independent_verification_sha256':independent.get('verification_sha256'),'source_sha256':primary.get('source_sha256'),'registry_sha256':primary.get('registry_sha256'),'promotion':False,'g5_graduation_preserved':True,'single_payload_marker_free_separation_earned_on_frozen_scope':bool(primary.get('single_payload_marker_free_separation_earned_on_frozen_scope')) if ok else False,'single_payload_collision_earned_on_frozen_scope':bool(primary.get('single_payload_collision_earned_on_frozen_scope')) if ok else False,'global_single_payload_minimality_claim':False,'topology_promoted':False,'g6_started':False,'next_authorized_stage':'G5:R7_DESIGN_AFTER_HUMAN_REVIEW' if ok else None,'failures':comparison.get('failures',[])};out['closeout_sha256']=canonical_sha256(out);return out
