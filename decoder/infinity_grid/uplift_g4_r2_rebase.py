from __future__ import annotations
"""Registered non-promoting G4:R2.REBASE bounded predictive hidden action-read audit."""
from importlib.resources import files
from collections import defaultdict, Counter, deque
from typing import Any, Mapping, Sequence
import json
from .canon import canonical_sha256
from .uplift_g3_r0 import _graph_metrics, _tree_canon
from .uplift_g3_r1 import _generate_tree_shapes

class G4R2RebaseError(RuntimeError): pass
_SPEC='resources/uplift/G4_R2_REBASE_PREDICTIVE_HIDDEN_READ_SPEC_V1.json'

def spec()->dict[str,Any]:
    o=json.loads(files('infinity_grid').joinpath(_SPEC).read_text())
    if o.get('schema_id')!='IG_G4_R2_REBASE_PREDICTIVE_HIDDEN_READ_SPEC_V1': raise G4R2RebaseError('bad spec schema')
    e=o.get('science_sha256'); p={k:v for k,v in o.items() if k!='science_sha256'}
    if canonical_sha256(p)!=e: raise G4R2RebaseError('spec hash mismatch')
    return o

def verify_authority(r1:Mapping[str,Any], replay:Mapping[str,Any], closeout:Mapping[str,Any])->dict[str,Any]:
    s=spec(); f=[]
    if r1.get('status')!='PASS' or r1.get('schema_id')!='IG_G4_R1_REBASE_FIBER_GRAFT_RESULT_V1': f.append('R1_RESULT')
    if str(r1.get('science_sha256'))!=s['authority']['g4_r1_primary_science_sha256']: f.append('R1_IDENTITY')
    if r1.get('classification')!='G4_R1_REBASE_FINITE_COMBINATORIAL_FIBER_AND_RELATION_VALUED_DECORATED_TREE_GRAFTING_LAW_EARNED_R2_REBASE_DESIGN_AUTHORIZED': f.append('R1_CLASSIFICATION')
    if r1.get('topology_promoted') is not False or r1.get('raw_h_promoted') is not False: f.append('R1_FIREWALL')
    if replay.get('status')!='PASS' or str(replay.get('science_sha256'))!=s['authority']['g4_r1_replay_science_sha256']: f.append('R1_REPLAY')
    if closeout.get('status')!='CERTIFIED_PASS' or str(closeout.get('science_sha256'))!=s['authority']['g4_r1_closeout_science_sha256']: f.append('R1_CLOSEOUT')
    o={'schema_id':'IG_G4_R2_REBASE_AUTHORITY_V1','status':'PASS' if not f else 'FAIL','failures':f,'r1_science_sha256':r1.get('science_sha256'),'r1_replay_science_sha256':replay.get('science_sha256'),'r1_closeout_science_sha256':closeout.get('science_sha256'),'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','promotion':False}
    o['science_sha256']=canonical_sha256(o)
    if f: raise G4R2RebaseError('authority failed: '+','.join(f))
    return o

def _dist(n:int,edges:Sequence[Sequence[int]])->list[list[int]]:
    adj=[[] for _ in range(n)]
    for a,b in edges: adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    D=[]
    for s in range(n):
        d=[-1]*n; d[s]=0; q=deque([s])
        while q:
            u=q.popleft()
            for w in adj[u]:
                if d[w]<0: d[w]=d[u]+1; q.append(w)
        D.append(d)
    return D

def _shell_rows(n:int,edges:Sequence[Sequence[int]])->list[tuple[int,...]]:
    D=_dist(n,edges); out=[]
    for v in range(n):
        md=max(D[v]) if D[v] else 0
        out.append(tuple(sum(1 for d in D[v] if d==k) for k in range(md+1)))
    return out

def action_signature(n:int,edges:Sequence[Sequence[int]],v:int)->tuple[Any,...]:
    rows=_shell_rows(n,edges); D=_dist(n,edges); groups:dict[tuple[int,...],Counter[int]]=defaultdict(Counter)
    for u,r in enumerate(rows): groups[r][D[v][u]]+=1
    return (rows[v],tuple((r,tuple(sorted(c.items()))) for r,c in sorted(groups.items())))

def state_read(n:int,edges:Sequence[Sequence[int]])->tuple[Any,...]:
    return tuple(sorted(action_signature(n,edges,v) for v in range(n)))

def _candidate_summary(n:int,edges:Sequence[Sequence[int]],name:str)->Any:
    m=_graph_metrics(n,edges); deg=tuple(int(x) for x in m['degree_sequence']); leaves=(1 if n==1 else sum(x==1 for x in deg))
    if name=='DEGREE_SEQUENCE': return deg
    if name=='SIX_SCALAR_TREE_SUMMARY': return (max(deg) if deg else 0,leaves,int(m['diameter']),int(m['radius']),int(m['articulations']),int(m['distance_sum']))
    if name=='SHELL_PROFILE_MULTISET': return tuple(sorted(_shell_rows(n,edges)))
    if name=='SHELL_CLASS_DISTANCE_HISTOGRAM_ACTION_BAG': return state_read(n,edges)
    raise G4R2RebaseError('unknown candidate '+name)

def _audit(max_n:int,candidates:Sequence[str])->dict[str,Any]:
    shapes=_generate_tree_shapes(max_n+1); rows=[]; ladder=[]; compression_seen=False
    # candidate state-separation census through registered parent scope
    for name in candidates:
        first=None; total_classes=0
        for n in range(1,max_n+1):
            g=defaultdict(list)
            for c,e in shapes[n].items(): g[_candidate_summary(n,e,name)].append(c)
            total_classes+=len(g); coll=[v for v in g.values() if len(v)>1]
            if first is None and coll: first={'g3_unit_count':n,'collision_group_count':len(coll),'first_group_size':len(coll[0]),'example_topology_canons':coll[0][:3]}
        ladder.append({'candidate':name,'first_state_collision':first,'total_state_classes_n1_to_nmax':total_classes,'state_separates_all_registered_shapes':first is None})
    selected=candidates[-1]
    for n in range(1,max_n+1):
        state_seen={}; total_vertices=0; action_classes=0; exact_child_count=0; predicted_child_count=0
        for c,e in sorted(shapes[n].items()):
            sr=state_read(n,e)
            if sr in state_seen and state_seen[sr]!=c: raise G4R2RebaseError(f'selected state read collision n={n}')
            state_seen[sr]=c
            by=defaultdict(list); exact=set()
            for v in range(n):
                sig=action_signature(n,e,v); child_edges=list(e)+[(v,n)]; cr=state_read(n+1,child_edges); by[sig].append(cr); exact.add(cr)
            for sig,vals in by.items():
                if len(set(vals))!=1: raise G4R2RebaseError(f'action read not predictive n={n}')
            pred={vals[0] for vals in by.values()}
            if pred!=exact: raise G4R2RebaseError(f'predicted child set mismatch n={n}')
            total_vertices+=n; action_classes+=len(by); exact_child_count+=len(exact); predicted_child_count+=len(pred)
        if action_classes<total_vertices: compression_seen=True
        rows.append({'g3_unit_count':n,'parent_topology_count':len(shapes[n]),'selected_state_class_count':len(state_seen),'attachment_vertex_count':total_vertices,'action_read_class_count':action_classes,'action_compression_fraction':(action_classes/total_vertices if total_vertices else 1.0),'summed_exact_distinct_child_read_count':exact_child_count,'summed_predicted_distinct_child_read_count':predicted_child_count,'prediction_exact':True})
    if not compression_seen: raise G4R2RebaseError('no strict action compression')
    o={'schema_id':'IG_G4_R2_REBASE_BOUNDED_PREDICTIVE_ACTION_READ_AUDIT_V1','status':'PASS','max_parent_g3_units':max_n,'candidate_ladder':ladder,'selected_action_read':selected,'rows':rows,'strict_action_compression_observed':compression_seen,'interpretation':'Exact bounded homogeneous-slice result only. The selected read is not claimed complete at all ranks or for arbitrary H-colour/edge-grade decoration.'}
    o['science_sha256']=canonical_sha256(o); return o

def finalize(*,r1:Mapping[str,Any],r1_replay:Mapping[str,Any],r1_closeout:Mapping[str,Any])->dict[str,Any]:
    s=spec(); auth=verify_authority(r1,r1_replay,r1_closeout); aud=_audit(int(s['bounded_scope']['max_g3_units']),s['candidate_ladder'])
    out={'schema_id':'IG_G4_R2_REBASE_PREDICTIVE_HIDDEN_READ_RESULT_V1','status':'PASS','stage_ref':'G4:R2.REBASE','classification':s['pass_classification'],'authority':auth,'bounded_scope':s['bounded_scope'],'candidate_ladder':aud['candidate_ladder'],'selected_read_definition':s['selected_action_read'],'predictive_audit':aud,'full_tree_canon_required_on_registered_scope':False,'minimality_claim':False,'all_ranks_claim':False,'promotion':False,'topology_promoted':False,'public_descriptor_changed':False,'raw_h_promoted':False,'g4_graduation_preserved':True,'g5_started':False,'next_authorized_stage':'G4:R3.REBASE_DESIGN','cost_control':s['cost_control'],'nonclaims':s['nonclaims']}
    out['science_sha256']=canonical_sha256(out); return out

def compare(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    checks={'primary_pass':primary.get('status')=='PASS','cold_pass':cold.get('status')=='PASS','classification_equal':primary.get('classification')==cold.get('classification'),'science_sha_equal':primary.get('science_sha256')==cold.get('science_sha256'),'predictive_audit_equal':primary.get('predictive_audit')==cold.get('predictive_audit')}
    ok=all(checks.values()); o={'schema_id':'IG_G4_R2_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256')}; o['science_sha256']=canonical_sha256(o); return o

def closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
    ok=replay.get('status')=='PASS'; o={'schema_id':'IG_G4_R2_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':primary.get('classification') if ok else None,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'replay_science_sha256':replay.get('science_sha256'),'g4_graduation_preserved':ok,'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG' if ok else None,'topology_promoted':False,'public_descriptor_changed':False,'g5_started':False,'next_authorized_stage':'G4:R3.REBASE_DESIGN' if ok else None}; o['science_sha256']=canonical_sha256(o); return o
