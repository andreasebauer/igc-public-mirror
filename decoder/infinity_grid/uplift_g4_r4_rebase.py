from __future__ import annotations
"""G4:R4.REBASE: bounded decoration-aware predictive action-read refinement."""
from importlib.resources import files
from collections import defaultdict
from itertools import product
from typing import Any,Mapping,Sequence
import json
from .canon import canonical_sha256
from .uplift_g3_r1 import _generate_tree_shapes
from .uplift_g4_r2_rebase import action_signature as r2_action_signature
class G4R4RebaseError(RuntimeError): pass
_SPEC='resources/uplift/G4_R4_REBASE_DECORATION_AWARE_PREDICTIVE_READ_SPEC_V1.json'
def spec()->dict[str,Any]:
 o=json.loads(files('infinity_grid').joinpath(_SPEC).read_text()); e=o.get('science_sha256'); p={k:v for k,v in o.items() if k!='science_sha256'}
 if o.get('schema_id')!='IG_G4_R4_REBASE_DECORATION_AWARE_PREDICTIVE_READ_SPEC_V1' or canonical_sha256(p)!=e: raise G4R4RebaseError('bad spec')
 return o
def verify_authority(r3:Mapping[str,Any],replay:Mapping[str,Any],closeout:Mapping[str,Any])->dict[str,Any]:
 s=spec(); f=[]
 if r3.get('schema_id')!='IG_G4_R3_REBASE_DECORATED_SCOPE_CHALLENGE_RESULT_V1' or r3.get('status')!='PASS': f.append('R3_RESULT')
 if r3.get('science_sha256')!=s['authority']['g4_r3_primary_science_sha256']: f.append('R3_IDENTITY')
 if r3.get('r2_generalization_to_full_r1_hidden_carrier') is not False: f.append('R3_BOUNDARY')
 if replay.get('status')!='PASS' or replay.get('science_sha256')!=s['authority']['g4_r3_replay_science_sha256']: f.append('R3_REPLAY')
 if closeout.get('status')!='CERTIFIED_PASS' or closeout.get('science_sha256')!=s['authority']['g4_r3_closeout_science_sha256']: f.append('R3_CLOSEOUT')
 o={'schema_id':'IG_G4_R4_REBASE_AUTHORITY_V1','status':'PASS' if not f else 'FAIL','failures':f,'r3_science_sha256':r3.get('science_sha256'),'promotion':False}; o['science_sha256']=canonical_sha256(o)
 if f: raise G4R4RebaseError('authority failed: '+','.join(f))
 return o
def _adj(n:int,edges:Sequence[Sequence[int]],grades:Sequence[int]):
 A=[[] for _ in range(n)]
 for (a,b),g in zip(edges,grades): a=int(a);b=int(b);g=int(g);A[a].append((b,g));A[b].append((a,g))
 return A
def _messages(n:int,edges:Sequence[Sequence[int]],colors:Sequence[int],grades:Sequence[int],depth:int):
 A=_adj(n,edges,grades); m=[(int(colors[v]),) for v in range(n)]
 for _ in range(depth): m=[(int(colors[v]),tuple(sorted((g,m[u]) for u,g in A[v]))) for v in range(n)]
 return m
def action_signature(n:int,edges:Sequence[Sequence[int]],colors:Sequence[int],grades:Sequence[int],v:int,depth:int):
 if depth==0: return (r2_action_signature(n,edges,v),int(colors[v]))
 return (r2_action_signature(n,edges,v),_messages(n,edges,colors,grades,depth)[v])
def state_read(n:int,edges:Sequence[Sequence[int]],colors:Sequence[int],grades:Sequence[int],depth:int):
 return tuple(sorted(action_signature(n,edges,colors,grades,v,depth) for v in range(n)))
def _audit(max_n:int)->dict[str,Any]:
 shapes=_generate_tree_shapes(max_n+1); ladder=[]; selected_rows=[]; compression=False
 for depth in range(0,5):
  first=None; rows=[]
  for n in range(1,max_n+1):
   groups=defaultdict(set); total=0
   for _,edges in sorted(shapes[n].items()):
    for colors in product(range(2),repeat=n):
     for grades in product(range(2),repeat=max(0,n-1)):
      msgs=_messages(n,edges,colors,grades,depth) if depth>0 else None
      for v in range(n):
       sig=(r2_action_signature(n,edges,v),msgs[v]) if depth>0 else (r2_action_signature(n,edges,v),int(colors[v]))
       ce=list(edges)+[(v,n)]; cc=list(colors)+[0]; cg=list(grades)+[0]
       groups[sig].add(state_read(n+1,ce,cc,cg,depth)); total+=1
   bad=[len(x) for x in groups.values() if len(x)>1]
   rr={'g3_unit_count':n,'attachment_instances':total,'action_read_class_count':len(groups),'nonpredictive_action_class_count':len(bad),'max_distinct_child_reads_in_one_action_class':max(bad) if bad else 1,'prediction_exact':not bad}
   rows.append(rr)
   if depth==4 and len(groups)<total: compression=True
   if bad and first is None: first={'g3_unit_count':n,'nonpredictive_action_class_count':len(bad),'max_distinct_child_reads_in_one_action_class':max(bad)}
   if bad: break
  name='R2_PLUS_ROOT_H_CLASS' if depth==0 else f'R2_PLUS_DECORATED_MESSAGE_DEPTH_{depth}'
  ladder.append({'candidate':name,'first_failure':first,'predictive_through_n5':first is None,'rows':rows})
  if depth==4: selected_rows=rows
 if ladder[-1]['first_failure'] is not None: raise G4R4RebaseError('selected depth4 read failed')
 if not any(x['first_failure'] is not None for x in ladder[:-1]): raise G4R4RebaseError('no earlier failure')
 if not compression: raise G4R4RebaseError('no action compression')
 o={'schema_id':'IG_G4_R4_REBASE_DECORATION_AWARE_PREDICTIVE_AUDIT_V1','status':'PASS','candidate_ladder':ladder,'selected_read':'R2_PLUS_DECORATED_MESSAGE_DEPTH_4_ACTION_BAG','selected_rows':selected_rows,'strict_action_compression_observed':compression,'interpretation':'Bounded exact result on exhaustive binary H-colour/edge-grade decorated trees through five G3 units. Depth is message-refinement depth, not a physical distance or geometry claim.'};o['science_sha256']=canonical_sha256(o);return o
def finalize(*,r3:Mapping[str,Any],r3_replay:Mapping[str,Any],r3_closeout:Mapping[str,Any])->dict[str,Any]:
 s=spec();auth=verify_authority(r3,r3_replay,r3_closeout);aud=_audit(int(s['bounded_scope']['max_parent_g3_units']))
 out={'schema_id':'IG_G4_R4_REBASE_DECORATION_AWARE_PREDICTIVE_READ_RESULT_V1','status':'PASS','stage_ref':'G4:R4.REBASE','classification':s['pass_classification'],'authority':auth,'bounded_scope':s['bounded_scope'],'predictive_audit':aud,'selected_read_definition':{'name':s['selected_read_name'],'vertex_action_signature':'R2 topology action signature plus four rounds of rooted decoration messages. Round 0 is the vertex H-class; each next round records the H-class and multiset of (incident edge grade, previous-round neighbor message).','state_read':'Multiset of all vertex action signatures.','prediction_rule':'Equal action signatures must yield equal child state reads after adjoining a fixed H-class-0, grade-0 leaf.'},'full_decorated_tree_canon_required_on_registered_scope':False,'minimality_within_frozen_ladder':True,'all_ranks_claim':False,'promotion':False,'topology_promoted':False,'public_descriptor_changed':False,'raw_h_promoted':False,'g4_graduation_preserved':True,'g5_started':False,'next_authorized_stage':'G4:R5.REBASE_DESIGN','cost_control':s['cost_control'],'nonclaims':s['nonclaims']};out['science_sha256']=canonical_sha256(out);return out
def compare(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
 checks={'primary_pass':primary.get('status')=='PASS','cold_pass':cold.get('status')=='PASS','classification_equal':primary.get('classification')==cold.get('classification'),'science_sha_equal':primary.get('science_sha256')==cold.get('science_sha256'),'audit_equal':primary.get('predictive_audit')==cold.get('predictive_audit')};ok=all(checks.values());o={'schema_id':'IG_G4_R4_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256')};o['science_sha256']=canonical_sha256(o);return o
def closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
 ok=replay.get('status')=='PASS';o={'schema_id':'IG_G4_R4_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':primary.get('classification') if ok else None,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'replay_science_sha256':replay.get('science_sha256'),'selected_read':primary.get('selected_read_definition',{}).get('name') if ok else None,'g4_graduation_preserved':ok,'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG' if ok else None,'topology_promoted':False,'public_descriptor_changed':False,'g5_started':False,'next_authorized_stage':'G4:R5.REBASE_DESIGN' if ok else None};o['science_sha256']=canonical_sha256(o);return o
