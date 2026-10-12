from __future__ import annotations
"""G4:R5.REBASE: targeted n=6 stabilization challenge for the R4 depth-4 decoration read."""
from importlib.resources import files
from collections import defaultdict
from itertools import product
from typing import Any,Mapping,Sequence
import json
from .canon import canonical_sha256
from .uplift_g4_r2_rebase import action_signature as r2_action_signature
class G4R5RebaseError(RuntimeError): pass
_SPEC='resources/uplift/G4_R5_REBASE_DEPTH_STABILIZATION_CHALLENGE_SPEC_V1.json'
def spec()->dict[str,Any]:
 o=json.loads(files('infinity_grid').joinpath(_SPEC).read_text()); e=o.get('science_sha256'); p={k:v for k,v in o.items() if k!='science_sha256'}
 if o.get('schema_id')!='IG_G4_R5_REBASE_DEPTH_STABILIZATION_CHALLENGE_SPEC_V1' or canonical_sha256(p)!=e: raise G4R5RebaseError('bad spec')
 return o
def verify_authority(r4:Mapping[str,Any],replay:Mapping[str,Any],closeout:Mapping[str,Any])->dict[str,Any]:
 s=spec(); f=[]
 if r4.get('schema_id')!='IG_G4_R4_REBASE_DECORATION_AWARE_PREDICTIVE_READ_RESULT_V1' or r4.get('status')!='PASS': f.append('R4_RESULT')
 if r4.get('science_sha256')!=s['authority']['g4_r4_primary_science_sha256']: f.append('R4_IDENTITY')
 if r4.get('selected_read_definition',{}).get('name')!='R2_PLUS_DECORATED_MESSAGE_DEPTH_4_ACTION_BAG': f.append('R4_SELECTED_READ')
 if replay.get('status')!='PASS' or replay.get('science_sha256')!=s['authority']['g4_r4_replay_science_sha256']: f.append('R4_REPLAY')
 if closeout.get('status')!='CERTIFIED_PASS' or closeout.get('science_sha256')!=s['authority']['g4_r4_closeout_science_sha256']: f.append('R4_CLOSEOUT')
 o={'schema_id':'IG_G4_R5_REBASE_AUTHORITY_V1','status':'PASS' if not f else 'FAIL','failures':f,'r4_science_sha256':r4.get('science_sha256'),'promotion':False}; o['science_sha256']=canonical_sha256(o)
 if f: raise G4R5RebaseError('authority failed: '+','.join(f))
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
 return (r2_action_signature(n,edges,v),_messages(n,edges,colors,grades,depth)[v])
def state_read(n:int,edges:Sequence[Sequence[int]],colors:Sequence[int],grades:Sequence[int],depth:int):
 msgs=_messages(n,edges,colors,grades,depth)
 return tuple(sorted((r2_action_signature(n,edges,v),msgs[v]) for v in range(n)))
def _path(n:int): return [(i,i+1) for i in range(n-1)]
def _audit_depth(depth:int)->dict[str,Any]:
 n=6; edges=_path(n); groups=defaultdict(set); witnesses={}; total=0
 for colors in product(range(2),repeat=n):
  for grades in product(range(2),repeat=n-1):
   msgs=_messages(n,edges,colors,grades,depth)
   for v in range(n):
    sig=(r2_action_signature(n,edges,v),msgs[v])
    ce=edges+[(v,n)]; cc=colors+(0,); cg=grades+(0,)
    child=state_read(n+1,ce,cc,cg,depth); groups[sig].add(child); total+=1
    if sig not in witnesses: witnesses[sig]=[]
    if len(witnesses[sig])<3: witnesses[sig].append({'colors':list(colors),'grades':list(grades),'attach_vertex':v,'child_read_sha256':canonical_sha256(child)})
 bad=[(sig,vals) for sig,vals in groups.items() if len(vals)>1]
 witness=None
 if bad:
  sig,vals=bad[0]; rows=witnesses[sig]
  # ensure two recorded rows with different child hashes where possible; derive by second pass only for this sig
  found=[]
  for colors in product(range(2),repeat=n):
   for grades in product(range(2),repeat=n-1):
    msgs=_messages(n,edges,colors,grades,depth)
    for v in range(n):
     s2=(r2_action_signature(n,edges,v),msgs[v])
     if s2!=sig: continue
     ce=edges+[(v,n)]; cc=colors+(0,); cg=grades+(0,); ch=state_read(n+1,ce,cc,cg,depth); h=canonical_sha256(ch)
     if all(x['child_read_sha256']!=h for x in found): found.append({'colors':list(colors),'grades':list(grades),'attach_vertex':v,'child_read_sha256':h})
     if len(found)>=2: break
    if len(found)>=2: break
   if len(found)>=2: break
  witness={'same_action_signature_sha256':canonical_sha256(sig),'representatives':found}
 return {'depth':depth,'attachment_instances':total,'action_read_class_count':len(groups),'nonpredictive_action_class_count':len(bad),'max_distinct_child_reads_in_one_action_class':max([len(v) for _,v in bad],default=1),'prediction_exact':not bad,'counterexample':witness,'strict_action_compression':len(groups)<total}
def audit()->dict[str,Any]:
 d4=_audit_depth(4); d5=_audit_depth(5)
 if d4['prediction_exact']: raise G4R5RebaseError('depth4 unexpectedly stable on P6')
 if not d5['prediction_exact']: raise G4R5RebaseError('depth5 failed on targeted P6 scope')
 if not d5['strict_action_compression']: raise G4R5RebaseError('depth5 has no compression')
 o={'schema_id':'IG_G4_R5_REBASE_DEPTH_STABILIZATION_AUDIT_V1','status':'PASS','underlying_shape':'P6','depth4':d4,'depth5':d5,'interpretation':'Exact targeted next-size challenge. It establishes that the R4 depth-4 message read does not stabilize at n=6 on P6, while depth 5 repairs this same bounded P6 scope. It does not establish an all-ranks depth law.'};o['science_sha256']=canonical_sha256(o);return o
def finalize(*,r4:Mapping[str,Any],r4_replay:Mapping[str,Any],r4_closeout:Mapping[str,Any])->dict[str,Any]:
 s=spec();auth=verify_authority(r4,r4_replay,r4_closeout);a=audit()
 out={'schema_id':'IG_G4_R5_REBASE_DEPTH_STABILIZATION_RESULT_V1','status':'PASS','stage_ref':'G4:R5.REBASE','classification':s['pass_classification'],'authority':auth,'bounded_scope':s['bounded_scope'],'stabilization_audit':a,'depth4_stable_at_n6_on_registered_scope':False,'depth5_repairs_registered_p6_scope':True,'all_ranks_claim':False,'promotion':False,'topology_promoted':False,'public_descriptor_changed':False,'raw_h_promoted':False,'g4_graduation_preserved':True,'g5_started':False,'next_authorized_stage':'G4:R6.REBASE_DESIGN','cost_control':s['cost_control'],'nonclaims':s['nonclaims']};out['science_sha256']=canonical_sha256(out);return out
def compare(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
 checks={'primary_pass':primary.get('status')=='PASS','cold_pass':cold.get('status')=='PASS','classification_equal':primary.get('classification')==cold.get('classification'),'science_sha_equal':primary.get('science_sha256')==cold.get('science_sha256'),'audit_equal':primary.get('stabilization_audit')==cold.get('stabilization_audit')};ok=all(checks.values());o={'schema_id':'IG_G4_R5_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256')};o['science_sha256']=canonical_sha256(o);return o
def closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
 ok=replay.get('status')=='PASS';o={'schema_id':'IG_G4_R5_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':primary.get('classification') if ok else None,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'replay_science_sha256':replay.get('science_sha256'),'g4_graduation_preserved':ok,'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG' if ok else None,'topology_promoted':False,'public_descriptor_changed':False,'g5_started':False,'next_authorized_stage':'G4:R6.REBASE_DESIGN' if ok else None};o['science_sha256']=canonical_sha256(o);return o
