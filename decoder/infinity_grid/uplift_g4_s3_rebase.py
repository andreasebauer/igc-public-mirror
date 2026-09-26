from __future__ import annotations
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
from .canon import canonical_sha256
from .g4_term_state import G3TermState
from .uplift_g4_s1_rebase import load_states
from .uplift_g4_s3 import ordered_two_reservation_signature

class G4S3RebaseError(RuntimeError): pass
_WCTX: dict[str,G3TermState] = {}

def _resource(name:str)->Path: return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))
def spec()->dict[str,Any]:
 o=json.loads(_resource('G4_S3_REBASE_HIGHER_ORDER_IRREDUCIBILITY_SPEC_V1.json').read_text())
 if o.get('schema_id')!='IG_G4_S3_REBASE_HIGHER_ORDER_IRREDUCIBILITY_SPEC_V1': raise G4S3RebaseError('bad spec')
 if canonical_sha256({k:v for k,v in o.items() if k!='science_sha256'})!=o.get('science_sha256'): raise G4S3RebaseError('spec hash')
 return o

def verify_authority(*,s0:Mapping[str,Any],s2:Mapping[str,Any],s2_replay:Mapping[str,Any],s2_closeout:Mapping[str,Any])->dict[str,Any]:
 f=[]
 if s0.get('schema_id')!='IG_G4_S0_REBASE_INTERFACE_RESULT_V1' or s0.get('status')!='PASS': f.append('S0')
 if s2.get('schema_id')!='IG_G4_S2_REBASE_MINIMAL_SUFFICIENT_READ_RESULT_V1' or s2.get('status')!='PASS': f.append('S2')
 if s2.get('minimal_sufficient_eligible_tier')!={'tier':4,'name':'REPAIRED_HIDDEN_STATE_H'}: f.append('CANDIDATE')
 if s2.get('next_authorized_stage')!='G4:S3.REBASE': f.append('UNLOCK')
 if s2_replay.get('certification')!='CERTIFIED_PASS' or s2_replay.get('primary_science_sha256')!=s2.get('science_sha256'): f.append('REPLAY')
 if s2_closeout.get('status')!='CERTIFIED_PASS' or s2_closeout.get('next_authorized_stage')!='G4:S3.REBASE': f.append('CLOSEOUT')
 if f: raise G4S3RebaseError('authority failed: '+','.join(f))
 out={'schema_id':'IG_G4_S3_REBASE_AUTHORITY_V1','status':'PASS','s0_science_sha256':s0.get('science_sha256'),'s2_science_sha256':s2.get('science_sha256'),'s2_replay_science_sha256':s2_replay.get('science_sha256'),'s2_closeout_science_sha256':s2_closeout.get('science_sha256'),'spec_science_sha256':spec()['science_sha256'],'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','lower_layer_rematerialization':False}
 out['science_sha256']=canonical_sha256(out); return out

def candidate_index(s0:Mapping[str,Any])->dict[str,Any]:
 c=s0['certified_term_corpus']; rows=[]
 for key,t in sorted(c['terms'].items()):
  ref=str(t['term']['term_ref']); caps=[int(x) for x in t['term']['total_caps']]; h=c['hidden_diagnostics'][key]['repaired_hidden_state']
  hv={'shell_profile_multiset':h['shell_profile_multiset'],'paired_support_load_distance_signature_multiset':h['paired_support_load_distance_signature_multiset']}
  ck=canonical_sha256({'caps7':caps,'repaired_hidden_state_H':hv})
  rows.append({'term_key':key,'term_ref':ref,'caps7':caps,'H':hv,'candidate_key_sha256':ck})
 if len(rows)!=4: raise G4S3RebaseError('expected four terms')
 out={'schema_id':'IG_G4_S3_REBASE_CANDIDATE_INDEX_V1','candidate':{'tier':4,'name':'REPAIRED_HIDDEN_STATE_H'},'rows':rows,'exact_term_count':4,'candidate_class_count':len({r['candidate_key_sha256'] for r in rows})}
 out['science_sha256']=canonical_sha256(out); return out

def init_worker(payload:Mapping[str,Any])->None:
 global _WCTX; _WCTX={str(k):G3TermState.from_wire(v) for k,v in payload['states'].items()}
def local_worker(payload:Mapping[str,Any])->dict[str,Any]:
 ref=str(payload['term_ref']); a=int(payload['first_type']); b=int(payload['second_type']); st=_WCTX[ref]
 sig=ordered_two_reservation_signature(st,a,b)
 return {'term_ref':ref,'first_type':a,'second_type':b,'signature':sig}

def local_sufficiency(*,cand:Mapping[str,Any],rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
 key_by_ref={r['term_ref']:r['candidate_key_sha256'] for r in cand['rows']}; expected=len(key_by_ref)*49
 if len(rows)!=expected: raise G4S3RebaseError(f'expected {expected} rows got {len(rows)}')
 grouped=defaultdict(list); vectors=defaultdict(dict)
 for r in rows:
  ref=str(r['term_ref']); a=int(r['first_type']); b=int(r['second_type']); sh=str(r['signature']['science_sha256'])
  grouped[(key_by_ref[ref],a,b)].append((ref,sh)); vectors[ref][(a,b)]=sh
 conflicts=[]
 for (ck,a,b),vals in grouped.items():
  u=sorted({h for _,h in vals})
  if len(u)>1: conflicts.append({'candidate_key_sha256':ck,'first_type':a,'second_type':b,'signature_sha256s':u,'term_refs':sorted(r for r,_ in vals)})
 classes=defaultdict(list)
 for ref,ck in key_by_ref.items(): classes[ck].append(ref)
 for ck,reps in classes.items():
  vhs=[canonical_sha256([vectors[r][(a,b)] for a in range(7) for b in range(7)]) for r in reps]
  if len(set(vhs))>1: conflicts.append({'failure':'FULL_VECTOR_CONFLICT','candidate_key_sha256':ck,'term_refs':sorted(reps),'vector_hashes':sorted(set(vhs))})
 out={'schema_id':'IG_G4_S3_REBASE_H_TWO_RESERVATION_SUFFICIENCY_V1','status':'PASS' if not conflicts else 'FAIL','candidate':{'tier':4,'name':'REPAIRED_HIDDEN_STATE_H'},'exact_term_count':len(key_by_ref),'candidate_class_count':len(classes),'ordered_endpoint_pair_count':49,'task_row_count':len(rows),'continuation_conflict_count':len(conflicts),'conflict_examples':conflicts[:16],'bounded_scope_note':'Exact on the frozen four-term rebase corpus; does not establish unseen-representative global minimality.'}
 out['science_sha256']=canonical_sha256(out); return out

def coverage(c:int,o:int,local_ok:bool)->dict[str,Any]:
 assignments=c**3; p3=assignments*(o**2); k3=assignments*(o**3); ok=bool(local_ok and c>=1 and o==31)
 out={'schema_id':'IG_G4_S3_REBASE_TRIPLE_FACTORIZATION_COVERAGE_V1','status':'PASS' if ok else 'FAIL','candidate_unit_class_count':c,'ordered_candidate_unit_assignments':assignments,'directed_operator_count':o,'public_context_counts':{'P3_CONNECTED_PATH':p3,'K3_TRIANGLE':k3,'TOTAL':p3+k3},'materialized_triple_contexts':False,'factorization_argument':'Every motif vertex has degree <=2. Certified S2.REBASE fixes the pair observer on H; the complete ordered two-reservation continuation is single-valued on each frozen H candidate class. Therefore P3/K3 branch-sensitive compatibility factors through H plus certified pair-edge values, with no additional triple-only hidden read on this registered scope.'}
 out['science_sha256']=canonical_sha256(out); return out

def finalize(*,s0:Mapping[str,Any],s2:Mapping[str,Any],s2_replay:Mapping[str,Any],s2_closeout:Mapping[str,Any],task_rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
 auth=verify_authority(s0=s0,s2=s2,s2_replay=s2_replay,s2_closeout=s2_closeout); cand=candidate_index(s0); loc=local_sufficiency(cand=cand,rows=task_rows); cov=coverage(int(cand['candidate_class_count']),31,loc['status']=='PASS'); ok=loc['status']=='PASS' and cov['status']=='PASS'
 out={'schema_id':'IG_G4_S3_REBASE_HIGHER_ORDER_RESULT_V1','status':'PASS' if ok else 'REVIEW_REQUIRED','stage_ref':'G4:S3.REBASE','classification':'G4_S3_REBASE_NO_IRREDUCIBLE_TRIPLE_RESIDUAL_ON_REGISTERED_P3_K3_H_SCOPE_S4_REBASE_UNLOCKED' if ok else 'G4_S3_REBASE_HIGHER_ORDER_RESIDUAL_S4_REBASE_LOCKED','authority':auth,'candidate_interface':cand,'local_two_reservation_sufficiency':loc,'triple_factorization_coverage':cov,'cost_control':{'new_science_kernels':196,'lower_layer_rematerialization':False,'triple_contexts_materialized':False},'public_descriptor_promoted':False,'hidden_state_promoted':False,'g4_s4_rebase_unlocked':ok,'g4_rebase_complete':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','g4_graduated':False,'next_authorized_stage':'G4:S4.REBASE' if ok else None,'nonclaims':spec()['nonclaims']}
 out['science_sha256']=canonical_sha256(out); return out

def compare(p:Mapping[str,Any],c:Mapping[str,Any])->dict[str,Any]:
 checks={'primary_pass':p.get('status')=='PASS','cold_pass':c.get('status')=='PASS','science_sha_equal':p.get('science_sha256')==c.get('science_sha256'),'classification_equal':p.get('classification')==c.get('classification'),'local_equal':p.get('local_two_reservation_sufficiency')==c.get('local_two_reservation_sufficiency'),'coverage_equal':p.get('triple_factorization_coverage')==c.get('triple_factorization_coverage')}; ok=all(checks.values()); out={'schema_id':'IG_G4_S3_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':p.get('science_sha256'),'cold_science_sha256':c.get('science_sha256')}; out['science_sha256']=canonical_sha256(out); return out

def closeout(p:Mapping[str,Any],c:Mapping[str,Any],r:Mapping[str,Any])->dict[str,Any]:
 ok=r.get('certification')=='CERTIFIED_PASS' and p.get('status')=='PASS' and c.get('status')=='PASS'; out={'schema_id':'IG_G4_S3_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':p.get('classification'),'primary_science_sha256':p.get('science_sha256'),'cold_science_sha256':c.get('science_sha256'),'replay_science_sha256':r.get('science_sha256'),'public_descriptor_promoted':False,'g4_s4_rebase_unlocked':ok,'g4_rebase_complete':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','next_authorized_stage':'G4:S4.REBASE' if ok else None}; out['science_sha256']=canonical_sha256(out); return out
