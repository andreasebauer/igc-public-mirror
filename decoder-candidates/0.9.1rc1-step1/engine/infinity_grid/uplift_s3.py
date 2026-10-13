from __future__ import annotations

"""G2:S3 triple compatibility / irreducibility audit.

S3 is deliberately non-promoting.  It asks whether any public fact of a connected
three-instance assembly (simple path P3 or triangle K3) can vary while all unary S0
interfaces and all complete S2 pair-outcome data remain fixed.

The finite sufficiency reduction is exact for this declared scope because every vertex
of P3/K3 has degree at most two.  S3 therefore checks two conditions:

1. S2 + the two endpoint unary interface identities uniquely determine the public
   endpoint reservation type(s) consumed by that pair edge.
2. Within every S0 interface class, the exact public two-reservation continuation
   table (the already-frozen D4 public operation table) is identical.

If both conditions hold, every degree<=2 connected triple joint legality and exact local
public post-reservation outcome is reconstructible from unary + pair data; hence there is
no irreducible S3 residual in the declared observer scope.
"""
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Mapping
import gzip, json, re
from .canon import canonical_sha256

class UpliftS3Error(RuntimeError): pass
_OP_RE=re.compile(r'^G1_PUBLIC_BRIDGE_RELATION_V1:(\d+)>(\d+)$')

def _sha(v:str,field:str)->str:
    v=str(v)
    if len(v)!=64 or any(c not in '0123456789abcdef' for c in v): raise UpliftS3Error(f'{field} must be lowercase sha256')
    return v

def verify_s3_authority(s2_closeout:Mapping[str,Any], *, expected_s2_source_sha256:str, expected_s2_science_sha256:str)->dict[str,Any]:
    failures=[]
    if s2_closeout.get('schema_id')!='IG_G2_S2_CLOSEOUT_V1': failures.append('SCHEMA')
    if s2_closeout.get('status')!='PASS': failures.append('STATUS')
    if s2_closeout.get('authorizes')!='G2:S3_TRIPLE_COMPATIBILITY_IRREDUCIBILITY_ONLY': failures.append('AUTHORIZATION')
    if s2_closeout.get('g2_graduated') is not False: failures.append('GRADUATION_FIREWALL')
    if s2_closeout.get('source_sha256')!=_sha(expected_s2_source_sha256,'expected_s2_source_sha256'): failures.append('SOURCE')
    if s2_closeout.get('s2_science_sha256')!=_sha(expected_s2_science_sha256,'expected_s2_science_sha256'): failures.append('SCIENCE')
    out={'schema_id':'IG_G2_S3_AUTHORITY_VERIFICATION_V1','status':'PASS' if not failures else 'FAIL','failures':failures,'s2_closeout_science_sha256':s2_closeout.get('science_sha256'),'g2_graduated':False}
    out['science_sha256']=canonical_sha256(out)
    if failures: raise UpliftS3Error('S3 authority verification failed: '+','.join(failures))
    return out

def endpoint_type_identifiability(repaired_records_path:str|Path, audit_signatures_path:str|Path)->dict[str,Any]:
    """Check that complete unary identities + S2 class fix pair endpoint type data."""
    seen:dict[tuple[str,tuple[str,str]],tuple[Any,...]]={}
    collisions=0; rows=0; failures=[]; class_records=Counter()
    with gzip.open(Path(repaired_records_path),'rt') as rf, gzip.open(Path(audit_signatures_path),'rt') as sf:
        while True:
            rl=rf.readline(); sl=sf.readline()
            if not rl and not sl: break
            if not rl or not sl: raise UpliftS3Error('aligned S1/audit streams differ in length')
            r=json.loads(rl); s=json.loads(sl); rows+=1
            if s.get('outcome_science_sha256')!=r.get('outcome_science_sha256'): raise UpliftS3Error(f'alignment mismatch row {rows}')
            if r.get('legality')!='LEGAL': raise UpliftS3Error('S3 declared R100 scope expects earned legal S1 records')
            m=_OP_RE.fullmatch(str(r.get('connection_operator_ref','')))
            if m is None: raise UpliftS3Error(f'operator parse failure row {rows}')
            a,b=map(int,m.groups()); li=str(r['left_interface_sha256']); ri=str(r['right_interface_sha256']); obs=_sha(s['realized_public_sha256'],'S2 observer')
            # Canonical endpoint annotation retains type association when interface ids differ;
            # for equal interfaces only the endpoint-type multiset is observable/relevant.
            if li<ri: anno=(li,a,ri,b)
            elif ri<li: anno=(ri,b,li,a)
            else: anno=(li,tuple(sorted((a,b))))
            key=(obs,tuple(sorted((li,ri))))
            class_records[key]+=1
            prev=seen.get(key)
            if prev is None: seen[key]=anno
            elif prev!=anno:
                collisions+=1
                if len(failures)<20: failures.append({'s2_realized_public_sha256':obs,'interface_pair':list(key[1]),'first':prev,'other':anno})
    out={'schema_id':'IG_G2_S3_PAIR_ENDPOINT_TYPE_IDENTIFIABILITY_V1','status':'PASS' if not failures else 'FAIL','record_count':rows,'s2_interface_pair_classes':len(seen),'classes_with_multiple_records':sum(v>1 for v in class_records.values()),'endpoint_annotation_conflict_count':collisions,'conflict_examples':failures,'hidden_internal_reads':False}
    out['science_sha256']=canonical_sha256(out); return out

def interface_two_reservation_sufficiency(s0_population:Mapping[str,Any], d4_rows:Mapping[str,Mapping[str,str]])->dict[str,Any]:
    by=defaultdict(list)
    for r in s0_population['interfaces']: by[str(r['interface_sha256'])].append(str(r['carrier_ref']))
    failures=[]; nontrivial=0
    for iface,refs in sorted(by.items()):
        if len(refs)>1: nontrivial+=1
        vectors=[]
        for ref in sorted(refs):
            row=d4_rows.get(ref)
            if row is None: raise UpliftS3Error(f'missing D4 row for carrier {ref}')
            vector=tuple(_sha(row[str(t)] if str(t) in row else row[t],f'D4 {ref} type {t}') for t in range(7))
            vectors.append((ref,vector))
        uniq={v for _,v in vectors}
        if len(uniq)>1 and len(failures)<20: failures.append({'interface_sha256':iface,'carrier_refs':refs,'vector_sha256s':[canonical_sha256(list(v)) for _,v in vectors]})
    out={'schema_id':'IG_G2_S3_INTERFACE_TWO_RESERVATION_SUFFICIENCY_V1','status':'PASS' if not failures else 'FAIL','interface_class_count':len(by),'nontrivial_interface_class_count':nontrivial,'carrier_count':sum(map(len,by.values())),'d4_vector_conflict_class_count':len(failures),'conflict_examples':failures,'public_operation':'reserve_external twice / frozen D4 exact continuation table','hidden_internal_reads':False}
    out['science_sha256']=canonical_sha256(out); return out

def triple_irreducibility_audit(*, s2_closeout:Mapping[str,Any], expected_s2_source_sha256:str, expected_s2_science_sha256:str, repaired_records_path:str|Path, audit_signatures_path:str|Path, s0_population:Mapping[str,Any], d4_rows:Mapping[str,Mapping[str,str]])->dict[str,Any]:
    auth=verify_s3_authority(s2_closeout,expected_s2_source_sha256=expected_s2_source_sha256,expected_s2_science_sha256=expected_s2_science_sha256)
    e=endpoint_type_identifiability(repaired_records_path,audit_signatures_path)
    d=interface_two_reservation_sufficiency(s0_population,d4_rows)
    passed=e['status']=='PASS' and d['status']=='PASS'
    classification='NO_IRREDUCIBLE_TRIPLE_RESIDUAL_IN_DECLARED_P3_K3_PUBLIC_SCOPE_S4_UNLOCKED' if passed else 'TRIPLE_RESIDUAL_OR_PAIR_ENDPOINT_AMBIGUITY_PRESENT_S4_LOCKED'
    out={
      'schema_id':'IG_G2_S3_TRIPLE_COMPATIBILITY_IRREDUCIBILITY_RESULT_V1','schema_version':'1.0.0','date':'2026-09-02','status':'PASS' if passed else 'REVIEW_REQUIRED','stage_ref':'G2:S3','classification':classification,
      'declared_triple_motifs':['P3_CONNECTED_PATH','K3_TRIANGLE'],'triple_instance_semantics':'three independently instantiated complete G1 carriers; carrier refs may repeat; simple pair-edge set; no parallel edge in S3',
      'joint_public_observer':'For each vertex, exact previous-layer public outcome after consuming its incident endpoint reservations (all canonical orders represented by D4), plus canonical relation-edge labels and joint legality.',
      'finite_reconstruction_argument':'P3/K3 have maximum degree 2. Unary S0 interface fixes exact two-reservation public continuation table; S2 pair data plus endpoint unary identities fixes each incident reservation type. Therefore joint legality and local public post-reservation facts are functions of unary+pair data.',
      'pair_endpoint_identifiability':e,'interface_two_reservation_sufficiency':d,'irreducible_residual_count':0 if passed else None,
      'hidden_internal_reads_in_promotable_reasoning':False,'observer_nonpromoting':True,'g2_graduated':False,'s4_unlocked':passed,
      'next_stage_if_independently_verified':'G2:S4' if passed else None,
      'nonclaims':['NOT_G2_STATE_DESCRIPTOR','NOT_S4_CLOSURE_RESULT','NOT_ARBITRARY_N_BODY_IRREDUCIBILITY','NO_G2_GRADUATION']}
    out['science_sha256']=canonical_sha256(out); return out
