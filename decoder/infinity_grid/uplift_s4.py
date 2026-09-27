from __future__ import annotations

"""G2:S4 one-step composition-closure audit.

S4 remains non-promoting. It asks whether certified S2 pair assemblies can be fed
back as complete units into the same public connection grammar without representative-
dependent behavior appearing immediately at the composition boundary.

The declared finite closure reduction has three parts:

1. Every certified repaired S1/S2 pair outcome retains positive public resources for
   every earned endpoint type in the frozen G1:R100 source population. Therefore every
   earned bridge operator remains publicly admissible when a pair assembly is reused as
   an input unit (legality closure in this exact population).
2. If an S2 class has multiple realized representatives, all representatives must have
   the same exact public D4 post-reservation continuation vector. D4 is exactly the input
   continuation information required by the repaired D4+Q2 connection grammar after
   one new bridge consumes an endpoint.
3. Every one of the 31 earned bridge operators is realized at least once with an actual
   S2 pair assembly as a complete child of a new lift, preserving the child intact and
   exposing the same public reserve_external capability.

This is ONE recursive binary composition step. It is not S5 finite-state sufficiency and
not S6 unbounded recursive closure.
"""
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import gzip, json, re
from .canon import canonical_sha256

class UpliftS4Error(RuntimeError): pass
_S4_IMPL_SPEC_RESOURCE='resources/uplift/G_UPLIFT_S4_COMPOSITION_CLOSURE_SPEC_V2.json'

def s4_implementation_spec()->dict[str,Any]:
    obj=json.loads(files('infinity_grid').joinpath(_S4_IMPL_SPEC_RESOURCE).read_text(encoding='utf-8'))
    expected=str(obj.get('science_sha256',''))
    payload={k:v for k,v in obj.items() if k!='science_sha256'}
    observed=canonical_sha256(payload)
    if expected!=observed: raise UpliftS4Error(f'S4 implementation spec hash mismatch: expected {expected}, observed {observed}')
    return obj

_OP_RE=re.compile(r'^G1_PUBLIC_BRIDGE_RELATION_V1:(\d+)>(\d+)$')

def _sha(v:str,field:str)->str:
    v=str(v)
    if len(v)!=64 or any(c not in '0123456789abcdef' for c in v): raise UpliftS4Error(f'{field} must be lowercase sha256')
    return v

def verify_s4_authority(certificate:Mapping[str,Any], *, expected_s3_source_sha256:str, expected_s3_result_sha256:str)->dict[str,Any]:
    failures=[]
    if certificate.get('schema_id')!='IG_G2_S4_UNLOCK_CERTIFICATE_V1': failures.append('SCHEMA')
    if certificate.get('status')!='EARNED': failures.append('STATUS')
    if certificate.get('authorizes')!='G2:S4_COMPOSITION_CLOSURE_ONLY': failures.append('AUTHORIZATION')
    if certificate.get('g2_graduated') is not False: failures.append('GRADUATION_FIREWALL')
    if certificate.get('source_sha256')!=_sha(expected_s3_source_sha256,'expected_s3_source_sha256'): failures.append('SOURCE')
    if certificate.get('s3_result_sha256')!=_sha(expected_s3_result_sha256,'expected_s3_result_sha256'): failures.append('SCIENCE')
    out={'schema_id':'IG_G2_S4_AUTHORITY_VERIFICATION_V1','status':'PASS' if not failures else 'FAIL','failures':failures,'s4_unlock_certificate_science_sha256':certificate.get('science_sha256'),'g2_graduated':False}
    out['science_sha256']=canonical_sha256(out)
    if failures: raise UpliftS4Error('S4 authority verification failed: '+','.join(failures))
    return out

def index_s2_collision_records(repaired_records_path:str|Path, audit_signatures_path:str|Path)->dict[str,Any]:
    """Return only S2 observer classes with >1 aligned realized records.

    The exact S2 class key is the REALIZED_PUBLIC_ASSEMBLY_V1 hash from the certified
    audit stream. Carrier/construction identity is retained only as realization provenance,
    never as a class key.
    """
    counts=Counter(); rows=0
    ap=Path(audit_signatures_path); rp=Path(repaired_records_path)
    with gzip.open(ap,'rt') as sf:
        for line in sf:
            s=json.loads(line); counts[_sha(s['realized_public_sha256'],'S2 observer')]+=1; rows+=1
    nontrivial={k for k,v in counts.items() if v>1}
    groups=defaultdict(list); rows2=0
    with gzip.open(rp,'rt') as rf, gzip.open(ap,'rt') as sf:
        while True:
            rl=rf.readline(); sl=sf.readline()
            if not rl and not sl: break
            if not rl or not sl: raise UpliftS4Error('aligned repaired/audit streams differ in length')
            r=json.loads(rl); s=json.loads(sl); rows2+=1
            if s.get('outcome_science_sha256')!=r.get('outcome_science_sha256'): raise UpliftS4Error(f'alignment mismatch row {rows2}')
            obs=_sha(s['realized_public_sha256'],'S2 observer')
            if obs in nontrivial:
                groups[obs].append({'row_index':rows2,'record':r})
    if rows2!=rows: raise UpliftS4Error('stream count changed between passes')
    hist=Counter(len(v) for v in groups.values())
    return {'schema_id':'IG_G2_S4_S2_COLLISION_INDEX_V1','status':'PASS','record_count':rows,'s2_class_count':len(counts),'nontrivial_s2_class_count':len(groups),'nontrivial_record_count':sum(map(len,groups.values())),'nontrivial_class_size_histogram':[{'record_count':k,'s2_class_count':v} for k,v in sorted(hist.items())],'groups':dict(groups)}

def resource_legality_closure(repaired_records_path:str|Path, s0_population:Mapping[str,Any])->dict[str,Any]:
    """Exact resource closure scan over all certified S1 pair outcomes.

    Pair output counters are derived solely from the frozen S0 reservation surfaces and
    the earned bridge endpoint types. If every output has every type positive, every
    earned bridge operator is publicly admissible for the next binary composition step.
    """
    iface={str(x['carrier_ref']):x for x in s0_population['interfaces']}
    mins=[None]*7; maxs=[None]*7; rows=0; malformed=[]
    with gzip.open(Path(repaired_records_path),'rt') as f:
        for line in f:
            r=json.loads(line); rows+=1
            if r.get('legality')!='LEGAL':
                if len(malformed)<20: malformed.append({'row':rows,'reason':'NONLEGAL'})
                continue
            m=_OP_RE.fullmatch(str(r.get('connection_operator_ref','')))
            if m is None: raise UpliftS4Error(f'operator parse failure row {rows}')
            a,b=map(int,m.groups()); L=iface[str(r['left_carrier_ref'])]; R=iface[str(r['right_carrier_ref'])]
            lc=L['one_endpoint_reservations'][a]['successor_total_free_by_type']; rc=R['one_endpoint_reservations'][b]['successor_total_free_by_type']
            if lc is None or rc is None: raise UpliftS4Error(f'legal row has absent public reservation row {rows}')
            caps=[int(lc[i])+int(rc[i]) for i in range(7)]
            for i,x in enumerate(caps):
                mins[i]=x if mins[i] is None else min(mins[i],x); maxs[i]=x if maxs[i] is None else max(maxs[i],x)
    all_positive=all(x is not None and x>0 for x in mins)
    out={'schema_id':'IG_G2_S4_PAIR_OUTPUT_RESOURCE_CLOSURE_V1','status':'PASS' if all_positive and not malformed else 'FAIL','record_count':rows,'minimum_total_free_by_type':mins,'maximum_total_free_by_type':maxs,'all_pair_outputs_positive_in_all_seven_types':all_positive,'malformed_examples':malformed,'consequence':'ALL_31_EARNED_BRIDGE_OPERATORS_REMAIN_PUBLICLY_ADMISSIBLE_FOR_NEXT_BINARY_STEP' if all_positive else 'NEXT_STEP_OPERATOR_DOMAIN_RESTRICTED','hidden_internal_reads':False}
    out['science_sha256']=canonical_sha256(out); return out

def d4_composition_congruence(collision_index:Mapping[str,Any], d4_vectors_by_row:Mapping[int, list[str]])->dict[str,Any]:
    examples=[]; checked=0; conflict_count=0
    for obs,items in sorted(collision_index['groups'].items()):
        vecs=[]
        for item in items:
            idx=int(item['row_index']); v=d4_vectors_by_row.get(idx)
            if v is None or len(v)!=7: raise UpliftS4Error(f'missing seven-type D4 vector for row {idx}')
            vv=tuple(_sha(x,f'D4 row {idx}') for x in v); vecs.append((idx,vv))
        checked+=1; uniq={v for _,v in vecs}
        if len(uniq)>1:
            conflict_count+=1
            if len(examples)<50:
                examples.append({'s2_realized_public_sha256':obs,'row_indices':[x for x,_ in vecs],'d4_vector_sha256s':[canonical_sha256(list(v)) for _,v in vecs]})
    out={'schema_id':'IG_G2_S4_S2_D4_COMPOSITION_CONGRUENCE_V1','status':'PASS' if conflict_count==0 else 'FAIL','nontrivial_s2_classes_checked':checked,'nontrivial_records_checked':sum(len(v) for v in collision_index['groups'].values()),'d4_conflict_class_count':conflict_count,'conflict_examples':examples,'conflict_examples_truncated':conflict_count>len(examples),'public_operation':'relation-valued G2 post-reservation branch-continuation projection on realized S2 pair assembly','hidden_internal_reads':False}
    out['science_sha256']=canonical_sha256(out); return out

def composition_closure_result(*, authority:Mapping[str,Any], resource_closure:Mapping[str,Any], d4_congruence:Mapping[str,Any], operator_basis:Mapping[str,Any])->dict[str,Any]:
    passed=authority.get('status')=='PASS' and resource_closure.get('status')=='PASS' and d4_congruence.get('status')=='PASS' and operator_basis.get('status')=='PASS'
    classification='S2_UNITS_CLOSED_UNDER_ONE_RECURSIVE_BINARY_COMPOSITION_STEP_S5_UNLOCKED' if passed else 'COMPOSITION_CLOSURE_RESIDUAL_PRESENT_S5_LOCKED'
    out={'schema_id':'IG_G2_S4_COMPOSITION_CLOSURE_RESULT_V1','schema_version':'1.0.0','date':'2026-09-02','status':'PASS' if passed else 'REVIEW_REQUIRED','stage_ref':'G2:S4','classification':classification,'closure_scope':'ONE_RECURSIVE_BINARY_COMPOSITION_STEP_USING_COMPLETE_S2_PAIR_ASSEMBLIES_AS_ATOMIC_INPUT_UNITS','candidate_grammar':'preserve complete input units; consume one public endpoint on each side through complete G2 owner-choice successor relation; add one earned G1 typed bridge relation; expose relation-valued G2 reserve surface while frozen lower-G reserve semantics remain unchanged','authority':dict(authority),'resource_legality_closure':dict(resource_closure),'s2_representative_d4_congruence':dict(d4_congruence),'earned_operator_realization_basis':dict(operator_basis),'hidden_internal_reads_in_promotable_reasoning':False,'g2_routing_semantics':'RELATION_VALUED_ALL_ELIGIBLE_OWNER_CHOICES_V1','s4_implementation_spec_sha256':s4_implementation_spec()['science_sha256'],'s5_unlocked':passed,'g2_graduated':False,'nonclaims':['NOT_FINITE_G2_STATE_DESCRIPTOR','NOT_UNBOUNDED_RECURSIVE_CLOSURE','NOT_ASSOCIATIVITY_OR_REBRACKETING_THEOREM','NOT_G2_GRADUATION']}
    out['science_sha256']=canonical_sha256(out); return out
