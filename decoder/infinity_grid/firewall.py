from __future__ import annotations
class FirewallError(RuntimeError): pass

EVIDENCE_ORIGINS={'DIRECT_EXECUTION','EXACT_DERIVATION','SEALED_REPLAY_EVIDENCE','DOCUMENTED_AUTHORITY','HISTORICAL_ATTESTATION'}
EVIDENCE_SCOPES={'EXACT_EXHAUSTIVE','EXACT_BOUNDED','BOUNDED_PREREGISTERED','THEOREM_CONDITIONAL','DOCUMENTARY','INCOMPLETE'}
EVIDENCE_DISPOSITIONS={'PASS','FAIL','OBSERVED','INHERITED','FALSIFIED','UNDEFINED','INSUFFICIENT','PROVISIONAL'}
EVIDENCE_AUTHORITIES={'NONE','RECONNAISSANCE_ONLY','SCOPED_AUDIT_AUTHORITY','FORMAL_THEOREM_AUTHORITY'}
AUTHORITY_RANK={'NONE':0,'RECONNAISSANCE_ONLY':1,'SCOPED_AUDIT_AUTHORITY':2,'FORMAL_THEOREM_AUTHORITY':3}

def _tuple(ev): return [ev.get(k) for k in ['origin','scope','disposition','authority','protocol_label']]

def enforce_plan(descriptor: dict, plan: dict):
    label=plan.get('protocol_label'); ev=plan.get('evidence',{})
    if ev.get('origin') not in EVIDENCE_ORIGINS: raise FirewallError(f"invalid evidence origin {ev.get('origin')!r}")
    if ev.get('scope') not in EVIDENCE_SCOPES: raise FirewallError(f"invalid evidence scope {ev.get('scope')!r}")
    if ev.get('disposition') not in EVIDENCE_DISPOSITIONS: raise FirewallError(f"invalid evidence disposition {ev.get('disposition')!r}")
    if ev.get('authority') not in EVIDENCE_AUTHORITIES: raise FirewallError(f"invalid evidence authority {ev.get('authority')!r}")
    if ev.get('protocol_label')!=label: raise FirewallError('evidence protocol_label must match plan protocol_label')
    if label not in descriptor.get('allowed_protocol_labels',[]): raise FirewallError(f"protocol label {label!r} not allowed")
    policy=descriptor.get('evidence_policy',{})
    ceiling=policy.get('max_authority','FORMAL_THEOREM_AUTHORITY')
    if AUTHORITY_RANK[ev['authority']]>AUTHORITY_RANK[ceiling]: raise FirewallError('evidence authority exceeds protocol ceiling')
    tuples=policy.get('allowed_tuples',[])
    if tuples and _tuple(ev) not in tuples: raise FirewallError('evidence tuple not licensed by protocol descriptor')
    req=descriptor.get('input_contract',{}).get('required_flags',{}); flags=plan.get('input_flags',{})
    for k,v in req.items():
        if flags.get(k)!=v: raise FirewallError(f"required input flag {k}={v!r} not satisfied")
    requested=plan.get('requested_claims',[])
    if descriptor.get('claim_generation')=='FORBIDDEN' and requested:
        raise FirewallError('protocol forbids claim generation')
    forbidden=[str(x).lower() for x in descriptor.get('forbidden_claims',[])]
    for claim in requested:
        text=str(claim).lower()
        for token in forbidden:
            if token and token in text: raise FirewallError(f"forbidden claim token {token!r}")
    return True

def enforce_claim(descriptor:dict, run_record:dict, claim:dict):
    ev=run_record.get('evidence',{}); authority=ev.get('authority'); status=claim.get('status'); ctype=str(claim.get('claim_type','')).upper(); text=(str(claim.get('statement',''))+' '+str(claim.get('claim_id',''))).lower()
    # Re-evaluate descriptor evidence contract independent of run caller.
    if ev.get('origin') not in EVIDENCE_ORIGINS or ev.get('scope') not in EVIDENCE_SCOPES or ev.get('disposition') not in EVIDENCE_DISPOSITIONS or ev.get('authority') not in EVIDENCE_AUTHORITIES:
        raise FirewallError('run evidence vocabulary invalid')
    if ev.get('protocol_label') not in descriptor.get('allowed_protocol_labels',[]): raise FirewallError('run evidence protocol label not licensed')
    policy=descriptor.get('evidence_policy',{}); ceiling=policy.get('max_authority','FORMAL_THEOREM_AUTHORITY')
    if AUTHORITY_RANK[ev['authority']]>AUTHORITY_RANK[ceiling]: raise FirewallError('run evidence exceeds protocol authority ceiling')
    tuples=policy.get('allowed_tuples',[])
    if tuples and _tuple(ev) not in tuples: raise FirewallError('run evidence tuple not licensed by protocol')
    forbidden=[str(x).lower() for x in descriptor.get('forbidden_claims',[])]
    for token in forbidden:
        if token and token in text: raise FirewallError(f'claim violates protocol forbidden token {token!r}')
    if status=='EARNED_SCOPED' and AUTHORITY_RANK.get(authority,0)<AUTHORITY_RANK['SCOPED_AUDIT_AUTHORITY']:
        raise FirewallError('earned scoped claim requires scoped audit or formal authority')
    if ctype in {'FORMAL_THEOREM','GRADUATION','MECHANISM_PROMOTION'} and AUTHORITY_RANK.get(authority,0)<AUTHORITY_RANK['FORMAL_THEOREM_AUTHORITY']:
        raise FirewallError(f'{ctype} requires formal theorem authority')
    if descriptor.get('protocol_id') in {'SCOUT','OSCOUT','SPECTROSCOPE'} and ctype in {'FORMAL_THEOREM','GRADUATION','MECHANISM_PROMOTION'}:
        raise FirewallError(f'{descriptor.get("protocol_id")} cannot issue {ctype}')
    return True
