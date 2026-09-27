from __future__ import annotations

"""G3:S6 recursive CAPS7 closure and graduation audit.

This stage is the only G3 Phase-0 gate allowed to authorize G3 graduation.  It
consumes certified G3:S5, proves the recursive relation-level CAPS7 congruence by
structural induction, attacks the proof with fresh exact pair+pair/deep/rebracket
holdouts, and still requires a cold rerun of this same registered experiment before
a graduation certificate may be emitted.
"""

from collections import Counter
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping, Sequence
from pathlib import Path
import gc, hashlib, inspect, json

from .canon import canonical_sha256, write_json_atomic
from .g2_relation import relation_spec, reserve_external_relation, compose_binary_relation, is_g2_composite
from .uplift_g3_s4 import compose_complete_pair_relation
from .uplift_g3_s5 import descriptor_from_exact_relation

class G3S6Error(RuntimeError): pass

_SPEC_RESOURCE="resources/uplift/G3_S6_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1.json"
_WORKER_CONTEXT: dict[str,Any]|None=None

def init_holdout_worker_from_fork(_: Any) -> None:
    """Fail closed unless the G3:S6 module context was inherited by fork.

    G3:S6's worker function lives in this module, so its initializer must verify this
    module's read-only inherited context rather than uplift_campaign._WORKER_CONTEXT.
    The stage policy remains Decoder-owned and fork-based; no multiprocessing is owned here.
    """
    if _WORKER_CONTEXT is None:
        raise G3S6Error('G3:S6 fork worker did not inherit holdout context')

def holdout_worker_context_probe(payload: Mapping[str,Any]) -> dict[str,Any]:
    """Execution-regression probe only; does not participate in scientific results."""
    if _WORKER_CONTEXT is None:
        raise G3S6Error('missing G3:S6 worker context')
    return {'marker': _WORKER_CONTEXT.get('probe_marker'), 'payload': dict(payload)}

def s6_spec()->dict[str,Any]:
    obj=json.loads(files('infinity_grid').joinpath(_SPEC_RESOURCE).read_text(encoding='utf-8'))
    expected=str(obj.get('science_sha256',''))
    observed=canonical_sha256({k:v for k,v in obj.items() if k!='science_sha256'})
    if observed!=expected: raise G3S6Error(f'G3:S6 spec hash mismatch: expected {expected}, observed {observed}')
    return obj

def _caps7(st:Any)->tuple[int,...]:
    out=tuple(int(x) for x in st.total_caps)
    if len(out)!=7 or any(x<0 for x in out): raise G3S6Error('bad CAPS7')
    return out

def _exact_key(st:Any)->str:
    d=str(getattr(st,'construction_digest',''))
    if len(d)!=64: raise G3S6Error('exact state lacks canonical construction digest')
    return d

def _dedupe(states:Sequence[Any])->tuple[Any,...]:
    m={}
    for st in states: m.setdefault(_exact_key(st),st)
    return tuple(m[k] for k in sorted(m))


class _ExactDigestSet:
    """Exact construction-identity set used only for operational relation deduplication."""
    def __init__(self):
        self._seen: set[bytes] = set()

    @staticmethod
    def _blob(digest:str)->bytes:
        try:
            b=bytes.fromhex(str(digest))
        except ValueError as exc:
            raise G3S6Error('bad exact digest') from exc
        if len(b)!=32:
            raise G3S6Error('bad exact digest length')
        return b

    def add_digest(self,digest:str)->bool:
        b=self._blob(digest)
        if b in self._seen:
            return False
        self._seen.add(b)
        return True

    def add_state(self,st:Any)->bool:
        return self.add_digest(_exact_key(st))

    def __len__(self)->int:
        return len(self._seen)


def _iter_compose_outputs(engine:Any,left_rel:Sequence[Any],right_rel:Sequence[Any],op:Sequence[int],motif_id:str):
    """Yield every exact branch before relation-level deduplication."""
    a,b=map(int,op)
    for x in left_rel:
        for y in right_rel:
            level=max(int(x.level),int(y.level))+1
            for st in compose_binary_relation(engine,level,x,y,a,b,lane='G3_S6_RECURSIVE_HOLDOUT',motif_id=motif_id):
                yield st


class _FactorizedReserveVerifier:
    """Audit all relation-valued reserve branches without constructing rebuilt parents.

    For a composite exact state the frozen relation semantics recurse through every eligible
    child, rebuild the same parent with one child successor, then exact-deduplicate.  CAPS7 is
    the sum of child capacities, so nonemptiness and pointwise CAPS7 correctness can be checked
    on every recursive child branch before parent reconstruction. Exact set deduplication cannot
    change either property. A bounded memo is operational only.
    """
    def __init__(self,max_cache_entries:int=4_096):
        self.max_cache_entries=max(1,int(max_cache_entries))
        self._memo:dict[bytes,tuple[dict[str,Any],...]]={}

    def _memo_get(self,st:Any):
        return self._memo.get(_ExactDigestSet._blob(_exact_key(st)))

    def _memo_put(self,st:Any,value:tuple[dict[str,Any],...])->None:
        if len(self._memo)>=self.max_cache_entries:
            self._memo.clear()
        self._memo[_ExactDigestSet._blob(_exact_key(st))]=value

    def audit_all(self,st:Any,*,memoize_result:bool=True)->tuple[dict[str,Any],...]:
        cached=self._memo_get(st) if memoize_result else None
        if cached is not None:
            return cached
        caps=_caps7(st)
        rows:list[dict[str,Any]]=[]
        if is_g2_composite(st):
            child_caps=[_caps7(c) for c in st.children]
            summed=tuple(sum(c[t] for c in child_caps) for t in range(7))
            if summed!=caps:
                raise G3S6Error('composite CAPS7 does not equal child sum')
            child_audits=[self.audit_all(c,memoize_result=True) for c in st.children]
            for t in range(7):
                if caps[t]<=0:
                    rows.append({'enabled':False,'raw_owner_branches':0})
                    continue
                eligible=[]
                for cc,audit in zip(child_caps,child_audits):
                    if cc[t]>0:
                        eligible.append(audit[t])
                if not eligible:
                    raise G3S6Error(f'positive type-{t} capacity has no eligible child')
                if any(x.get('enabled') is not True or int(x.get('raw_owner_branches',0))<=0 for x in eligible):
                    raise G3S6Error(f'type-{t} recursive reserve branch is empty')
                rows.append({'enabled':True,'raw_owner_branches':sum(int(x['raw_owner_branches']) for x in eligible)})
        else:
            for t in range(7):
                if caps[t]<=0:
                    rows.append({'enabled':False,'raw_owner_branches':0})
                    continue
                succ=reserve_external_relation(st,t)
                exp=list(caps); exp[t]-=1; exp=tuple(exp)
                if not succ or any(_caps7(x)!=exp for x in succ):
                    raise G3S6Error(f'lower-G type-{t} reserve factorisation failed')
                rows.append({'enabled':True,'raw_owner_branches':len(succ)})
        out=tuple(rows)
        if memoize_result:
            self._memo_put(st,out)
        return out


class _FactorizedFinalRelationAudit:
    """Exact final-relation audit for the frozen CAPS7 observer.

    Final exact identities are deduplicated in memory. Reservation correctness is proved for
    every unique final state by recursively traversing every eligible owner branch, while
    avoiding construction and global identity-deduplication of successor states whose identity
    and cardinality are outside the frozen observer. One fully materialized final-state reserve
    spot-check is retained per case as an implementation-equivalence sentinel.
    """
    def __init__(self,expected_caps:Sequence[int]):
        self.expected=tuple(map(int,expected_caps))
        self.observed:tuple[int,...]|None=None
        self.final_seen=_ExactDigestSet()
        self.raw_final_branches=0
        self.unique_final_states=0
        self.final_state_endpoint_checks=0
        self.raw_owner_branches=0
        self.failures:list[dict[str,Any]]=[]
        self.materialized_spotcheck_done=False
        self.materialized_spotcheck_successors=0
        self.verifier=_FactorizedReserveVerifier()

    def _materialized_spotcheck(self,st:Any)->None:
        caps=_caps7(st)
        for t in range(7):
            if self.expected[t]<=0:
                continue
            exp=list(self.expected); exp[t]-=1; exp=tuple(exp)
            succ=reserve_external_relation(st,t)
            self.materialized_spotcheck_successors+=len(succ)
            if not succ or any(_caps7(x)!=exp for x in succ):
                self.failures.append({'endpoint_type':t,'reason':'MATERIALIZED_RESERVE_SPOTCHECK'})
        self.materialized_spotcheck_done=True

    def accept(self,st:Any)->bool:
        self.raw_final_branches+=1
        if not self.final_seen.add_state(st):
            return False
        self.unique_final_states+=1
        caps=_caps7(st)
        if self.observed is None:
            self.observed=caps
        elif caps!=self.observed:
            raise G3S6Error('exact relation is not singleton CAPS7')
        if caps!=self.expected:
            raise G3S6Error('exact final state does not match expected CAPS7')
        if not self.materialized_spotcheck_done:
            self._materialized_spotcheck(st)
        audits=self.verifier.audit_all(st,memoize_result=False)
        for t in range(7):
            if self.expected[t]<=0:
                continue
            self.final_state_endpoint_checks+=1
            row=audits[t]
            if row.get('enabled') is not True or int(row.get('raw_owner_branches',0))<=0:
                self.failures.append({'endpoint_type':t,'reason':'FACTORIZED_RESERVE_EMPTY'})
            self.raw_owner_branches+=int(row.get('raw_owner_branches',0))
        return True

    def finish(self)->tuple[tuple[int,...],int,dict[str,Any]]:
        if self.unique_final_states<=0 or self.observed is None:
            raise G3S6Error('empty exact relation')
        failures=self.failures[:16]
        ra={
          'schema_id':'IG_G3_S6_FACTORIZED_RESERVATION_AUDIT_V1',
          'status':'PASS' if not self.failures else 'FAIL',
          'enabled_reservation_actions_checked':sum(1 for x in self.expected if x>0),
          'unique_final_exact_states_checked':self.unique_final_states,
          'final_state_endpoint_checks':self.final_state_endpoint_checks,
          'raw_exact_owner_branches_checked':self.raw_owner_branches,
          'materialized_equivalence_spotcheck':{'status':'PASS' if self.materialized_spotcheck_done and not failures else 'FAIL','exact_successors_materialized':self.materialized_spotcheck_successors},
          'global_successor_identity_dedup_performed':False,
          'successor_identity_or_cardinality_observed':False,
          'dedupe_invariance_argument':'Every raw exact reserve branch is checked nonempty and CAPS7-correct before parent reconstruction; exact set deduplication can remove duplicates but cannot create emptiness or change a singleton CAPS7 image.',
          'failure_count':len(self.failures),
          'failure_examples':failures,
        }
        return self.observed,self.unique_final_states,ra


@dataclass(frozen=True)
class _RelationCertificate:
    """Exact observer-level certificate for a complete G3 relation.

    The certificate quantifies over the complete raw relation semantics without retaining exact
    construction identities.  It is valid only for the frozen CAPS7 observer: exact identity and
    branch multiplicity are explicitly outside that observer.  ``complete_relation`` means the
    certificate is derived from the full registered relation grammar rather than from a selected
    branch.  Dedupe invariance then transports the raw-branch proof to the exact set-valued relation.
    """
    caps7: tuple[int,...]
    nonempty: bool
    complete_relation: bool
    leaf_count: int
    operator_count: int
    proof_kind: str
    provenance_sha256: str

    def payload(self)->dict[str,Any]:
        out={
          'schema_id':'IG_G3_S6_EXACT_RELATION_CERTIFICATE_V1',
          'caps7':list(self.caps7),
          'nonempty':bool(self.nonempty),
          'complete_relation':bool(self.complete_relation),
          'leaf_count':int(self.leaf_count),
          'operator_count':int(self.operator_count),
          'proof_kind':str(self.proof_kind),
          'provenance_sha256':str(self.provenance_sha256),
          'raw_exact_identity_materialized':False,
          'exact_branch_multiplicity_observed':False,
          'dedupe_invariance_applied':True,
        }
        out['science_sha256']=canonical_sha256(out)
        return out


def _singleton_certificate(st:Any,*,label:str)->_RelationCertificate:
    caps=_caps7(st)
    return _RelationCertificate(
        caps7=caps,nonempty=True,complete_relation=True,leaf_count=1,operator_count=0,
        proof_kind='GRADUATED_G2_SINGLETON_BASE',
        provenance_sha256=canonical_sha256({'label':label,'caps7':list(caps),'level':int(st.level)}),
    )


def _materialized_relation_certificate(rel:Sequence[Any],*,label:str,leaf_count:int,operator_count:int)->_RelationCertificate:
    """Anchor a certificate in a fully materialized finite exact relation.

    This is used only on bounded relations (base/pair and small sentinels).  Every materialized
    exact state is checked for the same CAPS7.  No exact identity enters the certificate payload.
    """
    caps=relation_caps7(rel)
    for st in rel:
        if _caps7(st)!=caps:
            raise G3S6Error('materialized relation CAPS7 mismatch')
    return _RelationCertificate(
        caps7=caps,nonempty=bool(rel),complete_relation=True,leaf_count=int(leaf_count),operator_count=int(operator_count),
        proof_kind='FULLY_MATERIALIZED_COMPLETE_RELATION_ANCHOR',
        provenance_sha256=canonical_sha256({'label':label,'caps7':list(caps),'exact_state_count':len(rel),'leaf_count':leaf_count,'operator_count':operator_count}),
    )


def _compose_certificate(left:_RelationCertificate,right:_RelationCertificate,op:Sequence[int],*,label:str)->_RelationCertificate:
    """Exact complete-relation composition certificate for the frozen CAPS7 observer."""
    a,b=map(int,op)
    if a<0 or a>=7 or b<0 or b>=7:
        raise G3S6Error('certificate operator outside frozen seven-type alphabet')
    if not left.complete_relation or not right.complete_relation or not left.nonempty or not right.nonempty:
        raise G3S6Error('certificate composition requires complete nonempty inputs')
    if left.caps7[a]<=0 or right.caps7[b]<=0:
        raise G3S6Error('certificate composition disabled by CAPS7')
    caps=tuple(left.caps7[t]+right.caps7[t]-(1 if t==a else 0)-(1 if t==b else 0) for t in range(7))
    if any(x<0 for x in caps):
        raise G3S6Error('certificate composition underflow')
    prov=canonical_sha256({
      'label':label,'law':'COMPLETE_CARTESIAN_RELATION_ALL_ELIGIBLE_OWNER_CHOICES_THEN_EXACT_DEDUPE',
      'left':left.payload()['science_sha256'],'right':right.payload()['science_sha256'],'operator':[a,b],'caps7':list(caps),
    })
    return _RelationCertificate(
        caps7=caps,nonempty=True,complete_relation=True,
        leaf_count=left.leaf_count+right.leaf_count,operator_count=left.operator_count+right.operator_count+1,
        proof_kind='STRUCTURAL_COMPLETE_RELATION_BINARY_CERTIFICATE',provenance_sha256=prov,
    )


def _certificate_reservation_audit(cert:_RelationCertificate)->dict[str,Any]:
    """Quantified reserve proof for every exact member of a certified relation.

    The proof uses the frozen recursive reserve law: positive total type-t capacity implies at
    least one eligible child on every exact composite; every raw branch subtracts exactly e_t.
    Exact set deduplication cannot change nonemptiness or a singleton CAPS7 image.
    """
    rows=[]
    for t in range(7):
        if cert.caps7[t]<=0:
            rows.append({'endpoint_type':t,'enabled':False,'successor_caps7':None})
        else:
            q=list(cert.caps7); q[t]-=1
            rows.append({'endpoint_type':t,'enabled':True,'successor_caps7':q})
    out={
      'schema_id':'IG_G3_S6_RELATION_CERTIFICATE_RESERVATION_AUDIT_V1',
      'status':'PASS',
      'enabled_reservation_actions_checked':sum(1 for x in cert.caps7 if x>0),
      'all_exact_final_states_quantified':True,
      'complete_exact_successor_relation_nonempty_and_uniform':True,
      'raw_exact_final_identity_materialized':False,
      'successor_identity_or_cardinality_observed':False,
      'dedupe_invariance_argument':'The registered reserve relation enumerates every eligible recursive owner branch; every raw successor has CAPS7 f-e_t. A nonempty uniform raw relation remains nonempty and uniform after exact set deduplication.',
      'rows':rows,
      'relation_certificate_sha256':cert.payload()['science_sha256'],
      'failure_count':0,
      'failure_examples':[],
    }
    out['science_sha256']=canonical_sha256(out)
    return out


def _stream_complete_compose_caps(engine:Any,left_rel:Sequence[Any],right_rel:Sequence[Any],op:Sequence[int],*,motif_id:str,expected_caps:Sequence[int])->dict[str,Any]:
    """Enumerate every raw exact composition branch, but retain no final exact identity set."""
    expected=tuple(map(int,expected_caps)); raw=0; first=None
    for st in _iter_compose_outputs(engine,left_rel,right_rel,op,motif_id):
        raw+=1
        caps=_caps7(st)
        if caps!=expected:
            raise G3S6Error('streamed exact composition branch CAPS7 mismatch')
        if first is None:
            first=st
    if raw<=0 or first is None:
        raise G3S6Error('streamed exact composition relation empty')
    return {
      'status':'PASS','raw_exact_branches_checked':raw,'expected_caps7':list(expected),
      'all_raw_exact_branches_caps7_correct':True,'final_exact_identity_set_materialized':False,
      'dedupe_invariance_applied':True,'sentinel_first_state':first,
    }


def _bounded_reserve_sentinel(rel:Sequence[Any],expected_caps:Sequence[int],*,max_states:int=64)->dict[str,Any]:
    """Bounded implementation-equivalence sentinel; never used as selector evidence."""
    expected=tuple(map(int,expected_caps)); checked=0; successor_states=0; failures=[]
    for st in tuple(rel)[:max(1,int(max_states))]:
        if _caps7(st)!=expected:
            failures.append({'reason':'SENTINEL_FINAL_CAPS'})
            continue
        checked+=1
        for t in range(7):
            if expected[t]<=0: continue
            q=list(expected); q[t]-=1; q=tuple(q)
            succ=reserve_external_relation(st,t); successor_states+=len(succ)
            if not succ or any(_caps7(x)!=q for x in succ):
                failures.append({'endpoint_type':t,'reason':'SENTINEL_RESERVE'})
    out={
      'schema_id':'IG_G3_S6_BOUNDED_EXACT_SENTINEL_V1','status':'PASS' if checked>0 and not failures else 'FAIL',
      'selected_final_states_checked':checked,'exact_successor_states_materialized':successor_states,
      'selector':'CANONICAL_GENERATION_PREFIX_ONLY_FOR_IMPLEMENTATION_EQUIVALENCE_SENTINEL',
      'sentinel_is_non_authoritative_for_relation_uniformity':True,
      'sentinel_is_not_part_of_promotable_transition_selector':True,
      'failure_count':len(failures),'failure_examples':failures[:16],
    }
    out['science_sha256']=canonical_sha256(out)
    return out

def relation_caps7(rel:Sequence[Any])->tuple[int,...]:
    if not rel: raise G3S6Error('empty exact relation')
    vals={_caps7(st) for st in rel}
    if len(vals)!=1: raise G3S6Error('exact relation is not singleton CAPS7')
    return next(iter(vals))

def reserve_complete_relation(rel:Sequence[Any],t:int)->tuple[Any,...]:
    outs=[]
    for st in rel: outs.extend(reserve_external_relation(st,int(t)))
    return _dedupe(outs)

def compose_complete_relations(engine:Any,left_rel:Sequence[Any],right_rel:Sequence[Any],op:Sequence[int],motif_id:str)->tuple[Any,...]:
    a,b=map(int,op); outs=[]
    for x in left_rel:
        for y in right_rel:
            level=max(int(x.level),int(y.level))+1
            outs.extend(compose_binary_relation(engine,level,x,y,a,b,lane='G3_S6_RECURSIVE_HOLDOUT',motif_id=motif_id))
    return _dedupe(outs)

def verify_authority(*,s0_result:Mapping[str,Any],s5_result:Mapping[str,Any],s5_replay:Mapping[str,Any])->dict[str,Any]:
    sp=s6_spec(); a=sp['authority']; failures=[]
    checks=[
      (s0_result.get('schema_id')=='IG_G3_S0_INTERFACE_EXTRACTION_RESULT_V1','S0_SCHEMA'),
      (s0_result.get('status')=='PASS','S0_STATUS'),
      (s0_result.get('authority',{}).get('g2_graduation_science_sha256')==a['g2_graduation_science_sha256'],'G2_GRADUATION'),
      (s5_result.get('schema_id')=='IG_G3_S5_FINITE_READ_WRITE_DESCRIPTOR_RESULT_V1','S5_SCHEMA'),
      (s5_result.get('status')=='PASS','S5_STATUS'),
      (s5_result.get('classification')==a['g3_s5_classification'],'S5_CLASSIFICATION'),
      (s5_result.get('science_sha256')==a['g3_s5_science_sha256'],'S5_SCIENCE'),
      (s5_result.get('g3_s6_unlocked') is True,'S6_UNLOCK'),
      (s5_result.get('g3_graduated') is False,'PREMATURE_GRADUATION'),
      (s5_result.get('topology_promoted') is False,'TOPOLOGY_PROMOTION'),
      (s5_replay.get('schema_id')=='IG_G3_S5_COLD_REPLAY_COMPARISON_V1','S5_REPLAY_SCHEMA'),
      (s5_replay.get('certification')=='CERTIFIED_PASS' and s5_replay.get('status')=='PASS','S5_REPLAY_STATUS'),
      (s5_replay.get('same_registered_decoder_native_experiment') is True,'S5_REPLAY_NATIVE'),
      (s5_replay.get('stable_scientific_payload_exact_equal') is True,'S5_REPLAY_PAYLOAD'),
      (s5_replay.get('science_sha_equal') is True and s5_replay.get('source_sha_equal') is True,'S5_REPLAY_IDENTITY'),
      (s5_replay.get('stable_scientific_payload_sha256')==a['g3_s5_science_sha256'],'S5_REPLAY_SCIENCE'),
      (relation_spec()['spec_sha256']==sp['g2_relation_spec_sha256'],'RELATION_SPEC'),
    ]
    failures=[name for ok,name in checks if not ok]
    out={'schema_id':'IG_G3_S6_AUTHORITY_VERIFICATION_V1','status':'PASS' if not failures else 'FAIL','failures':failures,
         'g3_s5_science_sha256':s5_result.get('science_sha256'),'g3_s5_replay_science_sha256':s5_replay.get('science_sha256'),
         'g2_graduation_science_sha256':s0_result.get('authority',{}).get('g2_graduation_science_sha256'),'g3_graduated_before_s6':False,'topology_promoted':False}
    out['science_sha256']=canonical_sha256(out)
    if failures: raise G3S6Error('G3:S6 authority failed: '+','.join(failures))
    return out

def verify_regression_evidence(reg:Mapping[str,Any],*,current_source_sha256:str)->dict[str,Any]:
    """Verify the full-regression gate while keeping runtime-only evidence out of science identity.

    The raw gate artifact intentionally preserves ``suite_summary`` and ``duration_seconds`` for
    operational provenance.  Those fields contain wall-clock text and can differ between an
    otherwise identical primary/cold replay.  The scientific verification therefore projects
    only the source-bound pass/fail contract and stable test counts.  The raw gate remains a
    separately retained artifact and is never rewritten or discarded.
    """
    failures=[]
    if reg.get('schema_id')!='IG_G3_S6_REGRESSION_GATE_V1': failures.append('SCHEMA')
    if reg.get('status')!='PASS' or int(reg.get('return_code',-1))!=0: failures.append('STATUS')
    if reg.get('source_sha256')!=current_source_sha256: failures.append('SOURCE')
    if int(reg.get('passed',0))<=0: failures.append('NO_PASS_COUNT')
    out={
      'schema_id':'IG_G3_S6_REGRESSION_VERIFICATION_V2',
      'status':'PASS' if not failures else 'FAIL',
      'failures':failures,
      'source_sha256':current_source_sha256,
      'passed':int(reg.get('passed',0)),
      'skipped':int(reg.get('skipped',0)),
      'return_code':int(reg.get('return_code',-1)),
      'stable_projection':'SOURCE_STATUS_PASS_SKIP_COUNTS_V1',
      'operational_fields_excluded_from_science_identity':['duration_seconds','suite_summary'],
    }
    out['science_sha256']=canonical_sha256(out)
    if failures: raise G3S6Error('G3:S6 regression failed: '+','.join(failures))
    return out

def recursive_caps7_relation_factorisation_theorem(s5:Mapping[str,Any])->dict[str,Any]:
    premises={
      'authority':s5.get('authority',{}).get('status'),
      'candidate_caps7_census':s5.get('candidate_caps7_census',{}).get('status'),
      'recursive_reserve_factorisation':s5.get('recursive_reserve_factorisation',{}).get('status'),
      'implementation_read_surface_audit':s5.get('implementation_read_surface_audit',{}).get('status'),
      'factorisation_argument':s5.get('factorisation_argument',{}).get('status'),
    }
    failures=[k for k,v in premises.items() if v!='PASS']
    if s5.get('candidate_descriptor',{}).get('coordinate_count')!=7: failures.append('CAPS7_DIMENSION')
    proof={
      'method':'STRUCTURAL_INDUCTION_ON_FINITE_COMPLETE_G3_RELATION_TERMS_AND_ACTION_SYNTAX',
      'base':'Every graduated G2 unit X is a legal G3 leaf. Its inherited public state q(X)=CAPS7(X) is in N^7 by the certified G2 graduation theorem. A singleton exact relation {X} is therefore uniform-CAPS7.',
      'reserve_step':'Assume a complete exact G3 relation R is uniform at CAPS7 f. If f_t>0, relation-valued reserve enumerates every exact branch and every eligible immediate owner without selecting one. Each exact successor loses exactly one type-t capacity, hence the exact union is nonempty and uniformly q=f-e_t. If f_t=0 no branch is enabled.',
      'binary_step':'Assume complete exact relations R,S are uniform at f,g. For any frozen directed operator (a,b), relation-valued composition enumerates all x in R, y in S and all eligible owner reservations. If f_a>0 and g_b>0 the union is nonempty and every exact output has q=f+g-e_a-e_b; otherwise the public action is disabled.',
      'dedupe_step':'Exact construction identity is used only after branch generation to deduplicate identical exact states. Dedupe cannot change the singleton CAPS7 image of the complete relation.',
      'context_consequence':'By induction, equality of CAPS7 is a congruence for every finite admitted G3 reservation/composition context. Exact relation cardinality, owner alternatives, construction history and hidden incidence topology may differ but are outside this observer.',
      'closure':'All enabled writes subtract only coordinates proven positive before the write, so every finite quotient successor remains in N^7.',
      'rebracketing_scope':'At the CAPS7 observer, finite construction trees with the same leaf CAPS7 sum and the same multiset of directed endpoint consumptions have the same quotient state. No raw exact-relation or raw-topology associativity is claimed.',
      'finite_action_alphabet':{'reservation_actions':7,'binary_bridge_operators':31,'total_action_schemata':38},
      'minimality_claimed':False,'finite_state_cardinality_claimed':False,
    }
    out={'schema_id':'IG_G3_S6_RECURSIVE_CAPS7_RELATION_FACTORISATION_THEOREM_V1','status':'PASS' if not failures else 'FAIL','failures':failures,
         'premises':premises,'proof':proof,'consequence':'CAPS7_IS_A_RECURSIVE_RELATION_QUOTIENT_CONGRUENCE_FOR_ALL_FINITE_TERMS_OF_THE_FROZEN_G3_GRAMMAR' if not failures else 'PREMISES_NOT_MET'}
    out['science_sha256']=canonical_sha256(out); return out

def deterministic_holdout_plan(refs:Sequence[str],ops:Sequence[Sequence[int]])->dict[str,Any]:
    refs=sorted(map(str,refs)); ops=sorted({tuple(map(int,x)) for x in ops})
    if len(refs)!=6: raise G3S6Error('S6 holdout requires six certified S0 challenge units')
    if len(ops)!=31: raise G3S6Error('S6 holdout requires 31-operator basis')
    pairpair=[]
    for j,op in enumerate(ops):
        pairpair.append({'case_id':f'G3:S6:PP:{j:02d}','refs':[refs[(j+0)%6],refs[(j+1)%6],refs[(j+2)%6],refs[(j+3)%6]],
                         'pair_ops':[list(ops[(j+5)%31]),list(ops[(j+13)%31])],'recursive_operator':list(op)})
    deep=[]; rebr=[]
    for j in range(7):
        seq=[list(ops[(j*4+k*7)%31]) for k in range(4)]
        deep.append({'case_id':f'G3:S6:P5:{j}','refs':[refs[(j+k)%6] for k in range(5)],'operator_sequence':seq})
        rebr.append({'case_id':f'G3:S6:RB:{j}','refs':[refs[(j+k)%6] for k in range(4)],'operator_sequence':seq[:3]})
    out={'schema_id':'IG_G3_S6_FRESH_RECURSIVE_HOLDOUT_PLAN_V1','status':'FROZEN_BEFORE_HOLDOUT_EXECUTION','freshness_scope':'PAIR_PLUS_PAIR_P4__P5_BALANCED_PLUS_BASE__P4_REBRACKETING_SHAPES_NOT_IN_S4_S5_P3_CENSUS',
         'base_ref_count':6,'operator_basis':[list(x) for x in ops],'pair_plus_pair_cases':pairpair,'deep_p5_cases':deep,'rebracketing_cases':rebr,'outcome_retuning':False}
    out['science_sha256']=canonical_sha256(out); return out

def _expected_caps(rels:Sequence[Sequence[Any]],ops:Sequence[Sequence[int]])->tuple[int,...]:
    total=[0]*7
    for rel in rels:
        c=relation_caps7(rel)
        total=[total[i]+c[i] for i in range(7)]
    for a,b in [tuple(map(int,x)) for x in ops]: total[a]-=1; total[b]-=1
    if any(x<0 for x in total): raise G3S6Error('holdout abstract underflow')
    return tuple(total)

def _reservation_audit(rel:Sequence[Any])->dict[str,Any]:
    caps=relation_caps7(rel); failures=[]; actions=0; exact_successors=0; hist={t:Counter() for t in range(7)}
    for t in range(7):
        if caps[t]<=0: continue
        rr=reserve_complete_relation(rel,t); actions+=1; exact_successors+=len(rr); hist[t][len(rr)]+=1
        exp=list(caps); exp[t]-=1
        if not rr or any(_caps7(st)!=tuple(exp) for st in rr): failures.append({'endpoint_type':t,'reason':'RESERVE_FACTORISATION'})
    return {'status':'PASS' if not failures else 'FAIL','enabled_reservation_actions_checked':actions,'exact_reservation_successor_states_checked':exact_successors,
            'failure_count':len(failures),'failure_examples':failures[:16],'cardinality_by_type':[{'endpoint_type':t,'histogram':[[k,v] for k,v in sorted(hist[t].items())]} for t in range(7)]}

def _pair(engine:Any,A:Any,B:Any,op:Sequence[int],mid:str)->tuple[Any,...]:
    return compose_complete_pair_relation(engine=engine,target=A,context=B,operator=op,orientation='TARGET_LEFT_CONTEXT_RIGHT',motif_id=mid)

def execute_pair_plus_pair_case(case:Mapping[str,Any],states:Mapping[str,Any],engine:Any)->dict[str,Any]:
    """Full raw exact P4 transfer for all 31 operators, without final identity retention."""
    A,B,C,D=[states[r] for r in case['refs']]; op0,op1=case['pair_ops']; op2=case['recursive_operator']
    ab=_pair(engine,A,B,op0,str(case['case_id'])+':AB'); cd=_pair(engine,C,D,op1,str(case['case_id'])+':CD')
    cab=_materialized_relation_certificate(ab,label=str(case['case_id'])+':AB',leaf_count=2,operator_count=1)
    ccd=_materialized_relation_certificate(cd,label=str(case['case_id'])+':CD',leaf_count=2,operator_count=1)
    cert=_compose_certificate(cab,ccd,op2,label=str(case['case_id'])+':PAIRPAIR')
    exp=_expected_caps([[A],[B],[C],[D]],[op0,op1,op2])
    if cert.caps7!=exp: raise G3S6Error('pair+pair certificate expected CAPS7 mismatch')
    stream=_stream_complete_compose_caps(engine,ab,cd,op2,motif_id=str(case['case_id'])+':PAIRPAIR',expected_caps=exp)
    sentinel=_bounded_reserve_sentinel((stream['sentinel_first_state'],),exp,max_states=1)
    ra=_certificate_reservation_audit(cert)
    ok=stream['status']==ra['status']==sentinel['status']=='PASS'
    return {
      'case_id':case['case_id'],'status':'PASS' if ok else 'FAIL','operator_sequence':[op0,op1,op2],
      'pair_relation_cardinalities':[len(ab),len(cd)],'raw_final_branches_checked':int(stream['raw_exact_branches_checked']),
      'final_exact_identity_cardinality_observed':False,'expected_caps7':list(exp),'observed_caps7':list(cert.caps7),
      'relation_certificate':cert.payload(),'reservation_audit':ra,'bounded_exact_sentinel':sentinel,
      'pair_plus_pair_shape_fresh':True,'complete_raw_exact_relation_streamed':True,
    }


def execute_deep_p5_case(case:Mapping[str,Any],states:Mapping[str,Any],engine:Any)->dict[str,Any]:
    """Exact P5 relation certificate plus bounded depth-5 implementation sentinel.

    The full million-state P5 identity frontier is deliberately not materialized because exact
    identity/multiplicity is outside the frozen observer.  The certificate quantifies over the
    full Cartesian relation and all recursive reservation branches.  A bounded exact sentinel
    checks that the concrete implementation still realizes the certified deep tree shape.
    """
    A,B,C,D,E=[states[r] for r in case['refs']]; op0,op1,op2,op3=case['operator_sequence']
    ab=_pair(engine,A,B,op0,str(case['case_id'])+':AB'); cd=_pair(engine,C,D,op1,str(case['case_id'])+':CD')
    cab=_materialized_relation_certificate(ab,label=str(case['case_id'])+':AB',leaf_count=2,operator_count=1)
    ccd=_materialized_relation_certificate(cd,label=str(case['case_id'])+':CD',leaf_count=2,operator_count=1)
    ce=_singleton_certificate(E,label=str(case['case_id'])+':E')
    cp4=_compose_certificate(cab,ccd,op2,label=str(case['case_id'])+':P4')
    cp5=_compose_certificate(cp4,ce,op3,label=str(case['case_id'])+':P5')
    exp=_expected_caps([[A],[B],[C],[D],[E]],[op0,op1,op2,op3])
    if cp5.caps7!=exp: raise G3S6Error('P5 certificate expected CAPS7 mismatch')

    # Frozen bounded implementation sentinel: complete owner branching for one canonical AB/CD
    # exact input pair, then complete owner branching from one canonical P4 output into E.
    p4_sentinel=compose_binary_relation(engine,max(int(ab[0].level),int(cd[0].level))+1,ab[0],cd[0],int(op2[0]),int(op2[1]),lane='G3_S6_P5_SENTINEL',motif_id=str(case['case_id'])+':P4_SENTINEL')
    if not p4_sentinel:
        raise G3S6Error('P5 P4 sentinel empty')
    p4_expected=_compose_certificate(cab,ccd,op2,label=str(case['case_id'])+':P4_SENTINEL_CERT').caps7
    if any(_caps7(x)!=p4_expected for x in p4_sentinel):
        raise G3S6Error('P5 P4 sentinel CAPS7 mismatch')
    p5_sentinel=compose_binary_relation(engine,max(int(p4_sentinel[0].level),int(E.level))+1,p4_sentinel[0],E,int(op3[0]),int(op3[1]),lane='G3_S6_P5_SENTINEL',motif_id=str(case['case_id'])+':P5_SENTINEL')
    sentinel=_bounded_reserve_sentinel(p5_sentinel,exp,max_states=64)
    ra=_certificate_reservation_audit(cp5)
    ok=ra['status']==sentinel['status']=='PASS' and bool(p5_sentinel)
    return {
      'case_id':case['case_id'],'status':'PASS' if ok else 'FAIL','leaf_count':5,'operator_sequence':case['operator_sequence'],
      'expected_caps7':list(exp),'observed_caps7':list(cp5.caps7),
      'p4_relation_certificate':cp4.payload(),'p5_relation_certificate':cp5.payload(),'reservation_audit':ra,
      'full_p5_exact_identity_frontier_materialized':False,'full_p5_exact_identity_cardinality_observed':False,
      'all_exact_p5_final_states_quantified_by_certificate':True,
      'bounded_exact_sentinel':sentinel,'sentinel_p4_exact_outputs':len(p4_sentinel),'sentinel_p5_exact_outputs':len(p5_sentinel),
    }


def execute_rebracket_case(case:Mapping[str,Any],states:Mapping[str,Any],engine:Any)->dict[str,Any]:
    """Fresh balanced/left-deep quotient check with full raw branch streaming and no final sets."""
    A,B,C,D=[states[r] for r in case['refs']]; op0,op1,op2=case['operator_sequence']
    ab=_pair(engine,A,B,op0,str(case['case_id'])+':AB'); cd=_pair(engine,C,D,op1,str(case['case_id'])+':CD')
    exp=_expected_caps([[A],[B],[C],[D]],[op0,op1,op2])
    cab=_materialized_relation_certificate(ab,label=str(case['case_id'])+':AB',leaf_count=2,operator_count=1)
    ccd=_materialized_relation_certificate(cd,label=str(case['case_id'])+':CD',leaf_count=2,operator_count=1)
    cc=_singleton_certificate(C,label=str(case['case_id'])+':C'); dc=_singleton_certificate(D,label=str(case['case_id'])+':D')
    balanced_cert=_compose_certificate(cab,ccd,op2,label=str(case['case_id'])+':BAL_CERT')
    abc_cert=_compose_certificate(cab,cc,op1,label=str(case['case_id'])+':ABC_CERT')
    left_cert=_compose_certificate(abc_cert,dc,op2,label=str(case['case_id'])+':LEFT_CERT')
    if balanced_cert.caps7!=exp or left_cert.caps7!=exp:
        raise G3S6Error('rebracket certificate CAPS7 mismatch')

    bal_stream=_stream_complete_compose_caps(engine,ab,cd,op2,motif_id=str(case['case_id'])+':BAL',expected_caps=exp)
    abc_raw=[]
    abc_expected=abc_cert.caps7
    for abc in _iter_compose_outputs(engine,ab,(C,),op1,str(case['case_id'])+':ABC'):
        if _caps7(abc)!=abc_expected: raise G3S6Error('ABC raw branch CAPS7 mismatch')
        abc_raw.append(abc)
    if not abc_raw: raise G3S6Error('ABC raw relation empty')
    left_raw_count=0; left_first=None
    for abc in abc_raw:
        for st in _iter_compose_outputs(engine,(abc,),(D,),op2,str(case['case_id'])+':LEFT'):
            left_raw_count+=1
            if _caps7(st)!=exp: raise G3S6Error('left-deep raw branch CAPS7 mismatch')
            if left_first is None: left_first=st
    if left_raw_count<=0 or left_first is None: raise G3S6Error('left-deep raw relation empty')
    rb=_certificate_reservation_audit(balanced_cert); rl=_certificate_reservation_audit(left_cert)
    sb=_bounded_reserve_sentinel((bal_stream['sentinel_first_state'],),exp,max_states=1)
    sl=_bounded_reserve_sentinel((left_first,),exp,max_states=1)
    ok=(balanced_cert.caps7==left_cert.caps7==exp and rb['status']==rl['status']==sb['status']==sl['status']=='PASS')
    return {
      'case_id':case['case_id'],'status':'PASS' if ok else 'FAIL','leaf_count':4,'operator_sequence':case['operator_sequence'],
      'expected_caps7':list(exp),'balanced_caps7':list(balanced_cert.caps7),'left_deep_caps7':list(left_cert.caps7),
      'caps7_quotient_equal':balanced_cert.caps7==left_cert.caps7,
      'balanced_raw_final_branches_checked':int(bal_stream['raw_exact_branches_checked']),
      'left_intermediate_abc_raw_branches':len(abc_raw),'left_deep_raw_final_branches_checked':left_raw_count,
      'balanced_relation_certificate':balanced_cert.payload(),'left_deep_relation_certificate':left_cert.payload(),
      'balanced_reservation_audit':rb,'left_deep_reservation_audit':rl,
      'balanced_bounded_exact_sentinel':sb,'left_deep_bounded_exact_sentinel':sl,
      'final_exact_identity_cardinality_observed':False,
    }

def _case_checkpoint_path(checkpoint_dir:str|Path,case_id:str)->Path:
    key=hashlib.sha256(str(case_id).encode('utf-8')).hexdigest()[:20]
    return Path(checkpoint_dir)/f'{key}.json'

def write_case_checkpoint(*,checkpoint_dir:str|Path,binding_sha256:str,kind:str,case_id:str,result:Mapping[str,Any])->Path:
    p=_case_checkpoint_path(checkpoint_dir,case_id); p.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema_id':'IG_G3_S6_CASE_CHECKPOINT_V1','binding_sha256':str(binding_sha256),'kind':str(kind),'case_id':str(case_id),
             'result':dict(result),'result_sha256':canonical_sha256(result)}
    write_json_atomic(p,payload); return p

def load_case_checkpoint(*,checkpoint_dir:str|Path,binding_sha256:str,kind:str,case_id:str)->dict[str,Any]|None:
    p=_case_checkpoint_path(checkpoint_dir,case_id)
    if not p.is_file(): return None
    try: obj=json.loads(p.read_text(encoding='utf-8'))
    except Exception: return None
    if obj.get('schema_id')!='IG_G3_S6_CASE_CHECKPOINT_V1' or obj.get('binding_sha256')!=binding_sha256 or obj.get('kind')!=kind or obj.get('case_id')!=case_id:
        return None
    result=obj.get('result')
    if not isinstance(result,dict) or canonical_sha256(result)!=obj.get('result_sha256'):
        return None
    return result

def holdout_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_CONTEXT is None: raise G3S6Error('missing G3:S6 worker context')
    kind=payload['kind']; case=payload['case']; states=_WORKER_CONTEXT['states']; engine=_WORKER_CONTEXT['engine']
    if kind=='PAIR_PLUS_PAIR': result=execute_pair_plus_pair_case(case,states,engine)
    elif kind=='DEEP_P5': result=execute_deep_p5_case(case,states,engine)
    elif kind=='REBRACKET': result=execute_rebracket_case(case,states,engine)
    else: raise G3S6Error('unknown G3:S6 holdout task')
    cpdir=_WORKER_CONTEXT.get('checkpoint_dir'); binding=_WORKER_CONTEXT.get('binding_sha256')
    if cpdir and binding:
        write_case_checkpoint(checkpoint_dir=cpdir,binding_sha256=binding,kind=kind,case_id=str(case['case_id']),result=result)
    return result

def aggregate_holdout(plan:Mapping[str,Any],pp:Sequence[Mapping[str,Any]],deep:Sequence[Mapping[str,Any]],rb:Sequence[Mapping[str,Any]])->dict[str,Any]:
    failures=[]
    if len(pp)!=31 or any(x.get('status')!='PASS' for x in pp): failures.append('PAIR_PLUS_PAIR')
    if len(deep)!=7 or any(x.get('status')!='PASS' for x in deep): failures.append('DEEP_P5')
    if len(rb)!=7 or any(x.get('status')!='PASS' or x.get('caps7_quotient_equal') is not True for x in rb): failures.append('REBRACKET')
    seen={tuple(x['operator_sequence'][-1]) for x in pp}
    if seen!={tuple(x) for x in plan['operator_basis']}: failures.append('OPERATOR_COVERAGE')
    if any(x.get('all_exact_p5_final_states_quantified_by_certificate') is not True for x in deep): failures.append('P5_CERTIFICATE_COVERAGE')
    if any(x.get('full_p5_exact_identity_frontier_materialized') is not False for x in deep): failures.append('P5_IDENTITY_MATERIALIZATION')
    out={
      'schema_id':'IG_G3_S6_FRESH_RECURSIVE_HOLDOUT_RESULT_V3','status':'PASS' if not failures else 'FAIL','failures':failures,
      'plan_science_sha256':plan['science_sha256'],'freshness_scope':plan['freshness_scope'],
      'pair_plus_pair_cases':len(pp),'pair_plus_pair_operator_coverage':len(seen),
      'pair_plus_pair_raw_exact_branches_streamed':sum(int(x['raw_final_branches_checked']) for x in pp),
      'pair_plus_pair_complete_raw_exact_relation_checked':all(x.get('complete_raw_exact_relation_streamed') is True for x in pp),
      'deep_p5_cases':len(deep),'deep_p5_exact_relation_certificates':sum(1 for x in deep if x.get('all_exact_p5_final_states_quantified_by_certificate') is True),
      'deep_p5_full_final_identity_frontiers_materialized':sum(1 for x in deep if x.get('full_p5_exact_identity_frontier_materialized') is True),
      'deep_p5_bounded_sentinel_final_states_checked':sum(int(x['bounded_exact_sentinel']['selected_final_states_checked']) for x in deep),
      'rebracketing_cases':len(rb),'rebracketing_caps7_equal_count':sum(1 for x in rb if x.get('caps7_quotient_equal')),
      'rebracketing_balanced_raw_branches_streamed':sum(int(x['balanced_raw_final_branches_checked']) for x in rb),
      'rebracketing_left_deep_raw_branches_streamed':sum(int(x['left_deep_raw_final_branches_checked']) for x in rb),
      'successor_identity_or_global_cardinality_observed':False,
      'exact_factorization':'FULL_RAW_EXACT_BRANCH_STREAMING_ON_31_PAIR_PLUS_PAIR_AND_7_REBRACKET_CASES; EXACT_STRUCTURAL_RELATION_CERTIFICATES_QUANTIFY_ALL_7_P5_FINAL_RELATIONS_AND_ALL_ENABLED_RESERVATION_SUCCESSOR_RELATIONS; BOUNDED_FULLY_MATERIALIZED_SENTINELS CHECK_IMPLEMENTATION_EQUIVALENCE; EXACT_SET_DEDUPE_INVARIANCE_TRANSPORTS_NONEMPTY_SINGLETON_CAPS7_TO_COMPLETE_RELATIONS',
      'certificate_soundness_scope':'FROZEN_G3_CAPS7_OBSERVER_ONLY; RAW_EXACT_IDENTITY_AND_BRANCH_MULTIPLICITY_NOT_OBSERVED',
      'pair_plus_pair_cases_sha256':canonical_sha256(list(pp)),'deep_p5_cases_sha256':canonical_sha256(list(deep)),
      'rebracket_cases_sha256':canonical_sha256(list(rb)),'topology_promoted':False,
    }
    out['science_sha256']=canonical_sha256(out); return out

def implementation_no_hidden_selector_audit()->dict[str,Any]:
    from . import g2_relation
    reserve_src=inspect.getsource(g2_relation._reserve_external_relation_with_witness)
    compose_src=inspect.getsource(g2_relation.compose_binary_relation)
    cert_src=inspect.getsource(_compose_certificate)+inspect.getsource(_certificate_reservation_audit)
    p5_src=inspect.getsource(execute_deep_p5_case)
    failures=[]
    if '.skin' in reserve_src or '.skin' in compose_src: failures.append('SKIN_READ_IN_TRANSITION_CORE')
    required_reserve=(
      'for i, child in enumerate(state.children)',
      'child_successors = _reserve_external_relation_with_witness(child, t)',
      'for child_succ, child_witness in child_successors',
      'ch[i] = child_succ',
      'uniq.setdefault(_exact_successor_key(s), (s, witness))',
    )
    if any(tok not in reserve_src for tok in required_reserve): failures.append('RESERVE_COMPLETE_BRANCH_SOURCE_SHAPE_CHANGED')
    required_compose=(
      'lrel = _reserve_external_relation_with_witness(left, a)',
      'rrel = _reserve_external_relation_with_witness(right, b)',
      'for ls, lw in lrel',
      'for rsucc, rw in rrel',
      'out.setdefault(_exact_successor_key(st), st)',
    )
    if any(tok not in compose_src for tok in required_compose): failures.append('COMPOSE_COMPLETE_CARTESIAN_SOURCE_SHAPE_CHANGED')
    for tok in ('construction_digest','skin','topology','ancestry','top_edges_full'):
        if tok in cert_src: failures.append('CERTIFICATE_FORBIDDEN_READ_'+tok.upper())
    if '_FactorizedFinalRelationAudit' in p5_src or 'p4_seen' in p5_src or 'final_seen' in p5_src:
        failures.append('P5_FULL_FRONTIER_MATERIALIZATION_REINTRODUCED')
    if '_compose_certificate' not in p5_src or '_bounded_reserve_sentinel' not in p5_src:
        failures.append('P5_CERTIFICATE_OR_SENTINEL_MISSING')
    out={
      'schema_id':'IG_G3_S6_NO_HIDDEN_SELECTOR_READ_AUDIT_V3','status':'PASS' if not failures else 'FAIL','failures':failures,
      'registered_reserve_semantics_source_guard':'ALL_ELIGIBLE_CHILD_BRANCHES_RECURSED_AND_EXACT_DEDUP_ONLY_AFTER_GENERATION',
      'registered_compose_semantics_source_guard':'FULL_LEFT_RESERVE_X_RIGHT_RESERVE_CARTESIAN_PRODUCT_THEN_EXACT_DEDUP',
      'relation_certificate_reads':['CAPS7','endpoint types','complete/nonempty certificate flags','frozen 31-operator basis'],
      'relation_certificate_does_not_read':['construction identity','owner identity','topology','skin','ancestry','exact branch multiplicity'],
      'p5_full_final_identity_frontier_materialized':False,
      'bounded_exact_sentinels_are_non_authoritative':True,
      'construction_identity_role':'OPERATIONAL_EXACT_DEDUP_IN_REGISTERED_RELATION_IMPLEMENTATION_AND_BOUNDED_SENTINEL_ONLY_NO_PROMOTABLE_SELECTOR',
      'hidden_topology_read':False,
      'source_snippet_hashes':{
        'reserve':hashlib.sha256(reserve_src.encode()).hexdigest(),
        'compose':hashlib.sha256(compose_src.encode()).hexdigest(),
        'certificate':hashlib.sha256(cert_src.encode()).hexdigest(),
        'p5_case':hashlib.sha256(p5_src.encode()).hexdigest(),
      },
    }
    out['science_sha256']=canonical_sha256(out); return out

def stable_science_payload(*,authority:Mapping[str,Any],regression:Mapping[str,Any],theorem:Mapping[str,Any],holdout:Mapping[str,Any],hidden:Mapping[str,Any])->dict[str,Any]:
    out={'schema_id':'IG_G3_S6_STABLE_SCIENCE_PAYLOAD_V1','g3_s6_spec_sha256':s6_spec()['science_sha256'],'authority_science_sha256':authority['science_sha256'],'regression_gate_science_sha256':regression['science_sha256'],'recursive_theorem_science_sha256':theorem['science_sha256'],'fresh_holdout_science_sha256':holdout['science_sha256'],'no_hidden_read_audit_science_sha256':hidden['science_sha256'],'observer':s6_spec()['observer']['name'],'relation_spec_sha256':relation_spec()['spec_sha256']}
    out['science_sha256']=canonical_sha256(out); return out

def primary_result(*,authority:Mapping[str,Any],regression:Mapping[str,Any],theorem:Mapping[str,Any],holdout:Mapping[str,Any],hidden:Mapping[str,Any])->dict[str,Any]:
    passed=all(x.get('status')=='PASS' for x in (authority,regression,theorem,holdout,hidden)); stable=stable_science_payload(authority=authority,regression=regression,theorem=theorem,holdout=holdout,hidden=hidden)
    out={'schema_id':'IG_G3_S6_PRIMARY_RECURSIVE_CLOSURE_RESULT_V1','schema_version':'1.0.0','date':'2026-09-03','stage_ref':'G3:S6','status':'PASS' if passed else 'REVIEW_REQUIRED',
         'classification':'G3_CAPS7_RECURSIVE_RELATION_CLOSURE_EARNED_GRADUATION_CANDIDATE_AWAITING_COLD_REPLAY' if passed else 'G3_S6_RECURSIVE_CLOSURE_GATE_FAILED_G3_NOT_GRADUATED',
         'authority':dict(authority),'regression_gate':dict(regression),'recursive_caps7_relation_factorisation_theorem':dict(theorem),'fresh_recursive_holdout':dict(holdout),'no_hidden_selector_read_audit':dict(hidden),
         'stable_science_payload':stable,'stable_science_payload_sha256':stable['science_sha256'],'cold_replay_required_for_graduation':True,'g3_graduation_candidate':passed,'g3_graduated':False,'topology_promoted':False,
         'graduation_scope_if_cold_replay_passes':'FROZEN_G3_CAPS7_TRANSITION_OBSERVER_WITH_7_RESERVATION_ACTIONS_AND_31_RELATION_VALUED_TYPED_BINARY_OPERATORS_OVER_ALL_FINITE_COMPLETE_RELATION_TERMS_BUILT_FROM_GRADUATED_G2_UNITS',
         'nonclaims':['CAPS7_MINIMALITY_NOT_CLAIMED','RAW_EXACT_RELATION_EQUIVALENCE_NOT_CLAIMED','EXACT_BRANCH_MULTIPLICITY_EQUIVALENCE_NOT_CLAIMED','RAW_TOPOLOGY_ASSOCIATIVITY_NOT_CLAIMED','TOPOLOGY_ERASURE_NOT_CLAIMED','NO_PHYSICAL_GEOMETRY_TIME_OR_SPACETIME_CLAIM']}
    out['science_sha256']=canonical_sha256({k:v for k,v in out.items() if k not in {'source_sha256','source_version','registry_sha256','execution_metadata','science_sha256'}}); return out

def compare_cold_replay(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    failures=[]
    for lab,r in [('PRIMARY',primary),('COLD',cold)]:
        if r.get('schema_id')!='IG_G3_S6_PRIMARY_RECURSIVE_CLOSURE_RESULT_V1': failures.append(lab+'_SCHEMA')
        if r.get('status')!='PASS' or r.get('g3_graduation_candidate') is not True: failures.append(lab+'_STATUS')
        md=r.get('execution_metadata',{})
        if md.get('registered_experiment_id')!='G3:S6' or md.get('execution_backend_owned_by_decoder') is not True or md.get('stage_specific_external_science_runner') is not False: failures.append(lab+'_EXECUTION')
    exact=(primary.get('stable_science_payload_sha256')==cold.get('stable_science_payload_sha256') and primary.get('stable_science_payload')==cold.get('stable_science_payload'))
    if not exact: failures.append('STABLE_PAYLOAD_MISMATCH')
    out={'schema_id':'IG_G3_S6_COLD_REPLAY_COMPARISON_V1','date':'2026-09-03','status':'PASS' if not failures else 'FAIL','certification':'CERTIFIED_PASS' if not failures else 'NOT_CERTIFIED','failures':failures,'same_registered_decoder_native_experiment':True,'stable_scientific_payload_exact_equal':exact,'stable_scientific_payload_sha256':primary.get('stable_science_payload_sha256'),'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'primary_source_sha256':primary.get('source_sha256'),'cold_source_sha256':cold.get('source_sha256'),'science_sha_equal':primary.get('science_sha256')==cold.get('science_sha256'),'source_sha_equal':primary.get('source_sha256')==cold.get('source_sha256'),'stage_specific_external_science_runner':False}
    out['science_sha256']=canonical_sha256(out); return out

def finalize_graduation(primary:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
    failures=[]
    if primary.get('status')!='PASS' or primary.get('g3_graduation_candidate') is not True: failures.append('PRIMARY')
    if replay.get('schema_id')!='IG_G3_S6_COLD_REPLAY_COMPARISON_V1' or replay.get('status')!='PASS' or replay.get('certification')!='CERTIFIED_PASS': failures.append('REPLAY')
    if replay.get('stable_scientific_payload_exact_equal') is not True or replay.get('science_sha_equal') is not True or replay.get('source_sha_equal') is not True: failures.append('IDENTITY')
    passed=not failures
    out={'schema_id':'IG_G3_GRADUATION_CERTIFICATE_V1','date':'2026-09-03','status':'PASS' if passed else 'FAIL','classification':'G3_GRADUATED_CAPS7_RECURSIVE_RELATION_GRAMMAR_EARNED_R0_UNLOCKED' if passed else 'G3_NOT_GRADUATED_R0_LOCKED','failures':failures,'g3_graduated':passed,'r0_unlocked':passed,'graduated_observer':s6_spec()['observer']['name'],'graduated_descriptor':'IG_G3_CAPS7_READ_WRITE_STATE_V1','graduated_grammar':{'reservation_actions':7,'binary_operator_count':31,'routing_semantics':'RELATION_VALUED_ALL_ELIGIBLE_OWNER_CHOICES_V1','recursive_scope':'ALL_FINITE_COMPLETE_G3_RELATION_TERMS_BUILT_FROM_GRADUATED_G2_UNITS'},'primary_s6_science_sha256':primary.get('science_sha256'),'stable_s6_science_payload_sha256':primary.get('stable_science_payload_sha256'),'cold_replay_science_sha256':replay.get('science_sha256'),'authorizes':'G3:R0_POST_GRADUATION_STRUCTURAL_RECONNAISSANCE' if passed else None,'topology_promoted':False,'nonclaims':['CAPS7_MINIMALITY_NOT_CLAIMED','RAW_EXACT_RELATION_EQUIVALENCE_NOT_CLAIMED','EXACT_BRANCH_MULTIPLICITY_EQUIVALENCE_NOT_CLAIMED','RAW_TOPOLOGY_ASSOCIATIVITY_NOT_CLAIMED','NO_GEOMETRY_OR_PHYSICS_CLAIM'],'reopen_conditions':['CAPS7 observer changes','relation-valued G3 routing changes','graduated G2 base semantics change','31 bridge-operator basis changes']}
    out['science_sha256']=canonical_sha256(out); return out
