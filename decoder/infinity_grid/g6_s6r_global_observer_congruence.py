from __future__ import annotations

"""Registered G6:S6R global observer-congruence theorem audit.

This operation is deliberately proof-first and fail-closed. It does not turn a
finite counterexample search or the S6 holdout panel into a universal theorem.
It verifies the frozen theorem registration, binds the earned CRW0/CRW1/S6R
results, discharges definitional/grammar lemmas, and asks whether the genuinely
new marker-free common-false-parent exclusion lemma has a bound proof object.
"""

import hashlib, json
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .v05_origin_guard import require_controller_execution_origin

PLAN_SCHEMA='IG_G6_S6R_GLOBAL_OBSERVER_CONGRUENCE_THEOREM_PREREGISTRATION_V1'
RESULT_SCHEMA='IG_G6_S6R_GLOBAL_OBSERVER_CONGRUENCE_THEOREM_RESULT_V1'
STAGE_ID='G6:S6R-GLOBAL'

class G6GlobalCongruenceError(RuntimeError): pass

def _sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _load_json(p:Path)->dict[str,Any]:
    obj=json.loads(p.read_text(encoding='utf-8'))
    if type(obj) is not dict: raise G6GlobalCongruenceError('JSON_OBJECT_REQUIRED')
    return obj

def _question_sha(plan:Mapping[str,Any])->str:
    # The preregistration self-binds by canonical hash with question_sha256 omitted.
    return canonical_sha256({k:v for k,v in plan.items() if k!='question_sha256'})

def run_global_observer_congruence_theorem(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,
                                           accepted_source_sha256:str,internal_execution_id:str)->dict[str,Any]:
    require_controller_execution_origin('g6-s6r-global-observer-congruence')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}
    plan=_load_json(Path(plan_path).resolve(strict=True))
    if plan.get('schema_id')!=PLAN_SCHEMA or plan.get('stage_id')!=STAGE_ID:
        raise G6GlobalCongruenceError('GLOBAL_PLAN_SCHEMA_STAGE')
    if _question_sha(plan)!=plan.get('question_sha256'):
        raise G6GlobalCongruenceError('GLOBAL_PLAN_HASH')
    if (plan.get('declared_leaf_domain') or {}).get('id')!='FOUR_FROZEN_S0_SEEDS_ONLY':
        raise G6GlobalCongruenceError('GLOBAL_LEAF_DOMAIN')
    obs=plan.get('frozen_observer') or {}
    if obs.get('q_definition')!='q(X)=O(X)=Rel_(0,0)(X,D2_PATH)' or obs.get('equality_authority')!='STRUCTURAL_EQUALITY_NOT_HASH_EQUALITY':
        raise G6GlobalCongruenceError('GLOBAL_OBSERVER_BINDING')

    auth=plan.get('authority') or {}
    for logical,key in [('crw0_result','crw0_result_file_sha256'),('crw1_result','crw1_result_file_sha256'),('s6r_result','s6r_a3_result_file_sha256')]:
        if _sha_file(paths[logical])!=auth.get(key):
            raise G6GlobalCongruenceError('GLOBAL_DEP_'+logical)
    crw0=_load_json(paths['crw0_result']); crw1=_load_json(paths['crw1_result']); s6=_load_json(paths['s6r_result'])
    if crw0.get('classification')!='PASS_OBSERVER_INVERSION_ON_DECLARED_SCOPE_REPRESENTATION_AMENDMENT_REVIEW_NEXT' or int(crw0.get('decode_failure_count',-1))!=0:
        raise G6GlobalCongruenceError('GLOBAL_CRW0_AUTHORITY')
    if crw1.get('classification')!='PASS_S5_OBSERVER_STATE_AND_WRITE_LAW_EARNED_ON_CERTIFIED_SCOPE' or crw1.get('descriptor_earned') is not True:
        raise G6GlobalCongruenceError('GLOBAL_CRW1_AUTHORITY')
    if s6.get('classification')!='BUDGET_INSUFFICIENT_GLOBAL_OBSERVER_CONGRUENCE_LEMMA_UNPROVED' or s6.get('recursive_closure_candidate_earned') is not False:
        raise G6GlobalCongruenceError('GLOBAL_S6R_AUTHORITY')

    # Proof audit. L1 and L5 are definitional consequences of the exact graft
    # construction and the declared generated-term grammar. L3/L4/L6 depend on L2.
    proof_object=paths.get('l2_proof_object')
    l2_status='UNRESOLVED'; l2_reason='NO_BOUND_FORMAL_OR_MACHINE_CHECKABLE_PROOF_OBJECT_FOR_COMMON_FALSE_PARENT_EXCLUSION'
    l2_digest=None
    if proof_object is not None:
        pobj=_load_json(proof_object)
        l2_digest=_sha_file(proof_object)
        # No proof format is admitted by this source version. Presence alone must
        # never promote a universal theorem.
        l2_reason='PROOF_OBJECT_PRESENT_BUT_NO_REGISTERED_L2_PROOF_CHECKER_IN_ACCEPTED_SOURCE'

    obligations=[
      {'id':'L1_TRUE_PARENT_SURVIVES','status':'PASS','basis':'DEFINITIONAL_EXACT_GRAFT_BRIDGE_DELETION'},
      {'id':'L2_COMMON_FALSE_PARENT_EXCLUSION','status':l2_status,'reason':l2_reason,'proof_object_sha256':l2_digest},
      {'id':'L3_GLOBAL_Q_INJECTIVITY','status':'UNRESOLVED','depends_on':['L2_COMMON_FALSE_PARENT_EXCLUSION']},
      {'id':'L4_WRITE_CONGRUENCE','status':'UNRESOLVED','depends_on':['L3_GLOBAL_Q_INJECTIVITY']},
      {'id':'L5_RECURSIVE_DOMAIN_PRESERVATION','status':'PASS','basis':'DECLARED_FINITE_GENERATED_TERM_GRAMMAR_CLOSED_UNDER_LEGAL_BINARY_COMPOSITION'},
      {'id':'L6_STRUCTURAL_INDUCTION','status':'UNRESOLVED','depends_on':['L3_GLOBAL_Q_INJECTIVITY','L4_WRITE_CONGRUENCE','L5_RECURSIVE_DOMAIN_PRESERVATION']},
    ]
    classification='GLOBAL_Q_CONGRUENCE_THEOREM_UNRESOLVED'
    result={
      'schema_id':RESULT_SCHEMA,'status':'REVIEW_REQUIRED','stage_id':STAGE_ID,
      'accepted_decoder_source_sha256':accepted_source_sha256,'internal_execution_id':internal_execution_id,
      'question_sha256':plan['question_sha256'],'declared_leaf_domain':plan['declared_leaf_domain'],
      'frozen_observer':obs,'proof_obligations':obligations,'classification':classification,
      'global_q_congruence_theorem_earned':False,'global_q_congruence_theorem_refuted':False,
      'finite_holdouts_used_as_universal_proof':False,'g6_public_layer_graduated':False,'g6_r0_started':False,
      'next_authorized':plan['next_on_unresolved'],
      'scientific_interpretation':'Current authority proves true-parent survival and generated-domain closure, and supplies strong finite inversion/write evidence, but it does not contain a machine-checkable proof of the marker-free common-false-parent exclusion lemma. The universal theorem therefore remains unresolved; no finite evidence is promoted to a theorem.',
      'nonclaims':plan['target_theorem']['nonclaims'],
    }
    result['result_sha256']=canonical_sha256(result)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out/'G6_S6R_GLOBAL_OBSERVER_CONGRUENCE_RESULT.json',result)
    (out/'READ_FIRST.txt').write_text('INFINITY GRID — G6:S6R GLOBAL OBSERVER CONGRUENCE THEOREM\n\n'+classification+'\n\nCritical unresolved lemma: L2_COMMON_FALSE_PARENT_EXCLUSION\nResult SHA-256: '+result['result_sha256']+'\n',encoding='utf-8')
    return result
