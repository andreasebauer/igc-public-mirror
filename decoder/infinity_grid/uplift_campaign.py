from __future__ import annotations

"""Persistent native G-uplift campaign engine (v0.30.6).

The campaign process owns one lower-G materialization session. Scientific stages are
registered resources, not chat-created runners. Decomposable work is expressed as
TaskSpecs and routed through infinity_grid.execution.execute_tasks.
"""
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import gzip, hashlib, json, os, re, time

from .canon import canonical_sha256, write_json_atomic
from .execution import ExecutionPolicy, TaskSpec, execute_tasks
from .runtime_telemetry import RuntimeTelemetry
from .materialized_discovery import MaterializedDiscoverySession
from .maturation_parallel import parallel_candidate_descriptors, rebuild_selected_states
from .records import source_sha256, utc_now
from . import regime_scanner as rs
from .g2_relation import post_reservation_public_branch_relation_projection, reserve_external_relation, compose_binary_relation, relation_spec
from .uplift_s4 import verify_s4_authority, index_s2_collision_records, resource_legality_closure, d4_composition_congruence, composition_closure_result, s4_implementation_spec
from .uplift_s5 import verify_s5_authority, verify_s0_base_reservation_law, scan_certified_pair_caps7, verify_s4_operator_caps7_factorisation, implementation_read_surface_audit, caps7_factorisation_argument, s5_result, s5_implementation_spec
from .uplift_toric import run_g2_toric_native
from .uplift_r0 import run_r0_recon, r0_spec
from .uplift_g3_s0 import run_g3_s0, phase0_spec as g3_phase0_spec, s0_spec as g3_s0_spec
from .uplift_g3_s1 import (
    s1_spec as g3_s1_spec, verify_s1_authority, reproduce_s0_challenge_states,
    pair_context_signature, finalize_s1_result,
)
from .uplift_g3_s2 import (
    s2_spec as g3_s2_spec, verify_s2_authority, load_s1_signature_index, certify_caps7_pair_observer_quotient,
)
from .uplift_g3_s3 import (
    s3_spec as g3_s3_spec, verify_s3_authority, ordered_two_reservation_signature, finalize_s3_result,
)
from .uplift_g3_s4 import (
    s4_spec as g3_s4_spec, verify_s4_authority as verify_g3_s4_authority,
    ordered_three_reservation_signature, compose_complete_pair_relation,
    recursive_relation_immediate_signature, recursive_reserve_capability_signature,
    finalize_s4_result as finalize_g3_s4_result,
)
from .uplift_g3_s5 import (
    s5_spec as g3_s5_spec, verify_s5_authority as verify_g3_s5_authority,
    certify_candidate_caps7_census as certify_g3_s5_candidate_caps7_census,
    verify_recursive_reserve_basis as verify_g3_s5_recursive_reserve_basis,
    implementation_read_surface_audit as g3_s5_implementation_read_surface_audit,
    caps7_factorisation_argument as g3_s5_caps7_factorisation_argument,
    finalize_s5_result as finalize_g3_s5_result,
)

from .uplift_g3_s6 import (
    s6_spec as g3_s6_spec, verify_authority as verify_g3_s6_authority,
    verify_regression_evidence as verify_g3_s6_regression_evidence,
    recursive_caps7_relation_factorisation_theorem, deterministic_holdout_plan as g3_s6_holdout_plan,
    aggregate_holdout as aggregate_g3_s6_holdout, implementation_no_hidden_selector_audit as g3_s6_hidden_audit,
    primary_result as g3_s6_primary_result,
)
from . import uplift_g3_s6
from .uplift_g3_r0 import run_g3_r0_recon, r0_spec as g3_r0_spec
from .uplift_g3_r1 import run_g3_r1_fiber_moduli_audit, r1_spec as g3_r1_spec
from .uplift_g3_r2 import (
    r2_spec as g3_r2_spec, verify_r2_authority, dense_adaptive_maturation,
    sentinel_rank_audit_worker, finalize_r2_result,
)
from .uplift_g4_s0 import run_g4_s0, phase0_spec as g4_phase0_spec, s0_spec as g4_s0_spec
from .uplift_g4_s1 import (
    s1_spec as g4_s1_spec, verify_s1_authority as verify_g4_s1_authority,
    load_s0_term_states as load_g4_s0_term_states,
    expand_pair_context_kernel_signature as expand_g4_pair_context_kernel_signature,
    finalize_s1_result as finalize_g4_s1_result,
)

from .uplift_g4_s1_rebase import (
    s1_rebase_spec as g4_s1_rebase_spec, verify_authority as verify_g4_s1_rebase_authority,
    load_states as load_g4_s1_rebase_states, finalize as finalize_g4_s1_rebase,
)
from .uplift_g4_s2 import (
    s2_spec as g4_s2_spec, verify_s2_authority as verify_g4_s2_authority,
    load_s1_signature_index as load_g4_s1_signature_index,
    certify_minimal_added_read as certify_g4_s2_minimal_added_read,
)
from .uplift_g4_s2_rebase import (
    spec as g4_s2_rebase_spec, load_signatures as load_g4_s1_rebase_signatures,
    audit as audit_g4_s2_rebase,
)
from .uplift_g4_s3_rebase import (
    spec as g4_s3_rebase_spec, verify_authority as verify_g4_s3_rebase_authority,
    finalize as finalize_g4_s3_rebase,
)
from .uplift_g4_s4_rebase import (
    spec as g4_s4_rebase_spec, finalize as finalize_g4_s4_rebase,
)
from .uplift_g4_s5_rebase import (
    spec as g4_s5_rebase_spec, finalize as finalize_g4_s5_rebase,
)
from .uplift_g4_s3 import (
    s3_spec as g4_s3_spec, verify_s3_authority as verify_g4_s3_authority,
    finalize_s3_result as finalize_g4_s3_result,
)
from .uplift_g4_s4 import (
    s4_spec as g4_s4_spec, verify_s4_authority as verify_g4_s4_authority,
    build_pair_basis as build_g4_s4_pair_basis, finalize_s4_result as finalize_g4_s4_result,
)
from .uplift_g4_s5 import (
    s5_spec as g4_s5_spec, verify_s5_authority as verify_g4_s5_authority,
    certify_candidate_census as certify_g4_s5_candidate_census,
    verify_reserve_factorisation as verify_g4_s5_reserve_factorisation,
    implementation_read_surface_audit as g4_s5_implementation_read_surface_audit,
    factorisation_argument as g4_s5_factorisation_argument,
    finalize_s5_result as finalize_g4_s5_result,
)
from .uplift_g4_s6 import (
    s6_spec as g4_s6_spec, verify_authority as verify_g4_s6_authority,
    verify_regression_evidence as verify_g4_s6_regression_evidence,
    recursive_descriptor_factorisation_theorem as g4_s6_recursive_theorem,
    tier1_index as g4_s6_tier1_index, aggregate_holdout as aggregate_g4_s6_holdout,
    implementation_no_hidden_selector_audit as g4_s6_hidden_audit,
    primary_result as g4_s6_primary_result, load_case_checkpoint as load_g4_s6_case_checkpoint,
)
from . import uplift_g4_s6
from .uplift_g3_r3 import (
    r3_spec as g3_r3_spec, verify_rebase_authority as verify_g3_r3_rebase_authority,
    structural_theorem_certificate as g3_r3_structural_theorem_certificate,
    finalize_r3_result as finalize_g3_r3_result,
)
from .uplift_g4_r0 import run_g4_r0_recon, r0_spec as g4_r0_spec
from .uplift_g5_r0 import run_g5_r0_recon, r0_spec as g5_r0_spec
from .uplift_g5_r1 import run_g5_r1_fiber_graft_audit, r1_spec as g5_r1_spec
from .uplift_g5_r2 import run_g5_r2_predictive_hidden_read, r2_spec as g5_r2_spec
from .uplift_g5_r3 import run_g5_r3_generic_message_congruence, r3_spec as g5_r3_spec
from .uplift_g5_r4 import run_g5_r4_marked_hidden_minimality, r4_spec as g5_r4_spec
from .uplift_g5_r5 import run_g5_r5_marker_free_one_step_separation, r5_spec as g5_r5_spec
from .uplift_g5_r6 import run_g5_r6_single_payload_separation, r6_spec as g5_r6_spec
from .uplift_g5_r7 import run_g5_r7_adversarial_escalation, r7_spec as g5_r7_spec

from .uplift_g4_s0_rebase import (
    run_s0_rebase as run_g4_s0_rebase, s0_rebase_spec as g4_s0_rebase_spec,
)
from .uplift_g3_r0 import flattened_g2_unit_incidence, _tree_canon
from .uplift_s6 import (
    verify_s6_authority, verify_regression_evidence, recursive_caps7_factorisation_argument,
    deterministic_holdout_plan, execute_depth2_holdout_case, execute_deep_chain_holdout_case,
    execute_rebracket_holdout_case, aggregate_fresh_holdout, implementation_no_hidden_selector_audit,
    s6_primary_result, s6_implementation_spec,
)

class UpliftCampaignError(RuntimeError): pass

_REGISTRY_NAME='G_UPLIFT_EXPERIMENT_REGISTRY_V1.json'
_WORKER_CONTEXT: dict[str,Any]|None=None
_OP_RE=re.compile(r'^G1_PUBLIC_BRIDGE_RELATION_V1:(\d+)>(\d+)$')

def _resource(name:str)->Path:
    return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def load_uplift_experiment_registry()->dict[str,Any]:
    obj=json.loads(_resource(_REGISTRY_NAME).read_text(encoding='utf-8'))
    if obj.get('schema_id')!='IG_G_UPLIFT_EXPERIMENT_REGISTRY_V1': raise UpliftCampaignError('bad uplift experiment registry schema')
    payload={k:v for k,v in obj.items() if k!='registry_sha256'}
    if canonical_sha256(payload)!=obj.get('registry_sha256'): raise UpliftCampaignError('uplift experiment registry hash mismatch')
    ids=[x.get('experiment_id') for x in obj.get('experiments',[])]
    if len(ids)!=len(set(ids)) or any(not x for x in ids): raise UpliftCampaignError('uplift experiment ids must be unique')
    known=set(ids)
    for row in obj['experiments']:
        for dep in row.get('dependencies',[]):
            if dep not in known: raise UpliftCampaignError(f"{row['experiment_id']}: unknown dependency {dep}")
    return obj

def list_registered_uplift_experiments()->dict[str,Any]:
    reg=load_uplift_experiment_registry()
    return {'schema_id':'IG_G_UPLIFT_REGISTERED_EXPERIMENT_LIST_V1','status':'PASS','registry_sha256':reg['registry_sha256'],'experiments':reg['experiments']}

def _science_stop_status(result:Mapping[str,Any])->str:
    return 'COMPLETE' if result.get('status')=='PASS' else 'SCIENTIFIC_STOP'

def _write_state(root:Path, **kw)->None:
    obj={'schema_id':'IG_G_UPLIFT_CAMPAIGN_STATE_V1','updated_utc':utc_now(),**kw}
    write_json_atomic(root/'CAMPAIGN_STATE.json',obj)

def _init_worker_from_fork(_:Any)->None:
    # The live materialized engine/carriers are inherited read-only by fork.
    if _WORKER_CONTEXT is None: raise UpliftCampaignError('fork worker did not inherit campaign context')

def _s4_d4_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_CONTEXT is None: raise UpliftCampaignError('S4 worker context absent')
    engine=_WORKER_CONTEXT['engine']; cmap=_WORKER_CONTEXT['carrier_map']; bridge_pairs=_WORKER_CONTEXT['bridge_pairs']
    idx=int(payload['row_index']); r=payload['record']
    m=_OP_RE.fullmatch(str(r['connection_operator_ref']))
    if m is None: raise UpliftCampaignError(f'bad operator row {idx}')
    a,b=map(int,m.groups()); L=cmap[str(r['left_carrier_ref'])]; R=cmap[str(r['right_carrier_ref'])]
    st=rs._build_lift(engine,101,[L,R],[(0,1)],bridge_pairs,'G2_S4_NATIVE',f'G2:S4:NATIVE:{idx}',force_pair=(a,b))
    if st is None: raise UpliftCampaignError(f'pair realization failed row {idx}')
    vec=[]
    for t in range(7):
        if int(st.total_caps[t])<=0: raise UpliftCampaignError(f'nonpositive pair capacity row {idx} type {t}')
        vec.append(post_reservation_public_branch_relation_projection(st,t)['science_sha256'])
    return {'row_index':idx,'d4_vector':vec,'pair_caps':[int(x) for x in st.total_caps]}


def _g3_s1_context_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_CONTEXT is None: raise UpliftCampaignError('G3:S1 worker context absent')
    engine=_WORKER_CONTEXT['engine']; states=_WORKER_CONTEXT['g3_s1_states']
    tr=str(payload['target_ref']); cr=str(payload['context_ref'])
    if tr not in states or cr not in states: raise UpliftCampaignError('G3:S1 task references unknown frozen challenge carrier')
    op=[int(x) for x in payload['operator']]; ori=str(payload['orientation'])
    sig=pair_context_signature(engine=engine,target=states[tr],context=states[cr],operator=op,orientation=ori)
    return {'target_ref':tr,'context_ref':cr,'operator':op,'orientation':ori,'operational_signature':sig}



def _g3_s3_local_continuation_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_CONTEXT is None: raise UpliftCampaignError('G3:S3 worker context absent')
    states=_WORKER_CONTEXT['g3_s3_states']
    ref=str(payload['carrier_ref']); a=int(payload['first_type']); b=int(payload['second_type'])
    if ref not in states: raise UpliftCampaignError('G3:S3 task references unknown frozen challenge carrier')
    sig=ordered_two_reservation_signature(states[ref],a,b)
    return {'carrier_ref':ref,'first_type':a,'second_type':b,'signature':sig}

def _g3_s4_local_three_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_CONTEXT is None: raise UpliftCampaignError('G3:S4 local-three worker context absent')
    states=_WORKER_CONTEXT['g3_s4_states']
    ref=str(payload['carrier_ref']); a=int(payload['first_type']); b=int(payload['second_type']); c=int(payload['third_type'])
    if ref not in states: raise UpliftCampaignError('G3:S4 local-three task references unknown frozen challenge carrier')
    sig=ordered_three_reservation_signature(states[ref],a,b,c)
    return {'carrier_ref':ref,'first_type':a,'second_type':b,'third_type':c,'signature':sig}


def _g3_s4_recursive_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_CONTEXT is None: raise UpliftCampaignError('G3:S4 recursive worker context absent')
    engine=_WORKER_CONTEXT['engine']; pair_relations=_WORKER_CONTEXT['g3_s4_pair_relations']; context=_WORKER_CONTEXT['g3_s4_context']
    i=int(payload['first_key_index']); j=int(payload['second_key_index'])
    if i not in pair_relations: raise UpliftCampaignError('G3:S4 unknown first pair key')
    op=[int(x) for x in payload['operator']]; ori=str(payload['orientation'])
    sig,_outs=recursive_relation_immediate_signature(engine=engine,pair_relation=pair_relations[i],context=context,operator=op,orientation=ori,motif_id=f'G3:S4:RECURSIVE:{i}:{j}:{ori}:{op[0]}>{op[1]}')
    return {'first_key_index':i,'second_key_index':j,'operator':op,'orientation':ori,'signature':sig}


def _g3_s4_reserve_basis_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_CONTEXT is None: raise UpliftCampaignError('G3:S4 reserve-basis worker context absent')
    engine=_WORKER_CONTEXT['engine']; pair_relations=_WORKER_CONTEXT['g3_s4_pair_relations']; context=_WORKER_CONTEXT['g3_s4_context']
    i=int(payload['first_key_index']); op=[int(x) for x in payload['operator']]; ori=str(payload['orientation'])
    sig,outs=recursive_relation_immediate_signature(engine=engine,pair_relation=pair_relations[i],context=context,operator=op,orientation=ori,motif_id=f'G3:S4:RESERVE_BASIS:{i}:{ori}:{op[0]}>{op[1]}')
    cap=recursive_reserve_capability_signature(outs)
    return {'first_key_index':i,'operator':op,'orientation':ori,'immediate_signature':sig,'reserve_capability':cap}


def _s6_holdout_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_CONTEXT is None: raise UpliftCampaignError('S6 worker context absent')
    engine=_WORKER_CONTEXT['engine']; carriers=_WORKER_CONTEXT['s6_sorted_carriers']; bridge_pairs=_WORKER_CONTEXT['bridge_pairs']
    kind=str(payload['kind']); case=payload['case']
    if kind=='DEPTH2':
        return execute_depth2_holdout_case(engine=engine,sorted_carriers=carriers,bridge_pairs=bridge_pairs,case=case)
    if kind=='DEEP_CHAIN':
        return execute_deep_chain_holdout_case(engine=engine,sorted_carriers=carriers,bridge_pairs=bridge_pairs,case=case)
    if kind=='REBRACKET':
        return execute_rebracket_holdout_case(engine=engine,sorted_carriers=carriers,bridge_pairs=bridge_pairs,case=case)
    raise UpliftCampaignError(f'unknown S6 holdout task kind {kind}')

class UpliftCampaignEngine:
    def __init__(self, run_root:str|Path, *, requested_workers:int|str|None='AUTO', lease_root:str|None=None, materialization_start_method:str='AUTO'):
        self.run_root=Path(run_root); self.run_root.mkdir(parents=True,exist_ok=True)
        self.registry=load_uplift_experiment_registry()
        # Materialization and native stage work have deliberately different optimal transports.
        # Deep G1 maturation defaults to the v0.28.6 in-process exact-state fast path.
        # v0.30.32 additionally permits an explicit Linux fork-inherited exact-state transport,
        # which parallelizes candidate recipes without serializing the recursive state DAG.
        # Native S4 continues to use fork after the lower-G population is materialized.
        self.materialization_policy=ExecutionPolicy(
            backend='AUTO',requested_workers=requested_workers,scheduler='COST_WEIGHTED_SHARDS',
            start_method=str(materialization_start_method),reserve_cores=0,lease_root=lease_root,owner='g-uplift-materialization-v0301')
        self.stage_policy=ExecutionPolicy(
            backend='AUTO',requested_workers=requested_workers,scheduler='COST_WEIGHTED_SHARDS',
            start_method='fork',reserve_cores=0,lease_root=lease_root,owner='g-uplift-stage-v0301')
        # Backward-compatible public name used by native stage handlers.
        self.policy=self.stage_policy
        self.session:MaterializedDiscoverySession|None=None
        self._r100_carriers:list[Any]|None=None; self._carrier_map:dict[str,Any]|None=None
        self._r100_growth_seed:tuple[Any,tuple[int,int]]|None=None
        self.materialization_count=0

    def __enter__(self):
        self.session=MaterializedDiscoverySession(execution_policy=self.materialization_policy); return self
    def __exit__(self,exc_type,exc,tb):
        if self.session is not None: self.session.close(); self.session=None
    def experiment(self,experiment_id:str)->dict[str,Any]:
        rows=[x for x in self.registry['experiments'] if x['experiment_id']==experiment_id]
        if len(rows)!=1: raise UpliftCampaignError(f'unknown registered uplift experiment {experiment_id}')
        return rows[0]
    def ensure_g1_r100_population(self)->list[Any]:
        if self._r100_carriers is not None: return self._r100_carriers
        if self.session is None: raise UpliftCampaignError('campaign engine not opened')
        # Deep G1 maturation deliberately uses the AUTO materialization policy.  This restores
        # the v0.28.6 exact in-process candidate path and avoids serializing/reconstructing the
        # recursive state DAG at every depth.  R100 itself is the complete 193-candidate cohort,
        # not the 24-state exploratory beam retained between depths.
        self.session.advance_to(99); prev=self.session.level_states[99]
        cand,_,_,center=parallel_candidate_descriptors(
            engine=self.session.engine,prev=prev,level=100,pairs=self.session.bridge_pairs,
            motifs=self.session.motifs,spec=self.session.spec,policy=self.materialization_policy)
        carriers=rebuild_selected_states(cand,engine=self.session.engine,prev=prev,level=100,pairs=self.session.bridge_pairs,motifs=self.session.motifs,center=center)
        carriers=sorted(carriers,key=lambda x:x.construction_digest)
        if len(carriers)!=193: raise UpliftCampaignError(f'G1:R100 expected 193 carriers, got {len(carriers)}')
        self._r100_carriers=carriers; self._carrier_map={str(x.construction_digest):x for x in carriers}; self.materialization_count+=1
        return carriers

    @property
    def carrier_map(self)->dict[str,Any]:
        self.ensure_g1_r100_population(); assert self._carrier_map is not None; return self._carrier_map

def _load_json(path:str|Path)->dict[str,Any]: return json.loads(Path(path).read_text(encoding='utf-8'))

def run_g2_s4_native(*, engine:UpliftCampaignEngine, s3_unlock_certificate:str|Path, s3_result:str|Path, s0_population:str|Path, repaired_records:str|Path, audit_signatures:str|Path)->dict[str,Any]:
    """Run registered G2:S4 inside one persistent campaign engine.

    Scientific REVIEW_REQUIRED is returned as a normal result and must not be converted
    into an operational process failure.
    """
    spec=engine.experiment('G2:S4')
    if spec.get('execution_mode')!='NATIVE_HANDLER': raise UpliftCampaignError('G2:S4 not registered native')
    stage_dir=engine.run_root/'stages'/'G2_S4'; stage_dir.mkdir(parents=True,exist_ok=True)
    cert=_load_json(s3_unlock_certificate); s3res=_load_json(s3_result); s0=_load_json(s0_population)
    auth=verify_s4_authority(cert,expected_s3_source_sha256=str(s3res['source_sha256']),expected_s3_result_sha256=str(s3res['sealed_science_sha256']))
    idx=index_s2_collision_records(repaired_records,audit_signatures)
    rc=resource_legality_closure(repaired_records,s0)
    carriers=engine.ensure_g1_r100_population()
    items=[]
    for obs in sorted(idx['groups']):
        for item in sorted(idx['groups'][obs],key=lambda x:int(x['row_index'])):
            items.append((obs,int(item['row_index']),item['record']))
    global _WORKER_CONTEXT
    _WORKER_CONTEXT={'engine':engine.session.engine,'carrier_map':engine.carrier_map,'bridge_pairs':list(engine.session.bridge_pairs)}
    binding=canonical_sha256({'stage':'G2:S4','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'s4_implementation_spec_sha256':s4_implementation_spec()['science_sha256'],'g2_relation_spec_sha256':relation_spec()['spec_sha256'],'s4_implementation_spec_sha256':s4_implementation_spec()['science_sha256']})
    tasks=[TaskSpec(task_id=f's4d4-{idx0:07d}',task_kind='G2_S4_D4_REALIZATION',binding_sha256=binding,payload={'row_index':idx0,'record':r},cost_weight=1.0) for _,idx0,r in items]
    batch=execute_tasks(tasks,worker_ref='infinity_grid.uplift_campaign:_s4_d4_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_campaign:_init_worker_from_fork',initializer_payload=None)
    vectors={int(v['row_index']):v['d4_vector'] for v in batch.results.values()}
    d4=d4_composition_congruence(idx,vectors)
    # 31-operator actual recursive-use basis on the persistent campaign engine.
    first_obs,first_idx,first_record=items[0]; m=_OP_RE.fullmatch(str(first_record['connection_operator_ref'])); assert m is not None
    a0,b0=map(int,m.groups()); L=engine.carrier_map[str(first_record['left_carrier_ref'])]; R=engine.carrier_map[str(first_record['right_carrier_ref'])]
    first_state=rs._build_lift(engine.session.engine,101,[L,R],[(0,1)],engine.session.bridge_pairs,'G2_S4_NATIVE','G2:S4:NATIVE:BASIS',force_pair=(a0,b0))
    if first_state is None: raise UpliftCampaignError('operator basis seed pair failed')
    context=carriers[0]; basis=[]; failures=[]
    for a,b in engine.session.bridge_pairs:
        before=[int(first_state.total_caps[i])+int(context.total_caps[i]) for i in range(7)]
        outputs=compose_binary_relation(engine.session.engine,102,first_state,context,a,b,lane='G2_S4_NATIVE_RECURSIVE_RELATION',motif_id=f'G2:S4:NATIVE:OP:{a}>{b}')
        if not outputs: failures.append({'operator':[a,b],'reason':'relation_realization_empty'}); continue
        exp=[before[i]-(1 if i==a else 0)-(1 if i==b else 0) for i in range(7)]
        output_rows=[]; ok=True
        for st2 in outputs:
            observed=[int(x) for x in st2.total_caps]
            reserve_branch_counts=[]
            for t in range(7):
                rel=reserve_external_relation(st2,t) if observed[t]>0 else tuple()
                reserve_branch_counts.append(len(rel))
                if observed[t]>0 and not rel: ok=False
            row_ok=observed==exp and len(st2.children)==2 and all((observed[t]<=0 or reserve_branch_counts[t]>0) for t in range(7))
            ok=ok and row_ok
            output_rows.append({'output_skin':str(st2.skin),'output_total_caps':observed,'reserve_relation_branch_counts':reserve_branch_counts})
        if not ok: failures.append({'operator':[a,b],'reason':'resource_or_relation_capability_mismatch'})
        basis.append({'operator':[int(a),int(b)],'status':'PASS' if ok else 'FAIL','relation_output_count':len(outputs),'outputs':output_rows})
    op={'schema_id':'IG_G2_S4_RELATION_VALUED_OPERATOR_RECURSIVE_REALIZATION_BASIS_V2','status':'PASS' if not failures and len(basis)==len(engine.session.bridge_pairs) else 'FAIL','operator_count':len(engine.session.bridge_pairs),'operators_realized':len(basis),'failures':failures,'basis_rows':basis,'input_pair_row_index':first_idx,'context_carrier_ref':context.construction_digest,'hidden_internal_reads_in_legality':False,'g2_relation_spec_sha256':relation_spec()['spec_sha256'],'s4_implementation_spec_sha256':s4_implementation_spec()['science_sha256']}; op['science_sha256']=canonical_sha256(op)
    result=composition_closure_result(authority=auth,resource_closure=rc,d4_congruence=d4,operator_basis=op)
    result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V2','registered_experiment_id':'G2:S4','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_batch':batch.metadata,'task_count':len(tasks),'requested_workers':engine.policy.requested_workers,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'g2_reservation_semantics':'RELATION_VALUED_V1','g2_relation_spec_sha256':relation_spec()['spec_sha256'],'s4_implementation_spec_sha256':s4_implementation_spec()['science_sha256']}
    result['native_engine_science_sha256']=canonical_sha256({k:v for k,v in result.items() if k not in {'science_sha256','native_engine_science_sha256'}})
    write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'D4_CONGRUENCE.json',d4); write_json_atomic(stage_dir/'RESOURCE_CLOSURE.json',rc); write_json_atomic(stage_dir/'OPERATOR_BASIS.json',op); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G2:S4',last_result_status=result['status'],classification=result['classification'],s5_unlocked=result['s5_unlocked'],g2_graduated=False)
    return result

def run_g2_s5_native(*, engine:UpliftCampaignEngine, s5_unlock_certificate:str|Path, s4_result:str|Path, s0_population:str|Path, repaired_records:str|Path, s4_operator_basis:str|Path, s2_revalidation:str|Path)->dict[str,Any]:
    """Run registered G2:S5 CAPS7 finite read/write quotient audit.

    S5 is intentionally a finite-coordinate factorisation audit, not a new lower-G
    materialization.  It consumes the already-certified S0/S1/S2/S4 evidence and
    therefore must not rebuild R100.  PASS unlocks S6 only.
    """
    spec=engine.experiment('G2:S5')
    if spec.get('execution_mode')!='NATIVE_HANDLER': raise UpliftCampaignError('G2:S5 not registered native')
    stage_dir=engine.run_root/'stages'/'G2_S5'; stage_dir.mkdir(parents=True,exist_ok=True)
    cert=_load_json(s5_unlock_certificate); s4res=_load_json(s4_result); s0=_load_json(s0_population)
    op=_load_json(s4_operator_basis); s2=_load_json(s2_revalidation)
    auth=verify_s5_authority(unlock_certificate=cert,s4_result=s4res)
    base=verify_s0_base_reservation_law(s0)
    census=scan_certified_pair_caps7(repaired_records,s0)
    # The frozen bridge basis is read from the certified S4 operator basis itself;
    # every basis row is checked against every exact relation-valued output branch.
    bridge_pairs=[tuple(map(int,row['operator'])) for row in op.get('basis_rows',[])]
    opfac=verify_s4_operator_caps7_factorisation(operator_basis=op,repaired_records_path=repaired_records,s0_population=s0,bridge_pairs=bridge_pairs)
    impl=implementation_read_surface_audit()
    arg=caps7_factorisation_argument(s0_base=base,pair_census=census,operator_factorisation=opfac,implementation_audit=impl)
    result=s5_result(authority=auth,s0_base=base,pair_census=census,operator_factorisation=opfac,implementation_audit=impl,argument=arg,s2_revalidation=s2)
    result['source_sha256']=source_sha256()
    result['source_version']='0.30.6'
    result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
        'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V3',
        'registered_experiment_id':'G2:S5',
        'registry_sha256':engine.registry['registry_sha256'],
        'campaign_materialization_count':engine.materialization_count,
        'lower_g_materialization_performed_for_s5':False,
        'parallel_stage_required':False,
        'execution_backend_owned_by_decoder':True,
        'stage_specific_external_science_runner':False,
        'g2_reservation_semantics':'RELATION_VALUED_V1',
        'g2_relation_spec_sha256':relation_spec()['spec_sha256'],
        's5_implementation_spec_sha256':s5_implementation_spec()['science_sha256'],
    }
    # Science identity deliberately excludes execution metadata/source packaging fields.
    write_json_atomic(stage_dir/'RESULT.json',result)
    write_json_atomic(stage_dir/'AUTHORITY.json',auth)
    write_json_atomic(stage_dir/'S0_BASE_RESERVATION_FACTORISATION.json',base)
    write_json_atomic(stage_dir/'CERTIFIED_PAIR_CAPS7_CENSUS.json',census)
    write_json_atomic(stage_dir/'S4_BINARY_WRITE_FACTORISATION.json',opfac)
    write_json_atomic(stage_dir/'IMPLEMENTATION_READ_SURFACE_AUDIT.json',impl)
    write_json_atomic(stage_dir/'CAPS7_FACTORISATION_ARGUMENT.json',arg)
    write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G2:S5',last_result_status=result['status'],classification=result['classification'],s6_unlocked=result['s6_unlocked'],g2_graduated=False)
    return result


def run_g2_s6_native(*, engine:UpliftCampaignEngine, s6_unlock_certificate:str|Path, s5_result:str|Path, regression_evidence:str|Path)->dict[str,Any]:
    """Run registered G2:S6 primary recursive closure / graduation-candidate audit.

    The primary run is not itself allowed to graduate G2.  It performs the general
    structural-induction gate plus a fresh exact recursive-state holdout using the complete
    R100 lower-G cohort.  Final graduation requires a separate cold replay and finalizer.
    """
    spec=engine.experiment('G2:S6')
    if spec.get('execution_mode')!='NATIVE_HANDLER': raise UpliftCampaignError('G2:S6 not registered native')
    stage_dir=engine.run_root/'stages'/'G2_S6'; stage_dir.mkdir(parents=True,exist_ok=True)
    cert=_load_json(s6_unlock_certificate); s5res=_load_json(s5_result); reg=_load_json(regression_evidence)
    current_source=source_sha256()
    auth=verify_s6_authority(unlock_certificate=cert,s5_result=s5res)
    rgate=verify_regression_evidence(reg,current_source_sha256=current_source)
    theorem=recursive_caps7_factorisation_argument(s5res)
    hidden=implementation_no_hidden_selector_audit()

    carriers=engine.ensure_g1_r100_population()
    sorted_carriers=sorted(carriers,key=lambda x:(tuple(int(v) for v in x.total_caps),str(x.construction_digest)))
    bridge_pairs=sorted({tuple(map(int,x)) for x in engine.session.bridge_pairs})
    plan=deterministic_holdout_plan(carrier_count=len(sorted_carriers),bridge_pairs=bridge_pairs)
    # Publish the outcome-blind exact plan before any holdout task is executed.
    write_json_atomic(stage_dir/'FRESH_HOLDOUT_PLAN.json',plan)

    global _WORKER_CONTEXT
    _WORKER_CONTEXT={'engine':engine.session.engine,'s6_sorted_carriers':sorted_carriers,'bridge_pairs':bridge_pairs}
    binding=canonical_sha256({
        'stage':'G2:S6','source_sha256':current_source,'registry_sha256':engine.registry['registry_sha256'],
        'authority_sha256':auth['science_sha256'],'s6_spec_sha256':s6_implementation_spec()['science_sha256'],
        'holdout_plan_sha256':plan['science_sha256'],'relation_spec_sha256':relation_spec()['spec_sha256']})
    tasks=[]
    for c in plan['depth2_cases']:
        tasks.append(TaskSpec(task_id=str(c['case_id']),task_kind='G2_S6_DEPTH2_HOLDOUT',binding_sha256=binding,payload={'kind':'DEPTH2','case':c},cost_weight=1.0))
    for c in plan['deep_chain_cases']:
        tasks.append(TaskSpec(task_id=str(c['case_id']),task_kind='G2_S6_DEEP_CHAIN_HOLDOUT',binding_sha256=binding,payload={'kind':'DEEP_CHAIN','case':c},cost_weight=5.0))
    for c in plan['rebracketing_cases']:
        tasks.append(TaskSpec(task_id=str(c['case_id']),task_kind='G2_S6_REBRACKET_HOLDOUT',binding_sha256=binding,payload={'kind':'REBRACKET','case':c},cost_weight=4.0))
    batch=execute_tasks(tasks,worker_ref='infinity_grid.uplift_campaign:_s6_holdout_worker',policy=engine.stage_policy,initializer_ref='infinity_grid.uplift_campaign:_init_worker_from_fork',initializer_payload=None)
    depth2=[batch.results[str(c['case_id'])] for c in plan['depth2_cases']]
    deep=[batch.results[str(c['case_id'])] for c in plan['deep_chain_cases']]
    rb=[batch.results[str(c['case_id'])] for c in plan['rebracketing_cases']]
    holdout=aggregate_fresh_holdout(plan=plan,depth2_results=depth2,deep_results=deep,rebracket_results=rb)
    result=s6_primary_result(authority=auth,regression=rgate,theorem=theorem,holdout=holdout,hidden_read_audit=hidden)
    result['source_sha256']=current_source
    result['source_version']='0.30.6'
    result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
        'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V4','registered_experiment_id':'G2:S6',
        'registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
        'parallel_batch':batch.metadata,'task_count':len(tasks),'requested_workers':engine.policy.requested_workers,
        'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,
        'g2_reservation_semantics':'RELATION_VALUED_V1','g2_relation_spec_sha256':relation_spec()['spec_sha256'],
        's6_implementation_spec_sha256':s6_implementation_spec()['science_sha256'],
        'primary_run_can_graduate_g2':False,
    }
    write_json_atomic(stage_dir/'RESULT.json',result)
    write_json_atomic(stage_dir/'AUTHORITY.json',auth)
    write_json_atomic(stage_dir/'REGRESSION_GATE.json',rgate)
    write_json_atomic(stage_dir/'RECURSIVE_CAPS7_FACTORISATION_THEOREM.json',theorem)
    write_json_atomic(stage_dir/'FRESH_HOLDOUT_RESULT.json',holdout)
    write_json_atomic(stage_dir/'NO_HIDDEN_SELECTOR_READ_AUDIT.json',hidden)
    write_json_atomic(stage_dir/'DEPTH2_CASE_RESULTS.json',{'schema_id':'IG_G2_S6_DEPTH2_CASE_RESULTS_V1','cases':depth2})
    write_json_atomic(stage_dir/'DEEP_CHAIN_CASE_RESULTS.json',{'schema_id':'IG_G2_S6_DEEP_CHAIN_CASE_RESULTS_V1','cases':deep})
    write_json_atomic(stage_dir/'REBRACKET_CASE_RESULTS.json',{'schema_id':'IG_G2_S6_REBRACKET_CASE_RESULTS_V1','cases':rb})
    write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G2:S6',last_result_status=result['status'],classification=result['classification'],g2_graduation_candidate=result['g2_graduation_candidate'],g2_graduated=False,r0_unlocked=False)
    return result

def run_g2_r0_native(*, engine:UpliftCampaignEngine, graduation_certificate:str|Path)->dict[str,Any]:
    """Run registered non-promoting G2:R0 structural reconnaissance inside Decoder.

    R0 is authorized only by the final G2 graduation certificate. It may read exact G2
    incidence topology as an explicitly frozen reconnaissance observer, but cannot alter
    CAPS7, routing semantics, G2 graduation, or launch G3.
    """
    spec=engine.experiment('G2:R0.STRUCTURAL_AUDIT')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G2:R0 structural audit not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G2_R0_STRUCTURAL_AUDIT'; stage_dir.mkdir(parents=True,exist_ok=True)
    cert=_load_json(graduation_certificate)
    result=run_r0_recon(engine=engine,graduation_certificate=cert)
    result['source_sha256']=source_sha256()
    result['source_version']='0.30.7'
    result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
        'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V5',
        'registered_experiment_id':'G2:R0.STRUCTURAL_AUDIT',
        'registry_sha256':engine.registry['registry_sha256'],
        'campaign_materialization_count':engine.materialization_count,
        'parallel_stage_required':False,
        'execution_backend_owned_by_decoder':True,
        'stage_specific_external_science_runner':False,
        'promotion':False,
        'g2_graduation_preserved':True,
        'g3_started':False,
        'r0_spec_science_sha256':r0_spec()['science_sha256'],
    }
    write_json_atomic(stage_dir/'RESULT.json',result)
    write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G2:R0',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,r0_complete=result['status']=='PASS',g3_started=False)
    return result


def run_g3_s0_native(*, engine:UpliftCampaignEngine, graduation_certificate:str|Path, r0_result:str|Path)->dict[str,Any]:
    """Run registered G3:S0 inside Decoder without promoting topology."""
    spec=engine.experiment('G3:S0')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G3:S0 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_S0'; stage_dir.mkdir(parents=True,exist_ok=True)
    grad=_load_json(graduation_certificate); r0=_load_json(r0_result)
    result=run_g3_s0(engine=engine,graduation_certificate=grad,r0_result=r0)
    result['source_sha256']=source_sha256(); result['source_version']='0.30.9'; result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
      'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V6','registered_experiment_id':'G3:S0',
      'registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
      'parallel_stage_required':False,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,
      'promotion':False,'g2_graduation_preserved':True,'g3_graduated':False,
      'g3_phase0_spec_sha256':g3_phase0_spec()['science_sha256'],'g3_s0_spec_sha256':g3_s0_spec()['science_sha256']}
    write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:S0',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_started=True,g3_s1_unlocked=result.get('g3_s1_unlocked',False),g3_graduated=False)
    return result


def run_g3_s1_native(*, engine:UpliftCampaignEngine, s0_result:str|Path)->dict[str,Any]:
    """Run registered G3:S1 topology-read pair-context audit inside Decoder."""
    spec=engine.experiment('G3:S1')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G3:S1 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_S1'; stage_dir.mkdir(parents=True,exist_ok=True)
    s0=_load_json(s0_result)
    auth=verify_s1_authority(s0)
    states,repro=reproduce_s0_challenge_states(engine=engine,s0_result=s0)
    refs=sorted(states)
    bridge_pairs=sorted({tuple(map(int,x)) for x in engine.session.bridge_pairs})
    if len(bridge_pairs)!=31: raise UpliftCampaignError('G3:S1 requires 31 bridge operators')
    global _WORKER_CONTEXT
    _WORKER_CONTEXT={'engine':engine.session.engine,'g3_s1_states':states}
    binding=canonical_sha256({'stage':'G3:S1','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'s1_spec_sha256':g3_s1_spec()['science_sha256']})
    tasks=[]
    for tr in refs:
        for cr in refs:
            for a,b in bridge_pairs:
                for ori in ('TARGET_LEFT_CONTEXT_RIGHT','CONTEXT_LEFT_TARGET_RIGHT'):
                    tasks.append(TaskSpec(task_id=f'g3s1-{len(tasks):05d}',task_kind='G3_S1_PAIR_CONTEXT',binding_sha256=binding,payload={'target_ref':tr,'context_ref':cr,'operator':[a,b],'orientation':ori},cost_weight=1.0))
    batch=execute_tasks(tasks,worker_ref='infinity_grid.uplift_campaign:_g3_s1_context_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_campaign:_init_worker_from_fork',initializer_payload=None)
    rows=[batch.results[k] for k in sorted(batch.results)]
    result=finalize_s1_result(s0_result=s0,reproduction=repro,task_rows=rows)
    result['source_sha256']=source_sha256(); result['source_version']='0.30.11'; result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
      'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V7','registered_experiment_id':'G3:S1',
      'registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
      'parallel_stage_required':True,'parallel_batch':batch.metadata,'task_count':len(tasks),'requested_workers':engine.policy.requested_workers,
      'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,
      'g2_graduation_preserved':True,'g3_graduated':False,'topology_promoted':False,
      'g3_phase0_spec_sha256':g3_phase0_spec()['science_sha256'],'g3_s1_spec_sha256':g3_s1_spec()['science_sha256']}
    write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    write_json_atomic(stage_dir/'CHALLENGE_REPRODUCTION.json',repro)
    with gzip.open(stage_dir/'PAIR_CONTEXT_SIGNATURES.json.gz','wt',encoding='utf-8',compresslevel=6) as fh:
        json.dump({'schema_id':'IG_G3_S1_PAIR_CONTEXT_SIGNATURE_INDEX_V1','task_count':len(rows),'rows':rows},fh,sort_keys=True,separators=(',',':'))
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:S1',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_started=True,g3_s2_unlocked=result.get('g3_s2_unlocked',False),g3_graduated=False)
    return result


def run_g3_s2_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s1_result:str|Path, s1_signature_index:str|Path)->dict[str,Any]:
    """Run registered G3:S2 deterministic CAPS7 pair-observer quotient inside Decoder."""
    spec=engine.experiment('G3:S2')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G3:S2 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_S2'; stage_dir.mkdir(parents=True,exist_ok=True)
    s0=_load_json(s0_result); s1=_load_json(s1_result); sigs=load_s1_signature_index(s1_signature_index)
    result=certify_caps7_pair_observer_quotient(s0_result=s0,s1_result=s1,signature_index=sigs)
    result['source_sha256']=source_sha256(); result['source_version']='0.30.12'; result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
      'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V8','registered_experiment_id':'G3:S2',
      'registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
      'parallel_stage_required':False,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,
      'promotion':False,'g2_graduation_preserved':True,'g3_graduated':False,'topology_promoted':False,
      'g3_phase0_spec_sha256':g3_phase0_spec()['science_sha256'],'g3_s2_spec_sha256':g3_s2_spec()['science_sha256'],
      'input_s1_science_sha256':s1.get('science_sha256'),'input_s1_signature_task_count':sigs.get('task_count')}
    write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    write_json_atomic(stage_dir/'AUTHORITY.json',result['authority']); write_json_atomic(stage_dir/'QUOTIENT_TABLE.json',{'schema_id':'IG_G3_S2_QUOTIENT_TABLE_V1','rows':result['quotient_table']})
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:S2',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_started=True,g3_s3_unlocked=result.get('g3_s3_unlocked',False),g3_graduated=False)
    return result



def run_g3_s3_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s1_result:str|Path, s1_signature_index:str|Path, s2_result:str|Path)->dict[str,Any]:
    """Run registered G3:S3 exact P3/K3 higher-order irreducibility reduction inside Decoder."""
    spec=engine.experiment('G3:S3')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G3:S3 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_S3'; stage_dir.mkdir(parents=True,exist_ok=True)
    s0=_load_json(s0_result); s1=_load_json(s1_result); s2=_load_json(s2_result); sigs=load_s1_signature_index(s1_signature_index)
    auth=verify_s3_authority(s0_result=s0,s1_result=s1,s2_result=s2)
    states,repro=reproduce_s0_challenge_states(engine=engine,s0_result=s0)
    refs=sorted(states)
    global _WORKER_CONTEXT
    _WORKER_CONTEXT={'engine':engine.session.engine,'g3_s3_states':states}
    binding=canonical_sha256({'stage':'G3:S3','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'s3_spec_sha256':g3_s3_spec()['science_sha256']})
    tasks=[]
    for ref in refs:
        for a in range(7):
            for b in range(7):
                tasks.append(TaskSpec(task_id=f'g3s3-{len(tasks):04d}',task_kind='G3_S3_LOCAL_TWO_RESERVATION_CONTINUATION',binding_sha256=binding,payload={'carrier_ref':ref,'first_type':a,'second_type':b},cost_weight=1.0))
    batch=execute_tasks(tasks,worker_ref='infinity_grid.uplift_campaign:_g3_s3_local_continuation_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_campaign:_init_worker_from_fork',initializer_payload=None)
    rows=[batch.results[k] for k in sorted(batch.results)]
    result=finalize_s3_result(s0_result=s0,s1_result=s1,s2_result=s2,signature_index=sigs,task_rows=rows,reproduction=repro)
    result['source_sha256']=source_sha256(); result['source_version']='0.30.14'; result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
      'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V9','registered_experiment_id':'G3:S3',
      'registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
      'parallel_stage_required':True,'parallel_batch':batch.metadata,'task_count':len(tasks),'requested_workers':engine.policy.requested_workers,
      'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,
      'g2_graduation_preserved':True,'g3_graduated':False,'topology_promoted':False,
      'g3_phase0_spec_sha256':g3_phase0_spec()['science_sha256'],'g3_s3_spec_sha256':g3_s3_spec()['science_sha256'],
      'input_s2_science_sha256':s2.get('science_sha256'),'input_s1_signature_task_count':sigs.get('task_count')}
    write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    write_json_atomic(stage_dir/'AUTHORITY.json',auth); write_json_atomic(stage_dir/'CHALLENGE_REPRODUCTION.json',repro)
    with gzip.open(stage_dir/'LOCAL_TWO_RESERVATION_CONTINUATIONS.json.gz','wt',encoding='utf-8',compresslevel=6) as fh:
        json.dump({'schema_id':'IG_G3_S3_LOCAL_CONTINUATION_INDEX_V1','task_count':len(rows),'rows':rows},fh,sort_keys=True,separators=(',',':'))
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:S3',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_started=True,g3_s4_unlocked=result.get('g3_s4_unlocked',False),g3_graduated=False)
    return result

def run_g3_s4_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s1_result:str|Path, s2_result:str|Path, s3_result:str|Path, s3_replay_comparison:str|Path)->dict[str,Any]:
    """Run registered G3:S4 one-step relation-valued composition-closure audit inside Decoder."""
    spec=engine.experiment('G3:S4')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G3:S4 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_S4'; stage_dir.mkdir(parents=True,exist_ok=True)
    s0=_load_json(s0_result); s1=_load_json(s1_result); s2=_load_json(s2_result); s3=_load_json(s3_result); s3replay=_load_json(s3_replay_comparison)
    auth=verify_g3_s4_authority(s0_result=s0,s1_result=s1,s2_result=s2,s3_result=s3,s3_replay_comparison=s3replay)
    states,repro=reproduce_s0_challenge_states(engine=engine,s0_result=s0)
    refs=sorted(states)
    if len(refs)!=6: raise UpliftCampaignError('G3:S4 expected six reproduced S0 carriers')
    target=states[refs[0]]; context0=states[refs[1]]; recursive_context=states[refs[2]]

    # Bind the complete canonical pair-relation basis to the certified S2 public observer.
    qrows=sorted(list(s2.get('quotient_table',[])),key=lambda r:(str(r['orientation']),tuple(map(int,r['operator']))))
    if len(qrows)!=62: raise UpliftCampaignError('G3:S4 requires 62 certified S2 public pair keys')
    pair_relations={}; pair_rows=[]; pair_failures=[]
    for i,q in enumerate(qrows):
        op=[int(x) for x in q['operator']]; ori=str(q['orientation'])
        observed=pair_context_signature(engine=engine.session.engine,target=target,context=context0,operator=op,orientation=ori)
        expected_sha=str(q['observer_signature_sha256'])
        rel=compose_complete_pair_relation(engine=engine.session.engine,target=target,context=context0,operator=op,orientation=ori,motif_id=f'G3:S4:PAIR_CANDIDATE:{i}:{ori}:{op[0]}>{op[1]}')
        caps_set=sorted({tuple(int(x) for x in st.total_caps) for st in rel})
        ok=(observed.get('science_sha256')==expected_sha and len(rel)==16 and len(caps_set)==1)
        if not ok:
            pair_failures.append({'key_index':i,'operator':op,'orientation':ori,'observed_s2_signature_sha256':observed.get('science_sha256'),'expected_s2_signature_sha256':expected_sha,'pair_relation_cardinality':len(rel),'caps7_state_count':len(caps_set)})
        pair_relations[i]=rel
        pair_rows.append({'key_index':i,'operator':op,'orientation':ori,'s2_observer_signature_sha256':expected_sha,'canonical_observer_signature_sha256':observed.get('science_sha256'),'pair_relation_cardinality':len(rel),'public_output_caps7_set':[list(x) for x in caps_set]})
    pair_basis={'schema_id':'IG_G3_S4_COMPLETE_PAIR_CANDIDATE_BASIS_V1','status':'PASS' if not pair_failures else 'FAIL','public_pair_context_key_count':len(qrows),'canonical_exact_target_ref':refs[0],'canonical_exact_context_ref':refs[1],'exact_refs_used_for_reproducible_materialisation_only':True,'construction_identity_is_public_selector':False,'expected_relation_cardinality_per_key':16,'failure_count':len(pair_failures),'failure_examples':pair_failures[:16],'rows':pair_rows}
    pair_basis['science_sha256']=canonical_sha256(pair_basis)
    if pair_failures: raise UpliftCampaignError('G3:S4 canonical pair basis failed S2 binding')

    global _WORKER_CONTEXT
    _WORKER_CONTEXT={'engine':engine.session.engine,'g3_s4_states':states,'g3_s4_pair_relations':pair_relations,'g3_s4_context':recursive_context}
    binding=canonical_sha256({'stage':'G3:S4','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'s4_spec_sha256':g3_s4_spec()['science_sha256'],'pair_basis_sha256':pair_basis['science_sha256']})

    local_tasks=[]
    for ref in refs:
        for a in range(7):
            for b in range(7):
                for c in range(7):
                    local_tasks.append(TaskSpec(task_id=f'g3s4-l3-{len(local_tasks):05d}',task_kind='G3_S4_LOCAL_THREE_RESERVATION',binding_sha256=binding,payload={'carrier_ref':ref,'first_type':a,'second_type':b,'third_type':c},cost_weight=1.0))
    local_batch=execute_tasks(local_tasks,worker_ref='infinity_grid.uplift_campaign:_g3_s4_local_three_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_campaign:_init_worker_from_fork',initializer_payload=None)
    local_rows=[local_batch.results[k] for k in sorted(local_batch.results)]

    recursive_tasks=[]
    for i,_q1 in enumerate(qrows):
        for j,q2 in enumerate(qrows):
            recursive_tasks.append(TaskSpec(task_id=f'g3s4-r-{i:02d}-{j:02d}',task_kind='G3_S4_RECURSIVE_P3_MATERIALISATION',binding_sha256=binding,payload={'first_key_index':i,'second_key_index':j,'operator':[int(x) for x in q2['operator']],'orientation':str(q2['orientation'])},cost_weight=8.0))
    recursive_batch=execute_tasks(recursive_tasks,worker_ref='infinity_grid.uplift_campaign:_g3_s4_recursive_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_campaign:_init_worker_from_fork',initializer_payload=None)
    recursive_rows=[recursive_batch.results[k] for k in sorted(recursive_batch.results)]

    # Actual reserve-capability basis: all 31 directed operators on one complete recursive candidate relation.
    op_rows=[]; seen_ops=set()
    for q in qrows:
        op=tuple(map(int,q['operator']))
        if op in seen_ops: continue
        seen_ops.add(op); op_rows.append(q)
    reserve_tasks=[]
    first_idx=0
    for k,q in enumerate(op_rows):
        reserve_tasks.append(TaskSpec(task_id=f'g3s4-basis-{k:02d}',task_kind='G3_S4_RECURSIVE_RESERVE_CAPABILITY_BASIS',binding_sha256=binding,payload={'first_key_index':first_idx,'operator':[int(x) for x in q['operator']],'orientation':'TARGET_LEFT_CONTEXT_RIGHT'},cost_weight=32.0))
    reserve_batch=execute_tasks(reserve_tasks,worker_ref='infinity_grid.uplift_campaign:_g3_s4_reserve_basis_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_campaign:_init_worker_from_fork',initializer_payload=None)
    reserve_rows=[reserve_batch.results[k] for k in sorted(reserve_batch.results)]

    result=finalize_g3_s4_result(s0_result=s0,s1_result=s1,s2_result=s2,s3_result=s3,s3_replay_comparison=s3replay,local_three_rows=local_rows,pair_basis=pair_basis,recursive_rows=recursive_rows,reserve_basis_rows=reserve_rows,reproduction=repro)
    result['source_sha256']=source_sha256(); result['source_version']='0.30.15'; result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
      'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V10','registered_experiment_id':'G3:S4','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
      'parallel_stage_required':True,'local_three_parallel_batch':local_batch.metadata,'recursive_parallel_batch':recursive_batch.metadata,'reserve_basis_parallel_batch':reserve_batch.metadata,
      'task_count':len(local_tasks)+len(recursive_tasks)+len(reserve_tasks),'local_three_task_count':len(local_tasks),'recursive_task_count':len(recursive_tasks),'reserve_basis_task_count':len(reserve_tasks),'requested_workers':engine.policy.requested_workers,
      'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g2_graduation_preserved':True,'g3_graduated':False,'topology_promoted':False,
      'g3_phase0_spec_sha256':g3_phase0_spec()['science_sha256'],'g3_s4_spec_sha256':g3_s4_spec()['science_sha256'],'input_s3_science_sha256':s3.get('science_sha256'),'input_s3_replay_stable_payload_sha256':s3replay.get('stable_scientific_payload_sha256'),
      'canonical_pair_basis_exact_refs_used_only_for_reproducible_materialisation':True,'bruteforce_830304_exact_triples_executed':False,
    }
    write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'AUTHORITY.json',auth); write_json_atomic(stage_dir/'CHALLENGE_REPRODUCTION.json',repro); write_json_atomic(stage_dir/'PAIR_CANDIDATE_BASIS.json',pair_basis)
    with gzip.open(stage_dir/'LOCAL_THREE_RESERVATION_CONTINUATIONS.json.gz','wt',encoding='utf-8',compresslevel=6) as fh:
        json.dump({'schema_id':'IG_G3_S4_LOCAL_THREE_CONTINUATION_INDEX_V1','task_count':len(local_rows),'rows':local_rows},fh,sort_keys=True,separators=(',',':'))
    with gzip.open(stage_dir/'RECURSIVE_P3_MATERIALISATION_ROWS.json.gz','wt',encoding='utf-8',compresslevel=6) as fh:
        json.dump({'schema_id':'IG_G3_S4_RECURSIVE_P3_MATERIALISATION_INDEX_V1','task_count':len(recursive_rows),'rows':recursive_rows},fh,sort_keys=True,separators=(',',':'))
    write_json_atomic(stage_dir/'RECURSIVE_RESERVE_OPERATOR_BASIS.json',{'schema_id':'IG_G3_S4_RECURSIVE_RESERVE_OPERATOR_BASIS_V1','rows':reserve_rows})
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:S4',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_started=True,g3_s5_unlocked=result.get('g3_s5_unlocked',False),g3_graduated=False)
    return result



def run_g3_s5_native(*, engine:UpliftCampaignEngine, s4_result:str|Path, s4_replay_comparison:str|Path, s4_pair_basis:str|Path, s4_recursive_rows:str|Path, s4_reserve_basis:str|Path)->dict[str,Any]:
    """Run registered G3:S5 finite CAPS7 read/write descriptor audit.

    S5 consumes the certified S4 evidence. It performs no lower-G/G3 rematerialisation and
    does not extend recursive depth; S6 owns that graduation gate.
    """
    spec=engine.experiment('G3:S5')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not True:
        raise UpliftCampaignError('G3:S5 not registered as descriptor-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_S5'; stage_dir.mkdir(parents=True,exist_ok=True)
    s4=_load_json(s4_result); replay=_load_json(s4_replay_comparison); pair=_load_json(s4_pair_basis); reserve=_load_json(s4_reserve_basis)
    auth=verify_g3_s5_authority(s4_result=s4,s4_replay_comparison=replay)
    census=certify_g3_s5_candidate_caps7_census(pair_basis=pair,recursive_rows_path=s4_recursive_rows)
    reserve_fac=verify_g3_s5_recursive_reserve_basis(reserve)
    impl=g3_s5_implementation_read_surface_audit()
    arg=g3_s5_caps7_factorisation_argument(candidate_census=census,reserve_factorisation=reserve_fac,implementation_audit=impl)
    result=finalize_g3_s5_result(authority=auth,candidate_census=census,reserve_factorisation=reserve_fac,implementation_audit=impl,factorisation_argument=arg)
    result['source_sha256']=source_sha256(); result['source_version']='0.30.16'; result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
      'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V11','registered_experiment_id':'G3:S5','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
      'parallel_stage_required':False,'task_count':0,'requested_workers':engine.policy.requested_workers,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,
      'promotion':True,'g2_graduation_preserved':True,'g3_graduated':False,'topology_promoted':False,'heavy_s4_census_rebuilt':False,
      'g3_phase0_spec_sha256':g3_phase0_spec()['science_sha256'],'g3_s5_spec_sha256':g3_s5_spec()['science_sha256'],'input_s4_science_sha256':s4.get('science_sha256'),
      'input_s4_certification':replay.get('certification'),'input_s4_science_sha_equal':replay.get('science_sha_equal'),
    }
    write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'AUTHORITY.json',auth)
    write_json_atomic(stage_dir/'CANDIDATE_CAPS7_CENSUS.json',census); write_json_atomic(stage_dir/'RECURSIVE_RESERVE_FACTORISATION.json',reserve_fac); write_json_atomic(stage_dir/'IMPLEMENTATION_READ_SURFACE_AUDIT.json',impl); write_json_atomic(stage_dir/'CAPS7_FACTORISATION_ARGUMENT.json',arg)
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:S5',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_started=True,g3_s6_unlocked=result.get('g3_s6_unlocked',False),g3_graduated=False,topology_promoted=False)
    return result



def run_g3_s6_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s5_result:str|Path, s5_replay_comparison:str|Path, regression_evidence:str|Path)->dict[str,Any]:
    """Run registered G3:S6 primary recursive-closure / graduation-candidate audit."""
    spec=engine.experiment('G3:S6')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not True:
        raise UpliftCampaignError('G3:S6 not registered as promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_S6'; stage_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=stage_dir/'case_checkpoints'; checkpoint_dir.mkdir(parents=True,exist_ok=True)
    s0=_load_json(s0_result); s5=_load_json(s5_result); replay=_load_json(s5_replay_comparison); reg=_load_json(regression_evidence)
    current_source=source_sha256()
    auth=verify_g3_s6_authority(s0_result=s0,s5_result=s5,s5_replay=replay)
    rgate=verify_g3_s6_regression_evidence(reg,current_source_sha256=current_source)
    theorem=recursive_caps7_relation_factorisation_theorem(s5)
    hidden=g3_s6_hidden_audit()
    states,repro=reproduce_s0_challenge_states(engine=engine,s0_result=s0)
    refs=sorted(states)
    ops=sorted({tuple(map(int,x)) for x in s5.get('recursive_reserve_factorisation',{}).get('operator_basis',[])})
    plan=g3_s6_holdout_plan(refs,ops)
    write_json_atomic(stage_dir/'FRESH_HOLDOUT_PLAN.json',plan)
    binding=canonical_sha256({'stage':'G3:S6','source_sha256':current_source,'registry_sha256':engine.registry['registry_sha256'],
                              'authority_sha256':auth['science_sha256'],'s6_spec_sha256':g3_s6_spec()['science_sha256'],
                              'holdout_plan_sha256':plan['science_sha256'],'relation_spec_sha256':relation_spec()['spec_sha256']})
    uplift_g3_s6._WORKER_CONTEXT={'engine':engine.session.engine,'states':states,'checkpoint_dir':str(checkpoint_dir),'binding_sha256':binding}
    task_defs=[]
    for c in plan['pair_plus_pair_cases']:
        task_defs.append(('PAIR_PLUS_PAIR',c,1.0))
    for c in plan['deep_p5_cases']:
        task_defs.append(('DEEP_P5',c,1.0))
    for c in plan['rebracketing_cases']:
        task_defs.append(('REBRACKET',c,1.0))
    recovered={}; tasks=[]
    for kind,c,cost in task_defs:
        cid=str(c['case_id'])
        cp=uplift_g3_s6.load_case_checkpoint(checkpoint_dir=checkpoint_dir,binding_sha256=binding,kind=kind,case_id=cid)
        if cp is not None:
            recovered[cid]=cp
        else:
            tasks.append(TaskSpec(task_id=cid,task_kind='G3_S6_'+kind,binding_sha256=binding,payload={'kind':kind,'case':c},cost_weight=cost))
    batch=execute_tasks(tasks,worker_ref='infinity_grid.uplift_g3_s6:holdout_worker',policy=engine.policy,
                        initializer_ref='infinity_grid.uplift_g3_s6:init_holdout_worker_from_fork',initializer_payload=None)
    all_results=dict(recovered); all_results.update(batch.results)
    missing=[str(c['case_id']) for _kind,c,_cost in task_defs if str(c['case_id']) not in all_results]
    if missing: raise UpliftCampaignError('G3:S6 missing case results after execution: '+','.join(missing))
    pp=[all_results[str(c['case_id'])] for c in plan['pair_plus_pair_cases']]
    deep=[all_results[str(c['case_id'])] for c in plan['deep_p5_cases']]
    rb=[all_results[str(c['case_id'])] for c in plan['rebracketing_cases']]
    checkpoint_index={'schema_id':'IG_G3_S6_CASE_CHECKPOINT_INDEX_V1','binding_sha256':binding,'case_count':len(task_defs),
                      'reused_case_count':len(recovered),'executed_case_count':len(tasks),
                      'case_result_sha256':{cid:canonical_sha256(all_results[cid]) for cid in sorted(all_results)}}
    checkpoint_index['science_sha256']=canonical_sha256(checkpoint_index)
    write_json_atomic(stage_dir/'CASE_CHECKPOINT_INDEX.json',checkpoint_index)
    holdout=aggregate_g3_s6_holdout(plan,pp,deep,rb)
    result=g3_s6_primary_result(authority=auth,regression=rgate,theorem=theorem,holdout=holdout,hidden=hidden)
    result['source_sha256']=current_source; result['source_version']='0.30.23'; result['registry_sha256']=engine.registry['registry_sha256']
    parallel_meta=dict(batch.metadata); parallel_meta.update({'reused_case_count':len(recovered),'executed_case_count':len(tasks),
                                                              'per_case_checkpointing':True,'checkpoint_binding_sha256':binding})
    result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V13','registered_experiment_id':'G3:S6',
        'registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
        'parallel_batch':parallel_meta,'task_count':len(task_defs),'requested_workers':engine.policy.requested_workers,
        'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':True,
        'primary_run_can_graduate_g3':False,'g2_graduation_preserved':True,'g3_graduated':False,'topology_promoted':False,
        'g3_phase0_spec_sha256':g3_phase0_spec()['science_sha256'],'g3_s6_spec_sha256':g3_s6_spec()['science_sha256'],
        'input_s5_science_sha256':s5.get('science_sha256'),'input_s5_replay_certification':replay.get('certification'),
        'challenge_reproduction_science_sha256':repro.get('science_sha256'),'exact_relation_certificate_engine':True,
        'p5_full_final_identity_frontier_materialized':False,'bounded_exact_sentinels':True,
        'successor_identity_cardinality_observed':False}
    write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']);
    write_json_atomic(stage_dir/'AUTHORITY.json',auth); write_json_atomic(stage_dir/'REGRESSION_GATE.json',rgate);
    write_json_atomic(stage_dir/'RECURSIVE_CAPS7_RELATION_FACTORISATION_THEOREM.json',theorem); write_json_atomic(stage_dir/'FRESH_HOLDOUT_RESULT.json',holdout);
    write_json_atomic(stage_dir/'NO_HIDDEN_SELECTOR_READ_AUDIT.json',hidden); write_json_atomic(stage_dir/'CHALLENGE_REPRODUCTION.json',repro);
    write_json_atomic(stage_dir/'PAIR_PLUS_PAIR_CASE_RESULTS.json',{'schema_id':'IG_G3_S6_PAIR_PLUS_PAIR_CASE_RESULTS_V2','cases':pp});
    write_json_atomic(stage_dir/'DEEP_P5_CASE_RESULTS.json',{'schema_id':'IG_G3_S6_DEEP_P5_CASE_RESULTS_V2','cases':deep});
    write_json_atomic(stage_dir/'REBRACKET_CASE_RESULTS.json',{'schema_id':'IG_G3_S6_REBRACKET_CASE_RESULTS_V2','cases':rb})
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:S6',last_result_status=result['status'],
                 classification=result['classification'],g2_graduated=True,g3_started=True,g3_graduation_candidate=result.get('g3_graduation_candidate',False),
                 g3_graduated=False,topology_promoted=False)
    return result

def run_g3_r0_native(*, engine:UpliftCampaignEngine, graduation_certificate:str|Path, s0_result:str|Path)->dict[str,Any]:
    """Run registered non-promoting G3:R0 hierarchical structural reconnaissance."""
    spec=engine.experiment('G3:R0.STRUCTURAL_AUDIT')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G3:R0 structural audit not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_R0_STRUCTURAL_AUDIT'; stage_dir.mkdir(parents=True,exist_ok=True)
    cert=_load_json(graduation_certificate); s0=_load_json(s0_result)
    result=run_g3_r0_recon(engine=engine,graduation_certificate=cert,s0_result=s0)
    result['source_sha256']=source_sha256(); result['source_version']='0.30.24'; result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
        'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V14',
        'registered_experiment_id':'G3:R0.STRUCTURAL_AUDIT',
        'registry_sha256':engine.registry['registry_sha256'],
        'campaign_materialization_count':engine.materialization_count,
        'parallel_stage_required':False,
        'execution_backend_owned_by_decoder':True,
        'stage_specific_external_science_runner':False,
        'promotion':False,
        'g2_graduation_preserved':True,
        'g3_graduation_preserved':True,
        'g4_started':False,
        'r0_spec_science_sha256':g3_r0_spec()['science_sha256'],
    }
    write_json_atomic(stage_dir/'RESULT.json',result)
    write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:R0',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_graduated=True,r0_complete=result['status']=='PASS',g4_started=False)
    return result


def run_g3_r1_native(*, engine:UpliftCampaignEngine, r0_result:str|Path, r0_replay_comparison:str|Path)->dict[str,Any]:
    """Run registered non-promoting G3:R1 finite fiber/moduli audit."""
    spec=engine.experiment('G3:R1.FIBER_MODULI_AUDIT')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G3:R1 fiber/moduli audit not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_R1_FIBER_MODULI_AUDIT'; stage_dir.mkdir(parents=True,exist_ok=True)
    r0=_load_json(r0_result); replay=_load_json(r0_replay_comparison)
    result=run_g3_r1_fiber_moduli_audit(engine=engine,r0_result=r0,r0_replay=replay)
    result['source_sha256']=source_sha256(); result['source_version']='0.30.25'; result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
        'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V15',
        'registered_experiment_id':'G3:R1.FIBER_MODULI_AUDIT',
        'registry_sha256':engine.registry['registry_sha256'],
        'campaign_materialization_count':engine.materialization_count,
        'parallel_stage_required':False,
        'execution_backend_owned_by_decoder':True,
        'stage_specific_external_science_runner':False,
        'promotion':False,
        'g3_graduation_preserved':True,
        'g4_started':False,
        'r1_spec_science_sha256':g3_r1_spec()['science_sha256'],
        'exact_r0_frontier_rematerialized':False,
        'structural_tree_shape_max_g2_units':g3_r1_spec()['shape_transition_census']['max_g2_units'],
    }
    write_json_atomic(stage_dir/'RESULT.json',result)
    write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:R1',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_graduated=True,r0_complete=True,r1_complete=result['status']=='PASS',g4_started=False)
    return result



def run_g3_r2_native(*, engine:UpliftCampaignEngine, r1_result:str|Path, r1_replay_comparison:str|Path)->dict[str,Any]:
    """Run registered non-promoting G3:R2 adaptive maturation sweep."""
    spec=engine.experiment('G3:R2.ADAPTIVE_MATURATION')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G3:R2 adaptive maturation not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_R2_ADAPTIVE_MATURATION'; stage_dir.mkdir(parents=True,exist_ok=True)
    r1=_load_json(r1_result); replay=_load_json(r1_replay_comparison)
    auth=verify_r2_authority(r1,replay)
    write_json_atomic(stage_dir/'AUTHORITY.json',auth)
    dense=dense_adaptive_maturation(r1,progress_path=stage_dir/'PROGRESS.json')
    write_json_atomic(stage_dir/'DENSE_MATURATION.json',dense)

    sentinel_rows=[]; batch_meta=None; stable_batch=None
    if dense.get('event') is None:
        cfg=g3_r2_spec()['sparse_sentinels']; binding=canonical_sha256({
            'stage':'G3:R2','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],
            'authority_sha256':auth['science_sha256'],'r2_spec_sha256':g3_r2_spec()['science_sha256'],
            'dense_sha256':dense['science_sha256']})
        tasks=[TaskSpec(task_id=f"G3_R2_SENTINEL_{int(n)}",task_kind='G3_R2_SPARSE_SENTINEL',binding_sha256=binding,
                        payload={'g2_unit_count':int(n),'r2_spec_science_sha256':g3_r2_spec()['science_sha256']},
                        cost_weight=float(n)) for n in cfg['g2_unit_counts']]
        batch=execute_tasks(tasks,worker_ref='infinity_grid.uplift_g3_r2:sentinel_rank_audit_worker',policy=engine.stage_policy)
        sentinel_rows=[batch.results[t.task_id] for t in tasks]
        batch_meta=batch.metadata
        stable_batch={'schema_id':'IG_G3_R2_SENTINEL_EXECUTION_SCIENCE_V1','task_count':len(tasks),
                      'task_ids':[t.task_id for t in tasks],
                      'sentinel_ranks':[int(x) for x in cfg['g2_unit_counts']],
                      'worker_ref':'infinity_grid.uplift_g3_r2:sentinel_rank_audit_worker'}
        stable_batch['science_sha256']=canonical_sha256(stable_batch)
        write_json_atomic(stage_dir/'SPARSE_SENTINELS.json',{'schema_id':'IG_G3_R2_SPARSE_SENTINEL_BATCH_V1','rows':sentinel_rows,'science_sha256':canonical_sha256(sentinel_rows)})

    result=finalize_r2_result(authority=auth,dense=dense,sentinel_rows=sentinel_rows,sentinel_batch_science=stable_batch)
    result['source_sha256']=source_sha256(); result['source_version']='0.30.26'; result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
        'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V17',
        'registered_experiment_id':'G3:R2.ADAPTIVE_MATURATION',
        'registry_sha256':engine.registry['registry_sha256'],
        'campaign_materialization_count':engine.materialization_count,
        'parallel_stage_required':dense.get('event') is None,
        'execution_backend_owned_by_decoder':True,
        'stage_specific_external_science_runner':False,
        'promotion':False,
        'g3_graduation_preserved':True,
        'g4_started':False,
        'r2_spec_science_sha256':g3_r2_spec()['science_sha256'],
        'exact_g3_relation_frontier_rematerialized':False,
        'dense_exhaustive_scope_end_rank':dense['exhaustive_scope_end_rank'],
        'sentinel_batch_execution_metadata':batch_meta,
    }
    write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:R2',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_graduated=True,r0_complete=True,r1_complete=True,r2_complete=result['status']=='PASS',g4_started=False)
    return result



def run_g3_r3_native(*, engine:UpliftCampaignEngine)->dict[str,Any]:
    """Run registered G3:R3 structural-algebra rebase after the delayed R18 break."""
    spec=engine.experiment('G3:R3.STRUCTURAL_ALGEBRA_REBASE')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G3:R3 structural rebase not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G3_R3_STRUCTURAL_ALGEBRA_REBASE'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G3:R3.STRUCTURAL_ALGEBRA_REBASE',workers_requested=engine.stage_policy.requested_workers if hasattr(engine.stage_policy,'requested_workers') else 4,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        auth=verify_g3_r3_rebase_authority(); write_json_atomic(stage_dir/'AUTHORITY.json',auth)
        theorem=g3_r3_structural_theorem_certificate(); write_json_atomic(stage_dir/'STRUCTURAL_THEOREM.json',theorem)
        telemetry.phase('SCIENCE_CENSUS')
        max_total=int(g3_r3_spec()['exhaustive_binary_graft_validation']['max_total_vertex_count'])
        binding=canonical_sha256({'stage':'G3:R3','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'r3_spec_sha256':g3_r3_spec()['science_sha256'],'theorem_sha256':theorem['science_sha256']})
        tasks=[TaskSpec(task_id=f'G3_R3_TOTAL_{n:02d}',task_kind='G3_R3_BINARY_GRAFT_TOTAL_RANK',binding_sha256=binding,payload={'total_vertex_count':n,'r3_spec_science_sha256':g3_r3_spec()['science_sha256']},cost_weight=float(n*n)) for n in range(2,max_total+1)]
        batch=execute_tasks(tasks,worker_ref='infinity_grid.uplift_g3_r3:binary_graft_exhaustive_total_rank_worker',policy=engine.stage_policy)
        rank_rows=[batch.results[t.task_id] for t in tasks]
        execution_science={'schema_id':'IG_G3_R3_BINARY_GRAFT_EXECUTION_SCIENCE_V1','task_ids':[t.task_id for t in tasks],'total_ranks':list(range(2,max_total+1)),'worker_ref':'infinity_grid.uplift_g3_r3:binary_graft_exhaustive_total_rank_worker','requested_workers':batch.metadata.get('requested_workers') if isinstance(batch.metadata,dict) else None}
        execution_science['science_sha256']=canonical_sha256(execution_science)
        write_json_atomic(stage_dir/'BINARY_GRAFT_EXHAUSTIVE_ROWS.json',{'schema_id':'IG_G3_R3_BINARY_GRAFT_EXHAUSTIVE_BATCH_V1','rows':rank_rows,'science_sha256':canonical_sha256(rank_rows)})
        telemetry.phase('RESULT_SEAL')
        result=finalize_g3_r3_result(authority=auth,theorem=theorem,rank_rows=rank_rows,execution_science=execution_science)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.45'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V28','registered_experiment_id':'G3:R3.STRUCTURAL_ALGEBRA_REBASE','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':True,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g3_public_caps7_graduation_preserved':True,'hidden_state_promoted_to_public_g3':False,'g4_forward_use_blocked_pending_rebase':True,'r3_spec_science_sha256':g3_r3_spec()['science_sha256'],'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True,'batch_execution_metadata':batch.metadata}
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G3:R3',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_graduated=True,g3_r3_complete=result.get('status')=='PASS',g4_started=False,g4_forward_use_blocked=True,g4_rebase_required=True)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise


def run_g4_s0_rebase_native(*, engine:UpliftCampaignEngine, r1_result:str|Path, r3_result:str|Path, r3_replay:str|Path, r3_closeout:str|Path)->dict[str,Any]:
    """Run G4:S0.REBASE from certified G3:R3 authority; no historical G4 result is trusted."""
    spec=engine.experiment('G4:S0.REBASE')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G4:S0.REBASE not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G4_S0_REBASE'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S0.REBASE',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        r1=_load_json(r1_result); r3=_load_json(r3_result); replay=_load_json(r3_replay); close=_load_json(r3_closeout)
        telemetry.phase('INPUT_MATERIALIZATION')
        result=run_g4_s0_rebase(r1_result=r1,r3_result=r3,r3_replay=replay,r3_closeout=close)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.46'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V29','registered_experiment_id':'G4:S0.REBASE','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':False,'requested_workers':1,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g3_public_caps7_graduation_preserved':True,'hidden_state_promoted':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','g4_rebase_complete':False,'g4_s0_rebase_spec_sha256':g4_s0_rebase_spec()['science_sha256'],'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'CERTIFIED_G3_TERM_CORPUS_REBASE.json',result['certified_term_corpus']); write_json_atomic(stage_dir/'AUTHORITY.json',result['authority'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S0.REBASE',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_graduated=True,g4_started=True,g4_rebase_required=True,g4_rebase_complete=False,g4_s1_rebase_unlocked=result.get('g4_s1_rebase_unlocked',False),historical_g4_forward_use_blocked=True,g4_graduated=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise

def run_g4_s0_native(*, engine:UpliftCampaignEngine, r1_result:str|Path, r2_result:str|Path, r2_replay:str|Path, r2_closeout:str|Path)->dict[str,Any]:
    """Run registered G4:S0 and freeze the reusable whole-G3 term basis for S1.

    S0 is the base-building stage.  It consumes certified G3/R1-R2 evidence and
    materializes the two exact-at-G3-boundary challenge terms directly from the
    certified G3 one-cross composition data.  It never reopens G1:R100/G2
    implementation archaeology.
    """
    spec=engine.experiment('G4:S0')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G4:S0 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G4_S0'; stage_dir.mkdir(parents=True,exist_ok=True)
    result=run_g4_s0(engine=engine,r1_result=_load_json(r1_result),r2_result=_load_json(r2_result),r2_replay=_load_json(r2_replay),r2_closeout=_load_json(r2_closeout))
    result['source_sha256']=source_sha256(); result['source_version']='0.30.37'; result['registry_sha256']=engine.registry['registry_sha256']
    result['execution_metadata']={
      'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V21','registered_experiment_id':'G4:S0',
      'registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
      'parallel_stage_required':False,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,
      'promotion':False,'g3_graduation_preserved':True,'g4_started':True,'g4_graduated':False,
      'g4_phase0_spec_sha256':g4_phase0_spec()['science_sha256'],'g4_s0_spec_sha256':g4_s0_spec()['science_sha256'],
      's0_materializes_reusable_base':True,
      'certified_term_corpus_sha256':result['certified_term_corpus']['science_sha256'],
      'carrier_input_mode':'CERTIFIED_G3_COMPOSITION_TERM_FROM_G3_AUTHORITY',
      'lower_layer_rematerialization':False,
      'g1_r100_materialization':False,
    }
    write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
    write_json_atomic(stage_dir/'CERTIFIED_G3_TERM_CORPUS.json',result['certified_term_corpus'])
    _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S0',last_result_status=result['status'],classification=result['classification'],g2_graduated=True,g3_graduated=True,g4_started=True,g4_s1_unlocked=result.get('g4_s1_unlocked',False),g4_graduated=False)
    return result


def run_g4_s1_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s0_replay_comparison:str|Path, s0_closeout:str|Path)->dict[str,Any]:
    """Run preregistered G4:S1 directly on the reusable G3 term basis frozen in S0.

    There is deliberately no lower-layer reconstruction path here.  G4 treats
    complete G3 terms as its previous-layer units and uses only their certified
    G3-boundary representation.  The frozen 248 scientific observations are
    evaluated through 124 unique exact computation kernels.
    """
    spec=engine.experiment('G4:S1')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G4:S1 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G4_S1'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S1',workers_requested=engine.policy.requested_workers,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        s0=_load_json(s0_result); replay=_load_json(s0_replay_comparison); closeout=_load_json(s0_closeout)
        auth=verify_g4_s1_authority(s0,replay,closeout)

        telemetry.phase('INPUT_MATERIALIZATION')
        states,repro=load_g4_s0_term_states(s0)
        refs=sorted(states)
        bridge_pairs=sorted({tuple(map(int,x)) for x in engine.session.bridge_pairs})
        if len(bridge_pairs)!=31: raise UpliftCampaignError('G4:S1 requires 31 bridge operators')

        telemetry.phase('TASK_GENERATION')
        binding=canonical_sha256({'stage':'G4:S1','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'s1_preregistered_spec_sha256':g4_s1_spec()['science_sha256']})
        kernels=[]; progress_units={}
        for lr in refs:
            for rr in refs:
                for a,b in bridge_pairs:
                    tid=f'g4s1k-{len(kernels):04d}'
                    kernels.append(TaskSpec(task_id=tid,task_kind='G4_S1_PAIR_KERNEL',binding_sha256=binding,payload={'left_ref':lr,'right_ref':rr,'operator':[a,b]},cost_weight=1.0))
                    progress_units[tid]=2
        scientific_task_count=len(refs)*len(refs)*len(bridge_pairs)*2
        if len(kernels)*2 != scientific_task_count:
            raise UpliftCampaignError('G4:S1 kernel/scientific-task accounting mismatch')

        # Tiny portable worker context: two four-node G3 terms.  No inherited deep DAG,
        # no R100/G2/G3 rematerialization, and no special fork-only start method.
        worker_init={'states':{ref:states[ref].to_wire() for ref in refs}}
        telemetry.phase('SCIENCE_CENSUS',tasks_total=scientific_task_count,kernels_total=len(kernels))
        batch=execute_tasks(
            kernels,
            worker_ref='infinity_grid.uplift_g4_s1:g4_s1_term_kernel_worker',
            policy=engine.policy,
            initializer_ref='infinity_grid.uplift_g4_s1:init_g4_s1_term_worker',
            initializer_payload=worker_init,
            telemetry=telemetry,
            scientific_task_units=progress_units,
        )

        telemetry.phase('MERGE')
        rows=[]
        for k in sorted(batch.results):
            kr=batch.results[k]
            lr=str(kr['left_ref']); rr=str(kr['right_ref']); op=[int(x) for x in kr['operator']]
            measurement=kr['kernel_measurement']
            rows.append({'target_ref':lr,'context_ref':rr,'operator':op,'orientation':'TARGET_LEFT_CONTEXT_RIGHT','operational_signature':expand_g4_pair_context_kernel_signature(measurement,orientation='TARGET_LEFT_CONTEXT_RIGHT')})
            rows.append({'target_ref':rr,'context_ref':lr,'operator':op,'orientation':'CONTEXT_LEFT_TARGET_RIGHT','operational_signature':expand_g4_pair_context_kernel_signature(measurement,orientation='CONTEXT_LEFT_TARGET_RIGHT')})
        rows=sorted(rows,key=lambda r:(str(r['target_ref']),str(r['context_ref']),str(r['orientation']),tuple(r['operator'])))
        if len(rows)!=scientific_task_count: raise UpliftCampaignError('G4:S1 expanded scientific task table incomplete')

        telemetry.phase('INTERPRETATION')
        result=finalize_g4_s1_result(s0_result=s0,authority=auth,reproduction=repro,task_rows=rows)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.37'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={
          'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V21','registered_experiment_id':'G4:S1',
          'registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
          'parallel_stage_required':True,'parallel_batch':batch.metadata,
          'task_count':scientific_task_count,'kernel_count':len(kernels),'kernel_dedup_ratio':scientific_task_count/len(kernels),
          'requested_workers':engine.policy.requested_workers,
          'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,
          'g3_graduation_preserved':True,'g4_graduated':False,'topology_promoted':False,'shell_profile_promoted':False,
          'g4_phase0_spec_sha256':g4_phase0_spec()['science_sha256'],'g4_s1_preregistered_spec_sha256':g4_s1_spec()['science_sha256'],
          'input_s0_science_sha256':s0.get('science_sha256'),
          'input_s0_term_corpus_sha256':s0['certified_term_corpus']['science_sha256'],
          'execution_optimization':'S0_FROZEN_G3_TERM_BASE_PLUS_124_KERNEL_DEDUP_V1',
          'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True,
          'carrier_input_mode':'CERTIFIED_G3_TERM_CORPUS_FROM_S0',
          'lower_layer_rematerialization':False,'g1_r100_materialization':False,
          'special_fork_materialization':False,
        }

        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
        write_json_atomic(stage_dir/'CHALLENGE_BASE_LOAD.json',repro)
        with gzip.open(stage_dir/'PAIR_CONTEXT_SIGNATURES.json.gz','wt',encoding='utf-8',compresslevel=1) as fh:
            json.dump({'schema_id':'IG_G4_S1_PAIR_CONTEXT_SIGNATURE_INDEX_V2','task_count':len(rows),'rows':rows},fh,sort_keys=True,separators=(',',':'))
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S1',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_s2_unlocked=result.get('g4_s2_unlocked',False),g4_graduated=False)
        telemetry.complete()
        return result
    except BaseException as exc:
        telemetry.fail(exc)
        raise



def run_g4_s1_rebase_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s0_replay_comparison:str|Path, s0_closeout:str|Path)->dict[str,Any]:
    spec=engine.experiment('G4:S1.REBASE')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False: raise UpliftCampaignError('G4:S1.REBASE registry contract')
    stage_dir=engine.run_root/'stages'/'G4_S1_REBASE'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S1.REBASE',workers_requested=engine.policy.requested_workers,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD'); s0=_load_json(s0_result); replay=_load_json(s0_replay_comparison); close=_load_json(s0_closeout); auth=verify_g4_s1_rebase_authority(s0,replay,close)
        telemetry.phase('INPUT_MATERIALIZATION'); states,keyrefs,loadmeta=load_g4_s1_rebase_states(s0); refs=sorted(states); bridge_pairs=sorted({tuple(map(int,x)) for x in engine.session.bridge_pairs})
        if len(refs)!=4 or len(bridge_pairs)!=31: raise UpliftCampaignError('G4:S1.REBASE frozen basis mismatch')
        telemetry.phase('TASK_GENERATION'); binding=canonical_sha256({'stage':'G4:S1.REBASE','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'spec_sha256':g4_s1_rebase_spec()['science_sha256']})
        kernels=[]; progress={}
        for lr in refs:
          for rr in refs:
            for a,b in bridge_pairs:
              tid=f'g4s1r-{len(kernels):04d}'; kernels.append(TaskSpec(task_id=tid,task_kind='G4_S1_REBASE_PAIR_KERNEL',binding_sha256=binding,payload={'left_ref':lr,'right_ref':rr,'operator':[a,b]},cost_weight=1.0)); progress[tid]=2
        if len(kernels)!=496: raise UpliftCampaignError('G4:S1.REBASE kernel count mismatch')
        telemetry.phase('SCIENCE_CENSUS',tasks_total=992,kernels_total=496)
        batch=execute_tasks(kernels,worker_ref='infinity_grid.uplift_g4_s1_rebase:kernel_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_g4_s1_rebase:init_worker',initializer_payload={'states':{r:states[r].to_wire() for r in refs}},telemetry=telemetry,scientific_task_units=progress)
        telemetry.phase('MERGE'); rows=[]
        for k in sorted(batch.results):
          kr=batch.results[k]; lr,rr=str(kr['left_ref']),str(kr['right_ref']); op=[int(x) for x in kr['operator']]; meas=kr['kernel_measurement']
          rows.append({'target_ref':lr,'context_ref':rr,'operator':op,'orientation':'TARGET_LEFT_CONTEXT_RIGHT','operational_signature':expand_g4_pair_context_kernel_signature(meas,orientation='TARGET_LEFT_CONTEXT_RIGHT')})
          rows.append({'target_ref':rr,'context_ref':lr,'operator':op,'orientation':'CONTEXT_LEFT_TARGET_RIGHT','operational_signature':expand_g4_pair_context_kernel_signature(meas,orientation='CONTEXT_LEFT_TARGET_RIGHT')})
        rows=sorted(rows,key=lambda r:(r['target_ref'],r['context_ref'],r['orientation'],tuple(r['operator'])))
        telemetry.phase('INTERPRETATION'); result=finalize_g4_s1_rebase(s0=s0,authority=auth,load_meta=loadmeta,rows=rows); result['source_sha256']=source_sha256(); result['source_version']='0.30.47'; result['registry_sha256']=engine.registry['registry_sha256']; result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V30','registered_experiment_id':'G4:S1.REBASE','parallel_stage_required':True,'parallel_batch':batch.metadata,'requested_workers':engine.policy.requested_workers,'kernel_count':496,'scientific_observation_count':992,'lower_layer_rematerialization':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','promotion':False,'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL'); write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'TERM_LOAD.json',loadmeta)
        with gzip.open(stage_dir/'PAIR_CONTEXT_SIGNATURES.json.gz','wt',encoding='utf-8',compresslevel=1) as fh: json.dump({'schema_id':'IG_G4_S1_REBASE_PAIR_CONTEXT_SIGNATURE_INDEX_V1','task_count':len(rows),'rows':rows},fh,sort_keys=True,separators=(',',':'))
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S1.REBASE',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_rebase_required=True,g4_rebase_complete=False,g4_s2_rebase_unlocked=True,historical_g4_forward_use_blocked=True,g4_graduated=False)
        telemetry.complete(); return result
    except BaseException as exc: telemetry.fail(exc); raise



def run_g4_s2_rebase_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s1_result:str|Path, s1_signature_index:str|Path, s1_replay_comparison:str|Path, s1_closeout:str|Path)->dict[str,Any]:
    """Lightweight corrected G4:S2 rebase audit; reuses S1.REBASE rows and runs zero new science kernels."""
    spec=engine.experiment('G4:S2.REBASE')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G4:S2.REBASE registry contract')
    stage_dir=engine.run_root/'stages'/'G4_S2_REBASE'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S2.REBASE',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        s0=_load_json(s0_result); s1=_load_json(s1_result); rep=_load_json(s1_replay_comparison); clo=_load_json(s1_closeout)
        telemetry.phase('INPUT_MATERIALIZATION')
        sig=load_g4_s1_rebase_signatures(s1_signature_index)
        telemetry.phase('INTERPRETATION',tasks_total=len(sig.get('rows',[])),kernels_total=0)
        result=audit_g4_s2_rebase(s0=s0,s1=s1,s1_replay=rep,s1_closeout=clo,signature_index=sig)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.49'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V31','registered_experiment_id':'G4:S2.REBASE','parallel_stage_required':False,'new_science_kernels':0,'reused_s1_observer_rows':len(sig.get('rows',[])),'lower_layer_rematerialization':False,'promotion':False,'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'TIER_QUOTIENT_AUDIT.json',{'schema_id':'IG_G4_S2_REBASE_TIER_QUOTIENT_AUDIT_V1','rows':result['tier_audit'],'science_sha256':canonical_sha256(result['tier_audit'])})
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S2.REBASE',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_rebase_required=True,g4_rebase_complete=False,g4_s3_rebase_unlocked=result.get('g4_s3_rebase_unlocked',False),historical_g4_forward_use_blocked=True,g4_graduated=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise



def run_g4_s3_rebase_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s2_result:str|Path, s2_replay_comparison:str|Path, s2_closeout:str|Path)->dict[str,Any]:
    """Corrected G4:S3 rebase; exact local two-reservation audit with factorized triple coverage."""
    spec=engine.experiment('G4:S3.REBASE')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False: raise UpliftCampaignError('G4:S3.REBASE registry contract')
    stage_dir=engine.run_root/'stages'/'G4_S3_REBASE'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S3.REBASE',workers_requested=engine.policy.requested_workers,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD'); s0=_load_json(s0_result); s2=_load_json(s2_result); rep=_load_json(s2_replay_comparison); clo=_load_json(s2_closeout); auth=verify_g4_s3_rebase_authority(s0=s0,s2=s2,s2_replay=rep,s2_closeout=clo)
        telemetry.phase('INPUT_MATERIALIZATION'); states,keyrefs,loadmeta=load_g4_s1_rebase_states(s0); refs=sorted(states)
        if len(refs)!=4: raise UpliftCampaignError('G4:S3.REBASE frozen term count mismatch')
        telemetry.phase('TASK_GENERATION'); binding=canonical_sha256({'stage':'G4:S3.REBASE','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'spec_sha256':g4_s3_rebase_spec()['science_sha256']}); tasks=[]
        for ref in refs:
          for a in range(7):
            for b in range(7):
              tasks.append(TaskSpec(task_id=f'g4s3r-{len(tasks):04d}',task_kind='G4_S3_REBASE_TWO_RESERVATION',binding_sha256=binding,payload={'term_ref':ref,'first_type':a,'second_type':b},cost_weight=1.0))
        if len(tasks)!=196: raise UpliftCampaignError('G4:S3.REBASE task count mismatch')
        telemetry.phase('SCIENCE_CENSUS',tasks_total=196,kernels_total=196)
        batch=execute_tasks(tasks,worker_ref='infinity_grid.uplift_g4_s3_rebase:local_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_g4_s3_rebase:init_worker',initializer_payload={'states':{r:states[r].to_wire() for r in refs}},telemetry=telemetry,scientific_task_units={t.task_id:1 for t in tasks})
        rows=[batch.results[k] for k in sorted(batch.results)]
        telemetry.phase('INTERPRETATION'); result=finalize_g4_s3_rebase(s0=s0,s2=s2,s2_replay=rep,s2_closeout=clo,task_rows=rows); result['source_sha256']=source_sha256(); result['source_version']='0.30.50'; result['registry_sha256']=engine.registry['registry_sha256']; result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V32','registered_experiment_id':'G4:S3.REBASE','parallel_stage_required':True,'parallel_batch':batch.metadata,'new_science_kernels':196,'lower_layer_rematerialization':False,'triple_contexts_materialized':False,'promotion':False,'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL'); write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'LOCAL_TWO_RESERVATION_ROWS.json',{'schema_id':'IG_G4_S3_REBASE_LOCAL_ROWS_V1','rows':rows,'science_sha256':canonical_sha256(rows)})
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S3.REBASE',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_rebase_required=True,g4_rebase_complete=False,g4_s4_rebase_unlocked=result.get('g4_s4_rebase_unlocked',False),historical_g4_forward_use_blocked=True,g4_graduated=False)
        telemetry.complete(); return result
    except BaseException as exc: telemetry.fail(exc); raise


def run_g4_s4_rebase_native(*, engine:UpliftCampaignEngine, s3_result:str|Path, s3_replay_comparison:str|Path, s3_closeout:str|Path)->dict[str,Any]:
    """Corrected G4:S4 rebase using exact symbolic post-recursive reserve factorization."""
    rs=engine.experiment('G4:S4.REBASE')
    if rs.get('execution_mode')!='NATIVE_HANDLER' or rs.get('promotion') is not False: raise UpliftCampaignError('G4:S4.REBASE registry contract')
    stage_dir=engine.run_root/'stages'/'G4_S4_REBASE'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S4.REBASE',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD'); s3=_load_json(s3_result); rep=_load_json(s3_replay_comparison); clo=_load_json(s3_closeout)
        telemetry.phase('INTERPRETATION',tasks_total=31**3,kernels_total=0)
        result=finalize_g4_s4_rebase(s3=s3,s3_replay=rep,s3_closeout=clo); result['source_sha256']=source_sha256(); result['source_version']='0.30.51'; result['registry_sha256']=engine.registry['registry_sha256']; result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V33','registered_experiment_id':'G4:S4.REBASE','parallel_stage_required':False,'new_parallel_science_kernels':0,'symbolic_checks':31**3,'lower_layer_rematerialization':False,'p3_contexts_materialized':False,'promotion':False,'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL'); write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'POST_RECURSIVE_RESERVE_AUDIT.json',result['post_recursive_reserve_audit'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S4.REBASE',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_rebase_required=True,g4_rebase_complete=False,g4_s5_rebase_unlocked=result.get('g4_s5_rebase_unlocked',False),historical_g4_forward_use_blocked=True,g4_graduated=False)
        telemetry.complete(); return result
    except BaseException as exc: telemetry.fail(exc); raise


def run_g4_s5_rebase_native(*, engine:UpliftCampaignEngine, s4_result:str|Path, s4_replay_comparison:str|Path, s4_closeout:str|Path, s3_result:str|Path)->dict[str,Any]:
    """Corrected G4:S5 rebase; symbolic finite read/write descriptor audit with zero new science kernels."""
    rs=engine.experiment('G4:S5.REBASE')
    if rs.get('execution_mode')!='NATIVE_HANDLER' or rs.get('promotion') is not True: raise UpliftCampaignError('G4:S5.REBASE registry contract')
    stage_dir=engine.run_root/'stages'/'G4_S5_REBASE'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S5.REBASE',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD'); s4=_load_json(s4_result); rep=_load_json(s4_replay_comparison); clo=_load_json(s4_closeout); s3=_load_json(s3_result)
        telemetry.phase('INTERPRETATION',tasks_total=0,kernels_total=0)
        result=finalize_g4_s5_rebase(s4=s4,s4_replay=rep,s4_closeout=clo,s3=s3); result['source_sha256']=source_sha256(); result['source_version']='0.30.52'; result['registry_sha256']=engine.registry['registry_sha256']; result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V34','registered_experiment_id':'G4:S5.REBASE','parallel_stage_required':False,'new_parallel_science_kernels':0,'lower_layer_rematerialization':False,'explicit_p3_materialization':False,'promotion':True,'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL'); write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'DESCRIPTOR_AUDIT.json',result['descriptor_audit'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S5.REBASE',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_rebase_required=True,g4_rebase_complete=False,g4_s6_rebase_unlocked=result.get('g4_s6_rebase_unlocked',False),historical_g4_forward_use_blocked=True,g4_graduated=False)
        telemetry.complete(); return result
    except BaseException as exc: telemetry.fail(exc); raise

def run_g4_s2_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s1_result:str|Path, s1_signature_index:str|Path, s1_replay_comparison:str|Path, s1_closeout:str|Path)->dict[str,Any]:
    """Run registered G4:S2 quotient/minimal added-read audit from certified S0/S1 only.

    This stage is intentionally light: it never reconstructs lower layers and does not
    rerun the S1 kernels.  It consumes the complete certified S1 248-row observer table
    and tests the preregistered descriptor ladder exactly.
    """
    spec=engine.experiment('G4:S2')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G4:S2 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G4_S2'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S2',workers_requested=engine.policy.requested_workers,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        s0=_load_json(s0_result); s1=_load_json(s1_result); replay=_load_json(s1_replay_comparison); closeout=_load_json(s1_closeout)
        auth=verify_g4_s2_authority(s0_result=s0,s1_result=s1,s1_replay=replay,s1_closeout=closeout)
        telemetry.phase('INPUT_MATERIALIZATION')
        sig_index=load_g4_s1_signature_index(s1_signature_index)
        telemetry.phase('INTERPRETATION')
        result=certify_g4_s2_minimal_added_read(s0_result=s0,s1_result=s1,s1_replay=replay,s1_closeout=closeout,signature_index=sig_index)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.39'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={
          'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V22','registered_experiment_id':'G4:S2',
          'registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
          'parallel_stage_required':False,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,
          'promotion':False,'g3_graduation_preserved':True,'g4_graduated':False,
          'g4_phase0_spec_sha256':g4_phase0_spec()['science_sha256'],'g4_s2_spec_sha256':g4_s2_spec()['science_sha256'],
          'input_s0_science_sha256':s0.get('science_sha256'),'input_s1_science_sha256':s1.get('science_sha256'),
          'input_s1_signature_task_count':int(sig_index.get('task_count',-1)),
          'carrier_input_mode':'CERTIFIED_G3_TERM_CORPUS_FROM_S0','lower_layer_rematerialization':False,'g1_r100_materialization':False,
          'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True,
        }
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
        write_json_atomic(stage_dir/'TIER_QUOTIENT_AUDIT.json',{'schema_id':'IG_G4_S2_TIER_QUOTIENT_AUDIT_V1','rows':result['tier_audit'],'science_sha256':canonical_sha256(result['tier_audit'])})
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S2',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_s3_unlocked=result.get('g4_s3_unlocked',False),g4_graduated=False)
        telemetry.complete()
        return result
    except BaseException as exc:
        telemetry.fail(exc)
        raise


def run_g4_s3_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s1_result:str|Path, s1_signature_index:str|Path, s1_replay_comparison:str|Path, s1_closeout:str|Path, s2_result:str|Path, s2_replay_comparison:str|Path, s2_closeout:str|Path)->dict[str,Any]:
    """Run preregistered G4:S3 P3/K3 higher-order compatibility reduction.

    The stage consumes only certified G4:S0/S1/S2 evidence.  It computes the
    complete ordered 7x7 local two-reservation continuation table for the two
    frozen G3 terms through Decoder-owned task execution, then applies the
    preregistered degree-2 P3/K3 factorisation.  No S3 science is run by merely
    registering this handler.
    """
    spec=engine.experiment('G4:S3')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G4:S3 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G4_S3'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S3',workers_requested=engine.policy.requested_workers,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        s0=_load_json(s0_result); s1=_load_json(s1_result); s1rep=_load_json(s1_replay_comparison); s1clo=_load_json(s1_closeout)
        s2=_load_json(s2_result); s2rep=_load_json(s2_replay_comparison); s2clo=_load_json(s2_closeout)
        sig_index=load_g4_s1_signature_index(s1_signature_index)
        auth=verify_g4_s3_authority(s0_result=s0,s1_result=s1,s1_replay=s1rep,s1_closeout=s1clo,s2_result=s2,s2_replay=s2rep,s2_closeout=s2clo,signature_index=sig_index)

        telemetry.phase('INPUT_MATERIALIZATION')
        states,repro=load_g4_s0_term_states(s0)
        refs=sorted(states)
        if len(refs)!=2: raise UpliftCampaignError('G4:S3 requires exactly two frozen S0 terms')

        telemetry.phase('TASK_GENERATION')
        binding=canonical_sha256({'stage':'G4:S3','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'s3_spec_sha256':g4_s3_spec()['science_sha256']})
        tasks=[]
        for ref in refs:
            for a in range(7):
                for b in range(7):
                    tasks.append(TaskSpec(task_id=f'g4s3-{len(tasks):04d}',task_kind='G4_S3_LOCAL_TWO_RESERVATION_CONTINUATION',binding_sha256=binding,payload={'term_ref':ref,'first_type':a,'second_type':b},cost_weight=1.0))
        if len(tasks)!=98: raise UpliftCampaignError('G4:S3 local task accounting mismatch')
        worker_init={'states':{ref:states[ref].to_wire() for ref in refs}}

        telemetry.phase('SCIENCE_CENSUS',tasks_total=len(tasks),kernels_total=len(tasks))
        batch=execute_tasks(
            tasks,
            worker_ref='infinity_grid.uplift_g4_s3:g4_s3_local_continuation_worker',
            policy=engine.policy,
            initializer_ref='infinity_grid.uplift_g4_s3:init_g4_s3_term_worker',
            initializer_payload=worker_init,
            telemetry=telemetry,
        )
        rows=[batch.results[k] for k in sorted(batch.results)]

        telemetry.phase('INTERPRETATION')
        result=finalize_g4_s3_result(
            s0_result=s0,s1_result=s1,s1_replay=s1rep,s1_closeout=s1clo,
            s2_result=s2,s2_replay=s2rep,s2_closeout=s2clo,signature_index=sig_index,
            task_rows=rows,reproduction=repro,
        )
        result['source_sha256']=source_sha256(); result['source_version']='0.30.40'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={
          'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V23','registered_experiment_id':'G4:S3',
          'registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,
          'parallel_stage_required':True,'parallel_batch':batch.metadata,'task_count':len(tasks),'kernel_count':len(tasks),
          'requested_workers':engine.policy.requested_workers,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,
          'promotion':False,'g3_graduation_preserved':True,'g4_graduated':False,'public_descriptor_promoted':False,'topology_promoted':False,'shell_profile_promoted':False,
          'g4_phase0_spec_sha256':g4_phase0_spec()['science_sha256'],'g4_s3_preregistered_spec_sha256':g4_s3_spec()['science_sha256'],
          'input_s0_science_sha256':s0.get('science_sha256'),'input_s1_science_sha256':s1.get('science_sha256'),'input_s2_science_sha256':s2.get('science_sha256'),
          'input_s1_signature_task_count':int(sig_index.get('task_count',-1)),
          'carrier_input_mode':'CERTIFIED_G3_TERM_CORPUS_FROM_S0','lower_layer_rematerialization':False,'g1_r100_materialization':False,
          'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True,
        }

        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata'])
        write_json_atomic(stage_dir/'AUTHORITY.json',auth); write_json_atomic(stage_dir/'CHALLENGE_BASE_LOAD.json',repro)
        with gzip.open(stage_dir/'LOCAL_TWO_RESERVATION_CONTINUATIONS.json.gz','wt',encoding='utf-8',compresslevel=1) as fh:
            json.dump({'schema_id':'IG_G4_S3_LOCAL_TWO_RESERVATION_INDEX_V1','task_count':len(rows),'rows':rows},fh,sort_keys=True,separators=(',',':'))
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S3',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_s4_unlocked=result.get('g4_s4_unlocked',False),g4_graduated=False)
        telemetry.complete()
        return result
    except BaseException as exc:
        telemetry.fail(exc)
        raise



def run_g4_s4_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s1_result:str|Path, s1_signature_index:str|Path, s2_result:str|Path, s3_result:str|Path, s3_replay_comparison:str|Path, s3_closeout:str|Path)->dict[str,Any]:
    """Run registered G4:S4 one-step relation-valued composition closure."""
    spec=engine.experiment('G4:S4')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G4:S4 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G4_S4'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S4',workers_requested=engine.policy.requested_workers,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        s0=_load_json(s0_result); s1=_load_json(s1_result); s2=_load_json(s2_result); s3=_load_json(s3_result); s3rep=_load_json(s3_replay_comparison); s3clo=_load_json(s3_closeout)
        sig_index=load_g4_s1_signature_index(s1_signature_index)
        auth=verify_g4_s4_authority(s0_result=s0,s1_result=s1,s2_result=s2,s3_result=s3,s3_replay=s3rep,s3_closeout=s3clo)
        telemetry.phase('INPUT_MATERIALIZATION')
        states,repro=load_g4_s0_term_states(s0); refs=sorted(states); bridge_pairs=sorted({tuple(map(int,x)) for x in engine.session.bridge_pairs})
        if len(refs)!=2 or len(bridge_pairs)!=31: raise UpliftCampaignError('G4:S4 frozen base/operator accounting mismatch')
        pair_basis,pair_relations,pair_rows=build_g4_s4_pair_basis(states=states,bridge_pairs=bridge_pairs,s1_index=sig_index)
        if pair_basis.get('status')!='PASS' or len(pair_rows)!=124: raise UpliftCampaignError('G4:S4 pair basis failed')
        binding=canonical_sha256({'stage':'G4:S4','source_sha256':source_sha256(),'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'s4_spec_sha256':g4_s4_spec()['science_sha256'],'pair_basis_sha256':pair_basis['science_sha256']})
        worker_init={'states':{ref:states[ref].to_wire() for ref in refs},'pair_rows':pair_rows}

        telemetry.phase('TASK_GENERATION')
        local=[]
        for ref in refs:
            for a in range(7):
                for b in range(7):
                    for c in range(7):
                        local.append(TaskSpec(task_id=f'g4s4-l3-{len(local):04d}',task_kind='G4_S4_LOCAL_THREE_RESERVATION',binding_sha256=binding,payload={'term_ref':ref,'a':a,'b':b,'c':c},cost_weight=1.0))
        recursive=[]
        for prow in pair_rows:
            i=int(prow['pair_key_index'])
            for third in refs:
                for a,b in bridge_pairs:
                    recursive.append(TaskSpec(task_id=f'g4s4-r-{len(recursive):05d}',task_kind='G4_S4_RECURSIVE_P3',binding_sha256=binding,payload={'pair_key_index':i,'third_ref':third,'operator':[a,b]},cost_weight=2.0))
        if len(local)!=686 or len(recursive)!=7688: raise UpliftCampaignError('G4:S4 preregistered task accounting mismatch')
        # One canonical complete recursive relation per directed operator for post-recursive reserve closure.
        reserve=[TaskSpec(task_id=f'g4s4-b-{k:02d}',task_kind='G4_S4_RESERVE_BASIS',binding_sha256=binding,payload={'pair_key_index':0,'third_ref':refs[0],'operator':[a,b]},cost_weight=4.0) for k,(a,b) in enumerate(bridge_pairs)]

        telemetry.phase('SCIENCE_CENSUS',tasks_total=len(local)+len(recursive)+len(reserve),kernels_total=len(local)+len(recursive)+len(reserve))
        lb=execute_tasks(local,worker_ref='infinity_grid.uplift_g4_s4:g4_s4_local_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_g4_s4:init_g4_s4_local_worker',initializer_payload={'states':worker_init['states']},telemetry=telemetry)
        rb=execute_tasks(recursive,worker_ref='infinity_grid.uplift_g4_s4:g4_s4_recursive_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_g4_s4:init_g4_s4_recursive_worker',initializer_payload=worker_init,telemetry=telemetry)
        bb=execute_tasks(reserve,worker_ref='infinity_grid.uplift_g4_s4:g4_s4_reserve_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_g4_s4:init_g4_s4_recursive_worker',initializer_payload=worker_init,telemetry=telemetry)
        local_rows=[lb.results[k] for k in sorted(lb.results)]; recursive_rows=[rb.results[k] for k in sorted(rb.results)]; reserve_rows=[bb.results[k] for k in sorted(bb.results)]
        telemetry.phase('INTERPRETATION')
        result=finalize_g4_s4_result(s0_result=s0,s1_result=s1,s2_result=s2,s3_result=s3,s3_replay=s3rep,s3_closeout=s3clo,reproduction=repro,local_rows=local_rows,pair_basis=pair_basis,recursive_rows=recursive_rows,reserve_rows=reserve_rows)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.41'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V24','registered_experiment_id':'G4:S4','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':True,'local_parallel_batch':lb.metadata,'recursive_parallel_batch':rb.metadata,'reserve_parallel_batch':bb.metadata,'task_count':len(local)+len(recursive)+len(reserve),'local_task_count':len(local),'recursive_task_count':len(recursive),'reserve_task_count':len(reserve),'requested_workers':engine.policy.requested_workers,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g3_graduation_preserved':True,'g4_graduated':False,'public_descriptor_promoted':False,'topology_promoted':False,'shell_profile_promoted':False,'g4_phase0_spec_sha256':g4_phase0_spec()['science_sha256'],'g4_s4_preregistered_spec_sha256':g4_s4_spec()['science_sha256'],'input_s3_science_sha256':s3.get('science_sha256'),'carrier_input_mode':'CERTIFIED_G3_TERM_CORPUS_FROM_S0','lower_layer_rematerialization':False,'g1_r100_materialization':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'AUTHORITY.json',auth); write_json_atomic(stage_dir/'PAIR_CANDIDATE_BASIS.json',pair_basis)
        with gzip.open(stage_dir/'LOCAL_THREE_RESERVATIONS.json.gz','wt',encoding='utf-8',compresslevel=1) as fh: json.dump({'schema_id':'IG_G4_S4_LOCAL_THREE_INDEX_V1','task_count':len(local_rows),'rows':local_rows},fh,sort_keys=True,separators=(',',':'))
        with gzip.open(stage_dir/'RECURSIVE_P3_CONTEXTS.json.gz','wt',encoding='utf-8',compresslevel=1) as fh: json.dump({'schema_id':'IG_G4_S4_RECURSIVE_P3_INDEX_V1','task_count':len(recursive_rows),'rows':recursive_rows},fh,sort_keys=True,separators=(',',':'))
        write_json_atomic(stage_dir/'RESERVE_BASIS.json',{'schema_id':'IG_G4_S4_RESERVE_BASIS_INDEX_V1','task_count':len(reserve_rows),'rows':reserve_rows})
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S4',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_s5_unlocked=result.get('g4_s5_unlocked',False),g4_graduated=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise

def run_g4_s5_native(*, engine:UpliftCampaignEngine, s3_result:str|Path, s4_result:str|Path, s4_replay_comparison:str|Path, s4_closeout:str|Path, s4_pair_basis:str|Path, s4_recursive_rows:str|Path, s4_reserve_basis:str|Path)->dict[str,Any]:
    """Run registered G4:S5 finite CAPS7 + Tier-1-class-bag descriptor audit.

    S5 consumes certified S4 evidence without rebuilding the heavy S4 census.
    Recursive closure and graduation remain exclusively G4:S6.
    """
    spec=engine.experiment('G4:S5')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not True:
        raise UpliftCampaignError('G4:S5 not registered as promoting native handler')
    stage_dir=engine.run_root/'stages'/'G4_S5'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S5',workers_requested=engine.policy.requested_workers,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        s3=_load_json(s3_result); s4=_load_json(s4_result); rep=_load_json(s4_replay_comparison); clo=_load_json(s4_closeout)
        pair=_load_json(s4_pair_basis); reserve_basis=_load_json(s4_reserve_basis)
        auth=verify_g4_s5_authority(s3_result=s3,s4_result=s4,s4_replay=rep,s4_closeout=clo)
        telemetry.phase('SCIENCE_CENSUS',tasks_total=0,kernels_total=0)
        census=certify_g4_s5_candidate_census(s3_result=s3,pair_basis=pair,recursive_rows_path=s4_recursive_rows)
        reserve=verify_g4_s5_reserve_factorisation(reserve_basis=reserve_basis)
        impl=g4_s5_implementation_read_surface_audit()
        arg=g4_s5_factorisation_argument(census=census,reserve=reserve,implementation=impl)
        telemetry.phase('INTERPRETATION')
        result=finalize_g4_s5_result(authority=auth,census=census,reserve=reserve,implementation=impl,argument=arg)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.42'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V25','registered_experiment_id':'G4:S5','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':False,'task_count':0,'requested_workers':engine.policy.requested_workers,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':True,'g3_graduation_preserved':True,'g4_graduated':False,'public_descriptor_promoted':bool(result.get('public_descriptor_promoted')),'promoted_descriptor':result.get('promoted_descriptor'),'topology_promoted':False,'shell_profile_promoted':False,'heavy_s4_census_rebuilt':False,'g4_phase0_spec_sha256':g4_phase0_spec()['science_sha256'],'g4_s5_preregistered_spec_sha256':g4_s5_spec()['science_sha256'],'input_s4_science_sha256':s4.get('science_sha256'),'input_s4_certification':rep.get('certification'),'input_s4_closeout_status':clo.get('status'),'carrier_input_mode':'CERTIFIED_S4_EVIDENCE_ONLY','lower_layer_rematerialization':False,'g1_r100_materialization':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'AUTHORITY.json',auth); write_json_atomic(stage_dir/'CANDIDATE_DESCRIPTOR_CENSUS.json',census); write_json_atomic(stage_dir/'RESERVE_FACTORISATION.json',reserve); write_json_atomic(stage_dir/'IMPLEMENTATION_READ_SURFACE_AUDIT.json',impl); write_json_atomic(stage_dir/'FACTORISATION_ARGUMENT.json',arg)
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S5',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_s6_unlocked=result.get('g4_s6_unlocked',False),g4_graduated=False,public_descriptor_promoted=result.get('public_descriptor_promoted',False),topology_promoted=False,shell_profile_promoted=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise


def run_g4_s6_native(*, engine:UpliftCampaignEngine, s0_result:str|Path, s3_result:str|Path, s4_pair_basis:str|Path, s5_result:str|Path, s5_replay_comparison:str|Path, s5_closeout:str|Path, regression_evidence:str|Path)->dict[str,Any]:
    """Run registered G4:S6 recursive-closure / graduation-candidate audit.

    Primary PASS never graduates G4. Only the cold-certified S6 comparison may
    authorize the separate G4 graduation certificate.
    """
    spec=engine.experiment('G4:S6')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not True:
        raise UpliftCampaignError('G4:S6 not registered as graduation-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G4_S6'; stage_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=stage_dir/'case_checkpoints'; checkpoint_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:S6',workers_requested=engine.policy.requested_workers,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        s0=_load_json(s0_result); s3=_load_json(s3_result); pair=_load_json(s4_pair_basis); s5=_load_json(s5_result); replay=_load_json(s5_replay_comparison); closeout=_load_json(s5_closeout); reg=_load_json(regression_evidence)
        current_source=source_sha256()
        auth=verify_g4_s6_authority(s0_result=s0,s3_result=s3,s4_pair_basis=pair,s5_result=s5,s5_replay=replay,s5_closeout=closeout)
        rgate=verify_g4_s6_regression_evidence(reg,current_source_sha256=current_source)
        theorem=g4_s6_recursive_theorem(s5)
        hidden=g4_s6_hidden_audit()
        states,load_meta=load_g4_s0_term_states(s0); alpha,ref_labels,_vals=g4_s6_tier1_index(s3)
        ops=sorted({tuple(map(int,r['operator'])) for r in pair.get('rows',[])})
        plan=dict(g4_s6_spec()['fresh_holdout']['full_plan'])
        frozen_plan={'schema_id':'IG_G4_S6_FRESH_HOLDOUT_PLAN_V1','status':'FROZEN_BY_PREREGISTRATION_SPEC','freshness_scope':g4_s6_spec()['fresh_holdout']['freshness_scope'],'pair_plus_pair_cases':plan['pair_plus_pair_cases'],'deep_p5_cases':plan['deep_p5_cases'],'rebracketing_cases':plan['rebracketing_cases'],'outcome_retuning':False}
        frozen_plan['science_sha256']=canonical_sha256(frozen_plan)
        write_json_atomic(stage_dir/'FRESH_HOLDOUT_PLAN.json',frozen_plan)
        binding=canonical_sha256({'stage':'G4:S6','source_sha256':current_source,'registry_sha256':engine.registry['registry_sha256'],'authority_sha256':auth['science_sha256'],'s6_spec_sha256':g4_s6_spec()['science_sha256'],'holdout_plan_sha256':frozen_plan['science_sha256'],'s0_term_corpus_sha256':s0.get('certified_term_corpus',{}).get('science_sha256'),'s3_index_sha256':s3.get('candidate_interface_index',{}).get('science_sha256'),'s4_pair_basis_sha256':pair.get('science_sha256')})
        uplift_g4_s6._WORKER_CONTEXT={'states':states,'ref_labels':ref_labels,'alphabet':alpha,'operator_basis':ops,'checkpoint_dir':str(checkpoint_dir),'binding_sha256':binding}
        task_defs=[]
        for c in frozen_plan['pair_plus_pair_cases']: task_defs.append(('PAIR_PLUS_PAIR',c,1.0))
        for c in frozen_plan['deep_p5_cases']: task_defs.append(('DEEP_P5',c,6.0))
        for c in frozen_plan['rebracketing_cases']: task_defs.append(('REBRACKET',c,4.0))
        recovered={}; tasks=[]
        for kind,c,cost in task_defs:
            cid=str(c['case_id']); cp=load_g4_s6_case_checkpoint(checkpoint_dir=checkpoint_dir,binding_sha256=binding,kind=kind,case_id=cid)
            if cp is not None: recovered[cid]=cp
            else: tasks.append(TaskSpec(task_id=cid,task_kind='G4_S6_'+kind,binding_sha256=binding,payload={'kind':kind,'case':c},cost_weight=cost))
        telemetry.phase('SCIENCE_CENSUS',tasks_total=len(task_defs),kernels_total=len(task_defs))
        batch=execute_tasks(tasks,worker_ref='infinity_grid.uplift_g4_s6:g4_s6_holdout_worker',policy=engine.policy,initializer_ref='infinity_grid.uplift_g4_s6:init_holdout_worker_from_fork',initializer_payload=None)
        all_results=dict(recovered); all_results.update(batch.results)
        missing=[str(c['case_id']) for _kind,c,_cost in task_defs if str(c['case_id']) not in all_results]
        if missing: raise UpliftCampaignError('G4:S6 missing case results after execution: '+','.join(missing))
        pp=[all_results[str(c['case_id'])] for c in frozen_plan['pair_plus_pair_cases']]
        deep=[all_results[str(c['case_id'])] for c in frozen_plan['deep_p5_cases']]
        rb=[all_results[str(c['case_id'])] for c in frozen_plan['rebracketing_cases']]
        cpidx={'schema_id':'IG_G4_S6_CASE_CHECKPOINT_INDEX_V1','binding_sha256':binding,'case_count':len(task_defs),'reused_case_count':len(recovered),'executed_case_count':len(tasks),'case_result_sha256':{cid:canonical_sha256(all_results[cid]) for cid in sorted(all_results)}}
        cpidx['science_sha256']=canonical_sha256(cpidx); write_json_atomic(stage_dir/'CASE_CHECKPOINT_INDEX.json',cpidx)
        holdout=aggregate_g4_s6_holdout(pair_plus_pair=pp,deep_p5=deep,rebracket=rb)
        telemetry.phase('INTERPRETATION')
        result=g4_s6_primary_result(authority=auth,regression=rgate,theorem=theorem,holdout=holdout,hidden=hidden)
        result['source_sha256']=current_source; result['source_version']='0.30.43'; result['registry_sha256']=engine.registry['registry_sha256']
        pm=dict(batch.metadata); pm.update({'reused_case_count':len(recovered),'executed_case_count':len(tasks),'per_case_checkpointing':True,'checkpoint_binding_sha256':binding})
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V26','registered_experiment_id':'G4:S6','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_batch':pm,'task_count':len(task_defs),'requested_workers':engine.policy.requested_workers,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':True,'primary_run_can_graduate_g4':False,'g3_graduation_preserved':True,'g4_graduated':False,'r0_unlocked':False,'public_descriptor_promoted':True,'promoted_descriptor':'CAPS7_PLUS_TIER1_CLASS_BAG','topology_promoted':False,'shell_profile_promoted':False,'g4_phase0_spec_sha256':g4_phase0_spec()['science_sha256'],'g4_s6_spec_sha256':g4_s6_spec()['science_sha256'],'input_s5_science_sha256':s5.get('science_sha256'),'input_s5_replay_certification':replay.get('certification'),'input_s5_closeout_status':closeout.get('status'),'s0_term_load_science_sha256':load_meta.get('science_sha256'),'lower_layer_rematerialization':False,'g1_r100_materialization':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'AUTHORITY.json',auth); write_json_atomic(stage_dir/'REGRESSION_GATE.json',rgate); write_json_atomic(stage_dir/'RECURSIVE_DESCRIPTOR_FACTORISATION_THEOREM.json',theorem); write_json_atomic(stage_dir/'FRESH_HOLDOUT_RESULT.json',holdout); write_json_atomic(stage_dir/'NO_HIDDEN_SELECTOR_READ_AUDIT.json',hidden); write_json_atomic(stage_dir/'PAIR_PLUS_PAIR_CASE_RESULTS.json',{'schema_id':'IG_G4_S6_PAIR_PLUS_PAIR_CASE_RESULTS_V1','cases':pp}); write_json_atomic(stage_dir/'DEEP_P5_CASE_RESULTS.json',{'schema_id':'IG_G4_S6_DEEP_P5_CASE_RESULTS_V1','cases':deep}); write_json_atomic(stage_dir/'REBRACKET_CASE_RESULTS.json',{'schema_id':'IG_G4_S6_REBRACKET_CASE_RESULTS_V1','cases':rb})
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:S6',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_graduation_candidate=result.get('g4_graduation_candidate',False),g4_graduated=False,r0_unlocked=False,public_descriptor_promoted=True,topology_promoted=False,shell_profile_promoted=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise


def run_g4_r0_native(*, engine:UpliftCampaignEngine, graduation_certificate:str|Path, s0_result:str|Path, s3_result:str|Path)->dict[str,Any]:
    """Run registered non-promoting G4:R0 structural reconnaissance."""
    spec=engine.experiment('G4:R0.STRUCTURAL_AUDIT')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G4:R0 structural audit not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G4_R0_STRUCTURAL_AUDIT'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G4:R0.STRUCTURAL_AUDIT',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        grad=_load_json(graduation_certificate); s0=_load_json(s0_result); s3=_load_json(s3_result)
        telemetry.phase('SCIENCE_CENSUS')
        result=run_g4_r0_recon(engine=engine,graduation_certificate=grad,s0_result=s0,s3_result=s3)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.44'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V27','registered_experiment_id':'G4:R0.STRUCTURAL_AUDIT','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':False,'requested_workers':1,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g4_graduation_preserved':True,'g5_started':False,'r0_spec_science_sha256':g4_r0_spec()['science_sha256'],'public_descriptor':'CAPS7_PLUS_TIER1_CLASS_BAG','topology_promoted':False,'shell_profile_promoted':False,'lower_layer_historical_rematerialization':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'AUTHORITY.json',result['authority']); write_json_atomic(stage_dir/'HIERARCHICAL_TREE_THEOREM.json',result['hierarchical_tree_theorem']); write_json_atomic(stage_dir/'TREE_SHAPE_REALISABILITY.json',result['bounded_tree_shape_realisability'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G4:R0',last_result_status=result['status'],classification=result['classification'],g3_graduated=True,g4_started=True,g4_graduated=True,r0_complete=result.get('status')=='PASS',public_descriptor_promoted=True,topology_promoted=False,shell_profile_promoted=False,g5_started=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise


def run_g5_r0_native(*, engine:UpliftCampaignEngine, graduation_decision:str|Path, independent_verification:str|Path, primary_s6:str|Path)->dict[str,Any]:
    """Run registered non-promoting G5:R0 structural reconnaissance."""
    spec=engine.experiment('G5:R0.STRUCTURAL_AUDIT')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G5:R0 structural audit not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G5_R0_STRUCTURAL_AUDIT'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G5:R0.STRUCTURAL_AUDIT',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        grad=_load_json(graduation_decision); ver=_load_json(independent_verification); prim=_load_json(primary_s6)
        telemetry.phase('SCIENCE_CENSUS')
        result=run_g5_r0_recon(engine=engine,graduation_decision=grad,independent_verification=ver,primary_s6=prim)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.71'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V28','registered_experiment_id':'G5:R0.STRUCTURAL_AUDIT','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':False,'requested_workers':1,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g5_graduation_preserved':True,'g6_started':False,'r0_spec_science_sha256':g5_r0_spec()['science_sha256'],'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','topology_promoted':False,'descriptor_changed':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result); write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']); write_json_atomic(stage_dir/'AUTHORITY.json',result['authority']); write_json_atomic(stage_dir/'HIDDEN_TREE_THEOREM.json',result['hierarchical_tree_theorem']); write_json_atomic(stage_dir/'TREE_SHAPE_CENSUS.json',{'schema_id':'IG_G5_R0_TREE_SHAPE_CENSUS_V1','rows':result['bounded_homogeneous_tree_shape_census']}); write_json_atomic(stage_dir/'TOPOLOGY_SEPARATION_WITNESS.json',result['first_same_public_nonisomorphic_tree_witness'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G5:R0',last_result_status=result['status'],classification=result['classification'],g5_graduated=True,r0_complete=result.get('status')=='PASS',topology_promoted=False,g6_started=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise



def run_g5_r1_native(*, engine:UpliftCampaignEngine, r0_primary:str|Path, r0_verification:str|Path, r0_closeout:str|Path)->dict[str,Any]:
    """Run registered non-promoting G5:R1 hidden-fiber/grafting audit."""
    spec=engine.experiment('G5:R1.FIBER_GRAFT_AUDIT')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G5:R1 fiber/graft audit not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G5_R1_FIBER_GRAFT_AUDIT'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G5:R1.FIBER_GRAFT_AUDIT',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        rp=_load_json(r0_primary); rv=_load_json(r0_verification); rc=_load_json(r0_closeout)
        telemetry.phase('SCIENCE_CENSUS')
        result=run_g5_r1_fiber_graft_audit(engine=engine,r0_primary=rp,r0_verification=rv,r0_closeout=rc)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.72'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V29','registered_experiment_id':'G5:R1.FIBER_GRAFT_AUDIT','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':False,'requested_workers':1,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g5_graduation_preserved':True,'g6_started':False,'r1_spec_science_sha256':g5_r1_spec()['science_sha256'],'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','topology_promoted':False,'descriptor_changed':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result);write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']);write_json_atomic(stage_dir/'AUTHORITY.json',result['authority']);write_json_atomic(stage_dir/'EXACT_FIBER_WITNESS.json',result['exact_fiber_witness']);write_json_atomic(stage_dir/'HIDDEN_GRAFTING_LAW.json',result['hidden_composition_law']);write_json_atomic(stage_dir/'RELATION_VALUED_GRAFT_WITNESS.json',result['relation_valued_graft_witness']);write_json_atomic(stage_dir/'HOMOGENEOUS_SHAPE_CENSUS.json',result['bounded_homogeneous_shape_census'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G5:R1',last_result_status=result['status'],classification=result['classification'],g5_graduated=True,r1_complete=result.get('status')=='PASS',topology_promoted=False,g6_started=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise

def run_g5_r2_native(*, engine:UpliftCampaignEngine, r1_primary:str|Path, r1_verification:str|Path, r1_closeout:str|Path)->dict[str,Any]:
    """Run registered non-promoting G5:R2 predictive hidden action-read audit."""
    spec=engine.experiment('G5:R2.PREDICTIVE_HIDDEN_ACTION_READ')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G5:R2 predictive hidden read not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G5_R2_PREDICTIVE_HIDDEN_ACTION_READ'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G5:R2.PREDICTIVE_HIDDEN_ACTION_READ',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        rp=_load_json(r1_primary); rv=_load_json(r1_verification); rc=_load_json(r1_closeout)
        telemetry.phase('SCIENCE_CENSUS')
        result=run_g5_r2_predictive_hidden_read(engine=engine,r1_primary=rp,r1_verification=rv,r1_closeout=rc)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.74'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V30','registered_experiment_id':'G5:R2.PREDICTIVE_HIDDEN_ACTION_READ','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':False,'requested_workers':1,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g5_graduation_preserved':True,'g6_started':False,'r2_spec_science_sha256':g5_r2_spec()['science_sha256'],'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','topology_promoted':False,'compact_hidden_read_promoted':False,'descriptor_changed':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result);write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']);write_json_atomic(stage_dir/'AUTHORITY.json',result['authority']);write_json_atomic(stage_dir/'FIXED_DEPTH_LOWER_BOUND.json',result['global_fixed_depth_lower_bound']);write_json_atomic(stage_dir/'FRESH_N8_COLOR_PANEL.json',result['fresh_n8_color_panel']);write_json_atomic(stage_dir/'ENDPOINT_TYPED_N5_PANEL.json',result['endpoint_typed_n5_panel']);write_json_atomic(stage_dir/'ALL_31_OPERATOR_SENTINEL.json',result['all_operator_endpoint_sentinel'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G5:R2',last_result_status=result['status'],classification=result['classification'],g5_graduated=True,r2_complete=result.get('status')=='PASS',topology_promoted=False,compact_hidden_read_promoted=False,g6_started=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise


def run_g5_r3_native(*, engine:UpliftCampaignEngine, r2_primary:str|Path, r2_verification:str|Path, r2_closeout:str|Path)->dict[str,Any]:
    """Run registered non-promoting G5:R3 generic message/future-congruence audit."""
    spec=engine.experiment('G5:R3.GENERIC_ENDPOINT_TYPED_MESSAGE_CONGRUENCE')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G5:R3 message congruence not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G5_R3_GENERIC_ENDPOINT_TYPED_MESSAGE_CONGRUENCE'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G5:R3.GENERIC_ENDPOINT_TYPED_MESSAGE_CONGRUENCE',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        rp=_load_json(r2_primary); rv=_load_json(r2_verification); rc=_load_json(r2_closeout)
        telemetry.phase('SCIENCE_CENSUS')
        result=run_g5_r3_generic_message_congruence(engine=engine,r2_primary=rp,r2_verification=rv,r2_closeout=rc)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.76'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V31','registered_experiment_id':'G5:R3.GENERIC_ENDPOINT_TYPED_MESSAGE_CONGRUENCE','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':False,'requested_workers':1,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g5_graduation_preserved':True,'g6_started':False,'r3_spec_science_sha256':g5_r3_spec()['science_sha256'],'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','topology_promoted':False,'compact_hidden_read_promoted':False,'descriptor_changed':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result);write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']);write_json_atomic(stage_dir/'AUTHORITY.json',result['authority']);write_json_atomic(stage_dir/'R2_BRIDGE.json',result['r2_bridge']);write_json_atomic(stage_dir/'GENERIC_MESSAGE_THEOREM.json',result['generic_message_theorem']);write_json_atomic(stage_dir/'EXHAUSTIVE_SUPPORT_PANEL.json',result['exhaustive_support_panel']);write_json_atomic(stage_dir/'ALL_31_OPERATOR_SENTINEL.json',result['all_operator_sentinel'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G5:R3',last_result_status=result['status'],classification=result['classification'],g5_graduated=True,r3_complete=result.get('status')=='PASS',topology_promoted=False,compact_hidden_read_promoted=False,g6_started=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise

def run_g5_r4_native(*, engine:UpliftCampaignEngine, r3_primary:str|Path, r3_verification:str|Path, r3_closeout:str|Path)->dict[str,Any]:
    """Run registered non-promoting G5:R4 marked-leaf hidden minimality audit."""
    spec=engine.experiment('G5:R4.MARKED_LEAF_HIDDEN_MINIMALITY')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G5:R4 marked-leaf minimality not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G5_R4_MARKED_LEAF_HIDDEN_MINIMALITY'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G5:R4.MARKED_LEAF_HIDDEN_MINIMALITY',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        rp=_load_json(r3_primary); rv=_load_json(r3_verification); rc=_load_json(r3_closeout)
        telemetry.phase('SCIENCE_CENSUS')
        result=run_g5_r4_marked_hidden_minimality(engine=engine,r3_primary=rp,r3_verification=rv,r3_closeout=rc)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.77'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V32','registered_experiment_id':'G5:R4.MARKED_LEAF_HIDDEN_MINIMALITY','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':False,'requested_workers':1,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g5_graduation_preserved':True,'g6_started':False,'r4_spec_science_sha256':g5_r4_spec()['science_sha256'],'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','topology_promoted':False,'exact_hidden_canon_promoted_to_public':False,'descriptor_changed':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result);write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']);write_json_atomic(stage_dir/'AUTHORITY.json',result['authority']);write_json_atomic(stage_dir/'MARKED_LEAF_SEPARATION_THEOREM.json',result['marked_leaf_separation_theorem']);write_json_atomic(stage_dir/'HOMOGENEOUS_TOPOLOGY_SUPPORT.json',result['homogeneous_topology_support']);write_json_atomic(stage_dir/'ALL_31_EXISTING_OPERATOR_SMALL_SUPPORT.json',result['all_31_existing_operator_small_support']);write_json_atomic(stage_dir/'ALL_31_MARKER_OPERATOR_SENTINEL.json',result['all_31_marker_operator_sentinel'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G5:R4',last_result_status=result['status'],classification=result['classification'],g5_graduated=True,r4_complete=result.get('status')=='PASS',topology_promoted=False,g6_started=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise


def run_g5_r5_native(*, engine:UpliftCampaignEngine, r4_primary:str|Path, r4_verification:str|Path, r4_closeout:str|Path)->dict[str,Any]:
    """Run registered non-promoting G5:R5 marker-free one-step separation audit."""
    spec=engine.experiment('G5:R5.MARKER_FREE_ONE_STEP_SEPARATION')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G5:R5 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G5_R5_MARKER_FREE_ONE_STEP_SEPARATION'; stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G5:R5.MARKER_FREE_ONE_STEP_SEPARATION',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD')
        rp=_load_json(r4_primary); rv=_load_json(r4_verification); rc=_load_json(r4_closeout)
        telemetry.phase('SCIENCE_CENSUS')
        result=run_g5_r5_marker_free_one_step_separation(engine=engine,r4_primary=rp,r4_verification=rv,r4_closeout=rc)
        result['source_sha256']=source_sha256(); result['source_version']='0.30.78'; result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V33','registered_experiment_id':'G5:R5.MARKER_FREE_ONE_STEP_SEPARATION','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':False,'requested_workers':1,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g5_graduation_preserved':True,'g6_started':False,'r5_spec_science_sha256':g5_r5_spec()['science_sha256'],'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','topology_promoted':False,'exact_hidden_canon_promoted_to_public':False,'descriptor_changed':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL')
        write_json_atomic(stage_dir/'RESULT.json',result);write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']);write_json_atomic(stage_dir/'AUTHORITY.json',result['authority']);write_json_atomic(stage_dir/'FRESH_PRIMARY_PANEL.json',result['fresh_primary_panel']);write_json_atomic(stage_dir/'ENDPOINT_TYPED_SENTINEL_PANEL.json',result['endpoint_typed_sentinel_panel'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G5:R5',last_result_status=result['status'],classification=result['classification'],g5_graduated=True,r5_complete=result.get('status')=='PASS',topology_promoted=False,g6_started=False)
        telemetry.complete(); return result
    except BaseException as exc:
        telemetry.fail(exc); raise


def run_g5_r6_native(*, engine:UpliftCampaignEngine, r5_primary:str|Path, r5_verification:str|Path, r5_closeout:str|Path)->dict[str,Any]:
    """Run registered non-promoting G5:R6 single-payload marker-free separation audit."""
    spec=engine.experiment('G5:R6.SINGLE_PAYLOAD_MARKER_FREE_SEPARATION')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G5:R6 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G5_R6_SINGLE_PAYLOAD_MARKER_FREE_SEPARATION';stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G5:R6.SINGLE_PAYLOAD_MARKER_FREE_SEPARATION',workers_requested=1,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD');rp=_load_json(r5_primary);rv=_load_json(r5_verification);rc=_load_json(r5_closeout)
        telemetry.phase('SCIENCE_CENSUS');result=run_g5_r6_single_payload_separation(engine=engine,r5_primary=rp,r5_verification=rv,r5_closeout=rc)
        result['source_sha256']=source_sha256();result['source_version']='0.30.79';result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V34','registered_experiment_id':'G5:R6.SINGLE_PAYLOAD_MARKER_FREE_SEPARATION','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':False,'requested_workers':1,'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g5_graduation_preserved':True,'g6_started':False,'r6_spec_science_sha256':g5_r6_spec()['science_sha256'],'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','topology_promoted':False,'exact_hidden_canon_promoted_to_public':False,'descriptor_changed':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL');write_json_atomic(stage_dir/'RESULT.json',result);write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']);write_json_atomic(stage_dir/'AUTHORITY.json',result['authority']);write_json_atomic(stage_dir/'FRESH_N9_PANEL.json',result['fresh_n9_panel']);write_json_atomic(stage_dir/'ENDPOINT_TYPED_N5_PANEL.json',result['endpoint_typed_n5_panel'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G5:R6',last_result_status=result['status'],classification=result['classification'],g5_graduated=True,r6_complete=result.get('status')=='PASS',topology_promoted=False,g6_started=False)
        telemetry.complete();return result
    except BaseException as exc:
        telemetry.fail(exc);raise


def run_g5_r7_native(*, engine:UpliftCampaignEngine, r6_primary:str|Path, r6_verification:str|Path, r6_closeout:str|Path)->dict[str,Any]:
    """Run registered non-promoting G5:R7 larger-domain single-payload adversarial escalation."""
    spec=engine.experiment('G5:R7.SINGLE_PAYLOAD_ADVERSARIAL_ESCALATION')
    if spec.get('execution_mode')!='NATIVE_HANDLER' or spec.get('promotion') is not False:
        raise UpliftCampaignError('G5:R7 not registered as non-promoting native handler')
    stage_dir=engine.run_root/'stages'/'G5_R7_SINGLE_PAYLOAD_ADVERSARIAL_ESCALATION';stage_dir.mkdir(parents=True,exist_ok=True)
    telemetry=RuntimeTelemetry(engine.run_root,'G5:R7.SINGLE_PAYLOAD_ADVERSARIAL_ESCALATION',workers_requested=4,wall_interval_seconds=30.0)
    try:
        telemetry.phase('AUTHORITY_LOAD');rp=_load_json(r6_primary);rv=_load_json(r6_verification);rc=_load_json(r6_closeout)
        telemetry.phase('SCIENCE_CENSUS')
        result=run_g5_r7_adversarial_escalation(engine=engine,r6_primary=rp,r6_verification=rv,r6_closeout=rc,telemetry=telemetry)
        result['source_sha256']=source_sha256();result['source_version']='0.30.80';result['registry_sha256']=engine.registry['registry_sha256']
        result['execution_metadata']={'schema_id':'IG_G_UPLIFT_NATIVE_EXECUTION_METADATA_V35','registered_experiment_id':'G5:R7.SINGLE_PAYLOAD_ADVERSARIAL_ESCALATION','registry_sha256':engine.registry['registry_sha256'],'campaign_materialization_count':engine.materialization_count,'parallel_stage_required':True,'requested_workers':4,'parallel_batch':result.pop('parallel_batch_metadata'),'execution_backend_owned_by_decoder':True,'stage_specific_external_science_runner':False,'promotion':False,'g5_graduation_preserved':True,'g6_started':False,'r7_spec_science_sha256':g5_r7_spec()['science_sha256'],'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG','topology_promoted':False,'exact_hidden_canon_promoted_to_public':False,'descriptor_changed':False,'runtime_telemetry_artifacts':['RUNTIME_STATUS.json','RUNTIME_TIMINGS.json'],'runtime_telemetry_not_science':True}
        telemetry.phase('RESULT_SEAL');write_json_atomic(stage_dir/'RESULT.json',result);write_json_atomic(stage_dir/'EXECUTION_METADATA.json',result['execution_metadata']);write_json_atomic(stage_dir/'AUTHORITY.json',result['authority']);write_json_atomic(stage_dir/'TASK_PARTITION_CERTIFICATE.json',result['task_partition_certificate']);write_json_atomic(stage_dir/'FRESH_N10_PANEL.json',result['fresh_n10_panel']);write_json_atomic(stage_dir/'ENDPOINT_TYPED_N6_PANEL.json',result['endpoint_typed_n6_panel'])
        _write_state(engine.run_root,status=_science_stop_status(result),current_stage='G5:R7',last_result_status=result['status'],classification=result['classification'],g5_graduated=True,r7_complete=result.get('status')=='PASS',topology_promoted=False,g6_started=False)
        telemetry.complete();return result
    except BaseException as exc:
        telemetry.fail(exc);raise


def run_native_campaign_stage(*, run_root:str|Path, experiment_id:str, inputs:Mapping[str,str|Path], requested_workers:int|str|None='AUTO', lease_root:str|None=None)->dict[str,Any]:
    root=Path(run_root); _write_state(root,status='RUNNING',current_stage=experiment_id,g2_graduated=False)
    with UpliftCampaignEngine(root,requested_workers=requested_workers,lease_root=lease_root,materialization_start_method='AUTO') as engine:
        if experiment_id=='G2:S4': result=run_g2_s4_native(engine=engine,**inputs)
        elif experiment_id=='G2:S5': result=run_g2_s5_native(engine=engine,**inputs)
        elif experiment_id=='G2:S6': result=run_g2_s6_native(engine=engine,**inputs)
        elif experiment_id=='G2:S5.TORIC_AUDIT': result=run_g2_toric_native(engine=engine,**inputs)
        elif experiment_id=='G2:R0.STRUCTURAL_AUDIT': result=run_g2_r0_native(engine=engine,**inputs)
        elif experiment_id=='G3:S0': result=run_g3_s0_native(engine=engine,**inputs)
        elif experiment_id=='G3:S1': result=run_g3_s1_native(engine=engine,**inputs)
        elif experiment_id=='G3:S2': result=run_g3_s2_native(engine=engine,**inputs)
        elif experiment_id=='G3:S3': result=run_g3_s3_native(engine=engine,**inputs)
        elif experiment_id=='G3:S4': result=run_g3_s4_native(engine=engine,**inputs)
        elif experiment_id=='G3:S5': result=run_g3_s5_native(engine=engine,**inputs)
        elif experiment_id=='G3:S6': result=run_g3_s6_native(engine=engine,**inputs)
        elif experiment_id=='G3:R0.STRUCTURAL_AUDIT': result=run_g3_r0_native(engine=engine,**inputs)
        elif experiment_id=='G3:R1.FIBER_MODULI_AUDIT': result=run_g3_r1_native(engine=engine,**inputs)
        elif experiment_id=='G3:R2.ADAPTIVE_MATURATION': result=run_g3_r2_native(engine=engine,**inputs)
        elif experiment_id=='G3:R3.STRUCTURAL_ALGEBRA_REBASE': result=run_g3_r3_native(engine=engine,**inputs)
        elif experiment_id=='G4:S0.REBASE': result=run_g4_s0_rebase_native(engine=engine,**inputs)
        elif experiment_id=='G4:S0': result=run_g4_s0_native(engine=engine,**inputs)
        elif experiment_id=='G4:S1.REBASE': result=run_g4_s1_rebase_native(engine=engine,**inputs)
        elif experiment_id=='G4:S1': result=run_g4_s1_native(engine=engine,**inputs)
        elif experiment_id=='G4:S2.REBASE': result=run_g4_s2_rebase_native(engine=engine,**inputs)
        elif experiment_id=='G4:S3.REBASE': result=run_g4_s3_rebase_native(engine=engine,**inputs)
        elif experiment_id=='G4:S4.REBASE': result=run_g4_s4_rebase_native(engine=engine,**inputs)
        elif experiment_id=='G4:S5.REBASE': result=run_g4_s5_rebase_native(engine=engine,**inputs)
        elif experiment_id=='G4:S2': result=run_g4_s2_native(engine=engine,**inputs)
        elif experiment_id=='G4:S3': result=run_g4_s3_native(engine=engine,**inputs)
        elif experiment_id=='G4:S4': result=run_g4_s4_native(engine=engine,**inputs)
        elif experiment_id=='G4:S5': result=run_g4_s5_native(engine=engine,**inputs)
        elif experiment_id=='G4:S6': result=run_g4_s6_native(engine=engine,**inputs)
        elif experiment_id=='G4:R0.STRUCTURAL_AUDIT': result=run_g4_r0_native(engine=engine,**inputs)
        elif experiment_id=='G5:R0.STRUCTURAL_AUDIT': result=run_g5_r0_native(engine=engine,**inputs)
        elif experiment_id=='G5:R1.FIBER_GRAFT_AUDIT': result=run_g5_r1_native(engine=engine,**inputs)
        elif experiment_id=='G5:R2.PREDICTIVE_HIDDEN_ACTION_READ': result=run_g5_r2_native(engine=engine,**inputs)
        elif experiment_id=='G5:R3.GENERIC_ENDPOINT_TYPED_MESSAGE_CONGRUENCE': result=run_g5_r3_native(engine=engine,**inputs)
        elif experiment_id=='G5:R4.MARKED_LEAF_HIDDEN_MINIMALITY': result=run_g5_r4_native(engine=engine,**inputs)
        elif experiment_id=='G5:R5.MARKER_FREE_ONE_STEP_SEPARATION': result=run_g5_r5_native(engine=engine,**inputs)
        elif experiment_id=='G5:R6.SINGLE_PAYLOAD_MARKER_FREE_SEPARATION': result=run_g5_r6_native(engine=engine,**inputs)
        elif experiment_id=='G5:R7.SINGLE_PAYLOAD_ADVERSARIAL_ESCALATION': result=run_g5_r7_native(engine=engine,**inputs)
        else: raise UpliftCampaignError(f'native handler not implemented for {experiment_id}')
    return {'schema_id':'IG_G_UPLIFT_NATIVE_CAMPAIGN_RUN_RESULT_V1','status':_science_stop_status(result),'experiment_id':experiment_id,'scientific_result_status':result.get('status'),'classification':result.get('classification'),'science_sha256':result.get('science_sha256'),'native_engine_science_sha256':result.get('native_engine_science_sha256'),'g2_graduated':bool(result.get('g2_graduated',result.get('g2_graduation_preserved',False))),'g3_graduated':bool(result.get('g3_graduated',result.get('g3_graduation_preserved',False))),'r0_unlocked':bool(result.get('r0_unlocked',False)),'r0_complete':bool(result.get('r0_complete', result.get('status')=='PASS' and experiment_id.endswith('R0.STRUCTURAL_AUDIT'))),'run_root':str(root)}
