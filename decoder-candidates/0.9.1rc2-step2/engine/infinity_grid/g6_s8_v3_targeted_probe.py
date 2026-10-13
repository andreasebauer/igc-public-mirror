from __future__ import annotations

"""G6:S8 V3 targeted sentinel probe.

This executor is intentionally narrow: one frozen S7 class, depth 3, and the
first eight factor-swap-normalized execution contexts.  It does not run the
retired V2 depth-3..6 census and cannot graduate G6 by itself.
"""

import hashlib, json, sqlite3
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .g6_s8_intrinsic_descriptor import _reuse_rows, _reference_partition
from .g6_s8_evaluators import _multiset_signature
from .v05_origin_guard import require_controller_execution_origin
from .v05_stage_runtime import TaskSpec, StageRuntimeError

STAGE_ID='G6:S8'
PLAN_SCHEMA='IG_G6_S8_PREREGISTRATION_V3'
EVALUATOR='infinity_grid.g6_s8_evaluators:s8_s7_class_recursive_prefix_comparator_evaluator'

class G6S8V3Error(RuntimeError): pass

def _sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def _load(path:str|Path)->dict[str,Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def _validate_plan(plan:Mapping[str,Any],*,accepted_source_sha256:str)->dict[str,Any]:
    if type(plan) is not dict or plan.get('schema_id')!=PLAN_SCHEMA or plan.get('stage_id')!=STAGE_ID:
        raise G6S8V3Error('S8V3_PLAN_SCHEMA')
    core=plan.get('scientific_core'); exe=plan.get('execution_contract'); lineage=plan.get('lineage')
    if type(core) is not dict or type(exe) is not dict or type(lineage) is not dict: raise G6S8V3Error('S8V3_PLAN_SECTIONS')
    if canonical_sha256(core)!=plan.get('scientific_core_sha256'): raise G6S8V3Error('S8V3_CORE_HASH')
    if canonical_sha256(exe)!=plan.get('execution_contract_sha256'): raise G6S8V3Error('S8V3_EXEC_HASH')
    if canonical_sha256(lineage)!=plan.get('lineage_sha256'): raise G6S8V3Error('S8V3_LINEAGE_HASH')
    if plan.get('question_sha256')!=plan.get('scientific_core_sha256'): raise G6S8V3Error('S8V3_QUESTION_HASH_ALIAS')
    tmp=dict(plan); full=tmp.pop('full_preregistration_sha256',None)
    if canonical_sha256(tmp)!=full: raise G6S8V3Error('S8V3_FULL_HASH')
    auth=exe.get('executor_authority') or {}
    if auth.get('accepted_source_sha256')!=accepted_source_sha256: raise G6S8V3Error('S8V3_EXECUTOR_SOURCE_BINDING')
    a=exe.get('authorized_execution') or {}
    if a.get('targeted_probe_authorized') is not True or a.get('full_v2_depth3_to6_sweep_authorized') is not False: raise G6S8V3Error('S8V3_EXEC_SCOPE')
    if int(a.get('workers',0))!=4 or a.get('sampling_allowed') is not False: raise G6S8V3Error('S8V3_EXEC_WORKERS')
    t=core.get('targeted_probe') or {}
    if t.get('status')!='FROZEN_PROSPECTIVE_SENTINEL' or int(t.get('future_depth',0))!=3 or int(t.get('execution_context_count',0))!=8: raise G6S8V3Error('S8V3_TARGET_SCOPE')
    if t.get('stop_on_first_valid_split') is not True: raise G6S8V3Error('S8V3_STOP_RULE')
    if len(t.get('member_state_tokens') or ())!=3 or len(set(t['member_state_tokens']))!=3: raise G6S8V3Error('S8V3_TARGET_MEMBERS')
    rows=list(t.get('execution_contexts') or ())
    if [int(x.get('execution_context_index',-1)) for x in rows]!=list(range(8)): raise G6S8V3Error('S8V3_CONTEXT_ORDER')
    b=exe.get('frozen_task_budgets') or {}
    if b.get('budget_exhaustion_outcome')!='E1_FROZEN_TASK_BUDGET_EXCEEDED' or b.get('transient_pause_outcome')!='E2_TRANSIENT_EXECUTION_PAUSE': raise G6S8V3Error('S8V3_OUTCOME_CODES')
    if 'COUNTS_AS_N_PROFILE_COORDINATE_EQUIVALENTS' not in str(b.get('family_profile_budget_unit')): raise G6S8V3Error('S8V3_PROFILE_BUDGET_UNIT')
    return dict(plan)

def _validate_artifacts(plan:Mapping[str,Any], artifacts:Mapping[str,str|Path])->dict[str,dict[str,Any]]:
    expected={'s7_result','s7_manifest','s7_memberships','s7_survivors','s7_context_normalization','reuse_input'}
    if set(artifacts)!=expected: raise G6S8V3Error('S8V3_ARTIFACT_SET')
    sa=(plan['scientific_core'].get('scientific_authority') or {})
    sha_key={
      's7_result':'s7_result_file_sha256','s7_manifest':'s7_manifest_file_sha256','s7_memberships':'s7_memberships_file_sha256',
      's7_survivors':'s7_survivors_file_sha256','s7_context_normalization':'s7_context_normalization_file_sha256','reuse_input':'reuse_input_file_sha256'}
    out={}
    for name in sorted(expected):
        p=Path(artifacts[name]).resolve(strict=True)
        if _sha_file(p)!=sa.get(sha_key[name]): raise G6S8V3Error('S8V3_ARTIFACT_HASH:'+name)
        out[name]=_load(p)
    if out['s7_result'].get('result_sha256')!=sa.get('s7_result_sha256'): raise G6S8V3Error('S8V3_S7_RESULT')
    reuse=out['reuse_input']; payload=canonical_sha256({k:v for k,v in reuse.items() if k!='reuse_payload_sha256'})
    if reuse.get('reuse_payload_sha256')!=payload or payload!=sa.get('reuse_input_payload_sha256'): raise G6S8V3Error('S8V3_REUSE_PAYLOAD')
    ctx=out['s7_context_normalization']; rows=list(ctx.get('execution_coordinates') or ())
    frozen=list(plan['scientific_core']['targeted_probe']['execution_contexts'])
    for i,want in enumerate(frozen):
        if i>=len(rows): raise G6S8V3Error('S8V3_CONTEXT_MISSING')
        got=rows[i]
        if int(got.get('execution_context_index',-1))!=i or str(got.get('seed_ref'))!=str(want.get('seed_ref')) or tuple(got.get('operator') or ())!=tuple(want.get('operator') or ()) or got.get('position')!='LEFT':
            raise G6S8V3Error('S8V3_CONTEXT_BINDING')
    return out

def _task(plan:Mapping[str,Any], members:list[dict[str,Any]], outer:int, *, basis_refs:tuple[str,...], operators:tuple[tuple[int,int],...])->TaskSpec:
    core=plan['scientific_core']; exe=plan['execution_contract']; obs=core['observer']; b=exe['frozen_task_budgets']
    payload={
      's7_class_id':core['targeted_probe']['s7_class_id'],'class_members':members,'future_depth':3,'outer_execution_context_index':int(outer),
      'recursive_prefix_schedule':[1,8,32,124], 'basis_refs':list(basis_refs),
      'operator_basis':[list(x) for x in operators],
      'max_recursive_states':int(b['max_recursive_states_per_task']),'max_exact_relation_calls':int(b['max_exact_relation_calls_per_task']),
      'max_profile_coordinate_equivalents':int(b['max_exact_relation_profile_coordinate_equivalents_per_task']),
      'max_worker_rss_bytes':int(b['max_worker_rss_bytes_per_task']),'v3_budget_semantics':True,
    }
    return TaskSpec(task_id=f'S8V3-E2G-D3-O{outer:03d}',task_kind='G6_S8_V3_TARGETED_CLASS_COMPARISON',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=3.0)

def _metrics(runtime,phase_id:str)->dict[str,Any]:
    db=runtime.runtime_root()/'phases'/phase_id/'partition.sqlite3'
    con=sqlite3.connect(db)
    try:
        row=con.execute('SELECT task_id,metrics_json FROM task_results ORDER BY task_id').fetchone()
        if row is None: raise G6S8V3Error('S8V3_NO_TASK_RESULT')
        if con.execute('SELECT COUNT(*) FROM task_results').fetchone()[0]!=1: raise G6S8V3Error('S8V3_TASK_RESULT_COUNT')
        return {'task_id':str(row[0]),'metrics':json.loads(str(row[1]))}
    finally: con.close()

def _semantic_witness(m:Mapping[str,Any])->dict[str,Any]:
    return {'s7_class_id':str(m['s7_class_id']),'a_state_token':str(m['a_state_token']),'b_state_token':str(m['b_state_token']),'recursive_prefix_count':int(m['recursive_prefix_count']),'separator_a_signature':m['separator_a_signature'],'separator_b_signature':m['separator_b_signature']}

def _run_one(runtime, task:TaskSpec, *, phase_id:str, workers:int):
    return runtime.run_structural_partition(phase_id=phase_id,tasks=[task],evaluator_ref=EVALUATOR,requested_workers=int(workers),max_tasks=1,stream_shard_task_limit=1)



def _tupleize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleize(x) for x in value)
    if type(value) is dict:
        return {str(k): _tupleize(v) for k, v in value.items()}
    return value


def _generation_store(runtime, phase_id: str) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    db = runtime._phase_root(phase_id) / 'state_store.sqlite3'
    if not db.is_file():
        raise G6S8V3Error('S8V3_STAGE_GENERATION_STORE_MISSING:' + phase_id)
    con = sqlite3.connect(db)
    try:
        rows: dict[str, dict[str, Any]] = {}
        for token, digest, b, canonical_size, state_json in con.execute(
            'SELECT state_token,index_digest,canonical_bytes,canonical_size_bytes,state_json FROM states ORDER BY canonical_bytes,state_token'
        ):
            rows[str(token)] = {
                'state_token': str(token), 'identity_sha256': str(digest),
                'identity_canonical_bytes': bytes(b), 'canonical_size_bytes': int(canonical_size),
                'state_json': str(state_json),
            }
        by_task: dict[str, list[str]] = {}
        for tid, token in con.execute('SELECT task_id,state_token FROM occurrences ORDER BY task_id,occurrence_id'):
            by_task.setdefault(str(tid), []).append(str(token))
        return rows, by_task
    finally:
        con.close()


def _partition_signatures(runtime, phase_id: str) -> dict[str, tuple[Any, ...]]:
    db = runtime._phase_root(phase_id) / 'partition.sqlite3'
    if not db.is_file():
        raise G6S8V3Error('S8V3_STAGE_PARTITION_STORE_MISSING:' + phase_id)
    con = sqlite3.connect(db)
    try:
        class_bytes = {str(t): bytes(b) for t, b in con.execute('SELECT class_token,representative_signature_bytes FROM classes')}
        out = {}
        for tid, token in con.execute('SELECT task_id,class_token FROM task_results ORDER BY task_id'):
            b = class_bytes.get(str(token))
            if b is None:
                raise G6S8V3Error('S8V3_STAGE_PUBLIC_SIGNATURE_BYTES_MISSING:' + str(tid))
            val = _tupleize(json.loads(b.decode('utf-8')))
            if not isinstance(val, tuple):
                raise G6S8V3Error('S8V3_STAGE_PUBLIC_SIGNATURE_SHAPE:' + str(tid))
            out[str(tid)] = val
        return out
    finally:
        con.close()


def _relation_generation_task(*, task_id: str, state_tree: Mapping[str, Any], execution_context_index: int,
                              basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...], max_worker_rss_bytes: int) -> TaskSpec:
    payload = {
        'state_tree': state_tree, 'execution_context_index': int(execution_context_index),
        'basis_refs': list(basis_refs), 'operator_basis': [list(x) for x in operators],
        'max_worker_rss_bytes': int(max_worker_rss_bytes),
    }
    return TaskSpec(task_id=task_id, task_kind='G6_S8_V3_STAGED_RELATION_GENERATION',
                    binding_sha256=canonical_sha256(payload), payload=payload,
                    cost_weight=max(1.0, float(state_tree.get('n', 1))))


def _public_task(*, task_id: str, state_tree: Mapping[str, Any]) -> TaskSpec:
    payload = {'state_tree': state_tree}
    return TaskSpec(task_id=task_id, task_kind='G6_S8_V3_STAGED_PUBLIC_READ',
                    binding_sha256=canonical_sha256(payload), payload=payload,
                    cost_weight=max(1.0, float(state_tree.get('n', 1))))


def _profile_extension_task(*, task_id: str, identity_bytes: bytes, state_json: str,
                            start: int, end: int, previous_counts: tuple[int, ...],
                            basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...],
                            max_worker_rss_bytes: int) -> TaskSpec:
    try:
        identity_json = identity_bytes.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise G6S8V3Error('S8V3_STAGE_IDENTITY_UTF8') from exc
    payload = {
        'grandchild_identity_json': identity_json, 'grandchild_state_json': state_json,
        'start_context_index': int(start), 'end_context_index': int(end),
        'previous_counts': list(int(x) for x in previous_counts),
        'basis_refs': list(basis_refs), 'operator_basis': [list(x) for x in operators],
        'max_worker_rss_bytes': int(max_worker_rss_bytes),
    }
    size_hint = max(1, len(identity_bytes) + len(state_json))
    return TaskSpec(task_id=task_id, task_kind='G6_S8_V3_STAGED_PROFILE_EXTENSION',
                    binding_sha256=canonical_sha256(payload), payload=payload,
                    cost_weight=max(1.0, (size_hint / 2048.0) * max(1, end-start)))


def _stage_pause_from_error(exc: StageRuntimeError, *, outer: int, phase_id: str) -> dict[str, Any] | None:
    text = str(exc).upper()
    if any(token in text for token in ('RESOURCE','MEMORY','WORKSPACE','NO_SPACE','TIMEOUT','LEASE','MAX_WORKER_RSS_BYTES')):
        return {'outcome':'E2_TRANSIENT_EXECUTION_PAUSE','outer_execution_context_index':int(outer),
                'reason':'STAGE_RUNTIME:'+str(exc),'phase_id':phase_id}
    return None


def _profile_store(runtime, phase_id: str) -> dict[bytes, tuple[int, ...]]:
    out: dict[bytes, tuple[int, ...]] = {}
    for row in runtime.iter_generated_states(phase_id=phase_id):
        key = bytes(row['identity_canonical_bytes'])
        counts = tuple(int(x) for x in ((row.get('state') or {}).get('profile_counts') or ()))
        if key in out and out[key] != counts:
            raise G6S8V3Error('S8V3_STAGE_PROFILE_IDENTITY_CONFLICT:' + phase_id)
        out[key] = counts
    return out


def _e1_metrics(*, outer: int, prefix: int, cid: str, counters: Mapping[str, int], reason: str) -> dict[str, Any]:
    return {
        'v3_outcome':'E1_FROZEN_TASK_BUDGET_EXCEEDED','pause_reason':reason,'future_depth':3,
        'outer_execution_context_index':int(outer),'recursive_prefix_count':int(prefix),
        's7_class_id':cid,'split_found':False, **{str(k):int(v) for k,v in counters.items()},
        'staged_execution':True,
    }


def _run_staged_context(*, runtime, plan: Mapping[str, Any], members: list[dict[str, Any]], outer: int,
                        basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...],
                        workers: int, phase_tag: str) -> dict[str, Any]:
    """Equality-exact staged execution of the frozen depth-3 outer component.

    Scientific tuples are byte-equivalent to ``outer_component_projection``.
    Only execution is decomposed: exact states are deduplicated by complete
    canonical bytes, layers are checkpointed in StageRuntime stores, and all
    independent generation/profile work is distributed across the requested pool.
    """
    cid = str(plan['scientific_core']['targeted_probe']['s7_class_id'])
    budgets = plan['execution_contract']['frozen_task_budgets']
    max_states = int(budgets['max_recursive_states_per_task'])
    max_relations = int(budgets['max_exact_relation_calls_per_task'])
    max_profiles = int(budgets['max_exact_relation_profile_coordinate_equivalents_per_task'])
    max_rss = int(budgets['max_worker_rss_bytes_per_task'])
    counters = {'recursive_state_count':0,'exact_relation_call_count':0,'logical_profile_coordinate_count':0,
                'outer_unique_child_count':0,'grandchild_unique_count':0,'grandchild_occurrence_count':0}

    member_rows = sorted(members, key=lambda r: str(r['state_token']))
    parent_task_member: dict[str, str] = {}
    parent_tasks = []
    for i, row in enumerate(member_rows):
        tid = f'{phase_tag}-PARENT-M{i:02d}'
        parent_task_member[tid] = str(row['state_token'])
        parent_tasks.append(_relation_generation_task(task_id=tid,state_tree=row['state_tree'],execution_context_index=outer,
                                                       basis_refs=basis_refs,operators=operators,max_worker_rss_bytes=max_rss))
    parent_phase = f'{phase_tag}-PARENT-GEN'
    try:
        runtime.run_content_indexed_generation(phase_id=parent_phase,tasks=parent_tasks,
            evaluator_ref='infinity_grid.g6_s8_evaluators:s8_v3_relation_generation_evaluator',requested_workers=workers,max_tasks=3)
    except StageRuntimeError as exc:
        pause=_stage_pause_from_error(exc,outer=outer,phase_id=parent_phase)
        if pause is not None: return dict(pause,metrics=dict(counters,staged_execution=True))
        raise
    outer_rows, parent_occ = _generation_store(runtime,parent_phase)
    counters['exact_relation_call_count'] += len(parent_tasks)
    counters['outer_unique_child_count'] = len(outer_rows)
    counters['recursive_state_count'] += len(outer_rows)
    if counters['exact_relation_call_count'] > max_relations:
        return {'outcome':'E1_FROZEN_TASK_BUDGET_EXCEEDED','metrics':_e1_metrics(outer=outer,prefix=1,cid=cid,counters=counters,reason='MAX_EXACT_RELATION_CALLS')}
    if counters['recursive_state_count'] > max_states:
        return {'outcome':'E1_FROZEN_TASK_BUDGET_EXCEEDED','metrics':_e1_metrics(outer=outer,prefix=1,cid=cid,counters=counters,reason='MAX_RECURSIVE_STATES')}

    parent_children: dict[str, list[str]] = {str(r['state_token']):[] for r in member_rows}
    for tid, member_token in parent_task_member.items():
        parent_children[member_token] = list(parent_occ.get(tid, []))

    # Read each globally deduplicated outer child once, in parallel.
    child_order = sorted(outer_rows, key=lambda tok: bytes(outer_rows[tok]['identity_canonical_bytes']))
    pub_task_child: dict[str, str] = {}
    public_tasks=[]
    for i, tok in enumerate(child_order):
        state=json.loads(outer_rows[tok]['state_json']); tree=state.get('tree') if type(state) is dict else None
        if type(tree) is not dict: raise G6S8V3Error('S8V3_STAGE_OUTER_CHILD_TREE')
        tid=f'{phase_tag}-PUBLIC-C{i:06d}'; pub_task_child[tid]=tok; public_tasks.append(_public_task(task_id=tid,state_tree=tree))
    pub_phase=f'{phase_tag}-PUBLIC'
    if public_tasks:
        try:
            runtime.run_structural_partition(phase_id=pub_phase,tasks=public_tasks,
                evaluator_ref='infinity_grid.g6_s7_evaluators:s7_public_state_evaluator',requested_workers=workers,
                max_tasks=len(public_tasks),stream_shard_task_limit=1)
        except StageRuntimeError as exc:
            pause=_stage_pause_from_error(exc,outer=outer,phase_id=pub_phase)
            if pause is not None: return dict(pause,metrics=dict(counters,staged_execution=True))
            raise
        pub_by_task=_partition_signatures(runtime,pub_phase)
        child_public={pub_task_child[tid]:sig for tid,sig in pub_by_task.items()}
    else:
        child_public={}

    grandchild_universe: dict[bytes, str] = {}
    inner_incidence: dict[tuple[str,int], list[bytes]] = {}
    profiles: dict[bytes, tuple[int,...]] = {}
    prev=0
    for prefix in (1,8,32,124):
        new_indices=range(prev,prefix)
        inner_task_meta: dict[str, tuple[str,int]] = {}
        inner_tasks=[]
        for ci,tok in enumerate(child_order):
            state=json.loads(outer_rows[tok]['state_json']); tree=state.get('tree') if type(state) is dict else None
            if type(tree) is not dict: raise G6S8V3Error('S8V3_STAGE_OUTER_CHILD_TREE')
            for j in new_indices:
                tid=f'{phase_tag}-INNER-P{prefix:03d}-C{ci:06d}-I{j:03d}'
                inner_task_meta[tid]=(tok,int(j))
                inner_tasks.append(_relation_generation_task(task_id=tid,state_tree=tree,execution_context_index=j,
                    basis_refs=basis_refs,operators=operators,max_worker_rss_bytes=max_rss))
        inner_phase=f'{phase_tag}-INNER-P{prefix:03d}'
        if inner_tasks:
            try:
                runtime.run_content_indexed_generation(phase_id=inner_phase,tasks=inner_tasks,
                    evaluator_ref='infinity_grid.g6_s8_evaluators:s8_v3_relation_generation_evaluator',requested_workers=workers,
                    max_tasks=len(inner_tasks))
            except StageRuntimeError as exc:
                pause=_stage_pause_from_error(exc,outer=outer,phase_id=inner_phase)
                if pause is not None:return dict(pause,metrics=dict(counters,staged_execution=True,recursive_prefix_count=prefix))
                raise
            rows, by_task=_generation_store(runtime,inner_phase)
            counters['exact_relation_call_count'] += len(inner_tasks)
            for tid,(ctok,j) in inner_task_meta.items():
                keys=[]
                for local_tok in by_task.get(tid,[]):
                    row=rows[local_tok]; key=bytes(row['identity_canonical_bytes']); keys.append(key)
                    old=grandchild_universe.get(key)
                    if old is None: grandchild_universe[key]=str(row['state_json'])
                    elif old != str(row['state_json']): raise G6S8V3Error('S8V3_STAGE_GRANDCHILD_STATE_CONFLICT')
                inner_incidence[(ctok,j)]=keys
                counters['grandchild_occurrence_count'] += len(keys)
        counters['grandchild_unique_count']=len(grandchild_universe)
        if counters['exact_relation_call_count'] > max_relations:
            return {'outcome':'E1_FROZEN_TASK_BUDGET_EXCEEDED','metrics':_e1_metrics(outer=outer,prefix=prefix,cid=cid,counters=counters,reason='MAX_EXACT_RELATION_CALLS')}

        # Extend every accumulated exact grandchild only over missing profile coordinates.
        gkeys=sorted(grandchild_universe)
        profile_tasks=[]; task_key={}; profile_coordinate_delta=0
        for gi,key in enumerate(gkeys):
            previous=profiles.get(key,())
            start=len(previous)
            if start not in (0,prev):
                raise G6S8V3Error('S8V3_STAGE_PROFILE_PREFIX_CONTINUITY')
            tid=f'{phase_tag}-PROFILE-P{prefix:03d}-G{gi:07d}'; task_key[tid]=key
            profile_tasks.append(_profile_extension_task(task_id=tid,identity_bytes=key,state_json=grandchild_universe[key],
                start=start,end=prefix,previous_counts=previous,basis_refs=basis_refs,operators=operators,max_worker_rss_bytes=max_rss))
            profile_coordinate_delta += prefix-start
        if counters['logical_profile_coordinate_count'] + profile_coordinate_delta > max_profiles:
            counters['logical_profile_coordinate_count'] += profile_coordinate_delta
            return {'outcome':'E1_FROZEN_TASK_BUDGET_EXCEEDED','metrics':_e1_metrics(outer=outer,prefix=prefix,cid=cid,counters=counters,reason='MAX_PROFILE_COORDINATE_EQUIVALENTS')}
        prof_phase=f'{phase_tag}-PROFILE-P{prefix:03d}'
        if profile_tasks:
            try:
                runtime.run_content_indexed_generation(phase_id=prof_phase,tasks=profile_tasks,
                    evaluator_ref='infinity_grid.g6_s8_evaluators:s8_v3_profile_extension_evaluator',requested_workers=workers,
                    max_tasks=len(profile_tasks))
            except StageRuntimeError as exc:
                pause=_stage_pause_from_error(exc,outer=outer,phase_id=prof_phase)
                if pause is not None:return dict(pause,metrics=dict(counters,staged_execution=True,recursive_prefix_count=prefix))
                raise
            new_profiles=_profile_store(runtime,prof_phase)
            if set(new_profiles)!=set(gkeys): raise G6S8V3Error('S8V3_STAGE_PROFILE_COVERAGE')
            if any(len(v)!=prefix for v in new_profiles.values()): raise G6S8V3Error('S8V3_STAGE_PROFILE_LENGTH')
            profiles=new_profiles
        counters['logical_profile_coordinate_count'] += profile_coordinate_delta
        counters['recursive_state_count'] += len(child_order) + len(gkeys)
        if counters['recursive_state_count'] > max_states:
            return {'outcome':'E1_FROZEN_TASK_BUDGET_EXCEEDED','metrics':_e1_metrics(outer=outer,prefix=prefix,cid=cid,counters=counters,reason='MAX_RECURSIVE_STATES')}

        child_sig: dict[str, tuple[Any,...]] = {}
        for tok in child_order:
            root_pub=child_public[tok]
            blocks=[]
            for j in range(prefix):
                keys=inner_incidence.get((tok,j))
                if keys is None: raise G6S8V3Error('S8V3_STAGE_INNER_INCIDENCE_MISSING')
                vectors=[profiles[k] for k in keys]
                blocks.append(_multiset_signature(vectors))
            child_sig[tok]=('ORDINARY_BRANCH_MULTISET_COMPACT_D2_PROJECTION',2,prefix,root_pub,tuple(blocks))
        parent_sig: dict[str, tuple[Any,...]] = {}
        for row in member_rows:
            mt=str(row['state_token']); vals=[child_sig[tok] for tok in parent_children[mt]]
            parent_sig[mt]=('S8_RECURSIVE_OUTER_COMPONENT_EXECUTION_PROJECTION',3,int(outer),prefix,_multiset_signature(vals))
        rep_token=str(member_rows[0]['state_token']); rep_sig=parent_sig[rep_token]
        for row in member_rows[1:]:
            tok=str(row['state_token']); sig=parent_sig[tok]
            if sig != rep_sig:
                metrics=dict(counters,future_depth=3,outer_execution_context_index=int(outer),recursive_prefix_count=int(prefix),
                             split_found=True,s7_class_id=cid,a_state_token=rep_token,b_state_token=tok,
                             separator_a_signature=rep_sig,separator_b_signature=sig,p124_full_equality_equivalent=True,
                             staged_execution=True,requested_workers=int(workers),prefix_coordinate_reuse=True,
                             exact_global_state_dedup=True)
                return {'outcome':'T_SPLIT_WITNESS','metrics':metrics}
        prev=prefix
    metrics=dict(counters,future_depth=3,outer_execution_context_index=int(outer),recursive_prefix_count=124,
                 split_found=False,s7_class_id=cid,class_member_count=len(member_rows),p124_full_equality_equivalent=True,
                 staged_execution=True,requested_workers=int(workers),prefix_coordinate_reuse=True,exact_global_state_dedup=True)
    return {'outcome':'T_NO_SPLIT_THIS_CONTEXT','metrics':metrics}


def run_g6_s8_v3_targeted_probe(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,accepted_source_sha256:str,internal_execution_id:str,runtime)->dict[str,Any]:
    require_controller_execution_origin('run_g6_s8_v3_targeted_probe')
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    plan=_validate_plan(_load(plan_path),accepted_source_sha256=accepted_source_sha256)
    resolved=_validate_artifacts(plan,artifacts)
    reuse_rows=_reuse_rows(resolved['reuse_input']); _ref, groups=_reference_partition(resolved['s7_memberships'],reuse_rows)
    target=plan['scientific_core']['targeted_probe']; cid=str(target['s7_class_id']); tokens=list(target['member_state_tokens'])
    if sorted(groups.get(cid,[]))!=sorted(tokens): raise G6S8V3Error('S8V3_TARGET_CLASS_BINDING')
    members=[{'state_token':t,'state_tree':reuse_rows[t]['state']} for t in sorted(tokens)]
    ctx_rows=list(resolved['s7_context_normalization'].get('execution_coordinates') or ())
    if len(ctx_rows)!=124: raise G6S8V3Error('S8V3_CONTEXT_COUNT')
    basis_refs=tuple(dict.fromkeys(str(r['seed_ref']) for r in ctx_rows))
    if len(basis_refs)!=4 or tuple(sorted(basis_refs))!=basis_refs: raise G6S8V3Error('S8V3_BASIS_REFS')
    operators=tuple(tuple(int(x) for x in r['operator']) for r in ctx_rows[:31])
    if len(operators)!=31 or len(set(operators))!=31: raise G6S8V3Error('S8V3_OPERATOR_BASIS')
    for bi,ref in enumerate(basis_refs):
        block=ctx_rows[bi*31:(bi+1)*31]
        if any(str(r['seed_ref'])!=ref for r in block) or tuple(tuple(int(x) for x in r['operator']) for r in block)!=operators:
            raise G6S8V3Error('S8V3_CONTEXT_CARTESIAN_BASIS')
    write_json_atomic(out/'G6_S8_V3_ACCEPTED_PREREGISTRATION.json',plan)
    progress=[]; witness=None; final_outcome=None; pause=None
    for outer in range(8):
        phase_tag=f'G6_S8_V3_E2G_D3_O{outer:03d}_STAGED'
        row=_run_staged_context(runtime=runtime,plan=plan,members=members,outer=outer,basis_refs=basis_refs,operators=operators,workers=4,phase_tag=phase_tag)
        m=row['metrics']; outcome=row['outcome']
        if outcome in ('E1_FROZEN_TASK_BUDGET_EXCEEDED','E2_TRANSIENT_EXECUTION_PAUSE'):
            pause={'outcome':outcome,'outer_execution_context_index':outer,'reason':str(m.get('pause_reason',row.get('reason',''))),'phase_id':phase_tag,'metrics':m}; final_outcome=outcome; break
        entry={'outer_execution_context_index':outer,'phase_id':phase_tag,'split_found':bool(m.get('split_found')),
               'recursive_prefix_count':int(m.get('recursive_prefix_count',124)),'metrics_sha256':canonical_sha256(m),
               'staged_execution':True,'requested_workers':4}
        progress.append(entry)
        write_json_atomic(out/'G6_S8_V3_TARGETED_PROGRESS.json',{'schema_id':'IG_G6_S8_V3_TARGETED_PROGRESS_V1','completed_contexts':progress,'pause':pause,'witness':witness,'payload_sha256':canonical_sha256([progress,pause,witness])})
        if m.get('split_found') is True:
            initial=_semantic_witness(m)
            # Required preregistered confirmation: same frozen context, staged 1-worker and 4-worker replay.
            r1=_run_staged_context(runtime=runtime,plan=plan,members=members,outer=outer,basis_refs=basis_refs,operators=operators,workers=1,phase_tag=f'{phase_tag}_REPLAY_W1')
            r4=_run_staged_context(runtime=runtime,plan=plan,members=members,outer=outer,basis_refs=basis_refs,operators=operators,workers=4,phase_tag=f'{phase_tag}_REPLAY_W4')
            m1=r1['metrics']; m4=r4['metrics']
            if m1.get('split_found') is not True or m4.get('split_found') is not True: raise G6S8V3Error('S8V3_WITNESS_REPLAY_NOT_SPLIT')
            w1=_semantic_witness(m1); w4=_semantic_witness(m4)
            if canonical_sha256(initial)!=canonical_sha256(w1) or canonical_sha256(initial)!=canonical_sha256(w4): raise G6S8V3Error('S8V3_WITNESS_REPLAY_MISMATCH')
            witness={'outcome':'T_SPLIT_WITNESS','outer_execution_context_index':outer,'phase_id':phase_tag,'semantic_witness':initial,
                     'semantic_witness_sha256':canonical_sha256(initial),'replay_1_worker_sha256':canonical_sha256(w1),
                     'replay_4_worker_sha256':canonical_sha256(w4),'replay_agreement':True}
            final_outcome='T_SPLIT_WITNESS'; break
    if final_outcome is None: final_outcome='T_NO_SPLIT_IN_FROZEN_SENTINEL_CONTEXTS'
    result={'schema_id':'IG_G6_S8_V3_TARGETED_PROBE_RESULT_V1','status':'PASS' if final_outcome.startswith('T_') else 'PAUSED','outcome':final_outcome,
            'scientific_core_sha256':plan['scientific_core_sha256'],'execution_contract_sha256':plan['execution_contract_sha256'],
            'full_preregistration_sha256':plan['full_preregistration_sha256'],'s7_class_id':cid,'member_state_tokens':sorted(tokens),
            'contexts_completed':len(progress),'frozen_context_count':8,'witness':witness,'pause':pause,
            'nonclaims':plan['scientific_core']['nonclaims'],'internal_execution_id':internal_execution_id,
            'accepted_source_sha256':accepted_source_sha256,'execution_strategy':'STAGED_GLOBAL_DEDUP_INCREMENTAL_PREFIX_V1'}
    result['result_sha256']=canonical_sha256(result)
    write_json_atomic(out/'G6_S8_V3_TARGETED_PROBE_RESULT.json',result)
    files=[]
    for p in sorted(x for x in out.iterdir() if x.is_file()): files.append({'file_name':p.name,'sha256':_sha_file(p),'size_bytes':p.stat().st_size})
    manifest={'schema_id':'IG_G6_S8_V3_TARGETED_PROBE_MANIFEST_V1','status':result['status'],'result_sha256':result['result_sha256'],'files':files}; manifest['manifest_sha256']=canonical_sha256(manifest); write_json_atomic(out/'MANIFEST.json',manifest)
    return result
