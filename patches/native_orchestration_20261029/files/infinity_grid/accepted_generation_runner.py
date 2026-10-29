"""Native runtime-owned compare-before-use generation; no private scheduler."""
from pathlib import Path
import json,os,time
from .canon import canonical_sha256,write_json_atomic
from .library_service_protocol import decode,digest,SCHEMA
from .complete_result_storage import ResultStore,require,exact_equal
from .full_block_runner import recover_workspace,_assign,retire_phase
from .release_profile import output_contracts,build_release
from .accepted_result_ledger import Ledger,GENERATED,COMPARED,AVAILABLE,DIVERGED,AcceptanceStopped
from .result_acceptance import validate_contract
from .accepted_target_profile import validate_target,build_accepted_release
from .full_block_manifest import evaluator_for
from .execution import TaskSpec
from .runtime_capacity import measured_worker_allocation
from .accepted_dispatch_plan import wave_plan


def _fault(stage,tid):
 if os.environ.get('IG_ENGINEERING_ACCEPTANCE_ABORT')!=stage:return
 require(os.environ.get('IG_ENGINEERING_ACCEPTANCE_ACK')=='ACCEPTANCE_RECOVERY_SOFTWARE_FIXTURE','ACCEPTANCE_FAULT_ACK_REQUIRED')
 raise RuntimeError('ENGINEERING_ACCEPTANCE_ABORT:'+stage+':'+tid)


def run(runtime,parameters,input_artifacts,manifest):
 from .complete_result_storage import verification_operation
 from .acceptance_observer import session
 from .accepted_predecessor import session as predecessor_session
 from .validation_frontier import session as validation_session,implementation_binding
 # A yielded batch follows the controller's existing PAUSED/checkpoint path.
 with verification_operation(), session(runtime.runtime_root/"full_generation"/"telemetry"):
  from .accepted_dispatch_plan import validate_work_plan
  from .acceptance_observer import register_plan,bind_identity
  from .preservation import _ACTIVE as preservation_active
  active=preservation_active.get()
  if active is not None:
   workspace=Path(active['workspace'])
   try:
    controller=json.loads((workspace/'runtime/status/CONTROLLER_STATUS.json').read_text())
    bind_identity(**{k:controller.get(k) for k in ('job_id','attempt_id','source_sha256')})
   except (OSError,ValueError):pass
  bind_identity(manifest_sha256=canonical_sha256(manifest),source_sha256=manifest['source_sha256'])
  binding=dict(implementation_binding(),manifest_sha256=canonical_sha256(manifest),trust_policy='NATIVE_OWNED_CHECKPOINT_STATE_V1',checks=['J3_CONSERVATION','J3_HISTORICAL_COMPARISON'])
  with validation_session(runtime.runtime_root/"full_generation"/"validation_authority",binding,max_new_units=None), predecessor_session(input_artifacts,manifest):
   if 'operational_work_plan' in input_artifacts:
    plan=validate_work_plan(json.loads(Path(input_artifacts['operational_work_plan']).read_bytes()),manifest)
    register_plan(plan)
   return _run(runtime,parameters,input_artifacts,manifest)

def _run(runtime,parameters,input_artifacts,manifest):
 runtime._require_execution();require(canonical_sha256(manifest)==parameters['manifest_sha256'],'FULL_RUN_MANIFEST_BYTES')
 require(manifest['resources']['workers']==runtime.default_workers and manifest['resources']['memory_budget_bytes']==runtime.memory_budget_bytes and manifest['resources']['workspace_budget_bytes']==runtime.workspace_budget_bytes,'FULL_NATIVE_RESOURCE_BINDING')
 from .preservation import _ACTIVE
 active=_ACTIVE.get();require(active is not None and manifest['resources']==active['admission']['job']['resources'],'NATIVE_PRESERVATION_SESSION_REQUIRED')
 from .result_contracts import contract_for,policy
 require(policy(contract_for(active['workspace']))==manifest['preservation'],'FULL_NATIVE_PRESERVATION_BINDING')
 root=runtime.runtime_root/'full_generation';root.mkdir(exist_ok=True);recover_workspace(root)
 store=ResultStore(root/'content',create=not (root/'content').exists(),purpose=manifest['purpose'])
 if 'generation_content' in input_artifacts:
  from .generation_checkpoint import import_content_archive
  import_content_archive(store,Path(input_artifacts['generation_content']))
 target=validate_target(store,manifest);tasks={t['task_id']:t for t in manifest['tasks']};manifest_ref=store.document(manifest);executions=[]
 with Ledger(root/'tasks.sqlite',store,manifest) as ledger:
  initial=ledger.initial_count
  from .native_source_ledger import NativeSources
  native=NativeSources(ledger,target) if target.get('native_entries') else None
  def status():
   from .acceptance_observer import progress,_ACTIVE
   counts={key:0 for key in (GENERATED,COMPARED,AVAILABLE,DIVERGED)}
   for _,r in ledger.rows():counts[r['state']]+=1
   observer=_ACTIVE.get()
   elapsed=0.0 if observer is None else time.monotonic()-observer.started
   progress('comparison',counts[COMPARED]+counts[AVAILABLE],len(tasks),elapsed)
   progress('preservation',counts[AVAILABLE],len(tasks),elapsed)
   write_json_atomic(root/'STATUS.json',{'schema_id':'IG_ACCEPTED_JOB_STATUS_V1','manifest_sha256':manifest['manifest_sha256'],'target_ref':manifest['target_ref'],'states':counts,
    **({'available_native_sources':len(native.available()),'required_native_sources':len(native.entries)} if native else {}),'scientific_adoption':False})
  def native_fault(stage,asset_id):status();_fault(stage,asset_id)
  native_available={} if native is None else native.process(input_artifacts,fault=native_fault,budget_check=lambda:runtime._enforce_execution_budgets(root))
  from .accepted_predecessor import seed,producer_source,generation_tasks
  from .acceptance_observer import phase
  with phase("retained_producer_admission"):seed(ledger)
  with phase("retained_result_verification"):ledger.verify_retained()
  def prepare(t):
   require(set(t['depends_on'])<=set(ledger.available()),'ACCEPTED_PREMISE_NOT_AVAILABLE');args=decode(t['arguments']);available=ledger.available()
   for binding in t['bindings']:args=_assign(args,binding['input_path'],store.read_result_at(available[binding['task_id']],binding['output_path']))
   for ref in (t['producer_source_ref'],t['runtime_ref'],*t['input_refs']):store.read(ref,runtime.memory_budget_bytes)
   if t['mode']=='FRESH':
    require(store.document_read(t['producer_source_ref'])=={'schema_id':'IG_FULL_BLOCK_PRODUCER_SOURCE_PIN_V1','records_source_sha256':producer_source(t,manifest['source_sha256'])},'FRESH_PRODUCER_SOURCE_PIN')
    require(store.document_read(t['runtime_ref'])=={'schema_id':'IG_FULL_BLOCK_PRODUCER_RUNTIME_PIN_V1','manifest_sha256':manifest['runtime_manifest_sha256']},'FRESH_PRODUCER_RUNTIME_PIN')
   c=validate_contract(store,t,t['validation_ref']);input_pin=digest(args);require(input_pin==c['input_digest'],'ACCEPTANCE_FROZEN_INPUT_DOMAIN_CHANGED')
   record={'task_id':t['task_id'],'family':t['family'],'operation_id':t['operation_id'],'input_payload_ref':store.exact_payload(args),'input_digest':input_pin,
    'producer_source_ref':t['producer_source_ref'],'runtime_ref':t['runtime_ref'],'input_refs':t['input_refs'],'schema_version':t['schema_version'],'identity_version':SCHEMA,
    'required_fields':[[k] for k in output_contracts()['required_fields'][t['schema_version']]],'completion_checks':t['completion_checks'],
    'depends_on':t['depends_on'],'dependency_bindings':t['bindings'],'origin_mode':'FRESH_RECOMPUTE' if t['mode']=='FRESH' else 'VERIFIED_RESTORED_BLOCK'}
   return args,record
  def advance(tid):
   row=ledger.row(tid)
   if row['state']==GENERATED:
    row=ledger.compare(tid);status();_fault(COMPARED,tid)
   if row['state']==DIVERGED:
    status();raise AcceptanceStopped('DATA_COMPARISON_BLOCKED:'+tid+':'+row['assessment_ref']['sha256'])
   if row['state']==COMPARED:
    c=validate_contract(store,tasks[tid],tasks[tid]['validation_ref'])
    if c['save_policy']=='LOCAL_DURABLE':proof={'kind':'LOCAL_DURABLE','result_ref':row['result_ref'],'assessment_ref':row['assessment_ref']}
    else:return # A quiescent physical dispatch must reach ACK before more work.
    ledger.make_available(tid,save_proof=proof);status();_fault(AVAILABLE,tid)
  def save_compared(ready):
   pending={}
   for tid in ready:
    if tasks[tid]['mode']=='REFERENCE':continue
    row=ledger.row(tid)
    if row['state']==COMPARED:
     require(validate_contract(store,tasks[tid],tasks[tid]['validation_ref'])['save_policy']=='NATIVE_CHECKPOINT_ACK','ACCEPTED_PENDING_SAVE_POLICY')
     pending[tid]={'result_ref':row['result_ref'],'assessment_ref':row['assessment_ref']}
   if pending:
    from .acceptance_save import obtain_native_save_proofs
    proofs=obtain_native_save_proofs(store,pending)
    from .checkpoint_save_batch import proof_verification_batch
    with proof_verification_batch(store):
     for tid in pending:ledger.make_available(tid,save_proof=proofs[tid]);status();_fault(AVAILABLE,tid)
   require(all(tasks[tid]['mode']=='REFERENCE' or ledger.row(tid)['state']==AVAILABLE for tid in ready),'ACCEPTED_WAVE_NOT_AVAILABLE')
  remaining=set(tasks);scheduled=set();wave=0
  while remaining:
   ready=sorted(tid for tid in remaining if set(tasks[tid]['depends_on'])<=scheduled);require(ready,'ACCEPTED_TARGET_DEPENDENCY_CYCLE')
   plan=wave_plan(ledger,runtime,manifest,wave,ready)
   for tid in ready:
    t=tasks[tid];require(set(t['depends_on'])<=set(ledger.available()),'ACCEPTED_PREMISE_NOT_AVAILABLE')
    if t['mode']=='REFERENCE':
     # A reference is preserved and byte-checked, never exposed as a premise.
     require(t['reuse_result_ref'] is not None,'REFERENCE_DATA_REQUIRED');store.verify_result(t['reuse_result_ref']);continue
    row=ledger.row(tid)
    if row is not None:
     _,record=prepare(t);require(exact_equal(row['record'],record),'RETAINED_ARGUMENT_OR_RECORD_CHANGED');advance(tid)
    elif t['mode']=='REUSE':
     _,record=prepare(t);ledger.generated(t,record,t['reuse_result_ref']);status();_fault(GENERATED,tid);advance(tid)
   # Drain retained compared results before any new native dispatch. In
   # particular, resuming a pending checkpoint must not generate another batch.
   save_compared([tid for tid in ready if tasks[tid]['mode']!='REFERENCE' and ledger.row(tid) is not None])
   for batch in plan['batches']:
    if all(ledger.row(tid) is not None for tid in batch['task_ids']):continue
    evaluator=batch['evaluator']
    # Frozen complete batch includes retained members, preserving native phase
    # identity after a callback interruption inside this dispatch.
    phase_members=generation_tasks([tasks[tid] for tid in batch['task_ids']])
    prepared={};native_tasks=[]
    for t in phase_members:
     args,record=prepare(t);prepared[t['task_id']]=(args,record)
     payload={'recipe_id':t['task_id'],'operation_id':t['operation_id'],'input_payload_ref':record['input_payload_ref'],'input_sha256':record['input_digest']}
     native_tasks.append(TaskSpec(t['task_id'],'ACCEPTED_COMPLETE_TASK',digest(payload),payload))
    def on_committed_result(tid,state):
     require(tid in prepared and state['recipe_id']==tid and state['input_sha256']==digest(prepared[tid][0]),'ACCEPTANCE_NATIVE_TASK_BINDING')
     ledger.generated(tasks[tid],prepared[tid][1],state['stored_result_ref']);status();_fault(GENERATED,tid);advance(tid)
    phase='ACCEPTED_'+str(wave)+'_'+canonical_sha256([t.task_id for t in native_tasks])[:24]
    observation_key='native_worker_peak_bytes:'+evaluator+':'+canonical_sha256(sorted({t['operation_id'] for t in phase_members}))
    observed_row=ledger.db.execute('SELECT v FROM meta WHERE k=?',(observation_key,)).fetchone()
    observed=None if observed_row is None else int(observed_row[0])*2
    allocation=measured_worker_allocation(runtime.default_workers,len(native_tasks),runtime.memory_budget_bytes,observed_worker_bytes=observed,purpose=manifest['purpose'])
    from .acceptance_observer import detail
    detail(worker_allocation=allocation)
    generated=runtime.run_content_indexed_generation(phase_id=phase,tasks=native_tasks,evaluator_ref=evaluator,requested_workers=allocation['workers'],
      max_tasks=None,max_generated_occurrences=None,max_task_occurrences=None,max_result_bytes=None,max_work_seconds=None,
      complete_result_store={'directory':str(store.directory),'purpose':store.purpose},on_committed_result=on_committed_result)
    peak=generated.execution_metadata.get('worker_metric_totals',{}).get('worker_peak_rss_bytes')
    if peak is not None:
     with ledger.db:ledger.db.execute('INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)',(observation_key,str(max(int(peak),0 if observed_row is None else int(observed_row[0])))))
    for item in runtime.iter_generated_states(phase_id=phase):on_committed_result(item['state']['recipe_id'],item['state'])
    require(all(ledger.row(t.task_id)['state'] in (COMPARED,AVAILABLE) for t in native_tasks),'ACCEPTED_GROUP_NOT_COMPARED')
    retire_phase(runtime,phase,[t.task_id for t in native_tasks]);executions.append({'task_ids':[t.task_id for t in native_tasks],'allocation':allocation,'execution':generated.execution_metadata})
    save_compared(batch['task_ids'])
   save_compared(ready)
   remaining.difference_update(ready);scheduled.update(ready);wave+=1
  available=ledger.available();required={e['task_id'] for e in target['required_entries']};require(set(available)==required,'ACCEPTED_TARGET_INCOMPLETE')
  rows={tid:row for tid,row in ledger.rows()};records={tid:row['record'] for tid,row in rows.items()}
  source=store.document({'schema_id':'IG_FULL_JOB_SOURCE_PIN_V1','records_source_sha256':manifest['source_sha256']});environment=store.document({'schema_id':'IG_FULL_JOB_RUNTIME_PIN_V1','manifest_sha256':manifest['runtime_manifest_sha256']})
  data=None
  if records:
   data_manifest=store.document({'schema_id':'IG_COMPLETE_PRODUCER_JOB_V1','purpose':manifest['purpose'],'scope_id':manifest['scope_id'],'families':sorted({r['family'] for r in records.values()}),
    'tasks':[records[t['task_id']] for t in manifest['tasks'] if t['task_id'] in records],'source_ref':source,'environment_ref':environment})
   data=build_release(store,data_manifest,available,label=manifest['scope_id'])
  accepted=build_accepted_release(store,manifest,manifest_ref,data,rows,native_availability=native_available)
  result={'outcome':'PASS','manifest_sha256':manifest['manifest_sha256'],'complete_task_count':len(rows),'new_commits':len(rows)-initial,'reused_commits':initial,
   'available_task_count':len(available),**({'available_native_source_count':len(native_available),'native_source_disposition':'REUSED_SOURCE_VERIFIED','native_sources_grant_fresh_match':False} if native else {}),'target_ref':manifest['target_ref'],'release':data,'accepted_release':accepted,'execution':executions,
   'scientific_adoption':False,'scope':'COMPLETE_FROZEN_ACCEPTED_DATA_TARGET','purpose':manifest['purpose']}
  write_json_atomic(root/'RESULT.json',result);status();runtime._enforce_execution_budgets(root);return result
