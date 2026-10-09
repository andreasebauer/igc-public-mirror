"""Prepare immutable candidate release and preregister native qualification."""
from pathlib import Path
import hashlib,json,sys,shutil
B=Path(__file__).resolve().parent;P=B.parent/'continuation0232';Q=B.parent/'continuation0231'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def write(n,x):(B/n).write_text(json.dumps(x,indent=2)+'\n')
proposal=read(P/'SCOPED_ADMISSION_PROPOSAL.json');a=read(Q/'AUDIT.json');export=read(P/'AUDIT.json');native=read(Q/'NATIVE_RESULT.json')
assert proposal['status']=='PREPARED_NOT_ADMITTED' and a['all193_DAG_roundtrip'] and a['native_exact_state_identity']
assert native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED' and read(Q/'CHECKPOINT_PRESERVED.json')['pending_bytes']==0
assert export['status']=='PASS_INDEPENDENT_SEALED_EXPORT_AUDIT' and read(P/'SEAL_DRIVE_SAVE.json')['raw_readback_verified']
assert sha(Q/'CATALOG_0151.json')==proposal['base_catalog_sha256']
shutil.copyfile(Q/'CATALOG_0151.json',B/'CATALOG_0151.json');shutil.copyfile(P/'SCOPED_ADMISSION_PROPOSAL.json',B/'PROPOSAL.json')
gate=dict(schema_id='IG_HISTORICAL_G1_EXACT_PARENT_ADMISSION_GATE_V1',status='PASS_PRIOR_NATIVE_COLD_AND_SEALED_EXPORT_GATES',prior_native_completion_sha256=native['completion_sha256'],prior_native_result_sha256=native['result_sha256'],source_evidence={n:sha(p) for n,p in [('native_0231',Q/'NATIVE_RESULT.json'),('cold_0231',Q/'AUDIT.json'),('preservation_0231',Q/'CHECKPOINT_PRESERVED.json'),('seal_audit_0232',P/'AUDIT.json'),('seal_save_0232',P/'SEAL_DRIVE_SAVE.json')]},scientific_root_sha256=proposal['proposed_slice']['scientific_root_sha256'],pending_bytes=0,generation_calls=0)
write('ADMISSION_GATE.json',gate)
ad=dict(schema_id='IG_SCOPED_HISTORICAL_G1_EXACT_PARENT_ADMISSION_V1',decision='ACCEPTED_FOR_REUSE_WITHIN_HISTORICAL_V1_EXACT_PARENT_SOURCE_SCOPE',release_id='MASTER_DATA_V1_0152',predecessor_catalog_sha256=proposal['base_catalog_sha256'],proposal_sha256=sha(B/'PROPOSAL.json'),scientific_root_sha256=proposal['proposed_slice']['scientific_root_sha256'],scope=proposal['proposed_slice']['scope'],counts=proposal['proposed_slice']['counts'],archive=proposal['proposed_slice']['archive'],admission_gate_sha256=sha(B/'ADMISSION_GATE.json'),generation_calls=0)
write('SCOPED_ADMISSION.json',ad)
cat=read(B/'CATALOG_0151.json');sl=dict(proposal['proposed_slice']);sl['scoped_admission_sha256']=sha(B/'SCOPED_ADMISSION.json');cat['slices'].append(sl);cat.update(release_id='MASTER_DATA_V1_0152',previous_release_id='MASTER_DATA_V1_0151',previous_catalog_sha256=proposal['base_catalog_sha256'])
cat['coverage_summary'].update(scientific_slices=152,g1_exact_parent_DAG_available=True,g1_exact_parent_DAG_scope=ad['scope']['authority'],g1_exact_parent_DAG_counts=ad['counts'])
cat['usage']+=' Historical V1 terminal exact-parent source uses ExactParentReader; no new unified reader for earlier slices is asserted.'
write('CATALOG_0152.json',cat)
inputs=dict(previous_catalog=B/'CATALOG_0151.json',candidate_catalog=B/'CATALOG_0152.json',scoped_admission=B/'SCOPED_ADMISSION.json',proposal=B/'PROPOSAL.json',admission_gate=B/'ADMISSION_GATE.json',seal=P/'G1_HISTORICAL_V1_EXACT_PARENT_SEAL0232.zip')
h={k:sha(p) for k,p in inputs.items()};s=read(Q/'SPEC.json');s.update(job_id='MASTER.G1.EXACT.PARENT.ADMISSION.0233',project_source=str(B/'project'))
s['question']=dict(stage_id=s['job_id'],description='Qualify scoped historical V1 exact parent source admission and reader against unchanged151 prior slices',outcomes=['PASS'],stopping_rule='Stop first source, seal, native/cold gate, DAG, census, reader route, predecessor, runtime or preservation mismatch. Activate release only after native verification, independent cold verification and pending0. No candidate generation.')
s['execution']['parameters']=dict(bindings=h);s['execution']['evaluator_refs']=[]
s['inputs']=[dict(logical_name=k,path=str(p),sha256=h[k]) for k,p in inputs.items()]
checks=dict(outcome='PASS',master_release='MASTER_DATA_V1_0152',scientific_slices=152,prior_slices_unchanged=151,exact_parent_reader_qualified=True,scoped_admission_verified=True,carrier_routes_checked=193,DAG_nodes_checked=16528,generation_calls=0,new_DAG_decodes=0,G2_promotion=False,master_cursor_updated=False)
s['output_contract']['result_checks']=[dict(pointer='/'+k,equals=v) for k,v in checks.items()];write('SPEC.json',s)
shutil.copyfile(Q/'READBACKS.json',B/'READBACKS.json');write('SOURCE_ATTESTATION.json',dict(status='PASS_SCOPED_EXACT_PARENT_ADMISSION_PREPARATION',master_cursor_updated=False,prior_slices_unchanged=151,generation_calls=0))
print('PASS preregistered scoped native reader qualification; master151 still active')
