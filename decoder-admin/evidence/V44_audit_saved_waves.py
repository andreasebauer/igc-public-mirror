from pathlib import Path
import json,hashlib,zipfile,collections
P=Path('/workspace/scratch/6d6f5c6d37c8/v33_gate');D=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=P/'DECODER_V33_DEV140_TARGETED_PASS_2026-09-30.zip'
assert sha(archive)=='3abc74a44a98963bde60b644e177de675bd9780839fb418eed7c722392e04bb2'
z=zipfile.ZipFile(archive)
read=lambda n:json.loads(z.read(n))
spec=read('CAPTURE_SPEC.json');selectors=spec['execution']['nodes'];assert len(selectors)==5
assert spec['output_contract']['preservation']['validation_wave_selectors']==1
assert spec['resources']['workers']==1
attempts=sorted([json.loads(z.read(n)) for n in z.namelist() if n.startswith('attempts/') and n.endswith('.json')],key=lambda r:r['attempt_id'])
assert [r['status'] for r in attempts]==['PAUSED']*4+['COMPLETED']
assert len({r['source_sha256'] for r in attempts})==1
assert attempts[0]['source_sha256']=='c19e1dc9ff75f67ece4cf6b9b5875bbfce7b5be36ad3153751c5e276ca7f2ebd'
assert len({r['registration_sha256'] for r in attempts})==1
boundary=[json.loads(z.read(n)) for n in z.namelist() if '/save_boundaries/' in n and n.endswith('.json')]
assert len(boundary)==4
for i in range(4):
 assert attempts[i]['reason'].endswith('VALIDATION_SAVE_BOUNDARY:'+str(4-i))
 b=next(b for b in boundary if b['selectors']==selectors[i:i+1]);assert b['remaining_selectors']==selectors[i+1:]
rows=[json.loads(z.read(n)) for n in z.namelist() if '/nodes/' in n and n.endswith('.json')]
assert len(rows)==5 and {r['node'] for r in rows}==set(selectors)
assert all(r['finished'] and all(r['phases'][k]['outcome']=='passed' for k in ['setup','call','teardown']) for r in rows)
checked=0
for wave in range(1,6):
 assert all(read(f'WAVE_{wave}_INTEGRITY.json').values())
 a=read(f'WAVE_{wave}_ACK.json');assert a['outbox']['pending_objects']==[]
 assert a['obligations_acknowledged']==28
 prior=read(f'WAVE_{wave}_PRIOR_EVIDENCE.json')
 for path,h in prior.items():
  archived='native_evidence/'+path.removeprefix('runtime/runs/')
  assert hashlib.sha256(z.read(archived)).hexdigest()==h
  checked+=1
restore=read('RESTORE_VERIFICATION.json');assert restore['exact_completion_equal'] and restore['terminal_completion_proof'] and restore['tests_dispatched']==0
objects=read('ALL_READBACKS.json')
for r in objects:
 p=Path(r['readback']);assert p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256']
result=dict(status='AUDIT_PASS_NO_WORKLOAD_DISPATCH',source_sha256=attempts[0]['source_sha256'],v33_archive_sha256=sha(archive),attempt_statuses=[r['status'] for r in attempts],native_boundaries=4,distinct_passed_nodes=5,prior_evidence_hash_comparisons=checked,checkpoint_roles_acknowledged=140,physical_readbacks_reverified=len(objects),saved_restore=restore,remaining=['Native resume refusal while logical roles pending but physical bytes already saved','Native failed/skipped first selector blocks later selector','Long selector spanning two ordinary checkpoint intervals with active genuine save loop, reserve/latency measurements and full-boundary corruption refusal'],qualification='OPEN',new_tests_dispatched=0)
(D/'V44_AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
