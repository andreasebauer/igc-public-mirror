from pathlib import Path
from collections import Counter
import hashlib,json,zipfile

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
R=HERE/'capture178/results'
native=json.loads((R/'NATIVE_RESULT.json').read_text())
counts=dict(Counter(r['status'] for r in native['result']['result']['nodes']))
cold=json.loads((R/'FINAL_COLD.json').read_text())
preserved=json.loads((R/'FINAL_PRESERVED.json').read_text())
if preserved['status']!='PRESERVED' or preserved['pending_bytes'] or cold['status']!='PASS':
 raise RuntimeError('SAVE_AND_COLD_GATE_NOT_CLOSED')
profile=json.loads((HERE/'PROFILE_RECONCILIATION.json').read_text())
qualified=counts=={'PASS':23} and native['status']=='COMPLETED'
scope={'status':'PASS_PROFILE_CONTRACT_GATE' if qualified else 'HELD_PROFILE_CONTRACT_GATE',
 'counts':counts,'registered_test_files':4,'profile_test_files':181,'fixture_roles':8,
 'source_sha256':profile['source_sha256'],'package_unchanged':True,
 'full_profile_native_collection':False,'full_profile_qualified':False,
 'full_p0':False,'full_rc':False,'independent_host':False,'live_controller_changed':False}
(HERE/'RESULT_SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n')
report=f'''INFINITY GRID 0.8LIB — QUALIFICATION178 FULL-PROFILE CONTRACT RECONCILIATION
2026-10-12

{scope['status']}; native {native['status']}/{native['execution_status']}; counts {json.dumps(counts)}.
Four captured test files: qualification contract,selector-stop control,native O6 identity,project-stage fixture. This is a23-test gate,not execution or native collection of the full181-file profile.

Profile now covers all181 active top-level test files exactly once. Added tests/test_validation_selector_stop.py,previously omitted. The negative control proves first-nonzero-selector stopping retains the executed prefix and does not run the later selector. Functional/workspace/representative/timing group ordering preserved; timing remains one worker. Existing full-profile worker settings were not promoted as qualified topology by this bounded gate.

Eight explicit fixture roles. step2_parent_snapshot now binds the cold-adopted173 archive526c2d5b5617d054ed228434dcfd70b8bd574d0def075ad574c0ddfd99954a10,14778814 bytes. Original historical archive713452bf109d17d4b377f9084e61383f8397c1a418012be2f41e09eff17f3bd8 remains separately recorded and unchanged. Native O6 material archivecb6f48641eb99d374d19d5dc8d5bfada13e12cf1ecb20ed0e66958629ec08f1b,533342 bytes is explicit. Private failed_evidence pin unchanged; native stage/prerun/representative prerequisites remain open. Runtime target aligned to verified V5 manifest;full dependency/install/independent-host fields remain open,not silently certified.

Only qualification/PROFILE.json and tests/test_qualification_contract.py changed from qualification176/177 source. Production package unchanged:{profile['package_sha256']}.
Parent source:{profile['parent_source_sha256']}.
Successor source:{profile['source_sha256']}.
Profile hash:{profile['profile_sha256']}.
Historical profile copies/source-reference retained;current status explicitly UNQUALIFIED_PROFILE_SUCCESSOR178.

Eight capture objects saved/reused with exact actual raw readbacks before dispatch. Pinned Python3.13.5/SQLite3.51.3;one outer worker,4GiB bound,no automatic global deadline,>=10GiB free at preparation. Nested project fixture uses spawn;fork refusal and altered O6 evidence refusal retained. No scientific observer reruns,master admissions or live-controller changes.

PRESERVED,zero pending bytes;{preserved['physical_objects_verified']} physical objects verified. Repeated full exports byte-identical. Cold restore:{cold['state_files_exact']} state files and modes plus complete project snapshot exact. Checkpoint:{cold['checkpoint_sha256']}. Slim export:{cold['slim_export_sha256']}. Same-host independent-directory scope only.

NEXT
Freeze native full-profile collection and recover/qualify the remaining declared fixtures in prerequisite order: saved_stage_one and genuine pre-execution archive,saved_stage_four,saved_stage_outcome,saved_representative_qualification,private failed_evidence. Reuse existing exact earned fixtures only when source/input contracts permit;do not manufacture completion or prerun history. Then run the full registered groups,retain nonpasses and stop at first unexplained mismatch. Reconcile remaining P0 operational recovery/save/runtime gates and independent-host acceptance separately. Previous177159-pass checkpoint stays authoritative for its original source/input scope;no duplicate totals. Full P0/full RC remain OPEN.

RESTORE
Use capture178/results/FINAL_CHECKPOINT_SLIM.zip with exact provider objects listed by FINAL_PROVIDER_READBACKS.json. Check size/hash before preservation.restore_checkpoint(...,objects=object_directory). Saved source object contains the full181-file profile. Inherited runtime/source pointers included. Do not blindly dispatch prepare.py on existing roots;terminal gate is already complete/preserved. Code/registrations mirrored on the recorded GitHub branch with exact readbacks.
'''
(HERE/'READ_FIRST.txt').write_text(report)
out=BASE/'deliverables178';out.mkdir(exist_ok=True)
(out/'IG_08LIB_QUALIFICATION178_PROFILE_2026-10-12.txt').write_text(report)
entries=[(p,str(p.relative_to(HERE))) for p in HERE.rglob('*') if p.is_file() and not p.is_relative_to(HERE/'source')]
entries += [(HERE/'source/qualification/PROFILE.json','source/qualification/PROFILE.json'),
 (HERE/'source/tests/test_qualification_contract.py','source/tests/test_qualification_contract.py'),
 (BASE/'successor175/RUNTIME_RESTORE.json','runtime/RUNTIME_RESTORE.json'),
 (BASE/'successor175/PREDECESSOR_SOURCE_RUNTIME_POINTERS.json','runtime/PREDECESSOR_SOURCE_RUNTIME_POINTERS.json')]
manifest={n:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p,n in entries}
with zipfile.ZipFile(out/'IG_08LIB_QUALIFICATION178_PROFILE_HANDOFF_2026-10-12.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p,n in entries:z.write(p,n)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'scope':scope,'cold':cold}))
