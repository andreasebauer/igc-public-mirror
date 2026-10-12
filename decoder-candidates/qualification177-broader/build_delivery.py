from pathlib import Path
from collections import Counter
import hashlib, json, zipfile

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
R=HERE/'capture177/results'
result=json.loads((R/'NATIVE_RESULT.json').read_text())
nodes=result['result']['result']['nodes']
counts=dict(Counter(x['status'] for x in nodes))
cold=json.loads((R/'FINAL_COLD.json').read_text())
preservation=json.loads((R/'FINAL_PRESERVED.json').read_text())
assert preservation['status']=='PRESERVED' and preservation['pending_bytes']==0
assert cold['status']=='PASS'
all_pass=counts=={'PASS':159} and result['status']=='COMPLETED'
status='PASS_BOUNDED_BROADER_REGRESSION' if all_pass else 'HELD_BOUNDED_REGRESSION'
scope={'status':status,'counts':counts,'test_files_registered':15,
 'source_sha256':'2a01a45eff7003f8f8842ab39f011f4268fccd19cef2e714e1aff7ad322df9a5',
 'production_package_changed':False,'full_p0':False,'full_rc':False,
 'independent_host':False,'live_controller_changed':False,
 'scientific_execution':False,'prior_passing_credit_not_added_again':True}
(HERE/'RESULT_SCOPE.json').write_text(json.dumps(scope,indent=2)+'\n')
report=f'''INFINITY GRID 0.8LIB — QUALIFICATION177 BROADER CAPTURED REGRESSION
2026-10-12

{status}. Native terminal {result['status']}/{result['execution_status']}.
Counts: {json.dumps(counts,sort_keys=True)}; {len(nodes)} emitted test results.
Registered same fifteen-file scope as174, now on exact qualification176 source with both native O6 material archive and saved adopted Step2 parent. No source edits or runtime substitutions. Original passing/failed captures preserved. Passing counts are not accumulated as duplicate qualification credit.

Source 2a01a45eff7003f8f8842ab39f011f4268fccd19cef2e714e1aff7ad322df9a5.
Package be372e1dc9aea9b6a2258ad895aa0334e35aea4f309c6263fd866898d81d0c9c unchanged.
Eight capture roles admitted only after exact actual raw provider readbacks; four physical objects reused and four newly uploaded. Parent14778814 bytes,sha256526c2d5b5617d054ed228434dcfd70b8bd574d0def075ad574c0ddfd99954a10; O6 outer archive533342 bytes,sha256cb6f48641eb99d374d19d5dc8d5bfada13e12cf1ecb20ed0e66958629ec08f1b.

Pinned Python3.13.5/SQLite3.51.3,one outer worker,4GiB memory bound,at least10GiB free before capture,no automatic global runtime deadline. First nonzero selector boundary stopping contract retained. Nested project fixture uses spawn and retains explicit fork refusal. No observer rerun,master admission or science promotion.

Final checkpoint PRESERVED,zero pending bytes; {preservation['physical_objects_verified']} physical objects verified. Repeated full exports identical. Independent-directory cold restore matches {cold['state_files_exact']} state files and modes plus full project snapshot. Checkpoint {cold['checkpoint_sha256']}; slim export {cold['slim_export_sha256']}; full export {cold['full_export_sha256']}. Same host only.

Scope: a bounded broader engineering regression,not full profile/RC or independent-host acceptance. Full P0 remains OPEN. Original audit also includes automated observer interrupted-save/publication recovery and acceptance-critical assertion/runtime boundaries; these are not silently closed by this decoder subset. Reconcile those obligation records before claiming P0 completion. Read-only inventory on this source finds180 profile selectors versus181 active test files: tests/test_validation_selector_stop.py is missing from PROFILE.json. The newly portable native_o6_material_root input also needs explicit full-profile reconciliation. Next: amend/freeze the full qualification profile and fixture contracts in a versioned successor,then register prerequisite-ordered native continuation and stop on unexplained nonpass. Independent-host qualification still needs a genuinely separate host,not another directory.

Code/registration published at the recorded GitHub branch/commit with exact file readbacks. Original failed174 and focused175/176 remain retained.

RESTORE
Use capture177/results/FINAL_CHECKPOINT_SLIM.zip plus exact raw objects in FINAL_PROVIDER_READBACKS.json. Verify every SHA256/size before preservation.restore_checkpoint(...,objects=object_directory). Source and pinned runtime pointers inherited from176/175 are included. Do not blindly invoke prepare.py on existing disposable roots. Terminal capture is already completed/preserved; no earned scientific work needs replay.
'''
(HERE/'READ_FIRST.txt').write_text(report)
out=BASE/'deliverables177';out.mkdir(exist_ok=True)
(out/'IG_08LIB_QUALIFICATION177_BROADER_2026-10-12.txt').write_text(report)
entries=[(p,str(p.relative_to(HERE))) for p in HERE.rglob('*')
         if p.is_file() and not p.is_relative_to(HERE/'inputs')]
extras=[(BASE/'successor175/RUNTIME_RESTORE.json','runtime/RUNTIME_RESTORE.json'),
 (BASE/'successor175/PREDECESSOR_SOURCE_RUNTIME_POINTERS.json','runtime/PREDECESSOR_SOURCE_RUNTIME_POINTERS.json'),
 (BASE/'successor176/SOURCE_DIFF.json','source/SOURCE_DIFF.json'),
 (BASE/'successor176/source/tests/test_p0_global_identity.py','source/tests/test_p0_global_identity.py'),
 (BASE/'successor176/source/tests/test_decoder06_project_stage.py','source/tests/test_decoder06_project_stage.py'),
 (BASE/'recovery176/PROOF.json','native_o6/RECOVERY_PROOF.json')]
entries+=extras
manifest={n:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p,n in entries}
with zipfile.ZipFile(out/'IG_08LIB_QUALIFICATION177_BROADER_HANDOFF_2026-10-12.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p,n in entries:z.write(p,n)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':status,'counts':counts,'cold':cold}))
