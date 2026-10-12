from pathlib import Path
import hashlib, json, zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
results = HERE / 'capture176/results'
native = json.loads((results/'NATIVE_RESULT.json').read_text())
nodes = native['result']['result']['nodes']
assert len(nodes) == 17 and all(x['status']=='PASS' for x in nodes)
assert native['status']=='COMPLETED' and native['execution_status']=='FINISHED'
cold = json.loads((results/'FINAL_COLD.json').read_text())
preserved = json.loads((results/'FINAL_PRESERVED.json').read_text())
assert cold['status']=='PASS' and preserved['status']=='PRESERVED' and preserved['pending_bytes']==0
proof = json.loads((BASE/'recovery176/PROOF.json').read_text())
report = f'''INFINITY GRID 0.8LIB — QUALIFICATION176 NATIVE O6 WITNESS
2026-10-12

COMPLETED/FINISHED: 17 PASS, 0 FAIL, 0 SKIPPED in two captured test files.
Native O6 resource collision witness now PASS rather than skipped. Nine prior identity controls, a new raw-evidence mutation refusal, and all six project-stage tests passed in this exact successor source scope.

Recovered genuine retained parent evidence from the exact-source material archive, not from regenerated observers. Outer archive SHA256 {proof['outer_sha256']}; inner replay archive SHA256 {proof['inner_sha256']}. Ten native parent records loaded; both exact witness parent identifiers and h6 resource hashes verified. Full recovery member hashes are included. The original source and source archive remain unchanged.

Successor test reads only the enclosing saved capture's native_o6_material_root artifact, checks source/package/save bindings and outer/inner hashes, safely extracts the immutable replay root in its own temporary directory, and checks both native parent resource hashes. Missing capture artifact or hash mismatch raises; no obsolete absolute directory or skip fallback remains. A one-byte altered archive is rejected before extraction. Production package unchanged; only tests/test_p0_global_identity.py changed from qualification175.

Successor source 2a01a45eff7003f8f8842ab39f011f4268fccd19cef2e714e1aff7ad322df9a5.
Production package be372e1dc9aea9b6a2258ad895aa0334e35aea4f309c6263fd866898d81d0c9c.
Capture 0d11178b24d1abfa010aa1a6c52aa59a9c86f9e75a6ba195202911c5e0251538.

Seven real capture objects saved and exact raw provider readbacks admitted before dispatch. Same pinned V5 runtime: Python3.13.5/SQLite3.51.3; one outer worker,4GiB memory bound,no automatic outer runtime deadline. Nested project fixture remains spawn;fork and missing/wrong runtime-binding refusal controls retained. Test-local synthetic receipts are fixture controls,not remote durability evidence.

Final checkpoint PRESERVED,zero pending bytes. Nineteen physical objects raw-verified. Repeated full exports identical. Cold restore matched {cold['state_files_exact']} state files and modes plus complete project snapshot. Checkpoint {cold['checkpoint_sha256']}; slim export {cold['slim_export_sha256']}. Independent directory on same host only,not independent-host acceptance.

Published successor scripts/test registration on GitHub branch decoder/qualification176-native-o6-20261012 with exact readbacks. Both known qualification174 fixture blockers are now closed in affected successor tests. Qualification174's other passing rows retain their original scope;this is not a combined full regression or release qualification.

NEXT
Register the broader fifteen-file regression on this exact successor source with both retained Step2 adopted-parent and native O6 inputs; retain prerequisite ordering and stop at first nonzero selector. Preserve earned prior evidence without duplicate passing credit. Then continue missing full-profile/RC obligations and independent-host acceptance. Full P0,full RC and independent-host qualification remain OPEN. Live controller unchanged;no master admission,observer rerun or science promotion.

RESTORE
Use capture176/results/FINAL_CHECKPOINT_SLIM.zip and exact object bytes named by FINAL_PROVIDER_READBACKS.json. Fetch each Drive object,check SHA256 and size,then restore through the captured source preservation.restore_checkpoint(...,objects=object_directory). Runtime recovery pointers and verified-runtime evidence inherited from175 are included. Fail closed on any mismatch. Old prepare.py names an existing disposable capture root;do not blindly dispatch it. The captured terminal result is already complete.
'''
(HERE/'READ_FIRST.txt').write_text(report)
out=BASE/'deliverables176';out.mkdir(exist_ok=True)
(out/'IG_08LIB_QUALIFICATION176_NATIVE_O6_2026-10-12.txt').write_text(report)
files=[p for p in HERE.rglob('*') if p.is_file() and not p.is_relative_to(HERE/'source')]
files += [HERE/'source/tests/test_p0_global_identity.py',HERE/'source/tests/test_decoder06_project_stage.py']
extra=[(p,'recovery176/'+p.name) for p in (BASE/'recovery176').glob('*') if p.is_file()]
extra += [(BASE/'successor175/RUNTIME_RESTORE.json','runtime/RUNTIME_RESTORE.json'),(BASE/'successor175/PREDECESSOR_SOURCE_RUNTIME_POINTERS.json','runtime/PREDECESSOR_SOURCE_RUNTIME_POINTERS.json')]
entries=[(p,str(p.relative_to(HERE))) for p in files]+extra
manifest={n:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,n in entries}
with zipfile.ZipFile(out/'IG_08LIB_QUALIFICATION176_NATIVE_O6_HANDOFF_2026-10-12.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p,n in entries:z.write(p,n)
    z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'DELIVERY_PREPARED','cold':cold,'files':len(entries)}))
