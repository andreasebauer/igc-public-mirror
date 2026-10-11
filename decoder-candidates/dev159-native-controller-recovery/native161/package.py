from pathlib import Path
import json,hashlib,zipfile
r=Path(__file__).resolve().parents[1];h=r/'native161';o=r/'deliverables/native161';o.mkdir(parents=True,exist_ok=True)
m=json.loads((h/'results/NATIVE_MATRIX.json').read_text());cold=json.loads((h/'results/FINAL_COLD.json').read_text());assert cold['task_count']==193
report='''IG_08LIB_DEV159_P0_NATIVE_CONTROLLER_RECOVERY_2026-10-11
PASS: captured dev159 candidate native O9 job executed through the genuine controller origin and shared structural-partition service. Production DAG initialization and native recipe construction were used; no origin guard or save gate was bypassed. All 193 results matched the registered frozen raw result hashes, including nested physical witnesses, V3 identities, historical execution-order policy/keys and pre-features.
Pause after exactly two durable commits: no completion published. Cold restore preserved both rows, checkpoint files and modes, and project claims. Recovery executed only the 191 missing recipes; the two prior rows remained unchanged.
Prepared-completion and completed-file publication faults were exercised. Publication retries created zero new execution attempts; total attempts remained two. All 193 final rows, prepared completion and final completion bytes were unchanged on reuse.
Final native preservation: PRESERVED, zero pending bytes. Final independent-directory cold restore: '''+str(cold['state_files_exact'])+''' files and their modes verified exactly; 193 task rows matched. This is same-host qualification, not independent-host certification.
Resources: one spawned worker; 4 GiB memory budget; no overall deadline; at least 10 GiB free at preparation. No live controller change or scientific admission.

Full P0 remains OPEN. Remaining: independent-host runtime/source/controller qualification, provider outage and unknown-write outcomes, full RC and authorized promotion/controller transition. This result closes the bounded native shared-service paused-recovery gate for the dev159 source overlay. Earlier dev155 evidence is no longer the only native recovery evidence.
Source identity: '''+json.loads((h/'PREREGISTRATION.json').read_text())['source_sha256']+'''. Qualification label dev159 identifies the restored candidate overlay; this does not by itself promote or retag the release.
GitHub: https://github.com/andreasebauer/igc-public-mirror/tree/decoder/dev156-matched-generation-protocol-20261011/decoder-candidates/dev159-native-controller-recovery

Restore: use results/FINAL_CHECKPOINT_SLIM.zip plus exact object bytes named <sha256>.bin. SOURCE_OBJECT_POINTERS.json maps every required digest to a verified Drive ID. Fetch raw bytes, verify size/hash, then call preservation.restore_checkpoint(archive,destination,archive_sha256,objects=download_directory). Pinned runtime is the dev151 bundled runtime specified in the predecessor source pointers included here. Independent-host gate must fail closed if host/runtime bindings differ.
'''
(o/'IG_08LIB_DEV159_P0_NATIVE_CONTROLLER_RECOVERY_2026-10-11.txt').write_text(report)
proofs={}
for stage in ['CAPTURE','PAUSE','FINAL']:
 for p in json.loads((h/'results'/(stage+'_PROVIDER_READBACKS.json')).read_text()):proofs[p['sha256']]={k:p[k] for k in ['sha256','size_bytes','drive_file_id','raw_exact_readback']}
(h/'SOURCE_OBJECT_POINTERS.json').write_text(json.dumps(proofs,indent=2,sort_keys=True))
files=[p for p in h.rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'RAW_PRIVATE' not in p.name]
files.append(r/'restored/matched_protocol/SOURCE_POINTERS.json')
manifest={str(p.relative_to(r)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
with zipfile.ZipFile(o/'IG_08LIB_DEV159_P0_NATIVE_CONTROLLER_RECOVERY_HANDOFF_2026-10-11.zip','w',zipfile.ZIP_DEFLATED) as z:
 z.writestr('READ_FIRST.txt',report);z.writestr('MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True))
 for p in files:z.write(p,str(p.relative_to(r)))
print(str(o))
