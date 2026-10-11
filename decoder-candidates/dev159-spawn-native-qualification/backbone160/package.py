from pathlib import Path
import json,hashlib,zipfile
root=Path(__file__).resolve().parents[1];o=root/'deliverables/backbone160';o.mkdir(parents=True,exist_ok=True)
log=(root/'backbone160/recovery_tests_retry.log').read_text();assert 'passed' in log and 'failed' not in log
report='''IG_08LIB_DEV159_P0_NATIVE_BACKBONES_2026-10-11
PASS: seven dev159 native O9 backbones were generated from production DAG reconstruction. All nested physical witnesses, V3 construction identities and historical execution-order keys matched the pinned saved compatibility evidence. No process adapter or origin-guard bypass was used.
Missing-only resume: seven checkpoints reused; zero new builds. Previous 193 raw checkpoints remain preserved.
Focused paused-recovery tests against dev159: '''+log.strip()+'''
The first pytest invocation failed collection because the bundled runtime removes PYTHONPATH; retry explicitly bound the source path before importing pytest. Both logs retained.

Full P0 remains OPEN. Shared execution-service native qualification was not performed. Its direct API requires native controller stage origin. SHARED_SERVICE_PREFLIGHT.txt records the captured-stage route, durable pause/restore and publication-retry acceptance requirements. Independent-host qualification remains open. Live controller unchanged; no scientific admission.

Next: capture the source-bound native stage job using registered context/recipes, pause after durable commits, cold-restore and run missing recipes only, then qualify publication retries. Seven dev159 backbone gate is now complete for this host; shared-service and independent-host gates remain.
Prior raw and corrected beam handoff Drive ID: 17RmDzF85_YAh-6k-W-oC1rXADkYDGiB2.
GitHub branch: decoder/dev156-matched-generation-protocol-20261011; directory decoder-candidates/dev159-spawn-native-qualification/backbone160.
'''
p=o/'IG_08LIB_DEV159_P0_NATIVE_BACKBONES_2026-10-11.txt';p.write_text(report)
files=[p for p in (root/'backbone160').glob('*') if p.is_file()]+[root/'worker159/RESULT.json',root/'worker159/ORDERING_CORRECTION.json']
manifest={str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
with zipfile.ZipFile(o/'IG_08LIB_DEV159_P0_NATIVE_BACKBONES_HANDOFF_2026-10-11.zip','w',zipfile.ZIP_DEFLATED) as z:
 z.writestr('READ_FIRST.txt',report);z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
 for p in files:z.write(p,str(p.relative_to(root)))
print(str(o))
