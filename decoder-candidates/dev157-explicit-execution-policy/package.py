import pathlib,json,hashlib,zipfile
R=pathlib.Path(__file__).resolve().parents[1];P=R/'integration157';D=R/'deliverables';prefix='IG_08LIB_DEV157_NATIVE_POLICY_RECOVERY_2026-10-11'
assert json.loads((P/'AUDIT.json').read_text())['status']=='PASS';assert json.loads((P/'RESUME_CHECK.json').read_text())['all_checkpoint_bytes_unchanged']
text='''DEV157 EXPLICIT EXECUTION POLICY AND NATIVE CHAIN RECOVERY — 2026-10-11

PASS for isolated scanner candidate and 193 saved O9 repeated reservation chains.

Integrated HISTORICAL_COORDINATE_V1 as explicit O7/Lift state metadata. Native V3 construction identities remain separate and unchanged. Native reserve_external now selects through the registered execution key, propagating policy to successor states. Native lift construction rejects mixed policies and propagates the owner policy. Scanner panel ordering and selection use the explicit execution key. The default is NATIVE_V3. No process-global method replacement is used.

All 193 native O9 chains, each with fourteen successive reservations, matched the previously qualified compatibility evidence (2702 steps): full nested physical snapshots, capacities, recursive counters, reservation witnesses, historical order keys, and V3 construction identities. Unknown policies and mixed child policies were rejected.

Registered recovery test: paused after two completed chains; resumed only 191 missing chains. The two prior checkpoints were unchanged. Final resume reused all 193 checkpoints, generated zero new chains/steps, and preserved every checkpoint byte. Checkpoints bind explicit policy, exact candidate source, registration and recipe. Cold audit verifies 193 checkpoints/2702 expected native trace steps and rejects modified witness data.

Limits: isolated scanner candidate; full release integration remains open. Parallel-worker packet serialization has not yet been migrated/qualified. Native raw O8/O9 generation, selection and backbone under this new candidate require qualification; this step uses rehydrated saved states. This is same-host verification, not independent-host qualification. Full P0 remains open. Live controller unchanged.

Next: propagate explicit policy through parallel candidate packets and resumable state serialization, fail closed on absent/mismatched historical policy, then qualify source-bound native raw generation and independent-host recovery.

Repository: https://github.com/andreasebauer/igc-public-mirror
Branch: decoder/dev156-matched-generation-protocol-20261011
Candidate path: decoder-candidates/dev157-explicit-execution-policy
Audit commit: c77eecf3f7af5ebe73019d4d6e9ca0d4fc6a3e3f

Restore exact dev156 using the prior source/runtime pointers, then replace only infinity_grid/regime_scanner.py with the dev157 source in CODE.zip. REGISTRATION.json contains the complete candidate source inventory hashes and all input pins. The active qualification runtime is /tmp/ig_o9_cold_20261011; restore it via tools/bootstrap_p0_runtime.py from the pinned baseline if absent.
'''
(D/(prefix+'.txt')).write_text(text)
def sha(b):return hashlib.sha256(b).hexdigest()
files={'READ_FIRST.txt':text.encode(),'matched_protocol/SOURCE_POINTERS.json':(R/'restored/matched_protocol/SOURCE_POINTERS.json').read_bytes()}
for folder in ['integration157','restored/o9_qualification','restored/panel_o9']:
 for p in (R/folder).rglob('*'):
  if p.is_file() and p.suffix!='.py' and '__pycache__' not in p.parts:files[str(p.relative_to(R))]=p.read_bytes()
files['MANIFEST.json']=json.dumps({k:{'bytes':len(v),'sha256':sha(v)} for k,v in files.items()},indent=2).encode()
with zipfile.ZipFile(D/(prefix+'_HANDOFF.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for k,v in files.items():z.writestr(k,v)
files={str(p.relative_to(R)):p.read_bytes() for p in P.glob('*.py')};files['decoder157/infinity_grid/regime_scanner.py']=(R/'decoder157/infinity_grid/regime_scanner.py').read_bytes();files['integration157/REGISTRATION.json']=(P/'REGISTRATION.json').read_bytes();files['MANIFEST.json']=json.dumps({k:{'bytes':len(v),'sha256':sha(v)} for k,v in files.items()},indent=2).encode()
with zipfile.ZipFile(D/(prefix+'_CODE.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for k,v in files.items():z.writestr(k,v)
for suffix in ['_HANDOFF.zip','_CODE.zip']:
 with zipfile.ZipFile(D/(prefix+suffix)) as z:
  assert z.testzip() is None
  for k,v in json.loads(z.read('MANIFEST.json')).items():assert sha(z.read(k))==v['sha256']
print('PACKAGED_AND_VERIFIED')
