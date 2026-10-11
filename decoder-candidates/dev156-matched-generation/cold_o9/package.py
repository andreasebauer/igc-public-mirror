import pathlib,json,hashlib,zipfile
R=pathlib.Path(__file__).resolve().parents[1];P=R/'cold_o9';D=R/'deliverables';prefix='IG_08LIB_DEV156_O9_COLD_RESTORE_2026-10-11'
assert json.loads((P/'RESULT.json').read_text())['status']=='PASS_COLD_O9_ROOT_RESTORATION';assert json.loads((P/'RESUME_CHECK.json').read_text())['unchanged']
text='''O9 COLD RESTORATION — 2026-10-11

PASS: after automated workspace pruning, recovered the O9 handoff and code bundles from Drive, checked every bundled manifest hash, reconstructed exact dev156 source (1558 files) from the pinned dev151 baseline and dev156 overlay, and restored a fresh bundled runtime (Python 3.13.5 / SQLite 3.51.3, full file/byte/mode checks).

All 62 saved O9 roots restored: 48 selected roots (24 per arm) and 14 backbone roots (seven per arm). Complete nested physical snapshots matched: skins, capacity vectors, leaf reservation counters and edge witnesses. Historical execution keys matched while native V3 identities remained distinct. All 5552 nested witness endpoints had valid child indexes, endpoint types and ordinal/counter bounds. A corrupted witness was rejected.

Zero candidate builds and zero reservation calls. Resume reused 62 restoration checkpoints with zero new restorations; every checkpoint remained byte-identical.

Limits: same host and the same native skin implementations, not an independent-host qualification or independent skin algorithm. Witness bounds checks do not by themselves prove complete reservation uniqueness. No claim of full historical raw-generation closure or full dev156 native release integration. Live controller unchanged. Full P0 remains open.

Next: integrate the now-qualified historical execution policy as explicit candidate metadata, preserving V3 identity; qualify native release recovery, then an independent host.

Code: https://github.com/andreasebauer/igc-public-mirror
Branch: decoder/dev156-matched-generation-protocol-20261011
Restoration registration commit: c3c238e86400dfca8b4893e0a558295ea005e9ea

Durable predecessor: IG_08LIB_DEV156_O9_PANEL_BACKBONE_2026-10-11_HANDOFF.zip
Drive ID: 1Sz4O9STG5-m87SrJYb98zYu7IGO1zJ6w
Source/runtime restore pointers are bundled in matched_protocol/SOURCE_POINTERS.json.
'''
(D/(prefix+'.txt')).write_text(text)
def sha(b):return hashlib.sha256(b).hexdigest()
files={'READ_FIRST.txt':text.encode(),'matched_protocol/SOURCE_POINTERS.json':(R/'restored/matched_protocol/SOURCE_POINTERS.json').read_bytes()}
for folder in ['cold_o9','restored/panel_o9','restored/o9_qualification']:
 for p in (R/folder).rglob('*'):
  if p.is_file() and p.suffix!='.py' and '__pycache__' not in p.parts:files[str(p.relative_to(R))]=p.read_bytes()
files['MANIFEST.json']=json.dumps({k:{'bytes':len(v),'sha256':sha(v)} for k,v in files.items()},indent=2).encode()
with zipfile.ZipFile(D/(prefix+'_HANDOFF.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for k,v in files.items():z.writestr(k,v)
files={str(p.relative_to(R)):p.read_bytes() for p in P.glob('*.py')};files['cold_o9/REGISTRATION.json']=(P/'REGISTRATION.json').read_bytes();files['MANIFEST.json']=json.dumps({k:{'bytes':len(v),'sha256':sha(v)} for k,v in files.items()},indent=2).encode()
with zipfile.ZipFile(D/(prefix+'_CODE.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for k,v in files.items():z.writestr(k,v)
for suffix in ['_HANDOFF.zip','_CODE.zip']:
 with zipfile.ZipFile(D/(prefix+suffix)) as z:
  assert z.testzip() is None
  for k,v in json.loads(z.read('MANIFEST.json')).items():assert sha(z.read(k))==v['sha256']
print('PACKAGED_AND_VERIFIED')
