import pathlib,json,hashlib,zipfile
R=pathlib.Path(__file__).resolve().parent;D=R/'deliverables';prefix='IG_08LIB_DEV156_O9_PANEL_BACKBONE_2026-10-11';P=R/'panel_o9'
def sha(b):return hashlib.sha256(b).hexdigest()
result=json.loads((P/'RESULT.json').read_text());assert result['status']=='PASS_O9_SELECTION_BACKBONE';assert json.loads((P/'AUDIT.json').read_text())['status']=='PASS';assert json.loads((P/'RESUME_CHECK.json').read_text())['unchanged']
text='''O9 SELECTION AND BACKBONE QUALIFICATION — 2026-10-11

PASS, finite isolated compatibility scope. Both arms rehydrated all 193 saved O9 raw candidates. Cap 220 was inactive. Dedup and beam selection retained the identical 24 motifs, historical execution order keys and mandatory topology twins. V3 identities remain distinct.

Seven frozen backbone motifs were built through each arm's native _build_lift (14 builds). All physical skins, capacities, recursive child counters, nested endpoint witnesses and historical order keys matched. Native construction uses the process-local historical reservation policy while preserving native V3 identities. Exact decoder156 release files and live controller remain unchanged.

Cold saved-checkpoint audit checked 386 dispositions and 14 backbone checkpoints; a modified nested counter was rejected. This is a same-host cold audit, not independent-host qualification or independent materialization of resource skins. Historical slim scanner has no full backbone driver; frozen motif catalogue was applied explicitly to each native builder.

Resume reused all 14 backbone checkpoints with zero new builds; all saved backbone checkpoint bytes remained unchanged. No O9 raw recipe regeneration and no O10 dispatch.

Full P0 remains open. Next: independent cold restoration of the O9 panel/backbone from durable source/runtime pointers, followed by release integration and independent-host qualification.

GitHub commit: adc653da330e480dee83598a97110705b389b97d
Branch: decoder/dev156-matched-generation-protocol-20261011
Repository: https://github.com/andreasebauer/igc-public-mirror
'''
(D/(prefix+'.txt')).write_text(text)
files={}
for directory in ['panel_o9','o9_qualification','panel_o8']:
 for p in (R/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.py':files[str(p.relative_to(R))]=p.read_bytes()
files['matched_protocol/SOURCE_POINTERS.json']=(R/'matched_protocol/SOURCE_POINTERS.json').read_bytes()
files['READ_FIRST.txt']=text.encode();files['REPORT.txt']=text.encode()
files['MANIFEST.json']=json.dumps({k:{'bytes':len(v),'sha256':sha(v)} for k,v in files.items()},indent=2).encode()
with zipfile.ZipFile(D/(prefix+'_HANDOFF.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for k,v in files.items():z.writestr(k,v)
code={}
for directory in ['panel_o9','o9_qualification','compatibility_o8','recursive_order']:
 for p in (R/directory).glob('*.py'):code[str(p.relative_to(R))]=p.read_bytes()
code['package_panel_o9.py']=(R/'package_panel_o9.py').read_bytes()
code['panel_o9/REGISTRATION.json']=(P/'REGISTRATION.json').read_bytes()
code['MANIFEST.json']=json.dumps({k:{'bytes':len(v),'sha256':sha(v)} for k,v in code.items()},indent=2).encode()
with zipfile.ZipFile(D/(prefix+'_CODE.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for k,v in code.items():z.writestr(k,v)
for suffix in ['_HANDOFF.zip','_CODE.zip']:
 with zipfile.ZipFile(D/(prefix+suffix)) as z:
  assert z.testzip() is None
  for k,v in json.loads(z.read('MANIFEST.json')).items():assert sha(z.read(k))==v['sha256']
print('PACKAGED_AND_VERIFIED')
