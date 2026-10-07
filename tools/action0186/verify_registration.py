from pathlib import Path
import sys,json,hashlib,zipfile
B=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent)
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
reg=json.loads((B/'PROJECT_REGISTRATION.json').read_bytes())
assert sha(B/'SPEC.json')==reg['spec_sha256']
assert set(reg['project_code'])=={str(p.relative_to(B)) for p in (B/'project').rglob('*.py')}
for n,h in reg['project_code'].items():assert sha(B/n)==h,n
assert sha(B/'CATALOG_0150.json')=='0e1296efea7bd007eebece394fbd966884f370c33bb5b849dded7d1dcbfc9c1f'
old=json.loads((B/'CATALOG_0150.json').read_bytes());candidate=json.loads((B/'CATALOG_0151_CANDIDATE.json').read_bytes());assert len(old['slices'])==150 and candidate['slices'][:-1]==old['slices'] and len(candidate['slices'])==151
assert candidate['publication_status']=='CANDIDATE_PENDING_NATIVE_VERIFICATION'
assert sha(B/'PREDECESSOR0185_HANDOFF.zip')=='98509e72bd805a6c0ba64ca10cc1427e05d3354ab4c50785496979f237940712'
with zipfile.ZipFile(B/'PREDECESSOR0185_HANDOFF.zip') as z:assert z.testzip() is None
pre=json.loads((B/'NATIVE_PREFLIGHT.json').read_bytes())
assert pre['status']=='PASS' and not pre['capture_called'] and not pre['handler_called']
for row in pre['architecture']['modules']:
 rel=Path(row['module_path']).relative_to(Path('/workspace/scratch/c89172f01c5f/register0186'))
 assert sha(B/rel)==row['sha256'],str(rel)
v=json.loads((B/'VALIDATION.json').read_bytes());assert not v['native_handler_called'] and v['scientific_slices']==150 and v['new_admissions']==0
print('PASS: frozen source/specification, predecessor150, candidate scope, native preflight and0185 dependency')
