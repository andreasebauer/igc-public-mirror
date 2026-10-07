"""Materialize exact registered bytes outside the mirrored filesystem before capture."""
from pathlib import Path
import sys,json,hashlib,shutil
B=Path(__file__).resolve().parent;root=Path(sys.argv[1]).resolve()
if not root.is_relative_to(Path('/tmp')) or root.exists():raise ValueError('FRESH_ISOLATED_ROOT_REQUIRED')
spec=json.loads((B/'SPEC.json').read_bytes())
reg=json.loads((B/'PROJECT_REGISTRATION.json').read_bytes())
bindings=json.loads(Path(sys.argv[2]).read_bytes()) if len(sys.argv)>2 else {}
if set(bindings)-{'engine_source','inputs'}:raise ValueError('UNKNOWN_RECOVERY_BINDING')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
if sha(B/'SPEC.json')!=reg['spec_sha256']:raise ValueError('SPEC_HASH')
for n,h in reg['project_code'].items():
 if sha(B/n)!=h:raise ValueError('SOURCE_HASH')
if set(reg['project_code'])!={str(p.relative_to(B)) for p in (B/'project').rglob('*.py')}:raise ValueError('SOURCE_CLOSURE')
engine=bindings.get('engine_source',spec['engine_source'])
sys.path.insert(0,engine)
from infinity_grid.v05_engineering_worker import engineering_source_tree_digest
if engineering_source_tree_digest(engine)!=reg['engine_source_sha256']:raise ValueError('ENGINE_HASH')
root.mkdir();shutil.copytree(B/'project',root/'project',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(engine,root/'engine',ignore=shutil.ignore_patterns('__pycache__'))
if engineering_source_tree_digest(root/'engine')!=reg['engine_source_sha256']:raise ValueError('COPIED_ENGINE_HASH')
spec['project_source']=str(root/'project');spec['engine_source']=str(root/'engine')
for group in [spec['inputs'],spec['environment']['artifacts']]:
 for x in group:
  old=Path(bindings.get('inputs',{}).get(x['logical_name'],x['path']))
  if sha(old)!=x['sha256']:raise ValueError('INPUT_HASH:'+x['logical_name'])
  dest=root/'inputs'/x['logical_name']/old.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,dest)
  if sha(dest)!=x['sha256']:raise ValueError('COPIED_INPUT_HASH')
  x['path']=str(dest)
(root/'SPEC.json').write_text(json.dumps(spec,indent=2)+'\n')
print(str(root/'SPEC.json'))
