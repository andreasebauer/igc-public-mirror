"""Read-only source and scientific callable provenance comparison."""
from pathlib import Path
import ast,json,hashlib,difflib
B=Path(__file__).resolve().parent;H=Path('/workspace/scratch/c89172f01c5f/binding0190/recovered/Infinity_Grid_Decoder_PHASE3_EMERGENCY_RESCUE_CHECKPOINT_V2_2026-09-02/source/Infinity_Grid_Algebra_Decoder_v0.28.8_TRUST_REPAIR_COMPLETE_2026-09-02/src/infinity_grid');N=Path('/tmp/ig_engine0204/infinity_grid');rows=[]
functions={'regime_scanner.py':['_state_pre_features','_farthest_select','O7State.construction_digest','O7State.base_key','O7State.skin','O7State.reserve_external','LiftState.construction_digest','LiftState.reserve_external'],'maturation_parallel.py':['_build_recipe_state','_candidate_worker','enumerate_candidate_recipes','_farthest_select_descriptors'],'materialized_discovery.py':['_get_process_o7_runtime']}
def definitions(p):
 a=ast.parse(p.read_text());d={}
 for x in a.body:
  if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)):d[x.name]=x
  elif isinstance(x,ast.ClassDef):
   for z in x.body:
    if isinstance(z,(ast.FunctionDef,ast.AsyncFunctionDef)):d[x.name+'.'+z.name]=z
 return d
for name,targets in functions.items():
 old=H/name;new=N/name;od=definitions(old);nd=definitions(new);diff='\n'.join(difflib.unified_diff(old.read_text().splitlines(),new.read_text().splitlines(),fromfile='historical/'+name,tofile='replay/'+name,n=3));(B/(name+'.diff.txt')).write_text(diff)
 funcs=[]
 for target in targets:
  x=ast.dump(od[target],include_attributes=False);y=ast.dump(nd[target],include_attributes=False);funcs.append({'callable':target,'AST_equal':x==y,'historical_AST_sha256':hashlib.sha256(x.encode()).hexdigest(),'replay_AST_sha256':hashlib.sha256(y.encode()).hexdigest()})
 rows.append({'module':name,'historical_path':str(old),'historical_sha256':hashlib.file_digest(old.open('rb'),'sha256').hexdigest(),'replay_path':str(new),'replay_sha256':hashlib.file_digest(new.open('rb'),'sha256').hexdigest(),'byte_equal':old.read_bytes()==new.read_bytes(),'callables':funcs})
(B/'SOURCE_COMPARISON.json').write_text(json.dumps({'status':'PASS_READ_ONLY_SOURCE_PROVENANCE_COMPARISON','generator_calls':0,'rows':rows},indent=2));print(json.dumps({r['module']:{v['callable']:v['AST_equal'] for v in r['callables']} for r in rows},indent=2))
