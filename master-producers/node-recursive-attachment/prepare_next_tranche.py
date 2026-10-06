from pathlib import Path
import sys,json,hashlib,shutil,ast
B=Path(__file__).resolve().parent.parent;tranche=int(sys.argv[1]);assert 2<=tranche<=6;R=B/'recursive_native'/f'tranche{tranche:02d}';assert not (R/'CAPTURE_SAVE_STATUS.json').exists();(R/'project').mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(B/'engine'))
from infinity_grid.canon import canonical_sha256,canonical_bytes
from infinity_grid.workflow_guard import preflight
from infinity_grid.v05_engineering_worker import engineering_source_tree_digest
sha=lambda b:hashlib.sha256(b).hexdigest()
def write(n,o):(R/n).write_text(json.dumps(o,indent=2)+'\n')
D=B/'l13_lineage';dry=json.loads((D/'DRY_VERIFICATION.json').read_text());assert dry['status']=='PASS_DRY_RECURSIVE_ATTACHMENT';plan=json.loads((D/'COMMON_RUNTIME_NEXT_PLAN.json').read_text());scope=plan['chunks'][tranche-1]
for n in ['PARENT_PACKET.json','PRIMITIVE_PACKET.json','DRY_RECIPE.json']:shutil.copy2(D/n,R/n)
assert sha((R/'DRY_RECIPE.json').read_bytes())==dry['recipe_sha256']
bound=json.loads((R/'DRY_RECIPE.json').read_text())
for key,file in [('parent_packet_sha256','PARENT_PACKET.json'),('primitive_packet_sha256','PRIMITIVE_PACKET.json')]:assert sha((R/file).read_bytes())==bound[key]
assert sha((D/'recursive_attachment.py').read_bytes())==bound['adapter_sha256']
assert sha((B/'unified139/CATALOG_0139.json').read_bytes())=='4753cee3dd1adb0b963f10ae11d98fb0c35640b43f3706f1d90b836150458102'
source=(D/'recursive_attachment.py').read_text().replace('from canonical_kernel import canonical_sha256','from infinity_grid.canon import canonical_sha256')
(R/'project/recursive_attachment.py').write_text(source);(R/'project/__init__.py').write_text('')
shutil.copy2(B/'recursive_native/handler_v2.py',R/'project/handler.py')
parents=json.loads((R/'PARENT_PACKET.json').read_text())['rows'][scope['parent_start']:scope['parent_stop_exclusive']];events=json.loads((R/'PRIMITIVE_PACKET.json').read_text())['events']
assert parents[0]['object_id']==scope['first_parent_id'] and parents[-1]['object_id']==scope['last_parent_id']
tree=ast.parse((D/'historical_run_level.py').read_text());ns={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'dtup','canon','bridge','compose'}],type_ignores=[]),'original_pure_constructor','exec'),ns)
def rec(r):return (r[0],tuple(tuple(p) for p in r[1]),r[2],tuple(r[3]) if isinstance(r[3],list) else r[3],r[4])
bs=set();lawful=0
for p in parents:
 for e in events:
  for a in range(4):
   for b in range(3):
    child=ns['compose'](rec(p['ordered_record']),a,rec(e['record']),b)
    if child is not None:lawful+=1;bs.add(canonical_sha256(child))
assert lawful==scope['expected_lawful_rooted_constructions']
recipe={'schema_id':'IG_RECURSIVE_COMMON_RUNTIME_TRANCHE_RECIPE_V1','scope':scope,'identity_recipe_sha256':dry['recipe_sha256'],'expected_projected_boundaries':len(bs),'projected_set_sha256':canonical_sha256(sorted(bs)),'parent_packet_sha256':sha((R/'PARENT_PACKET.json').read_bytes()),'primitive_packet_sha256':sha((R/'PRIMITIVE_PACKET.json').read_bytes()),'adapter_sha256':sha((R/'project/recursive_attachment.py').read_bytes()),'handler_sha256':sha((R/'project/handler.py').read_bytes()),'catalog_sha256':'4753cee3dd1adb0b963f10ae11d98fb0c35640b43f3706f1d90b836150458102','admission':False,'comparison':'Complete independently recovered original constructor outcomes + recursive parent/reference checks','stopping_rule':f"Exactly frozen parent indices {scope['parent_start']}..{scope['parent_stop_exclusive']-1} x13 native events x12 endpoints; stop on any hash, semantic, reference or count mismatch. Save and verify capture/checkpoint.",'resources':{'tasks':scope['tasks'],'endpoint_attempts':scope['endpoint_attempts'],'shard_rows':1024,'shard_byte_limit':4194304}}
write('RECIPE_CONTRACT.json',recipe)
q={'status':'PASS','engine_digest':engineering_source_tree_digest(B/'engine'),'adapter_sha256':recipe['adapter_sha256'],'handler_sha256':recipe['handler_sha256'],'expected_lawful':lawful,'expected_projected_boundaries':len(bs)};write('QUALIFICATION.json',q)
check=R/'preflight_source';shutil.copytree(B/'engine',check,ignore=shutil.ignore_patterns('__pycache__'));shutil.copytree(R/'project',check/'project');pf=preflight(check,[check/'project/recursive_attachment.py',check/'project/handler.py']);write('PREFLIGHT.json',pf);shutil.rmtree(check);assert pf['status']=='PASS',pf
spec=json.loads((B/'node_adapter/CAPTURE_SPEC.json').read_text());spec['job_id']=f'MASTER.QUALIFICATION.NODE.RECURSIVE.DEPTH2.T{tranche:02d}.V1';spec['project_source']=str(R/'project');spec['question']={'stage_id':f'MASTER:QUALIFICATION:NODE:RECURSIVE:DEPTH2:T{tranche:02d}:V1','description':f'Recursive attachment with complete parent formation and port-origin references, tranche{tranche}','outcomes':['PASS'],'stopping_rule':recipe['stopping_rule']};spec['execution']={'kind':'STAGE','handler_ref':'project.handler:handler','evaluator_refs':['project.recursive_attachment:evaluate'],'parameters':{'recipe_sha256':sha((R/'RECIPE_CONTRACT.json').read_bytes())}}
spec['inputs']=[{'logical_name':name,'path':str(R/file),'sha256':sha((R/file).read_bytes())} for name,file in [('parent_packet','PARENT_PACKET.json'),('primitive_packet','PRIMITIVE_PACKET.json'),('recipe_contract','RECIPE_CONTRACT.json'),('dry_recipe','DRY_RECIPE.json')]]
spec['output_contract']['result_checks']=[{'pointer':'/'+k,'equals':v} for k,v in {'outcome':'PASS','rooted_constructions':lawful,'formation_occurrences':lawful,'projected_boundaries':len(bs),'official_master_admission':False,'historical_l13_reproduced':False}.items()];write('CAPTURE_SPEC.json',spec);print(json.dumps(q))

ops=(B/'recursive_native/tranche01/native_ops.py').read_text().replace("store=B/'campaign/stores/node_recursive_depth2_t01'",f"store=B/'campaign/stores/node_recursive_depth2_t{tranche:02d}'").replace('tranche1 qualification',f'tranche{tranche} qualification')
(R/'native_ops.py').write_text(ops)
v=(B/'recursive_native/tranche01_repack/verify_saved.py').read_text().replace("count==scope['expected_lawful_rooted_constructions']==36352","count==scope['expected_lawful_rooted_constructions']").replace("len(boundaries)==763","len(boundaries)==recipe['expected_projected_boundaries']")
(R/'verify_saved.py').write_text(v)
shutil.copy2(B/'recursive_native/tranche01_repack/cold_ops.py',R/'cold_ops.py')
