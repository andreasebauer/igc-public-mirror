from pathlib import Path
import sys,json,hashlib,ast,itertools,copy,shutil
B=Path(__file__).resolve().parents[1];R=Path(__file__).resolve().parent
sys.path[:0]=[str(B/'engine'),str(B/'unified139'),str(R)]
from ig_master import UnifiedMasterReader
from infinity_grid.canon import canonical_sha256,canonical_bytes
from infinity_grid.workflow_guard import preflight
from infinity_grid.v05_engineering_worker import engineering_source_tree_digest
from project.node_attachment import evaluate,project
sha=lambda b:hashlib.sha256(b).hexdigest()
def write(name,obj):(R/name).write_text(json.dumps(obj,indent=2)+'\n')
mapping=json.loads((B/'node_input_audit/PRIMITIVE_MAPPING.json').read_text());events=[]
with UnifiedMasterReader(B/'unified139',canonical_directory=B/'canonical14',archive_directory=B/'unified139/archives') as reader:
 for row in mapping['rows']:
  parent=reader.lookup_j3(row['canonical_exact_tid'])['carrier'];all_events={e['event_id']:e for e in parent['events']}
  for matched in row['matched_events']:
   e=all_events[matched['event_id']];assert e['record']==matched['ordered_record']
   projected,ports,targets=project(e['record'])
   historical=(projected[0],tuple(tuple(p) for p in projected[1]),projected[2],tuple(projected[3]),projected[4])
   assert sha(repr(historical).encode())==row['historical_boundary_sha256']
   events.append({'j3_id':parent['j3_id'],'event_id':e['event_id'],'record':e['record'],'realizations':e['realizations'],'realization_count':len(e['realizations']),'selector':{'exact_tid':row['canonical_exact_tid'],'historical_rank':row['historical_rank_label'],'historical_boundary_sha256':row['historical_boundary_sha256']},'projected_to_native_ports':ports})
assert len(events)==len({e['event_id'] for e in events})==13
catalog=(B/'unified139/CATALOG_0139.json').read_bytes();assert sha(catalog)=='4753cee3dd1adb0b963f10ae11d98fb0c35640b43f3706f1d90b836150458102'
packet={'schema_id':'IG_ADMITTED_J3_NODE_PRIMITIVE_PACKET_V1','accepted_catalog_sha256':sha(catalog),'parent_root_sha256':'e1e2327070b2f9fc3bb548fc58a6961b70a0b885eb857d040ce0dcd5bfe4a351','primitive_mapping_sha256':sha((B/'node_input_audit/PRIMITIVE_MAPPING.json').read_bytes()),'events':sorted(events,key=lambda e:e['event_id']),'extraction':'Read-only admitted event subset, including all realization witnesses. No J3 production.'}
write('PRIMITIVE_PACKET.json',packet)
# Load only the original pure constructor functions, excluding module imports/CLI.
source=(B/'node_input_audit/sources/run_level.py').read_bytes();tree=ast.parse(source.decode());keep={'dtup','canon','bridge','compose'}
pure=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in keep],type_ignores=[]);oracle={};exec(compile(pure,'pinned_original_constructor_pure_functions','exec'),oracle)
attempted=lawful=0;projected_set=set();object_ids=set();formation_ids=set();rows=[]
def original_record(e):
 r=e['record'];return (r[0],tuple(tuple(p) for p in r[1]),r[2],r[3],r[4])
for a,b in itertools.product(packet['events'],repeat=2):
 payload={'left':a,'right':b,'recipe_sha256':'DRY_QUALIFICATION'};actual=evaluate(payload)['states'];by_bridge={tuple(x['state']['identity']['bridge_slots']):x['state'] for x in actual}
 for sa,sb in itertools.product(range(3),repeat=2):
  attempted+=1;expected=oracle['compose'](original_record(a),sa,original_record(b),sb)
  if expected is None:assert (sa,sb) not in by_bridge;continue
  lawful+=1;row=by_bridge[(sa,sb)];assert canonical_bytes(expected)==canonical_bytes(row['projected_boundary'])
  assert row['microscopic_realization_product_count']==len(a['realizations'])*len(b['realizations'])
  assert row['ordered_record'][3]==[a['record'][3],b['record'][3]]
  assert len(row['external_port_origins'])==4 and sorted(row['projected_port_to_ordered_port'])==list(range(4))
  assert [row['ordered_record'][1][i] for i in row['projected_port_to_ordered_port']]==row['projected_boundary'][1]
  assert [row['ordered_record'][3][i] for i in row['projected_target_to_component']]==row['projected_boundary'][3]
  assert row['object_id'] not in object_ids and row['formation_id'] not in formation_ids
  object_ids.add(row['object_id']);formation_ids.add(row['formation_id']);projected_set.add(canonical_sha256(row['projected_boundary']));rows.append(row)
# Independent explicit illegal, asymmetric, and legal selective bridge cases.
def synthetic(port):return {'record':[0,[port,[0,0],[0,0]],1,0,0],'j3_id':'test','event_id':'test','selector':{},'realization_count':1}
for pa,pb,legal in [([1,2],[4,0],False),([1,0],[4,2],False),([1,4],[4,1],True),([0,0],[0,0],True)]:
 result=evaluate({'left':synthetic(pa),'right':synthetic(pb),'recipe_sha256':'TEST'})
 assert any(x['state']['identity']['bridge_slots']==[0,0] for x in result['states'])==legal
assert lawful>len(projected_set) and len(rows)==len(object_ids)==len(formation_ids)
recipe={'schema_id':'IG_NODE_ATTACHMENT_DEPTH1_QUALIFICATION_RECIPE_V1','scope':{'attachment_depth':1,'ordered_roles':['seed','attachment'],'primitive_selector_count':9,'native_event_count':13,'ports_per_input':3,'attempted_endpoint_pairs':1521,'selection':'EXHAUSTIVE_WITHIN_DECLARED_INPUT_ALPHABET'},'identity':'IG_NODE_ATTACHMENT_ROOTED_CONSTRUCTION_V1','identity_semantics':'Rooted directed recipe occurrence with typed component roles, native event IDs, historical selector labels and chosen native bridge slots. No isomorphism quotient, no identification by projected boundary. One formation per rooted construction; implicit microscopic realization products refer to full admitted event witness sets.','input_sha256':sha((R/'PRIMITIVE_PACKET.json').read_bytes()),'accepted_catalog_sha256':sha(catalog),'historical_constructor_sha256':sha(source),'qualification_expected_lawful_occurrences':lawful,'qualification_expected_projected_boundaries':len(projected_set),'stopping_rule':'All 169 ordered native event pairs and all 9 endpoint pairs each, exactly one attachment depth; stop on hash/semantic/reference/coverage mismatch. No frontier selection or mature-node completeness claim.','resources':{'max_tasks':169,'max_occurrences':1521,'max_task_occurrences':9,'max_task_result_bytes':65536,'max_shard_bytes':1048576,'max_shards':6},'admission':'Qualification candidate only; scientific admission requires separately verified native export and cold restore.'}
write('RECIPE_CONTRACT.json',recipe)
check=R/'preflight_source';shutil.copytree(B/'engine',check,ignore=shutil.ignore_patterns('__pycache__'));shutil.copytree(R/'project',check/'project')
pf=preflight(check,[check/'project/node_attachment.py']);write('PREFLIGHT.json',pf);shutil.rmtree(check);assert pf['status']=='PASS'
report={'status':'PASS','execution_kind':'DRY_ADAPTER_QUALIFICATION_NOT_SCIENTIFIC_ADMISSION','attempted_endpoint_pairs':attempted,'lawful_rooted_constructions':lawful,'formation_occurrences':len(formation_ids),'projected_boundaries':len(projected_set),'projection_collisions_retained':lawful-len(projected_set),'native_source_binding':'All 13 events read from admitted J3; historical projected record hashes match all nine selectors.','independent_oracle':'Exact pure functions recovered from pinned original run_level.py; all endpoint outcomes checked.','negative_bridge_cases':4,'per_occurrence_checks':'Native endpoint identity, ordered destinations, projection preimages, unique formation IDs and parent option product counts.','input_sha256':sha((R/'PRIMITIVE_PACKET.json').read_bytes()),'recipe_sha256':sha((R/'RECIPE_CONTRACT.json').read_bytes()),'engine_digest':engineering_source_tree_digest(B/'engine'),'adapter_sha256':sha((R/'project/node_attachment.py').read_bytes()),'runtime_gate':'Native production must use registered common runtime and sealed capture. Dry rows are not exported or admitted.'}
write('QUALIFICATION.json',report);print(json.dumps(report))
old=json.loads((B/'campaign/MASTER_WORK/0139/permutation_p210_t024/FIRST_TRANCHE_SPEC.json').read_text())
old.update(job_id='MASTER.QUALIFICATION.NODE.ATTACHMENT.DEPTH1.V1',project_source=str(R/'project'))
old['question']={'stage_id':'MASTER:QUALIFICATION:NODE:ATTACHMENT:DEPTH1:V1','description':'Formation-preserving single primitive attachment over admitted nine-selector/13-event alphabet.','outcomes':['PASS'],'stopping_rule':recipe['stopping_rule']}
old['execution']={'kind':'STAGE','handler_ref':'project.node_attachment:handler','evaluator_refs':['project.node_attachment:evaluate'],'parameters':{'input_sha256':report['input_sha256'],'recipe_sha256':report['recipe_sha256'],'accepted_catalog_sha256':sha(catalog)}}
old['inputs']=[{'logical_name':name,'path':str(R/file),'sha256':sha((R/file).read_bytes())} for name,file in [('primitive_packet','PRIMITIVE_PACKET.json'),('recipe_contract','RECIPE_CONTRACT.json')]]
old['output_contract']['result_checks']=[{'pointer':'/'+key,'equals':v} for key,v in {'outcome':'PASS','rooted_constructions':lawful,'formation_occurrences':lawful,'projected_boundaries':len(projected_set),'official_master_admission':False,'existing_j3_regenerated':0,'historical_mature_node_census_reproduced':False}.items()]
write('CAPTURE_SPEC.json',old)
