from pathlib import Path
import json, hashlib, runpy, collections
R=Path.cwd(); B=R/'o7_additional_readiness0172'; B.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_bytes())
shaj=lambda d:sha(json.dumps(d,sort_keys=True,separators=(',',':')).encode())
def dump(n,d):(B/n).write_text(json.dumps(d,indent=2)+'\n')
# Validate saved literals using the existing pure reader, never the producer main.
v=runpy.run_path(str(R/'scope_reconcile0171/validate_recovery.py'))
roots=load(R/'scope_reconcile0171/NEW_SAVED_ROOT_STATES.json')
pb=load(R/'scope_reconcile0171/RECOVERED_PROFILE_BINDINGS.json')
profiles=load(R/pb['profiles']['recovered_path'])['records']
resources=load(R/'scope_reconcile0171/SURVIVOR_RESOURCE_HASHES.json')
selected=load(R/'o7_readiness0168/SELECTED.json')['prototypes']
twins=load(R/'o7_readiness0168/TWINS.json')['twins']
assert sha((R/'o7_integrate0170/CATALOG_0148.json').read_bytes())=='21e55dac42946a8949fb0cb3a5ce2bd3a5bf0c9624c734c539d754a89b3a0712'
binding_rows=load(R/'o7_readiness0168/O6_BINDINGS.json')
bindings={x['prototype_id']:{'role':'admitted_O6_component_binding','binding':x} for x in binding_rows}
for t in twins:
 bindings[t['twin_id']]={'role':'source_bound_external_adversarial_O6_root','literal_root':t,'admitted_O6_root':False,'O5_bindings':load(R/'o7_readiness0168/EXTERNAL_TWIN_BINDINGS.json')[0 if t['twin_id'].endswith('A') else 1]}
root_index=[];known=set(v['r'].records)
for row in roots:
 key=(row['lane'],row['rank'],row['digest']); assert key not in known;known.add(key)
 cfg=v['r'].lanes[row['lane']];ids=cfg.get('prototype_ids') or [cfg['prototype_id']]*cfg['owner_count']
 root_index.append({'root_key':list(key),'literal_payload_sha256':row['payload_sha256'],'parent_ids':ids,'parent_exact_colors':[v['colors'][x] for x in ids],'ordered_context_sha256':shaj({'ids':ids,'colors':[v['colors'][x] for x in ids]}),'source_checkpoint_sha256':row['checkpoint_sha256'],'source_checkpoint_chain':row['checkpoint_chain'],'record':row['record']})
objects={};occurrences=[];occ_keys=set();source_roots=set();owners=collections.Counter()
for i,row in enumerate(profiles):
 ids=row['parent_ids']; colors=row['parent_exact_colors'];edges=row['edges']
 # A saved component must be connected on all its declared owners.
 adjacent={j:set() for j in range(len(ids))}
 for e in edges:adjacent[e[0]].add(e[7]);adjacent[e[7]].add(e[0])
 seen={0};todo=[0]
 while todo:
  for j in adjacent[todo.pop()]-seen:seen.add(j);todo.append(j)
 assert len(ids)>=2 and len(seen)==len(ids)
 # An E7 digest alone cannot identify a carrier; owner identities are ordered.
 identity={'schema':'IG_O7_EXACT_COMPONENT_IDENTITY_V1','parent_ids':ids,'parent_exact_colors':colors,'E7_digest':row['state_digest']}
 object_id=shaj(identity); payload={'identity':identity,'edges':edges,'accounting':row['accounting'],'R7_skin_sha256':row['R7_skin_sha256'],'resource_sha256':resources[str(i)]['resource_sha256']}
 if object_id in objects:assert objects[object_id]['payload']==payload
 else:objects[object_id]={'object_id':object_id,'payload_sha256':shaj(payload),'payload':payload}
 key=(row['lane'],row['m7'],row['source_state_digest'],row['component_index']);assert key not in occ_keys;occ_keys.add(key)
 rootkey=key[:3];source_roots.add(rootkey);owners.update(ids)
 occurrences.append({'record_index':i,'occurrence_key':list(key),'object_id':object_id,'literal_payload_sha256':shaj(row),'source_root_available':rootkey in known,'source_root_role':'available_saved_whole_root_reference' if rootkey in known else 'external_unavailable_whole_root_reference','source_owner_index_mapping_available':False,'root_rank':row['m7'],'component_E7_edge_count':len(edges),'saved_profile_summary_role':'source_evidence_only_full_Counter_unavailable','record':row})
assert len(root_index)==24 and len(occurrences)==205 and len(objects)==164
used=set(owners)|{x for row in root_index for x in row['parent_ids']}; assert used<=set(bindings)
counts={'additional_whole_roots':24,'additional_root_E7_edges':sum(len(x['record']['edges']) for x in root_index),'additional_root_O6_owner_occurrences':sum(len(x['parent_ids']) for x in root_index),'component_occurrences':len(occurrences),'distinct_exact_component_objects':len(objects),'component_E7_edge_occurrences':sum(x['component_E7_edge_count'] for x in occurrences),'component_O6_owner_occurrences':sum(owners.values()),'distinct_source_root_references':len(source_roots),'available_source_root_references':sum(x in known for x in source_roots),'unavailable_source_root_references':sum(x not in known for x in source_roots),'used_admitted_O6_prototype_bindings':sum(bindings[x]['role']=='admitted_O6_component_binding' for x in used),'used_external_O6_twin_bindings':sum(bindings[x]['role']=='source_bound_external_adversarial_O6_root' for x in used)}
dump('ROOT_SCOPE.json',root_index);dump('COMPONENT_OBJECTS.json',list(objects.values()));dump('COMPONENT_OCCURRENCES.json',occurrences);dump('OWNER_BINDINGS.json',{x:bindings[x] for x in sorted(used)})
contract={'schema':'IG_ADDITIONAL_SAVED_O7_EXPORT_CONTRACT_V1','master_release':'MASTER_DATA_V1_0148','master_catalog_sha256':sha((R/'o7_integrate0170/CATALOG_0148.json').read_bytes()),'counts':counts,'root_identity':'lane/root rank/typed E7 digest plus ordered exact O6 parent context and literal payload hash','component_identity':'SHA256 canonical JSON of schema, ordered parent IDs, ordered exact colors, E7 digest; full physical payload hash checked separately','occurrence_identity':'lane/source whole-root rank/source whole-root digest/component index; retain all205 source records','profile_authority':'saved summaries only; full Counter entries and fresh certification unavailable','requirements':['lossless literal root and component records with source-chain hashes','exact ordered O6 bindings, external twin topology and admitted O5 bindings','all typed E7 coordinates, multiplicity, capacity, accounting and exact R7 resource hashes','205 occurrence keys retained independently of164 physical identities','explicit unavailable source root references and absent owner-index mappings','content-addressed scientific export, standalone reader, native capture/checkpoint/cold replay before admission'],'exclusions':['reconstruction of absent whole forests','complete action/parent ancestry','global canonical isomorphism','historical graduation or automatic O8','new scientific admission'],'generation_calls':0,'new_admissions':0}
dump('EXPORT_CONTRACT.json',contract)
inputs=[R/'scope_reconcile0171/NEW_SAVED_ROOT_STATES.json',R/pb['profiles']['recovered_path'],R/pb['survivors']['recovered_path'],R/'scope_reconcile0171/SOURCE_SEARCH.json',R/'scope_reconcile0171/SURVIVOR_RESOURCE_HASHES.json',R/'o7_export0169/EXPORT_BINDINGS.json',R/'o7_integrate0170/CATALOG_0148.json']
for n in ['SELECTED.json','TWINS.json','O6_BINDINGS.json','EXTERNAL_TWIN_BINDINGS.json']:inputs.append(R/'o7_readiness0168'/n)
dump('INPUT_PINS.json',{str(p.relative_to(R)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in inputs})
outputs=['ROOT_SCOPE.json','COMPONENT_OBJECTS.json','COMPONENT_OCCURRENCES.json','OWNER_BINDINGS.json','EXPORT_CONTRACT.json','INPUT_PINS.json']
result={'status':'FINITE_ADDITIONAL_O7_EXPORT_SCOPE_READY','counts':counts,'component_lane_counts':dict(collections.Counter(x['record']['lane'] for x in occurrences)),'component_parent_occurrences':dict(owners),'output_pins':{n:sha((B/n).read_bytes()) for n in outputs},'checks':['prior saved-byte/science-pin validation rerun','all literal typed-edge capacities/accounting and resource hashes revalidated','distinct occurrence keys and exact contextual physical payloads verified','root ranks kept separate from component edge counts','all used owner bindings classified and retained'],'generation_calls':0,'new_admissions':0,'next_scope':'WP5_ADDITIONAL_SAVED_O7_ROOT_AND_COMPONENT_NATIVE_EXPORT'}
dump('READINESS_VALIDATION.json',result);print(json.dumps(result,indent=2))
