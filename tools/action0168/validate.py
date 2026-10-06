from pathlib import Path
import json,hashlib,sys,ast,collections,types,struct
R=Path(__file__).resolve().parent.parent;B=R/'o7_readiness0168';B.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.loads(p.read_bytes())
def dump(n,d):(B/n).write_text(json.dumps(d,indent=2)+'\n')
def js(x):return json.loads(json.dumps(x))
def shaj(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
engine=next((R/'o7_readiness0167/sources').glob('*v0.2.4_C04*/*/02_CODE/o7_live_engine.py'))
selected=['nested_h6','iter_sites','site_at','materialize_owner','owner_free','eparts','accounting']
tree=ast.parse(engine.read_bytes());fn={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
module=ast.Module(body=[fn[n] for n in selected],type_ignores=[]);ns={'collections':collections,'PN':7};exec(compile(module,str(engine),'exec'),ns)
# No full engine import, orbit enumerator, generator or original main is executed.
frozen='import collections\nPN=7\n\n'+ast.unparse(module)+'\n';(B/'frozen_o7.py').write_text(frozen)
dump('PURE_EXTRACTION.json',{'original_sha256':sha(engine.read_bytes()),'pure_sha256':sha(frozen.encode()),'functions':selected,'functions_ast_sha256':{n:sha(ast.dump(fn[n],include_attributes=False).encode()) for n in selected}})
sys.path.insert(0,str(R/'o6_export0165'));from project.reader import CarrierReader
from project import frozen_o6 as m
pins=load(R/'o6_export0165/EXPORT_BINDINGS.json');reader=CarrierReader(R/'o6_export0165/SCIENTIFIC_EXPORT.zip',pins['archive_sha256'],pins['root_sha256'],R/'o5_export0162/SCIENTIFIC_EXPORT.zip',R/'o4_export0159/SCIENTIFIC_EXPORT.zip',R/'o3_export0156/SCIENTIFIC_EXPORT.zip')
pr=next((R/'o_readiness0158/sources').rglob('OSCOUT_O7_SELECTED_O6_PROTOTYPES.json')).parent
sources={'selected':pr/'OSCOUT_O7_SELECTED_O6_PROTOTYPES.json','twins':pr/'OSCOUT_O7_O6_TOPOLOGY_TWINS.json','pool':pr/'OSCOUT_O7_O6_UNIQUE_R6_FEATURE_POOL.json','selector':pr/'OSCOUT_O7_PROTOTYPE_SELECTION_v0.1.json','spec':pr.parent/'02_SPEC/OSCOUT_O7_LIVE_PREREGISTRATION_v0.1.json'}
for name,p in sources.items(): (B/(name.upper()+'.json')).write_bytes(p.read_bytes())
dump('INPUT_BINDINGS.json',{n:{'source':str(p.relative_to(R)),'sha256':sha(p.read_bytes())} for n,p in sources.items()})
alg=json.loads(reader.raw['algebra']);ports=[tuple(x) for x in alg['port_atoms']];ix={p:i for i,p in enumerate(ports)};pairs={(ix[tuple(a)],ix[tuple(b)]) for a,b in alg['ordered_compatible_port_pairs']}
selectedrows=load(sources['selected'])['prototypes'];bindings=[];parents={}
for p in selectedrows:
 loc=p['source_locator'];keys=[k for k in reader.records if k[0]==loc['source_lane'] and k[2]==loc['phase1_state_digest']];assert len(keys)==1;k=keys[0]
 assert reader.states[loc['phase1_selected_state_index']]==reader.records[k]
 owners=loc['source_component_owner_indices'];comps=[c for c in reader.components(*k) if c['source_owners']==owners];assert len(comps)==1;c=comps[0]
 resources=reader.resources(*k);h=tuple(resources[i] for i in owners)
 assert js(h)==p['ordered_resource_O5_carriers'] and js(c['edges'])==p['exact_E6_edges'] and c['proto_ids']==p['exact_O5_parent_ids']
 ca=m.component_invariants(c['proto_ids'],reader.P,tuple(tuple(e) for e in c['edges']),tuple(range(len(owners))))
 assert js(ca)==p['accounting'], (p['prototype_id'],ca,p['accounting'])
 assert sha(repr(tuple(sorted(m.canon_o5(x) for x in h))).encode())==p['canonical_R6_sha256']
 assert shaj({'proto_ids':p['exact_O5_parent_ids'],'E6_edges':p['exact_E6_edges']})==p['exact_representative_sha256']
 parents[p['prototype_id']]=types.SimpleNamespace(pid=p['prototype_id'],h6=h,accounting=p['accounting'])
 bindings.append({'prototype_id':p['prototype_id'],'source_key':list(k),'source_owners':owners,'root_payload_sha256':shaj(reader.records[k]),'ordered_resource_sha256':shaj(h),'canonical_R6_sha256':p['canonical_R6_sha256'],'exact_representative_sha256':p['exact_representative_sha256'],'O5_bindings':[reader.owner(*k,i) for i in owners],'role':'admitted_O6_component'})
# Literal twin witnesses are external adversarial roots, not admitted selected O6 states.
twinchecks=[]
for t in load(sources['twins'])['twins']:
 ids=t['exact_O5_parent_ids'];edges=tuple(tuple(e) for e in t['exact_E6_edges']);assert not m.validate_exact_state(ids,reader.P,edges,pairs)
 h=m.materialize(ids,reader.P,edges);assert js(h)==t['ordered_resource_O5_carriers']
 assert js(m.component_invariants(ids,reader.P,edges,tuple(range(len(ids)))))==t['accounting']
 assert sha(repr(tuple(sorted(m.canon_o5(x) for x in h))).encode())==t['canonical_R6_sha256']
 assert shaj({'proto_ids':ids,'E6_edges':js(edges)})==t['exact_representative_sha256']
 twinchecks.append({'id':t['twin_id'],'role':'source_bound_external_adversarial_O6_root','resource_sha256':shaj(h),'R6':t['canonical_R6_sha256'],'exact_sha256':t['exact_representative_sha256'],'O5_runtime_ids':ids})
assert len(twinchecks)==2 and twinchecks[0]['R6']==twinchecks[1]['R6'] and twinchecks[0]['exact_sha256']!=twinchecks[1]['exact_sha256']
lanes=load(sources['spec'])['frozen']['homogeneous_heterogeneous_lane_sizes']['lanes'];rows=load(R/'o7_readiness0167/SAVED_STATE_ROWS.json');components=[];resourcehashes={};ownerscount=0
for st in rows:
 lane=st['lane'];cfg=lanes[lane];ids=cfg.get('prototype_ids') or [cfg['prototype_id']]*cfg['owner_count'];ctx=types.SimpleNamespace(n=len(ids),parents=tuple(parents[x] for x in ids));edges=tuple(tuple(e) for e in st['record']['edges']);usage=collections.Counter()
 assert len(edges)==st['rank']<=cfg['max_m7'] and edges==tuple(sorted(edges))
 assert sha(b'IG-E7-STATE-v1|'+b''.join(struct.pack('>14I',*e) for e in edges))==st['digest']
 assert shaj(st['record'])==st['payload_sha256']
 for e in edges:
  assert len(e)==14 and all(type(x)==int and 0<=x<2**32 for x in e)
  c,p,a,d,q,b=ns['eparts'](e);assert 0<=c<d<ctx.n and (a,b) in pairs
  for owner,path,port in [(c,p,a),(d,q,b)]:
   point,free=ns['site_at'](ctx.parents[owner].h6,path);assert 0<=port<7;usage[(owner,path,port)]+=1;assert usage[(owner,path,port)]<=free[port]
 inv=ns['accounting'](ctx,edges);assert inv==st['record']['accounting'] and inv['ok']
 statekey=f"{lane}:{st['rank']}:{st['digest']}";resourcehashes[statekey]=[shaj(ns['materialize_owner'](parent.h6,edges,i)) for i,parent in enumerate(ctx.parents)];ownerscount+=ctx.n
 # E7 incidence components: saved construction materialization only, no canonicalization search.
 adj=[set() for _ in ids]
 for e in edges:adj[e[0]].add(e[7]);adj[e[7]].add(e[0])
 seen=set()
 for start in range(len(ids)):
  if start in seen:continue
  todo=[start];comp=[];seen.add(start)
  while todo:
   x=todo.pop();comp.append(x)
   for y in adj[x]:
    if y not in seen:seen.add(y);todo.append(y)
  comp=sorted(comp)
  if len(comp)<2:continue
  mapping={x:i for i,x in enumerate(comp)};ce=[]
  for e in edges:
   if e[0] in mapping and e[7] in mapping:ce.append((mapping[e[0]],*e[1:7],mapping[e[7]],*e[8:]))
  cc=types.SimpleNamespace(n=len(comp),parents=tuple(ctx.parents[i] for i in comp));ca=ns['accounting'](cc,ce);assert ca['ok']
  components.append({'state_key':statekey,'source_owners':comp,'prototype_ids':[ids[i] for i in comp],'edges':js(ce),'accounting':ca})
dump('O6_BINDINGS.json',bindings);dump('EXTERNAL_TWIN_BINDINGS.json',twinchecks);dump('COMPONENTS.json',components);dump('RESOURCE_PAYLOAD_SHA256.json',resourcehashes)
out={'status':'SAVED_O7_RECOVERED_SCOPE_SOURCE_BOUND_READY','master_release':'MASTER_DATA_V1_0147','catalog_sha256':sha((R/'o6_integrate0166/CATALOG_0147.json').read_bytes()),'counts':{'states':len(rows),'nonseed_states':sum(s['rank']>0 for s in rows),'external_seed_states':sum(s['rank']==0 for s in rows),'typed_E7_edges':sum(s['rank'] for s in rows),'O6_owner_occurrences':ownerscount,'nontrivial_components':len(components),'selected_O6_prototype_bindings':len(bindings),'external_adversarial_twin_roots_verified':2},'lane_ranks':{'HET4':[0,1,2,3,4,5],'HOM6':[0,1]},'checks':['eight selected O6 prototypes resolve to admitted roots, exact components, ordered resources and accounting','two literal O6 twins validated as external adversarial roots only','all recovered E7 coordinates, compatibility, capacity, accounting and component materializations pass'],'remaining_debts':['full final survivor/profile coverage missing','selector feature pool and preregistration selection need export preservation','historical closeout D1-D6 not certified','complete parent/action occurrence ancestry unavailable','inherited76 microscopic O2 panels missing'],'generation_calls':0,'new_admissions':0,'o7_graduated':False,'automatic_o8_authorized':False,'full_l0_to_g8_complete':False,'next_scope':'WP5_SAVED_O7_SCOPED_TYPED_EXPORT_AND_READER'}
dump('READINESS_VALIDATION.json',out);reader.close();print(json.dumps(out,indent=2))
