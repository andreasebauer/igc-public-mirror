from pathlib import Path
import json,hashlib,zipfile,copy,collections,ast,types,struct
from .previous_o6.reader import CarrierReader as O6Reader,ast_bytes
from .previous_o6 import frozen_o6 as o6
from . import frozen_o7 as m
sha=lambda b:hashlib.sha256(b).hexdigest()
shaj=lambda x:sha(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
js=lambda x:json.loads(json.dumps(x))
class CarrierReader:
 def __init__(self,path,archive_sha,root_sha,o6_archive,o5_archive,o4_archive,o3_archive):
  self.closed=False;self.z=None;self.o6=None
  try:
   if sha(Path(path).read_bytes())!=archive_sha:raise ValueError('ARCHIVE_HASH')
   self.z=zipfile.ZipFile(path);raw=self.z.read('ROOT.json')
   if sha(raw)!=root_sha:raise ValueError('ROOT_HASH')
   self.root=json.loads(raw);self.raw={};seen={'ROOT.json'}
   for item in self.root['content']:
    n='content/'+item['sha256']+'.blob';b=self.z.read(n)
    if sha(b)!=item['sha256'] or len(b)!=item['bytes'] or item['name'] in self.raw:raise ValueError('CONTENT_HASH')
    self.raw[item['name']]=b;seen.add(n)
   if set(self.z.namelist())!=seen or len(self.z.namelist())!=len(seen):raise ValueError('CONTENT_CLOSURE')
   expected={'states','occurrences','selected','twins','pool','selector','spec','readiness','contract','bindings','external_twins','components','resources','original_producer','pure_extraction','source_refs'}
   if set(self.raw)!=expected:raise ValueError('CONTENT_NAMES')
   ex=json.loads(self.raw['pure_extraction']);pure=Path(m.__file__).read_bytes()
   if sha(pure)!=ex['pure_sha256'] or sha(self.raw['original_producer'])!=ex['original_sha256']:raise ValueError('PURE_HASH')
   original={n.name:n for n in ast.parse(self.raw['original_producer']).body if isinstance(n,ast.FunctionDef)};frozen={n.name:n for n in ast.parse(pure).body if isinstance(n,ast.FunctionDef)}
   if set(frozen)!=set(ex['functions_ast_sha256']):raise ValueError('PURE_CLOSURE')
   for name,h in ex['functions_ast_sha256'].items():
    if sha(ast_bytes(original[name]))!=h or sha(ast_bytes(frozen[name]))!=h:raise ValueError('PURE_AST_IDENTITY')
   pins=self.root['O6_bindings'];self.o6=O6Reader(o6_archive,pins['archive_sha256'],pins['root_sha256'],o5_archive,o4_archive,o3_archive)
   self.bindings={p['prototype_id']:p for p in json.loads(self.raw['bindings'])};self.P={};self.base_components={};selected=json.loads(self.raw['selected'])['prototypes']
   if {p['prototype_id'] for p in selected}!=set(self.bindings) or len(selected)!=8:raise ValueError('SELECTED_CLOSURE')
   for p in selected:
    pid=p['prototype_id'];b=self.bindings[pid];k=tuple(b['source_key']);row=self.o6.records[k];owners=b['source_owners'];loc=p['source_locator']
    if self.o6.states[loc['phase1_selected_state_index']]!=row or k[0]!=loc['source_lane'] or k[2]!=loc['phase1_state_digest'] or owners!=loc['source_component_owner_indices'] or shaj(row)!=b['root_payload_sha256']:raise ValueError('SOURCE_LOCATOR')
    matches=[c for c in self.o6.components(*k) if c['source_owners']==owners]
    if len(matches)!=1:raise ValueError('COMPONENT_BINDING')
    c=matches[0];h=tuple(self.o6.resources(*k)[i] for i in owners)
    if js(h)!=p['ordered_resource_O5_carriers'] or shaj(h)!=b['ordered_resource_sha256'] or c['proto_ids']!=p['exact_O5_parent_ids'] or js(c['edges'])!=p['exact_E6_edges'] or js([self.o6.owner(*k,i) for i in owners])!=b['O5_bindings']:raise ValueError('NESTED_BINDING')
    if js(o6.component_invariants(c['proto_ids'],self.o6.P,tuple(tuple(e) for e in c['edges']),tuple(range(len(owners)))))!=p['accounting']:raise ValueError('O6_ACCOUNTING')
    if sha(repr(tuple(sorted(o6.canon_o5(x) for x in h))).encode())!=p['canonical_R6_sha256'] or shaj({'proto_ids':p['exact_O5_parent_ids'],'E6_edges':p['exact_E6_edges']})!=p['exact_representative_sha256']:raise ValueError('O6_IDENTITY')
    self.P[pid]=types.SimpleNamespace(pid=pid,h6=h,accounting=p['accounting']);self.base_components[pid]=c
   alg=json.loads(self.o6.raw['algebra']);ports=[tuple(x) for x in alg['port_atoms']];ix={p:i for i,p in enumerate(ports)};pairs={(ix[tuple(a)],ix[tuple(b)]) for a,b in alg['ordered_compatible_port_pairs']}
   # Preserve and verify saved external twin witnesses; never admit them as selected roots.
   twins=json.loads(self.raw['twins'])['twins'];tb=json.loads(self.raw['external_twins'])
   if len(twins)!=2 or {t['twin_id'] for t in twins}!={t['id'] for t in tb}:raise ValueError('TWIN_SCOPE')
   for t in twins:
    ids=t['exact_O5_parent_ids'];edges=tuple(tuple(e) for e in t['exact_E6_edges'])
    if o6.validate_exact_state(ids,self.o6.P,edges,pairs):raise ValueError('TWIN_CAPACITY')
    h=o6.materialize(ids,self.o6.P,edges)
    if js(h)!=t['ordered_resource_O5_carriers'] or js(o6.component_invariants(ids,self.o6.P,edges,tuple(range(len(ids)))))!=t['accounting'] or shaj({'proto_ids':ids,'E6_edges':js(edges)})!=t['exact_representative_sha256'] or sha(repr(tuple(sorted(o6.canon_o5(x) for x in h))).encode())!=t['canonical_R6_sha256']:raise ValueError('TWIN_BINDING')
   self.lanes=json.loads(self.raw['spec'])['frozen']['homogeneous_heterogeneous_lane_sizes']['lanes'];self.records={};self.contexts={};self.component_records={};resourcehashes=json.loads(self.raw['resources']);allcomps=[];owners=edgescount=0
   for st in json.loads(self.raw['states']):
    lane=st['lane'];rank=st['rank'];key=(lane,rank,st['digest']);row=st['record'];edges=tuple(tuple(e) for e in row['edges'])
    if lane not in ('HET4','HOM6') or key in self.records:raise ValueError('STATE_SCOPE')
    cfg=self.lanes[lane];ids=cfg.get('prototype_ids') or [cfg['prototype_id']]*cfg['owner_count'];ctx=types.SimpleNamespace(n=len(ids),parents=tuple(self.P[x] for x in ids));usage=collections.Counter()
    if len(edges)!=rank or edges!=tuple(sorted(edges)) or rank>cfg['max_m7'] or shaj(row)!=st['payload_sha256'] or sha(b'IG-E7-STATE-v1|'+b''.join(struct.pack('>14I',*e) for e in edges))!=st['digest']:raise ValueError('STATE_IDENTITY')
    for e in edges:
     if len(e)!=14 or any(type(x)!=int or x<0 or x>=2**32 for x in e):raise ValueError('EDGE_TYPE')
     c,p,a,d,q,b=m.eparts(e)
     if not 0<=c<d<ctx.n or (a,b) not in pairs:raise ValueError('ENDPOINT')
     for owner,path,port in [(c,p,a),(d,q,b)]:
      point,free=m.site_at(ctx.parents[owner].h6,path);usage[(owner,path,port)]+=1
      if usage[(owner,path,port)]>free[port]:raise ValueError('CAPACITY')
    if m.accounting(ctx,edges)!=row['accounting'] or not row['accounting']['ok']:raise ValueError('STATE_ACCOUNTING')
    label=f'{lane}:{rank}:{key[2]}'
    if [shaj(m.materialize_owner(p.h6,edges,i)) for i,p in enumerate(ctx.parents)]!=resourcehashes[label]:raise ValueError('RESOURCE_HASH')
    self.records[key]=row;self.contexts[key]=ctx;self.component_records[key]=[];adj=[set() for _ in ids]
    for e in edges:adj[e[0]].add(e[7]);adj[e[7]].add(e[0])
    seenowners=set()
    for start in range(ctx.n):
     if start in seenowners:continue
     todo=[start];comp=[];seenowners.add(start)
     while todo:
      x=todo.pop();comp.append(x)
      for y in adj[x]:
       if y not in seenowners:seenowners.add(y);todo.append(y)
     comp=sorted(comp)
     if len(comp)<2:continue
     mp={x:i for i,x in enumerate(comp)};ce=[(mp[e[0]],*e[1:7],mp[e[7]],*e[8:]) for e in edges if e[0] in mp and e[7] in mp];cc=types.SimpleNamespace(n=len(comp),parents=tuple(ctx.parents[i] for i in comp));ca=m.accounting(cc,ce)
     if not ca['ok']:raise ValueError('COMPONENT_ACCOUNTING')
     record={'state_key':label,'source_owners':comp,'prototype_ids':[ids[i] for i in comp],'edges':js(ce),'accounting':ca};self.component_records[key].append(record);allcomps.append(record)
    owners+=ctx.n;edgescount+=len(edges)
   if allcomps!=json.loads(self.raw['components']):raise ValueError('COMPONENT_CLOSURE')
   readiness=json.loads(self.raw['readiness']);counts=readiness['counts']
   if len(self.records)!=74 or edgescount!=192 or owners!=322 or len(allcomps)!=82 or counts!={'states':74,'nonseed_states':72,'external_seed_states':2,'typed_E7_edges':192,'O6_owner_occurrences':322,'nontrivial_components':82,'selected_O6_prototype_bindings':8,'external_adversarial_twin_roots_verified':2}:raise ValueError('SCOPE_COUNTS')
   self.report={'carriers':72,'seed_carriers':2,'typed_edges':192,'O6_owner_occurrences':322,'components':82,'selected_O6_bindings':8,'external_twin_roots':2,'generation_calls':0,'complete_derivation_lineage':False,'full_l0_to_g8_complete':False}
  except Exception:self.close();raise
 def _key(self,lane,rank,digest):
  if self.closed:raise ValueError('CLOSED_READER')
  key=(lane,rank,digest)
  if key not in self.records:raise KeyError(key)
  return key
 def lookup(self,lane,rank,digest):
  key=self._key(lane,rank,digest);return {'record':copy.deepcopy(self.records[key]),'external_seed':rank==0,'identity_scope':'source-bound lane/rank/binary E7 digest plus literal row hash; not global canonicality'}
 def owner(self,lane,rank,digest,index):
  key=self._key(lane,rank,digest);ctx=self.contexts[key]
  if type(index)!=int or not 0<=index<ctx.n:raise ValueError('OWNER_INDEX')
  pid=ctx.parents[index].pid;b=self.bindings[pid];k=tuple(b['source_key'])
  return copy.deepcopy({'owner_index':index,'prototype_id':pid,'binding':b,'O6_root':self.o6.lookup(*k),'O6_component':self.base_components[pid],'resource_O6':js(ctx.parents[index].h6),'nested_O5_owners':[self.o6.owner(*k,i) for i in b['source_owners']]})
 def resources(self,lane,rank,digest):
  key=self._key(lane,rank,digest);ctx=self.contexts[key];edges=self.records[key]['edges'];return tuple(m.materialize_owner(p.h6,edges,i) for i,p in enumerate(ctx.parents))
 def components(self,lane,rank,digest):return copy.deepcopy(self.component_records[self._key(lane,rank,digest)])
 def parents(self,lane,rank,digest):self._key(lane,rank,digest);raise ValueError('Complete parent/action occurrence ancestry unavailable')
 def close(self):
  self.closed=True
  if self.o6:self.o6.close()
  if self.z:self.z.close()
