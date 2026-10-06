from pathlib import Path
import json,hashlib,zipfile,copy,collections,io,gzip,csv
from . import frozen_o4 as m
from .previous_o3.reader import CarrierReader as O3Reader
sha=lambda b:hashlib.sha256(b).hexdigest()
class CarrierReader:
 def __init__(self,path,archive_sha,root_sha,o3_archive):
  self.closed=False;self.o3=None;self.z=None
  try:
   if sha(Path(path).read_bytes())!=archive_sha:raise ValueError('ARCHIVE_HASH')
   self.z=zipfile.ZipFile(path);raw=self.z.read('ROOT.json')
   if sha(raw)!=root_sha:raise ValueError('ROOT_HASH')
   self.root=json.loads(raw);self.raw={};seen={'ROOT.json'}
   if sha(Path(m.__file__).read_bytes())!=self.root['frozen_producer_sha256']:raise ValueError('PRODUCER_HASH')
   for item in self.root['content']:
    n='content/'+item['sha256']+'.blob';b=self.z.read(n)
    if sha(b)!=item['sha256'] or len(b)!=item['bytes']:raise ValueError('CONTENT_HASH')
    if item['name'] in self.raw:raise ValueError('DUPLICATE_CONTENT')
    self.raw[item['name']]=b;seen.add(n)
   if set(self.z.namelist())!=seen or len(self.z.namelist())!=len(seen):raise ValueError('CONTENT_CLOSURE')
   if set(self.raw)!={'states','prototypes','pin','spec','algebra','contract','readiness','payloads','source_manifest','selector','features','templates'}:raise ValueError('SCOPE')
   p=self.root['O3_bindings'];self.o3=O3Reader(o3_archive,p['archive_sha256'],p['root_sha256']);self.states=json.loads(self.raw['states'])['states'];self.prototypes=json.loads(self.raw['prototypes'])['selected'];self.pin=json.loads(self.raw['pin']);self.spec=json.loads(self.raw['spec']);self.readiness=json.loads(self.raw['readiness']);self.bindings={x['prototype_id']:x for x in self.readiness['O3_parent_bindings']};self.P={x['prototype_id']:x for x in self.prototypes}
   pin=self.pin;ordered=pin['ordered_resource_blocks'];N=sum(len(b) for b in ordered);d=sum(sum(s['p'])-2 for b in ordered for s in b);m3=len(pin['internal_o3_edges']);rho=sum(sum(s['p'][a]-s['f'][a] for a in range(7)) for b in ordered for s in b)//2;self.P['PHASE0_PINNED_ACTUAL_O3_COMPONENT']={'prototype_id':'PHASE0_PINNED_ACTUAL_O3_COMPONENT','ordered_resource_blocks':ordered,'N':N,'d':d,'m3':m3,'U2':rho-m3,'rho3':rho,'beta_flat3':rho-N+1,'beta3':m3-len(ordered)+1,'n3':len(ordered),'r3_rank':pin['source_r3_rank']}
   if set(self.P)!=set(self.bindings) or len(self.P)!=7:raise ValueError('PROTOTYPE_CLOSURE')
   self.base_components={}
   for pid,proto in self.P.items():
    b=self.bindings[pid];rank=b['source_rank'];h=b['source_exact_key'];comp=b['source_component_entities'];row=self.o3.lookup(rank,h)['record'];a=self.o3.audit;blocks=a.expanded(row);U=a.usage3(row,blocks);ordered=[[[list(p),[p[k]-u2[k]-U[v][i][k] for k in range(7)]] for i,(p,u2) in enumerate(blocks[v])] for v in comp];stored=[[[s['p'],s['f']] for s in block] for block in proto['ordered_resource_blocks']]
    if stored!=ordered or sha(self.o3.content_bytes('panel'+str(rank)))!=b['source_panel_sha256']:raise ValueError('O3_COMPONENT_COORDINATES')
    payload=sha(json.dumps({'entities':row['entities'],'edges':row['edges']},sort_keys=True,separators=(',',':')).encode())
    if payload!=b['root_payload_sha256']:raise ValueError('O3_ROOT_PAYLOAD')
    matches=[c for c in self.o3.components(rank,h) if c['source_entities']==comp]
    if len(matches)!=1 or matches[0]['diagnostic']['R3_sha256']!=b['R3_sha256']:raise ValueError('O3_COMPONENT_BINDING')
    self.base_components[pid]=matches[0]
   alg=json.loads(self.raw['algebra']);ports=[tuple(x) for x in alg['port_atoms']];ix={p:i for i,p in enumerate(ports)};pairs={(ix[tuple(x)],ix[tuple(y)]) for x,y in alg['ordered_compatible_port_pairs']};self.records={};self.component_records={};counts=collections.Counter();payloads=json.loads(self.raw['payloads'])
   for st in self.states:
    lane=st['lane'];rank=st['m4'];ids=tuple(st['proto_ids']);edges=tuple(tuple(e) for e in st['edges']);L=self.spec['lanes'][lane];expected=tuple(L['prototype_ids']) if lane=='HET4' else tuple([L['prototype_id']]*L['copies'])
    if ids!=expected or not 0<=rank<=L['max_m4'] or len(edges)!=rank or edges!=tuple(sorted(edges)) or m.state_digest(edges)!=st['digest']:raise ValueError('STATE_IDENTITY')
    key=(lane,rank,st['digest']);label=f'{lane}:{rank}:{st["digest"]}'
    if key in self.records or m.shaj(st)!=payloads[label]:raise ValueError('PAYLOAD_IDENTITY')
    for e in edges:
     if len(e)!=8 or not all(type(x)==int for x in e):raise ValueError('EDGE_ARITY')
     c,b,s,a,d,g,t,q=e
     if not(0<=c<d<len(ids)) or (a,q) not in pairs:raise ValueError('EDGE_ENDPOINT')
     if not(0<=b<len(self.P[ids[c]]['ordered_resource_blocks']) and 0<=g<len(self.P[ids[d]]['ordered_resource_blocks']) and 0<=s<len(self.P[ids[c]]['ordered_resource_blocks'][b]) and 0<=t<len(self.P[ids[d]]['ordered_resource_blocks'][g])):raise ValueError('SITE_ENDPOINT')
    for (c,b,s,a),used in m.usage(edges).items():
     if used>m.base_free(ids,self.P,c,b,s,a):raise ValueError('CAPACITY')
    components=[]
    for comp in m.graph_components(len(ids),edges):
     if len(comp)>1:
      inv=m.comp_invariants(ids,self.P,edges,comp)
      if not inv['ok']:raise ValueError('COMPONENT_ACCOUNTING')
      remap={v:i for i,v in enumerate(comp)};ee=[[remap[c],b,s,a,remap[d],g,t,q] for c,b,s,a,d,g,t,q in edges if c in remap and d in remap];components.append({'source_owners':list(comp),'proto_ids':[ids[c] for c in comp],'edges':ee,'accounting':inv})
    self.records[key]=st;self.component_records[key]=components;counts['states']+=1;counts['typed_E4_edges']+=len(edges);counts['O3_owner_occurrences']+=len(ids);counts['nontrivial_component_occurrences']+=len(components)
   if dict(counts)!=self.readiness['O4_saved_counts'] or set(payloads)!={f'{l}:{r}:{h}' for l,r,h in self.records}:raise ValueError('READINESS_COUNTS')
   seeds=sum(st['m4']==0 for st in self.states);self.report={'carriers':len(self.states)-seeds,'seed_carriers':seeds,'typed_edges':counts['typed_E4_edges'],'components':counts['nontrivial_component_occurrences'],'O3_owner_occurrences':counts['O3_owner_occurrences'],'prototype_bindings':len(self.P),'generation_calls':0,'complete_derivation_lineage':False,'microscopic_O2_source_closed':False,'missing_O2_panels':76}
  except BaseException:self.close();raise
 def ready(self):
  if self.closed:raise ValueError('READER_CLOSED')
 def lookup(self,lane,rank,digest):
  self.ready();return {'lane':lane,'rank':rank,'digest':digest,'record':copy.deepcopy(self.records[(lane,rank,digest)]),'external_seed':rank==0,'identity_scope':'lane/rank/edge digest plus full payload SHA256'}
 def owner(self,lane,rank,digest,index):
  row=self.lookup(lane,rank,digest)['record']
  if type(index)!=int or not 0<=index<len(row['proto_ids']):raise KeyError('OWNER_INDEX')
  pid=row['proto_ids'][index];b=self.bindings[pid];return {'owner_index':index,'prototype_id':pid,'resource_prototype':copy.deepcopy(self.P[pid]),'binding':copy.deepcopy(b),'O3_component':copy.deepcopy(self.base_components[pid]),'O3_root':self.o3.lookup(b['source_rank'],b['source_exact_key']),'external_microscopic_Qbank_boundary':True}
 def components(self,lane,rank,digest):self.lookup(lane,rank,digest);return copy.deepcopy(self.component_records[(lane,rank,digest)])
 def parents(self,*args):self.ready();raise ValueError('SAVED_DERIVATION_LINEAGE_NOT_AVAILABLE')
 def content_bytes(self,n):self.ready();return self.raw[n]
 def close(self):
  self.closed=True
  if self.z:self.z.close()
  if self.o3:self.o3.close()
