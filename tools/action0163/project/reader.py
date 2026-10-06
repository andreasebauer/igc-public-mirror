from pathlib import Path
import json,hashlib,zipfile,copy,collections,gzip,csv,io
from . import frozen_o5 as m
from .previous_o4.reader import CarrierReader as O4Reader
from .previous_o4 import frozen_o4 as o4
sha=lambda b:hashlib.sha256(b).hexdigest()
class CarrierReader:
 def __init__(self,path,archive_sha,root_sha,o4_archive,o3_archive):
  self.closed=False;self.o4=None;self.z=None
  try:
   if sha(Path(path).read_bytes())!=archive_sha:raise ValueError('ARCHIVE_HASH')
   self.z=zipfile.ZipFile(path);raw=self.z.read('ROOT.json')
   if sha(raw)!=root_sha:raise ValueError('ROOT_HASH')
   self.root=json.loads(raw);self.raw={};seen={'ROOT.json'}
   if sha(Path(m.__file__).read_bytes())!=self.root['frozen_producer_sha256']:raise ValueError('PRODUCER_HASH')
   for item in self.root['content']:
    n='content/'+item['sha256']+'.blob';b=self.z.read(n)
    if sha(b)!=item['sha256'] or len(b)!=item['bytes'] or item['name'] in self.raw:raise ValueError('CONTENT_HASH_OR_DUPLICATE')
    self.raw[item['name']]=b;seen.add(n)
   if set(self.z.namelist())!=seen or len(self.z.namelist())!=len(seen):raise ValueError('CONTENT_CLOSURE')
   if set(self.raw)!={'states','prototypes','pin','spec','algebra','contract','readiness','payloads','components','source_manifest','selector','features','templates'}:raise ValueError('SCOPE')
   p=self.root['O4_bindings'];self.o4=O4Reader(o4_archive,p['archive_sha256'],p['root_sha256'],o3_archive);self.states=json.loads(self.raw['states'])['states'];self.prototypes=json.loads(self.raw['prototypes'])['selected'];self.pin=json.loads(self.raw['pin']);self.spec=json.loads(self.raw['spec']);self.readiness=json.loads(self.raw['readiness']);self.bindings={x['prototype_id']:x for x in self.readiness['prototype_bindings']};self.P={x['prototype_id']:x for x in self.prototypes}
   pin=self.pin;A=pin['accounting'];pid='PHASE0_PINNED_ACTUAL_O4_COMPONENT';self.P[pid]=dict(prototype_id=pid,ordered_resource_o3_carriers=pin['ordered_resource_o3_carriers'],K3=A['K3'],N=A['N'],U2=A['U2'],m3=A['m3'],m4=A['m4'],d=A['d'],r4=A['r'],beta_flat4=A['beta_flat4'],beta4=A['beta4'])
   if set(self.P)!=set(self.bindings) or len(self.P)!=7:raise ValueError('PROTOTYPE_CLOSURE')
   self.base_components={}
   for pid,proto in self.P.items():
    b=self.bindings[pid];key=(b['source_lane'],b['source_rank'],b['source_digest']);row=self.o4.lookup(*key)['record'];owners=b['source_owners'];resources=o4.materialize(tuple(row['proto_ids']),self.o4.P,tuple(tuple(e) for e in row['edges']));ordered=json.loads(json.dumps([resources[i] for i in owners]));matches=[c for c in self.o4.components(*key) if c['source_owners']==owners]
    if ordered!=proto['ordered_resource_o3_carriers'] or m.shaj(ordered)!=b['ordered_resource_sha256'] or o4.shaj(row)!=b['root_payload_sha256'] or len(matches)!=1:raise ValueError('O4_COMPONENT_COORDINATES')
    c=matches[0]
    if c['edges']!=b['component_E4_edges'] or c['proto_ids']!=b['O3_prototype_ids'] or json.loads(json.dumps(c['accounting']))!=b['component_accounting'] or [self.o4.owner(*key,i)['binding'] for i in owners]!=b['O3_bindings']:raise ValueError('O4_TYPED_INCIDENCE_OR_NESTED_BINDING')
    for f in ['N','U2','m3','m4','d','beta4']:
     if proto[f]!=c['accounting'][f]:raise ValueError('PROTOTYPE_ACCOUNTING')
    if proto['r4']!=c['accounting']['r'] or proto['beta_flat4']!=c['accounting']['beta_flat'] or proto['K3']!=len(owners):raise ValueError('PROTOTYPE_RANK')
    if pid!='PHASE0_PINNED_ACTUAL_O4_COMPONENT':
     if proto['canonical_R4_sha256']!=m.shaj(m.canon_o4(m.proto_o4(proto))) or proto['internal_e4_edges']!=c['edges'] or proto['o3_prototype_ids']!=c['proto_ids']:raise ValueError('PROTOTYPE_PAYLOAD')
    self.base_components[pid]=c
   alg=json.loads(self.raw['algebra']);ports=[tuple(x) for x in alg['port_atoms']];ix={p:i for i,p in enumerate(ports)};pairs={(ix[tuple(x)],ix[tuple(y)]) for x,y in alg['ordered_compatible_port_pairs']};self.records={};self.component_records={};counts=collections.Counter();payloads=json.loads(self.raw['payloads']);allcomponents=[]
   for st in self.states:
    lane=st['lane'];rank=st['m5'];ids=tuple(st['proto_ids']);edges=tuple(tuple(e) for e in st['edges']);L=self.spec['lanes'][lane];expected=tuple(L['prototype_ids']) if lane=='HET4' else tuple([L['prototype_id']]*L['copies'])
    if ids!=expected or not 0<=rank<=L['max_m5'] or len(edges)!=rank or edges!=tuple(sorted(edges)) or m.state_digest(edges)!=st['digest']:raise ValueError('STATE_IDENTITY')
    key=(lane,rank,st['digest']);label=f'{lane}:{rank}:{st["digest"]}'
    if key in self.records or m.shaj(st)!=payloads[label]:raise ValueError('PAYLOAD_IDENTITY')
    for e in edges:
     if len(e)!=10 or not all(type(x)==int for x in e):raise ValueError('EDGE_ARITY')
     c,g,b,s,a,d,G,B,S,q=e
     if not 0<=c<d<len(ids) or (a,q) not in pairs:raise ValueError('EDGE_ENDPOINT')
     for owner,group,block,site,port in [(c,g,b,s,a),(d,G,B,S,q)]:
      resource=self.P[ids[owner]]['ordered_resource_o3_carriers']
      if not(0<=group<len(resource) and 0<=block<len(resource[group]) and 0<=site<len(resource[group][block]) and 0<=port<7):raise ValueError('SITE_ENDPOINT')
    for (c,g,b,s,a),used in m.usage(edges).items():
     if used>m.base_free(ids,self.P,c,g,b,s,a):raise ValueError('CAPACITY')
    components=[]
    for comp in m.graph_components(len(ids),edges):
     if len(comp)>1:
      inv=m.comp_invariants(ids,self.P,edges,comp)
      if not inv['ok']:raise ValueError('COMPONENT_ACCOUNTING')
      pids,ee=m.normalize_component(ids,edges,comp);c={'state_key':label,'source_owners':list(comp),'proto_ids':list(pids),'edges':[list(e) for e in ee],'accounting':json.loads(json.dumps(inv))};components.append(c);allcomponents.append(c)
    self.records[key]=st;self.component_records[key]=components;counts['states']+=1;counts['typed_E5_edges']+=len(edges);counts['O4_owner_occurrences']+=len(ids)
   seeds=sum(st['m5']==0 for st in self.states);actual=dict(counts,nontrivial_component_occurrences=len(allcomponents),nonseed_states=len(self.states)-seeds,external_seed_states=seeds)
   if actual!=self.readiness['counts'] or allcomponents!=json.loads(self.raw['components']) or set(payloads)!={f'{l}:{r}:{h}' for l,r,h in self.records}:raise ValueError('READINESS_COUNTS_OR_COMPONENTS')
   self.report={'carriers':len(self.states)-seeds,'seed_carriers':seeds,'typed_edges':counts['typed_E5_edges'],'components':len(allcomponents),'O4_owner_occurrences':counts['O4_owner_occurrences'],'prototype_bindings':len(self.P),'generation_calls':0,'complete_derivation_lineage':False,'microscopic_O2_source_closed':False,'missing_O2_panels':76}
  except BaseException:self.close();raise
 def ready(self):
  if self.closed:raise ValueError('READER_CLOSED')
 def lookup(self,lane,rank,digest):
  self.ready();return {'lane':lane,'rank':rank,'digest':digest,'record':copy.deepcopy(self.records[(lane,rank,digest)]),'external_seed':rank==0,'identity_scope':'lane/rank/edge digest plus full payload SHA256'}
 def owner(self,lane,rank,digest,index):
  row=self.lookup(lane,rank,digest)['record']
  if type(index)!=int or not 0<=index<len(row['proto_ids']):raise KeyError('OWNER_INDEX')
  pid=row['proto_ids'][index];b=self.bindings[pid];return {'owner_index':index,'prototype_id':pid,'resource_prototype':copy.deepcopy(self.P[pid]),'binding':copy.deepcopy(b),'O4_component':copy.deepcopy(self.base_components[pid]),'O4_root':self.o4.lookup(b['source_lane'],b['source_rank'],b['source_digest']),'nested_O3_owners':[self.o4.owner(b['source_lane'],b['source_rank'],b['source_digest'],i) for i in b['source_owners']],'external_microscopic_Qbank_boundary':True}
 def components(self,lane,rank,digest):self.lookup(lane,rank,digest);return copy.deepcopy(self.component_records[(lane,rank,digest)])
 def resources(self,lane,rank,digest):
  row=self.lookup(lane,rank,digest)['record'];return m.materialize(tuple(row['proto_ids']),self.P,tuple(tuple(e) for e in row['edges']))
 def parents(self,*args):self.ready();raise ValueError('SAVED_DERIVATION_LINEAGE_NOT_AVAILABLE')
 def content_bytes(self,n):self.ready();return self.raw[n]
 def close(self):
  self.closed=True
  if self.z:self.z.close()
  if self.o4:self.o4.close()
