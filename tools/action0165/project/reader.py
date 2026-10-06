from pathlib import Path
import json,hashlib,zipfile,copy,collections,ast
from . import frozen_o6 as m
from .previous_o5.reader import CarrierReader as O5Reader
sha=lambda b:hashlib.sha256(b).hexdigest()
def ast_bytes(node):
 def normalized(value):
  if isinstance(value,ast.AST):return {'node':type(value).__name__,'fields':{k:normalized(v) for k,v in ast.iter_fields(value) if v is not None and v!=[]}}
  if isinstance(value,list):return [normalized(v) for v in value]
  if isinstance(value,bytes):return {"bytes_hex":value.hex()}
  return value
 return json.dumps(normalized(node),sort_keys=True,separators=(',',':')).encode()

class CarrierReader:
 def __init__(self,path,archive_sha,root_sha,o5_archive,o4_archive,o3_archive):
  self.closed=False;self.o5=None;self.z=None
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
   if set(self.raw)!={'states','prototypes','pin','spec','algebra','contract','readiness','payloads','components','source_manifest','selector','templates','original_producer','pure_extraction'}:raise ValueError('SCOPE')
   extraction=json.loads(self.raw['pure_extraction'])
   if sha(self.raw['original_producer'])!=self.root['original_producer_sha256'] or extraction['original_sha256']!=self.root['original_producer_sha256'] or extraction['pure_sha256']!=self.root['frozen_producer_sha256']:raise ValueError('PURE_SOURCE_EXTRACTION_HASH')
   import ast
   original={n.name:n for n in ast.parse(self.raw['original_producer']).body if isinstance(n,ast.FunctionDef)};pure={n.name:n for n in ast.parse(Path(m.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)}
   if set(pure)!=set(extraction['functions_ast_sha256']):raise ValueError('PURE_FUNCTION_CLOSURE')
   for name,h in extraction['functions_ast_sha256'].items():
    if sha(ast_bytes(original[name]))!=h or sha(ast_bytes(pure[name]))!=h:raise ValueError('PURE_AST_IDENTITY')
   p=self.root['O5_bindings'];self.o5=O5Reader(o5_archive,p['archive_sha256'],p['root_sha256'],o4_archive,o3_archive);self.states=json.loads(self.raw['states'])['states'];self.pool=json.loads(self.raw['prototypes']);self.prototypes=self.pool['prototypes'];self.pin=json.loads(self.raw['pin']);self.spec=json.loads(self.raw['spec']);self.readiness=json.loads(self.raw['readiness']);self.bindings={x['runtime_alias'] or x['prototype_id']:x for x in self.readiness['prototype_pool_bindings']};self.pin_alias=self.readiness['pinned_runtime_alias']
   selection=json.loads(self.raw['selector']);ok,selected=m.verify_prototype_selection(self.pool,selection)
   if not ok or selected!=self.readiness['selected_prototype_ids'] or self.pin['prototype_id']!=selection['pinned_prototype_id']:raise ValueError('PROTOTYPE_SELECTION')
   self.P={};raw_prototypes={}
   for raw in [*self.prototypes,self.pin]:
    pid=self.pin_alias if raw is self.pin else raw['prototype_id'];self.P[pid]={'prototype_id':pid,'o5':m.nested_o5(raw['ordered_resource_o4_carriers']),'canonical_R5_sha256':raw['canonical_R5_sha256'],**raw['accounting']};raw_prototypes[pid]=raw
   if set(self.P)!=set(self.bindings) or len(self.P)!=133:raise ValueError('PROTOTYPE_CLOSURE')
   if self.P[self.pin_alias]!=dict(self.P[self.pin['prototype_id']],prototype_id=self.pin_alias):raise ValueError('PIN_ALIAS')
   self.base_components={}
   for pid,proto in self.P.items():
    b=self.bindings[pid];key=(b['source_lane'],b['source_rank'],b['source_digest']);row=self.o5.lookup(*key)['record'];owners=b['source_owners'];resources=self.o5.resources(*key);ordered=json.loads(json.dumps([resources[i] for i in owners]));matches=[c for c in self.o5.components(*key) if c['source_owners']==owners]
    if m.nested_o5(ordered)!=proto['o5'] or m.shaj(ordered)!=b['ordered_resource_sha256'] or m.shaj(row)!=b['root_payload_sha256'] or len(matches)!=1 or m.shaj(m.canon_o5(proto['o5']))!=proto['canonical_R5_sha256']:raise ValueError('O5_COMPONENT_COORDINATES')
    c=matches[0];raw=raw_prototypes[pid]
    if c['edges']!=b['component_E5_edges'] or c['proto_ids']!=b['O4_prototype_ids'] or c['accounting']!=b['component_accounting'] or [self.o5.owner(*key,i)['binding'] for i in owners]!=b['O4_bindings'] or raw['exact_E5_edges']!=c['edges'] or raw['source_o4_prototype_ids']!=c['proto_ids']:raise ValueError('O5_TYPED_INCIDENCE_OR_NESTED_BINDING')
    for f in ['K4','N','U2','m3','m4','m5','d','r','beta5','P','F','g']:
     if proto[f]!=c['accounting'][f]:raise ValueError('PROTOTYPE_ACCOUNTING')
    if proto['beta_flat5']!=c['accounting']['beta_flat']:raise ValueError('PROTOTYPE_RANK')
    sites=[s for o4 in proto['o5'] for o3 in o4 for block in o3 for s in block];free=[sum(s[1][a] for s in sites) for a in range(7)]
    if proto['free7']!=free or proto['total_free']!=sum(free) or proto['min_site_free']!=min(sum(s[1]) for s in sites) or proto['distinct_site_states']!=len(set(sites)):raise ValueError('PROTOTYPE_RESOURCE_FEATURES')
    if 'features' in raw and raw['features']!=[proto[f] for f in self.pool['feature_order'][:13]]+free:raise ValueError('PROTOTYPE_SELECTOR_FEATURES')
    self.base_components[pid]=c
   classes=set();occurrences=0
   for key in self.o5.records:
    resources=self.o5.resources(*key)
    for c in self.o5.components(*key):classes.add(m.shaj(m.canon_o5(tuple(resources[i] for i in c['source_owners']))));occurrences+=1
   if occurrences!=169 or len(classes)!=132 or classes!={p['canonical_R5_sha256'] for p in self.prototypes}:raise ValueError('R5_POOL_CLOSURE')
   alg=json.loads(self.raw['algebra']);ports=[tuple(x) for x in alg['port_atoms']];ix={p:i for i,p in enumerate(ports)};pairs={(ix[tuple(x)],ix[tuple(y)]) for x,y in alg['ordered_compatible_port_pairs']};self.records={};self.component_records={};counts=collections.Counter();payloads=json.loads(self.raw['payloads']);allcomponents=[]
   for st in self.states:
    lane=st['lane'];rank=st['m6'];ids=tuple(st['proto_ids']);edges=tuple(tuple(e) for e in st['edges']);L=self.spec['lanes'][lane];expected=tuple(L['prototype_ids']) if lane=='HET4' else tuple([self.pin_alias]*L['copies'])
    if ids!=expected or not 0<=rank<=L['max_m6'] or len(edges)!=rank or edges!=tuple(sorted(edges)):raise ValueError('STATE_IDENTITY')
    for e in edges:
     if len(e)!=12 or not all(type(x)==int and 0<=x<2**32 for x in e):raise ValueError('EDGE_ARITY_OR_ENCODING')
     c,j,k,v,i,a,d,J,K,V,I,b=e
     if not 0<=c<d<len(ids) or (a,b) not in pairs:raise ValueError('EDGE_ENDPOINT')
     for owner,o4,o3,block,site,port in [(c,j,k,v,i,a),(d,J,K,V,I,b)]:
      res=self.P[ids[owner]]['o5']
      if not(0<=o4<len(res) and 0<=o3<len(res[o4]) and 0<=block<len(res[o4][o3]) and 0<=site<len(res[o4][o3][block]) and 0<=port<7):raise ValueError('SITE_ENDPOINT')
    if m.state_digest(edges)!=st['digest']:raise ValueError('BINARY_EDGE_DIGEST')
    key=(lane,rank,st['digest']);label=f'{lane}:{rank}:{st["digest"]}'
    if key in self.records or m.shaj(st)!=payloads[label]:raise ValueError('PAYLOAD_IDENTITY')
    if m.validate_exact_state(ids,self.P,edges,pairs):raise ValueError('CAPACITY_OR_RESOURCE_RESERVATIONS')
    components=[]
    for comp in m.graph_components(len(ids),edges):
     if len(comp)>1:
      inv=m.component_invariants(ids,self.P,edges,comp)
      if not inv['ok']:raise ValueError('COMPONENT_ACCOUNTING')
      pids,ee=m.normalize_component(ids,edges,comp);c={'state_key':label,'source_owners':list(comp),'proto_ids':list(pids),'edges':[list(e) for e in ee],'accounting':json.loads(json.dumps(inv))};components.append(c);allcomponents.append(c)
    self.records[key]=st;self.component_records[key]=components;counts['states']+=1;counts['typed_E6_edges']+=len(edges);counts['O5_owner_occurrences']+=len(ids)
   seeds=sum(st['m6']==0 for st in self.states);actual=dict(counts,nontrivial_component_occurrences=len(allcomponents),nonseed_states=len(self.states)-seeds,external_seed_states=seeds)
   if actual!=self.readiness['counts'] or allcomponents!=json.loads(self.raw['components']) or set(payloads)!={f'{l}:{r}:{h}' for l,r,h in self.records}:raise ValueError('READINESS_COUNTS_OR_COMPONENTS')
   self.report={'carriers':len(self.states)-seeds,'seed_carriers':seeds,'typed_edges':counts['typed_E6_edges'],'components':len(allcomponents),'O5_owner_occurrences':counts['O5_owner_occurrences'],'prototype_bindings':len(self.P),'unique_R5_resource_classes':132,'O5_component_occurrences_checked':169,'prototype_selection_pass':True,'generation_calls':0,'complete_derivation_lineage':False,'microscopic_O2_source_closed':False,'missing_O2_panels':76}
  except BaseException:self.close();raise
 def ready(self):
  if self.closed:raise ValueError('READER_CLOSED')
 def lookup(self,lane,rank,digest):
  self.ready();return {'lane':lane,'rank':rank,'digest':digest,'record':copy.deepcopy(self.records[(lane,rank,digest)]),'external_seed':rank==0,'identity_scope':'lane/rank/binary edge digest plus full payload SHA256'}
 def owner(self,lane,rank,digest,index):
  row=self.lookup(lane,rank,digest)['record']
  if type(index)!=int or not 0<=index<len(row['proto_ids']):raise KeyError('OWNER_INDEX')
  pid=row['proto_ids'][index];b=self.bindings[pid];return {'owner_index':index,'prototype_id':pid,'resource_prototype':copy.deepcopy(self.P[pid]),'binding':copy.deepcopy(b),'O5_component':copy.deepcopy(self.base_components[pid]),'O5_root':self.o5.lookup(b['source_lane'],b['source_rank'],b['source_digest']),'nested_O4_owners':[self.o5.owner(b['source_lane'],b['source_rank'],b['source_digest'],i) for i in b['source_owners']],'external_microscopic_Qbank_boundary':True}
 def components(self,lane,rank,digest):self.lookup(lane,rank,digest);return copy.deepcopy(self.component_records[(lane,rank,digest)])
 def resources(self,lane,rank,digest):
  row=self.lookup(lane,rank,digest)['record'];return m.materialize(tuple(row['proto_ids']),self.P,tuple(tuple(e) for e in row['edges']))
 def parents(self,*args):self.ready();raise ValueError('SAVED_DERIVATION_LINEAGE_NOT_AVAILABLE')
 def content_bytes(self,n):self.ready();return self.raw[n]
 def close(self):
  self.closed=True
  if self.z:self.z.close()
  if self.o5:self.o5.close()
