from pathlib import Path
import json,hashlib,zipfile,copy,types,collections,struct
from .previous_o7.reader import CarrierReader as O7Reader,shaj,js,sha
from .previous_o7 import frozen_o7 as m

def container(tag,children):return sha((tag+'|'+'|'.join(sorted(children))).encode())
def resource_signature(h6):
 def leaf(site):return sha(b'SITE|'+json.dumps([list(site[0]),list(site[1])],separators=(',',':')).encode())
 return container('R6',[container('O5',[container('O4',[container('O3',[container('O2',[leaf(x) for x in block]) for block in o3]) for o3 in o4]) for o4 in o5]) for o5 in h6])

class CarrierReader:
 def __init__(self,path,archive_sha,root_sha,o7_archive,o6_archive,o5_archive,o4_archive,o3_archive):
  self.closed=False;self.z=None;self.base=None
  try:
   if sha(Path(path).read_bytes())!=archive_sha:raise ValueError('ARCHIVE_HASH')
   self.z=zipfile.ZipFile(path);raw=self.z.read('ROOT.json')
   if sha(raw)!=root_sha:raise ValueError('ROOT_HASH')
   self.root=json.loads(raw);self.raw={};seen={'ROOT.json'}
   if self.root['schema']!='IG_ADDITIONAL_SAVED_O7_EXPORT_V1':raise ValueError('SCHEMA')
   for x in self.root['content']:
    n='content/'+x['sha256']+'.blob';b=self.z.read(n)
    if sha(b)!=x['sha256'] or len(b)!=x['bytes'] or x['name'] in self.raw:raise ValueError('CONTENT_HASH')
    self.raw[x['name']]=b;seen.add(n)
   if set(self.z.namelist())!=seen or len(self.z.namelist())!=len(seen):raise ValueError('CONTENT_CLOSURE')
   self.data={n:json.loads(b) for n,b in self.raw.items()};pins=self.root['O7_bindings']
   names={'roots','objects','occurrences','owners','contract','readiness','input_pins','profiles','survivors','source_profile_bindings','source_search'}
   checkpoints={'checkpoint_'+x['source_checkpoint_sha256'] for x in self.data['roots']}
   if set(self.raw)!=names|checkpoints or len(checkpoints)!=2:raise ValueError('CONTENT_NAMES')
   if sha(self.raw['profiles'])!='77497fcd32952096bb13923d5cb8821a1725a8d8afad2eab7a18c2e95afb44a9' or sha(self.raw['survivors'])!='eac96afbd7f97855bc4b1db639ff447cf9ffb43fb52c759f8734a884d4300862':raise ValueError('SAVED_PROFILE_PINS')
   mapping={'ROOT_SCOPE.json':'roots','COMPONENT_OBJECTS.json':'objects','COMPONENT_OCCURRENCES.json':'occurrences','OWNER_BINDINGS.json':'owners','EXPORT_CONTRACT.json':'contract','INPUT_PINS.json':'input_pins'}
   if set(self.data['readiness']['output_pins'])!=set(mapping):raise ValueError('READINESS_OUTPUT_CLOSURE')
   for n,h in self.data['readiness']['output_pins'].items():
    if sha(self.raw[mapping[n]])!=h:raise ValueError('READINESS_OUTPUT_PIN')
   self.base=O7Reader(o7_archive,pins['archive_sha256'],pins['root_sha256'],o6_archive,o5_archive,o4_archive,o3_archive)
   self.P=dict(self.base.P);self.colors={x['prototype_id']:x['exact_representative_sha256'] for x in json.loads(self.base.raw['selected'])['prototypes']}
   self.twins={x['twin_id']:x for x in json.loads(self.base.raw['twins'])['twins']}
   for pid,t in self.twins.items():self.P[pid]=types.SimpleNamespace(pid=pid,h6=m.nested_h6(t['ordered_resource_O5_carriers']),accounting=t['accounting']);self.colors[pid]=t['exact_representative_sha256']
   alg=json.loads(self.base.o6.raw['algebra']);ports=[tuple(x) for x in alg['port_atoms']];ix={p:i for i,p in enumerate(ports)};self.pairs={(ix[tuple(a)],ix[tuple(b)]) for a,b in alg['ordered_compatible_port_pairs']}
   self.bindings=self.data['owners']
   for pid,b in self.bindings.items():
    if pid in self.base.bindings:
     if b['role']!='admitted_O6_component_binding' or b['binding']!=self.base.bindings[pid]:raise ValueError('OWNER_BINDING')
    elif pid in self.twins:
     tb=next(x for x in json.loads(self.base.raw['external_twins']) if x['id']==pid)
     if b['role']!='source_bound_external_adversarial_O6_root' or b['admitted_O6_root'] is not False or b['literal_root']!=self.twins[pid] or b['O5_bindings']!=tb:raise ValueError('EXTERNAL_OWNER_BINDING')
    else:raise ValueError('UNKNOWN_OWNER')
   original_profiles=self.data['profiles']['records']
   if original_profiles!=self.data['survivors']['records'] or len(original_profiles)!=205:raise ValueError('SURVIVOR_CLOSURE')
   self.records={};self.contexts={};known=set(self.base.records);rootowners=rootedges=0
   for st in self.data['roots']:
    k=tuple(st['root_key']);lane,rank,digest=k;row=st['record'];ids=st['parent_ids'];cfg=self.base.lanes[lane];expected=cfg.get('prototype_ids') or [cfg['prototype_id']]*cfg['owner_count']
    if lane!='HOM6' or rank not in (2,3) or ids!=expected or k in known or shaj(row)!=st['literal_payload_sha256'] or len(row['edges'])!=rank:raise ValueError('ROOT_SCOPE')
    if st['parent_exact_colors']!=[self.colors[x] for x in ids] or shaj({'ids':ids,'colors':st['parent_exact_colors']})!=st['ordered_context_sha256']:raise ValueError('ROOT_CONTEXT')
    checkpoint=self.data['checkpoint_'+st['source_checkpoint_sha256']]
    if shaj({x:y for x,y in checkpoint.items() if x!='payload_sha256'})!=checkpoint['payload_sha256'] or row not in checkpoint['result']['beam']:raise ValueError('CHECKPOINT_BINDING')
    for field,h in [('prereg_science_sha256','1459e670df72028d4f9978549d2da0f179029079ae0769953eddcd19b426bbd8'),('correction_science_sha256','e1ce490a8bbe29cc9d818a601539a5010f1ab9ada6a076b3e480c0b0a643741f'),('parent_o6_science_sha256','7b9e3ec0fbabcdb0fa46c2d5d153ac186cdd2d2343275a407e82b6fd14d9da37')]:
     if checkpoint[field]!=h:raise ValueError('CHECKPOINT_SCIENCE_PIN')
    if sha(self.raw['checkpoint_'+st['source_checkpoint_sha256']])!=st['source_checkpoint_sha256']:raise ValueError('CHECKPOINT_BYTE_PIN')
    self.contexts[k]=self._check(ids,row['edges'],row['accounting'],digest);self.records[k]=st;known.add(k);rootowners+=len(ids);rootedges+=rank
   self.objects={};self.object_contexts={}
   for obj in self.data['objects']:
    h=obj['object_id'];payload=obj['payload'];identity=payload['identity'];ids=identity['parent_ids']
    if h in self.objects or identity['schema']!='IG_O7_EXACT_COMPONENT_IDENTITY_V1' or shaj(identity)!=h or shaj(payload)!=obj['payload_sha256'] or identity['parent_exact_colors']!=[self.colors[x] for x in ids]:raise ValueError('OBJECT_IDENTITY')
    ctx=self._check(ids,payload['edges'],payload['accounting'],identity['E7_digest']);edges=payload['edges'];adj=[set() for _ in ids]
    for e in edges:adj[e[0]].add(e[7]);adj[e[7]].add(e[0])
    reached={0};todo=[0]
    while todo:
     for j in adj[todo.pop()]-reached:reached.add(j);todo.append(j)
    if len(ids)<2 or len(reached)!=len(ids):raise ValueError('CONNECTED_COMPONENT')
    resources=self._resources(ctx,edges)
    if shaj(resources)!=payload['resource_sha256'] or container('R7',[resource_signature(x) for x in resources])!=payload['R7_skin_sha256']:raise ValueError('COMPONENT_RESOURCE')
    self.objects[h]=obj;self.object_contexts[h]=ctx
   self.occurrences={};refs=set();used=set();owners=edges=0;observed=set()
   for i,occ in enumerate(self.data['occurrences']):
    k=tuple(occ['occurrence_key']);row=occ['record'];h=occ['object_id'];payload=self.objects[h]['payload'];identity=payload['identity']
    if k in self.occurrences or k!=(row['lane'],row['m7'],row['source_state_digest'],row['component_index']) or occ['record_index']!=i or row!=original_profiles[i] or shaj(row)!=occ['literal_payload_sha256']:raise ValueError('OCCURRENCE_IDENTITY')
    if (identity['parent_ids'],identity['parent_exact_colors'],identity['E7_digest'])!=(row['parent_ids'],row['parent_exact_colors'],row['state_digest']) or any(payload[x]!=row[x] for x in ['edges','accounting','R7_skin_sha256']):raise ValueError('OCCURRENCE_OBJECT')
    if row['component_owner_count']!=len(row['parent_ids']) or occ['root_rank']!=row['m7'] or occ['component_E7_edge_count']!=len(row['edges']) or occ['source_owner_index_mapping_available'] is not False or occ['saved_profile_summary_role']!='source_evidence_only_full_Counter_unavailable':raise ValueError('OCCURRENCE_SCOPE')
    available=k[:3] in known;role='available_saved_whole_root_reference' if available else 'external_unavailable_whole_root_reference'
    if occ['source_root_available']!=available or occ['source_root_role']!=role:raise ValueError('SOURCE_ROOT_ROLE')
    self.occurrences[k]=occ;refs.add(k[:3]);observed.add(h);used.update(row['parent_ids']);owners+=len(row['parent_ids']);edges+=len(row['edges'])
   if observed!=set(self.objects) or used!=set(self.bindings):raise ValueError('OBJECT_OWNER_CLOSURE')
   counts={'additional_whole_roots':len(self.records),'additional_root_E7_edges':rootedges,'additional_root_O6_owner_occurrences':rootowners,'component_occurrences':len(self.occurrences),'distinct_exact_component_objects':len(self.objects),'component_E7_edge_occurrences':edges,'component_O6_owner_occurrences':owners,'distinct_source_root_references':len(refs),'available_source_root_references':sum(k in known for k in refs),'unavailable_source_root_references':sum(k not in known for k in refs),'used_admitted_O6_prototype_bindings':len(used-set(self.twins)),'used_external_O6_twin_bindings':len(used&set(self.twins))}
   if counts!=self.data['contract']['counts'] or counts!=self.data['readiness']['counts'] or counts!={'additional_whole_roots':24,'additional_root_E7_edges':60,'additional_root_O6_owner_occurrences':144,'component_occurrences':205,'distinct_exact_component_objects':164,'component_E7_edge_occurrences':512,'component_O6_owner_occurrences':533,'distinct_source_root_references':164,'available_source_root_references':96,'unavailable_source_root_references':68,'used_admitted_O6_prototype_bindings':4,'used_external_O6_twin_bindings':2}:raise ValueError('SCOPE_COUNTS')
   self.report=dict(counts,generation_calls=0,scientific_master_admission=False,full_Counter_available=False,complete_derivation_lineage=False,full_l0_to_g8_complete=False)
  except Exception:self.close();raise
 def _check(self,ids,edges,accounting,digest):
  ctx=types.SimpleNamespace(n=len(ids),parents=tuple(self.P[x] for x in ids));usage=collections.Counter()
  if edges!=sorted(edges):raise ValueError('EDGE_ORDER')
  for e in edges:
   if len(e)!=14 or any(type(x)!=int or not 0<=x<2**32 for x in e):raise ValueError('EDGE_TYPE')
   c,p,a,d,q,b=m.eparts(e)
   if not 0<=c<d<ctx.n or (a,b) not in self.pairs:raise ValueError('EDGE_ENDPOINT')
   for owner,path,typ in [(c,p,a),(d,q,b)]:
    if len(path)!=5 or any(type(x)!=int or x<0 for x in path):raise ValueError('SITE_PATH')
    cap,free=m.site_at(ctx.parents[owner].h6,path);usage[(owner,path,typ)]+=1
    if usage[(owner,path,typ)]>free[typ]:raise ValueError('CAPACITY')
  if sha(b'IG-E7-STATE-v1|'+b''.join(struct.pack('>14I',*e) for e in edges))!=digest or m.accounting(ctx,edges)!=accounting or not accounting['ok']:raise ValueError('DIGEST_ACCOUNTING')
  return ctx
 def ready(self):
  if self.closed:raise ValueError('CLOSED_READER')
 def lookup(self,lane,rank,digest):self.ready();return copy.deepcopy(self.records[(lane,rank,digest)])
 def component(self,*key):self.ready();return copy.deepcopy(self.occurrences[tuple(key)])
 def object(self,object_id):self.ready();return copy.deepcopy(self.objects[object_id])
 def _resources(self,ctx,edges):return tuple(m.materialize_owner(p.h6,edges,i) for i,p in enumerate(ctx.parents))
 def resources(self,*key):self.ready();return self._resources(self.contexts[tuple(key)],self.records[tuple(key)]['record']['edges'])
 def component_resources(self,*key):
  occ=self.component(*key);h=occ['object_id'];return self._resources(self.object_contexts[h],self.objects[h]['payload']['edges'])
 def _owner(self,ctx,index):
  if type(index)!=int or not 0<=index<ctx.n:raise ValueError('OWNER_INDEX')
  pid=ctx.parents[index].pid;b=self.bindings[pid]
  if pid in self.base.bindings:
   bound=b['binding'];k=tuple(bound['source_key']);out={'owner_index':index,'prototype_id':pid,'role':b['role'],'binding':bound,'O6_root':self.base.o6.lookup(*k),'O6_component':self.base.base_components[pid],'resource_O6':js(ctx.parents[index].h6),'nested_O5_owners':[self.base.o6.owner(*k,j) for j in bound['source_owners']]}
  else:
   t=self.twins[pid];nested=[]
   for j,p in enumerate(t['exact_O5_parent_ids']):
    bound=self.base.o6.bindings[p];nested.append({'owner_index':j,'prototype_id':p,'resource_prototype':self.base.o6.P[p],'binding':bound,'O5_root':self.base.o6.o5.lookup(bound['source_lane'],bound['source_rank'],bound['source_digest']),'O5_component':self.base.o6.base_components[p]})
   out={'owner_index':index,'prototype_id':pid,'role':b['role'],'binding':b,'external_O6_root':t,'resource_O6':js(ctx.parents[index].h6),'nested_O5_owners':nested,'admitted_O6_root':False}
  return copy.deepcopy(out)
 def owner(self,lane,rank,digest,index):self.lookup(lane,rank,digest);return self._owner(self.contexts[(lane,rank,digest)],index)
 def component_owner(self,lane,rank,digest,index,owner_index):
  occ=self.component(lane,rank,digest,index);return self._owner(self.object_contexts[occ['object_id']],owner_index)
 def parents(self,*key):self.ready();raise ValueError('COMPLETE_PARENT_ACTION_ANCESTRY_UNAVAILABLE')
 def source_owner_mapping(self,*key):self.component(*key);raise ValueError('SOURCE_OWNER_INDEX_MAPPING_UNAVAILABLE')
 def close(self):
  self.closed=True
  if self.base:self.base.close()
  if self.z:self.z.close()
