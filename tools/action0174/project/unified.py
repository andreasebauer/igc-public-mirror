from pathlib import Path
import json,hashlib,types,copy
from .previous_o7.unified import O7MasterReader
from .additional_export.reader import CarrierReader
from .additional_export.previous_o7 import frozen_o7 as m
from .bindings import CATALOG_SHA256,PREVIOUS_SHA256,ADMISSION_SHA256

class O7ExtendedMasterReader:
 def __init__(self,catalog,additional_admission,additional_archive,*,previous_catalog,previous_arguments):
  self.closed=False;self.previous=None;self.additional=None
  try:
   def load(p,h):
    b=Path(p).read_bytes()
    if hashlib.sha256(b).hexdigest()!=h:raise ValueError('AUTHORITY_HASH')
    return json.loads(b)
   cat=load(catalog,CATALOG_SHA256);old=load(previous_catalog,PREVIOUS_SHA256);ad=load(additional_admission,ADMISSION_SHA256);s=cat['slices'][-1]
   if cat['release_id']!='MASTER_DATA_V1_0149' or cat['previous_catalog_sha256']!=PREVIOUS_SHA256 or cat['slices'][:-1]!=old['slices'] or len(old['slices'])!=148:raise ValueError('PREDECESSOR_CATALOG')
   if ad['decision']!='ACCEPTED_FOR_REUSE_WITHIN_ADDITIONAL_SAVED_O7_ROOT_AND_EXACT_CONTEXTUAL_COMPONENT_SCOPE' or ad['scientific_root_sha256']!=s['scientific_root_sha256'] or ad['scientific_archive_sha256']!=s['archive']['sha256'] or s['scoped_admission_sha256']!=ADMISSION_SHA256 or ad['predecessor_catalog_sha256']!=PREVIOUS_SHA256 or s['scope']!=ad['scope'] or s['counts']!=ad['counts']:raise ValueError('SCOPED_ADMISSION')
   self.previous=O7MasterReader(previous_catalog,**previous_arguments);a=previous_arguments['previous_arguments'];a5=a['previous_arguments'];a4=a5['previous_arguments'];a3=a4['previous_arguments']
   self.additional=CarrierReader(additional_archive,s['archive']['sha256'],s['scientific_root_sha256'],previous_arguments['o7_archive'],a['o6_archive'],a5['o5_archive'],a4['o4_archive'],a3['o3_archive'])
   for k,v in ad['counts'].items():
    if self.additional.report[k]!=v:raise ValueError('ADMISSION_COUNTS')
   self.catalog=cat;self.root_components={}
   for key,st in self.additional.records.items():
    ids=st['parent_ids'];edges=st['record']['edges'];ctx=self.additional.contexts[key];adj=[set() for _ in ids]
    for e in edges:adj[e[0]].add(e[7]);adj[e[7]].add(e[0])
    seen=set();components=[]
    for start in range(len(ids)):
     if start in seen:continue
     todo=[start];comp=[];seen.add(start)
     while todo:
      x=todo.pop();comp.append(x)
      for y in adj[x]-seen:seen.add(y);todo.append(y)
     comp=sorted(comp)
     if len(comp)<2:continue
     mp={x:j for j,x in enumerate(comp)};ce=[(mp[e[0]],*e[1:7],mp[e[7]],*e[8:]) for e in edges if e[0] in mp and e[7] in mp];cc=types.SimpleNamespace(n=len(comp),parents=tuple(ctx.parents[j] for j in comp));ac=m.accounting(cc,ce)
     if not ac['ok']:raise ValueError('ROOT_COMPONENT_ACCOUNTING')
     components.append({'state_key':f'{key[0]}:{key[1]}:{key[2]}','source_owners':comp,'prototype_ids':[ids[j] for j in comp],'edges':json.loads(json.dumps(ce)),'accounting':ac,'role':'deterministic_literal_root_incidence_component_not_original_saved_profile_mapping'})
    self.root_components[key]=components
  except BaseException:self.close();raise
 def ready(self):
  if self.closed:raise ValueError('READER_CLOSED')
 def lookup_o7(self,lane,rank,digest):
  self.ready();key=(lane,rank,digest)
  if key not in self.additional.records:return self.previous.lookup_o7(*key)
  st=self.additional.lookup(*key);return {'record':st['record'],'external_seed':False,'identity_scope':'saved root lane/rank/E7 digest,ordered exact parent context and full literal hash'}
 def o7_owner(self,lane,rank,digest,index):
  self.ready();key=(lane,rank,digest)
  return self.additional.owner(*key,index) if key in self.additional.records else self.previous.o7_owner(*key,index)
 def o7_components(self,lane,rank,digest):
  self.ready();key=(lane,rank,digest)
  return copy.deepcopy(self.root_components[key]) if key in self.additional.records else self.previous.o7_components(*key)
 def o7_resources(self,lane,rank,digest):
  self.ready();key=(lane,rank,digest)
  return self.additional.resources(*key) if key in self.additional.records else self.previous.o7_resources(*key)
 def o7_parents(self,*key):self.lookup_o7(*key);raise ValueError('COMPLETE_PARENT_ACTION_ANCESTRY_UNAVAILABLE')
 def lookup_o7_component(self,*key):self.ready();return self.additional.component(*key)
 def o7_component_object(self,object_id):self.ready();return self.additional.object(object_id)
 def o7_component_owner(self,*a):self.ready();return self.additional.component_owner(*a)
 def o7_component_resources(self,*key):self.ready();return self.additional.component_resources(*key)
 def o7_component_source_owner_mapping(self,*key):self.ready();return self.additional.source_owner_mapping(*key)
 def coverage_report(self):
  self.ready();r=self.previous.coverage_report();r.update(release_id='MASTER_DATA_V1_0149',scientific_slices=149,o7_additional_saved_roots_and_components=dict(self.additional.report),o7_total_saved_whole_root_routes=98,o7_new_literal_root_component_occurrences=sum(map(len,self.root_components.values())),o7_graduated=False,automatic_o8_authorized=False);return r
 def __getattr__(self,name):
  if name.startswith('_') or name not in O7MasterReader.__dict__ or not callable(O7MasterReader.__dict__[name]):raise AttributeError(name)
  self.ready();return getattr(self.previous,name)
 def close(self):
  self.closed=True
  if self.additional:self.additional.close()
  if self.previous:self.previous.close()
 def __enter__(self):self.ready();return self
 def __exit__(self,*exc):self.close()
