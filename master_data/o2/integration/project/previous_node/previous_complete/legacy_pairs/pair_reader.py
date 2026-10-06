"""Independent verified reader for pairs and their explicit master dependency."""
from pathlib import Path
import hashlib
from infinity_grid.storage_collections import ContentDirectory
from infinity_grid.storage_schema import canonical_bytes
from .prefix_contract import digest,parse
from .prefix_reader import PrefixReader,check,fields,_boundary,integer
from .pair_contract import CONTRACT,RECIPES,DATASET,PARENT_ROOT,pair_object_id,pair_formation_id

class PairReader:
 def __init__(self,directory,root_sha256,parent):
  self.directory=Path(directory);self.root_sha256=root_sha256;self.parent=parent;self._verified=False
  check(type(parent) is PrefixReader and parent.root_sha256==PARENT_ROOT,'PAIR_PARENT_SCOPE')
  check(self.directory.is_dir() and not self.directory.is_symlink(),'PAIR_DIRECTORY')
  f=self.directory/'ROOT.json';check(f.is_file() and not f.is_symlink() and f.stat().st_size<=65536,'PAIR_ROOT_FILE')
  raw=f.read_bytes();check(hashlib.sha256(raw).hexdigest()==root_sha256,'PAIR_ROOT_HASH');self.root=parse(raw,65536)
  fields(self.root,['schema_id','dataset_id','parent_root_sha256','contract_ref','families'])
  check(self.root['schema_id']=='IG_MASTER_PAIR_DATA_V1' and self.root['dataset_id']==DATASET and self.root['parent_root_sha256']==PARENT_ROOT,'PAIR_ROOT_SCOPE')
  self.store=ContentDirectory(self.directory/'content');self.seen={};self.total=0
 def read(self,ref):
  fields(ref,['sha256','size_bytes']);raw=self.store.read(ref,4194304)
  check(ref['sha256'] not in self.seen,'PAIR_DUPLICATE_CONTENT')
  self.seen[ref['sha256']]=ref;self.total+=len(raw);check(self.total<=8388608,'PAIR_BYTES_BOUND');return raw
 def table(self,ref,key):
  raw=self.read(ref);check(raw.endswith(b'\n'),'PAIR_TABLE_FRAMING')
  rows=[parse(line,65536) for line in raw.splitlines()];check(len(rows)<=729,'PAIR_TABLE_BOUND')
  ids=[r.get(key) for r in rows];check(all(type(x) is str and len(x)==64 for x in ids) and ids==sorted(set(ids)),'PAIR_DUPLICATE_OR_UNSORTED')
  return {r[key]:r for r in rows}
 def verify(self):
  if self._verified:return dict(self.report)
  self.parent.verify();check(self.read(self.root['contract_ref'])==canonical_bytes(CONTRACT),'PAIR_CONTRACT')
  fields(self.root['families'],['AB','BC']);self.objects={};self.formations={};self.by_object={}
  for family,recipe in RECIPES.items():
   info=self.root['families'][family];fields(info,['objects','formations'])
   objects=self.table(info['objects'],'object_id');forms=self.table(info['formations'],'formation_id')
   a,b=[self.parent.by_class[c] for c in recipe['classes']];slots=recipe['bridge_slots'];seen=set();by_obj={}
   for oid,obj in objects.items():
    fields(obj,['object_id','record']);_boundary(obj['record'],4,2)
    check(oid==pair_object_id(family,obj['record']),'PAIR_OBJECT_ID')
   for fid,row in forms.items():
    fields(row,['formation_id','object_id','components']);oid=row['object_id'];comps=row['components']
    check(oid in objects and type(comps) is list and len(comps)==2,'PAIR_DANGLING_OBJECT_OR_COMPONENT_ARITY')
    records=[]
    for comp,role,carrier in zip(comps,recipe['roles'],[a,b]):
     fields(comp,['role','j3_id','event_id'])
     check(comp['role']==role and comp['j3_id']==carrier['j3_id'],'PAIR_COMPONENT_BINDING')
     check(comp['event_id'] in self.parent.events[carrier['j3_id']],'PAIR_DANGLING_EVENT')
     records.append(self.parent.events[carrier['j3_id']][comp['event_id']]['record'])
    check(fid==pair_formation_id(family,oid,comps),'PAIR_FORMATION_ID')
    key=tuple(c['event_id'] for c in comps);check(key not in seen,'PAIR_DUPLICATE_FORMATION');seen.add(key)
    x,y=records;xp,xm=x[1][slots[0]];yp,ym=y[1][slots[1]]
    check((xm==0 or xm&yp) and (ym==0 or ym&xp),'PAIR_ILLEGAL_BRIDGE')
    labelled={role:record for role,record in zip(recipe['roles'],records)}
    expected=[x[0]|y[0],[labelled[role][1][slot] for role,slot in recipe['external_ports']],min(x[2],y[2]),[x[3],y[3]],int(x[4] and y[4])]
    check(objects[oid]['record']==expected,'PAIR_WIRING_OR_RECORD')
    by_obj.setdefault(oid,[]).append(row)
   # Verify completeness independently against stored input pairs; do not add rows.
   count=0
   for x in a['events']:
    for y in b['events']:
     xp,xm=x['record'][1][slots[0]];yp,ym=y['record'][1][slots[1]]
     if (xm==0 or xm&yp) and (ym==0 or ym&xp):count+=1
   check(len(forms)==count and set(objects)==set(by_obj),'PAIR_INCOMPLETE_CENSUS')
   self.objects[family]=objects;self.formations[family]=forms;self.by_object[family]=by_obj
  actual=set()
  for p in self.directory.rglob('*'):
   check(not p.is_symlink(),'PAIR_SYMLINK')
   if p.is_file():actual.add(str(p.relative_to(self.directory)))
   else:check(p==self.directory/'content' and p.is_dir(),'PAIR_UNDECLARED_DIRECTORY')
  check(actual=={'ROOT.json'}|{'content/'+s+'.blob' for s in self.seen},'PAIR_FILE_INVENTORY')
  self._verified=True;self.report={'status':'PAIR_SCIENTIFIC_CLOSURE_PASS','root_sha256':self.root_sha256,'parent_root_sha256':PARENT_ROOT,
   'families':{f:{'objects':len(self.objects[f]),'formations':len(self.formations[f])} for f in RECIPES},
   'scientific_files':len(actual),'scientific_bytes':self.total+(self.directory/'ROOT.json').stat().st_size,
   'dependency_scope':'PARENT_SLICE_REQUIRED; NO_DUPLICATED_FOUNDATION','acceptance':'NOT_GRANTED_BY_READER'}
  return dict(self.report)
 def lookup(self,family,oid):
  check(self._verified,'VERIFY_BEFORE_PAIR_LOOKUP');check(family in RECIPES and oid in self.objects[family],'PAIR_OBJECT_NOT_FOUND')
  occurrences=[]
  for row in self.by_object[family][oid]:
   comps=[]
   for c in row['components']:
    carrier=self.parent.carriers[c['j3_id']];event=self.parent.events[c['j3_id']][c['event_id']]
    comps.append({'role':c['role'],'carrier':carrier,'event':event,'primitive_source_states':[self.parent.primitive(s) for s in carrier['source_sids']]})
   occurrences.append({'formation':row,'components':comps})
  return parse(canonical_bytes({'object':self.objects[family][oid],'occurrences':occurrences,'recipe':RECIPES[family]}))
