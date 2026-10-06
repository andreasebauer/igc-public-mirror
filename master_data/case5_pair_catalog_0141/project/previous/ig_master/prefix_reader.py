"""Fail-closed scientific reader. No generator, historical input, or algebra imports.

A Reader is a verified in-memory snapshot. Every table is exhausted and all
scientific links resolved before lookup is enabled. Missing coverage is an error,
never a request to reconstruct an object. Decoding needs only this export profile
and the existing byte-integrity reader, not source archives or runtime receipts.
"""
from pathlib import Path
from itertools import combinations_with_replacement,product
from collections import Counter
import hashlib,re,stat
from .support import ContentDirectory
from .support import canonical_bytes
from .prefix_contract import (CONTRACT,RECIPE,FORMAT,digest,parse,event_id,object_id,j3_id,formation_id)

class PrefixError(ValueError):pass

def check(condition,message):
 if not condition:raise PrefixError(message)

def fields(row,names):
 check(type(row) is dict and set(row)==set(names),'SCIENTIFIC_FIELDS_MISMATCH')

def integer(v,low=0,high=None):
 check(type(v) is int and v>=low and (high is None or v<high),'SCIENTIFIC_INTEGER_OR_RANGE')

def validate_foundation(f):
 fields(f,['schema_id','scope','obligations','primitive_states','arrays'])
 check(f['schema_id']=='IG_FRESH_FOUNDATION_V1','FOUNDATION_SCHEMA')
 expected={'radius':2,'primitive_carrier':'LARGEST_BOTTOM_SCC_OF_LOW_GRAPH',
  'primitive_order':'SORTED_REPR_MEMORY_STATE','l1_carrier':'UNORDERED_TRIPLES_WITH_REPLACEMENT',
  'l1_observer':'ISOLATED_SELECTED_LABELLED_SUCCESSOR_SET',
  'candidate_equivalence':'ISOLATED_BEHAVIOR_PLUS_EVENT_ROOTED_CORRECTED_M1',
  'complete_compositional_boundary_claim':False,'option_width':10}
 check(canonical_bytes(f['scope'])==canonical_bytes(expected),'FOUNDATION_SCOPE_MISMATCH')
 check(f['obligations']==[[i,s] for i in range(3) for s in [-1,1]],'OBLIGATION_ALPHABET')
 a=f['arrays'];fields(a,['tri','cid','oc','ot','os','om','on','rk'])
 states=f['primitive_states'];check(type(states) is list and len(states)==66,'CORE_CARRIER_SCOPE')
 n=len(states);width=10;nt=n*(n+1)*(n+2)//6
 for key,values in a.items():
  check(type(values) is list,'ARRAY_TYPE')
  for value in values:integer(value)
  check(len(values)==(nt*3 if key=='tri' else nt if key=='cid' else n if key=='oc' else n*width),'ARRAY_LENGTH')
 check(max(a['cid'])<nt and len(set(a['cid']))==max(a['cid'])+1,'CANDIDATE_CLASS_RANGE')
 for tid,triple in enumerate(combinations_with_replacement(range(n),3)):
  check(a['tri'][3*tid:3*tid+3]==list(triple),'EXACT_TRIPLE_CENSUS')
 memories=[]
 for sid,state in enumerate(states):
  fields(state,['sid','memory_state','rank','moves'])
  integer(state['sid'],0,n);check(state['sid']==sid,'SID_ORDER');integer(state['rank'])
  memory=state['memory_state'];check(type(memory) is list and len(memory)==2,'MEMORY_SHAPE')
  for triad in memory:
   check(type(triad) is list and len(triad)==3,'MEMORY_TRIAD_SHAPE')
   for x in triad:integer(x,-2,3)
  memories.append(canonical_bytes(memory))
  check(type(state['moves']) is list and a['oc'][sid]==len(state['moves'])+1 and 1<=a['oc'][sid]<=width,'OPTION_CARDINALITY')
  for i in range(width):
   at=sid*width+i
   if i==0:expected_values=[sid,0,0,0,state['rank']]
   elif i<a['oc'][sid]:
    m=state['moves'][i-1]
    fields(m,['target_sid','delta','expected','active','fulfilled','missing','supply_mask','missing_mask','target_rank'])
    integer(m['target_sid'],0,n)
    for key in ['active','fulfilled','target_rank']:integer(m[key])
    check(type(m['missing']) is list and len(m['missing'])==3 and all(type(x) is bool for x in m['missing']),'MISSING_OBLIGATION_FLAGS')
    check(type(m['expected']) is list and len(m['expected'])==3,'EXPECTED_OBLIGATION_VECTOR')
    for x in m['expected']:integer(x,-1,2)
    check(m['target_rank']==states[m['target_sid']]['rank'],'MOVE_RANK_TARGET')
    for key in ['supply_mask','missing_mask']:integer(m[key],0,64)
    # Preserve the raw primitive details rather than only the P/M projection.
    check(type(m['delta']) is list and len(m['delta'])==3,'MOVE_DELTA')
    for x in m['delta']:integer(x,-1,2)
    prev,cur=memory;target=states[m['target_sid']]['memory_state']
    check(target==[cur,[cur[i]+m['delta'][i] for i in range(3)]],'MOVE_MEMORY_TARGET')
    sign=lambda x:(x>0)-(x<0)
    trend=[sign(cur[i]-prev[i]) for i in range(3)]
    check(m['expected']==[trend[2],trend[0],-trend[1]],'MOVE_EXPECTED_BINDING')
    supply=sum(1<<j for j,(axis,direction) in enumerate(f['obligations']) if sign(m['delta'][axis])==direction)
    missing=sum(1<<j for j,(axis,direction) in enumerate(f['obligations']) if m['missing'][axis] and m['expected'][axis]==direction)
    check(supply==m['supply_mask'] and missing==m['missing_mask'],'MOVE_MASK_BINDING')
    expected_values=[m['target_sid'],m['supply_mask'],m['missing_mask'],int(m['active']>0 and m['fulfilled']==0),m['target_rank']]
   else:expected_values=[0]*5
   check([a[k][at] for k in ['ot','os','om','on','rk']]==expected_values,'OPTION_ARRAY_BINDING')
 check(len(set(memories))==n,'DUPLICATE_PRIMITIVE_MEMORY')
 return f


def _boundary(record,ports,destination):
 check(type(record) is list and len(record)==5,'BOUNDARY_SHAPE')
 integer(record[0],0,64);integer(record[2]);integer(record[4],0,2)
 check(type(record[1]) is list and len(record[1])==ports,'BOUNDARY_PORTS')
 for pm in record[1]:
  check(type(pm) is list and len(pm)==2,'PORT_PAIR')
  for v in pm:integer(v,0,64)
 if destination==1:integer(record[3])
 else:
  check(type(record[3]) is list and len(record[3])==destination,'DESTINATION_INTERFACE')
  for v in record[3]:integer(v)


def _bridge(a,pa,b,pb):
 ap,am=a[1][pa];bp,bm=b[1][pb]
 return (am==0 or bool(am&bp)) and (bm==0 or bool(bm&ap))

class PrefixReader:
 def __init__(self,directory,root_sha256):
  self.directory=Path(directory);self.root_sha256=root_sha256;self._verified=False
  check(type(root_sha256) is str and re.fullmatch('[0-9a-f]{64}',root_sha256) is not None,'INVALID_ROOT_ID')
  check(not self.directory.is_symlink() and self.directory.is_dir(),'UNSAFE_DATA_ROOT')
  check(not (self.directory/'ROOT.json').is_symlink(),'UNSAFE_ROOT_FILE')
  root_stat=(self.directory/'ROOT.json').lstat()
  check(stat.S_ISREG(root_stat.st_mode) and root_stat.st_size<=65536,'ROOT_FILE_TYPE_OR_BOUND')
  raw=(self.directory/'ROOT.json').read_bytes()
  check(len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==root_sha256,'ROOT_DIGEST_MISMATCH')
  self.root=parse(raw,65536)
  fields(self.root,['schema_id','dataset_id','identity_profile','contract_ref','recipe_ref','foundation_ref','catalog','catalog_sha256','tables'])
  check(self.root['schema_id']==FORMAT and self.root['dataset_id']=='IG_MASTER_FRESH_PREFIX_CASE5_V1' and
        self.root['identity_profile']=='NATIVE_EXACT_ORDERED_BOUNDARY_PLUS_SCOPED_FORMATIONS_V1','ROOT_SCOPE_MISMATCH')
  fields(self.root['tables'],['j3','abc_objects','abc_formations'])
  self.store=ContentDirectory(self.directory/'content');self.seen={};self.total=0
 def _read(self,ref):
  fields(ref,['sha256','size_bytes']);sha=ref['sha256']
  check(sha not in self.seen or self.seen[sha]==ref,'CONFLICTING_CONTENT_REFERENCE')
  raw=self.store.read(ref,CONTRACT['limits']['content_bytes'])
  if sha not in self.seen:
   self.total+=len(raw);self.seen[sha]=ref
   check(self.total<=CONTRACT['limits']['total_bytes'] and len(self.seen)<=CONTRACT['limits']['files'],'SLICE_BUDGET')
  return raw
 def _json(self,ref):return parse(self._read(ref))
 def _table(self,name,key):
  t=self.root['tables'][name];fields(t,['name','key','rows','shards'])
  check(t['name']==name and t['key']==key,'TABLE_BINDING')
  integer(t['rows'],0,100001);check(type(t['shards']) is list and len(t['shards'])<=100,'SHARD_COUNT_BOUND')
  result={};last=None
  for shard in t['shards']:
   fields(shard,['content_ref','first_key','last_key','rows']);integer(shard['rows'],1,1001)
   raw=self._read(shard['content_ref']);check(raw.endswith(b'\n'),'SHARD_FRAMING')
   rows=[parse(line,65536) for line in raw.splitlines()]
   check(len(rows)==shard['rows'] and bool(rows),'SHARD_ROWS')
   check(rows[0].get(key)==shard['first_key'] and rows[-1].get(key)==shard['last_key'],'SHARD_BOUNDS')
   for row in rows:
    k=row.get(key);check(type(k) is str and re.fullmatch('[0-9a-f]{64}',k) is not None,'ROW_ID_TYPE')
    check(last is None or last<k,'DUPLICATE_OR_UNORDERED_RECORD');last=k;result[k]=row
  check(len(result)==t['rows'],'TABLE_COUNT')
  return result
 def _choice_record(self,sids,choices):
  a=self.foundation['arrays'];indices=[s*10+i for s,i in zip(sids,choices)]
  supply=[a['os'][x] for x in indices];missing=[a['om'][x] for x in indices];need=[a['on'][x] for x in indices]
  exposed=[]
  for i in range(3):
   satisfied=need[i]==0 or bool(missing[i]&(supply[(i+1)%3]|supply[(i+2)%3]))
   if not satisfied and (need[i]==0 or missing[i]==0):return None
   exposed.append(0 if satisfied else missing[i])
  targets=[a['ot'][x] for x in indices];target_tid=self.triple_index[tuple(sorted(targets))]
  rec=[supply[0]|supply[1]|supply[2],[[supply[i],exposed[i]] for i in range(3)],
       min(a['rk'][x] for x in indices),a['cid'][target_tid],int(all(x==0 for x in choices))]
  return rec,targets,target_tid
 def verify(self):
  if self._verified:return dict(self.report)
  check(canonical_bytes(self._json(self.root['contract_ref']))==canonical_bytes(CONTRACT),'UNSUPPORTED_DATA_CONTRACT')
  check(canonical_bytes(self._json(self.root['recipe_ref']))==canonical_bytes(RECIPE),'UNSUPPORTED_RECIPE')
  self.foundation=validate_foundation(self._json(self.root['foundation_ref']))
  a=self.foundation['arrays'];catalog={'schema_id':'IG_FRESH_FOUNDATION_CATALOG_V1','scope':self.foundation['scope'],
   'primitive_states_sha256':digest(self.foundation['primitive_states']),'arrays_sha256':digest(a),
   'class_id_semantics':'FIRST_OCCURRENCE_ORDER_IN_RAW_TRIPLE_CENSUS'}
  check(canonical_bytes(self.root['catalog'])==canonical_bytes(catalog) and digest(catalog)==self.root['catalog_sha256'],'CATALOG_BINDING')
  self.triple_index={tuple(a['tri'][3*i:3*i+3]):i for i in range(len(a['cid']))}
  reps={}
  for tid,cid in enumerate(a['cid']):reps.setdefault(cid,tid)
  self.carriers=self._table('j3','j3_id');check(len(self.carriers)==3,'J3_CARRIER_COUNT')
  self.by_class={};self.events={};realizations=0
  for carrier in self.carriers.values():
   fields(carrier,['j3_id','class_id','representative_tid','source_sids','events']);cid=carrier['class_id'];tid=carrier['representative_tid']
   integer(cid);integer(tid,0,len(a['cid']))
   check(cid in RECIPE['classes'] and cid not in self.by_class and tid==reps[cid],'J3_SCOPE_OR_REPRESENTATIVE')
   check(carrier['j3_id']==j3_id(self.root['catalog_sha256'],tid),'J3_IDENTITY')
   sids=a['tri'][3*tid:3*tid+3];check(carrier['source_sids']==sids,'EXACT_CARRIER_BINDING')
   check(type(carrier['events']) is list and len(carrier['events'])<=1000,'J3_EVENTS_BOUND')
   choices_seen=set();eventmap={};prior=None
   for event in carrier['events']:
    fields(event,['event_id','record','realizations']);rec=event['record'];_boundary(rec,3,1)
    eid=event['event_id'];check(eid==event_id(rec) and (prior is None or prior<eid),'EVENT_ID_OR_DUPLICATE');prior=eid
    check(type(event['realizations']) is list and 0<len(event['realizations'])<=1000,'REALIZATION_BOUND')
    eventmap[eid]=event
    for occurrence in event['realizations']:
     fields(occurrence,['option_indices','target_sids','target_tid']);choices=occurrence['option_indices']
     integer(occurrence['target_tid'],0,len(a['cid']))
     check(type(occurrence['target_sids']) is list and len(occurrence['target_sids'])==3,'TARGET_SHAPE')
     for target in occurrence['target_sids']:integer(target,0,len(self.foundation['primitive_states']))
     check(type(choices) is list and len(choices)==3,'CHOICE_SHAPE')
     for s,i in zip(sids,choices):integer(i,0,a['oc'][s])
     check(tuple(choices) not in choices_seen,'DUPLICATE_J3_REALIZATION');choices_seen.add(tuple(choices))
     computed=self._choice_record(sids,choices)
     check(computed is not None and computed[0]==rec and computed[1]==occurrence['target_sids'] and computed[2]==occurrence['target_tid'],'REALIZATION_OR_TARGET_BINDING')
     realizations+=1
   # Small exact completeness check over stored per-site option indices. It does
   # not call a generator, read an oracle, or create missing scientific records.
   allowed={c for c in product(*(range(a['oc'][s]) for s in sids)) if self._choice_record(sids,c) is not None}
   check(choices_seen==allowed,'INCOMPLETE_J3_RELATION')
   self.events[carrier['j3_id']]=eventmap;self.by_class[cid]=carrier
  check(set(self.by_class)==set(RECIPE['classes']),'J3_CLASS_SCOPE')
  self.objects=self._table('abc_objects','object_id');self.formations=self._table('abc_formations','formation_id')
  for oid,obj in self.objects.items():
   fields(obj,['object_id','record']);_boundary(obj['record'],5,3)
   check(oid==object_id(self.root['catalog_sha256'],obj['record']),'ABC_OBJECT_IDENTITY')
  self.by_object={};formation_keys=set()
  for fid,formation in self.formations.items():
   fields(formation,['formation_id','object_id','components']);oid=formation['object_id'];comps=formation['components']
   check(oid in self.objects,'DANGLING_ABC_OBJECT')
   check(type(comps) is list and len(comps)==3,'COMPONENT_ARITY');records=[]
   for comp,role,cid in zip(comps,RECIPE['component_roles'],RECIPE['classes']):
    fields(comp,['role','j3_id','event_id'])
    check(comp['role']==role and comp['j3_id']==self.by_class[cid]['j3_id'],'COMPONENT_SCOPE_OR_ROLE')
    check(comp['event_id'] in self.events[comp['j3_id']],'DANGLING_COMPONENT_EVENT')
    records.append(self.events[comp['j3_id']][comp['event_id']]['record'])
   check(fid==formation_id(oid,comps),'FORMATION_IDENTITY')
   key=tuple(c['event_id'] for c in comps);check(key not in formation_keys,'DUPLICATE_FORMATION');formation_keys.add(key)
   x,y,z=records;check(_bridge(x,0,y,0) and _bridge(y,1,z,0),'ILLEGAL_INTERNAL_BRIDGE')
   expected=[x[0]|y[0]|z[0],[x[1][1],x[1][2],y[1][2],z[1][1],z[1][2]],min(x[2],y[2],z[2]),[x[3],y[3],z[3]],int(x[4] and y[4] and z[4])]
   check(self.objects[oid]['record']==expected,'ATTACHMENT_OR_BOUNDARY_MISMATCH')
   self.by_object.setdefault(oid,[]).append(formation)
  check(set(self.by_object)==set(self.objects),'OBJECT_WITHOUT_FORMATION')
  A,B,C=[[e['record'] for e in self.by_class[c]['events']] for c in RECIPE['classes']]
  complete_count=sum(sum(_bridge(x,0,y,0) for x in A)*sum(_bridge(y,1,z,0) for z in C) for y in B)
  check(len(self.formations)==complete_count,'INCOMPLETE_ABC_FORMATION_CENSUS')
  actual=set()
  for p in self.directory.rglob('*'):
   check(not p.is_symlink(),'UNSAFE_SLICE_LINK')
   if p.is_file():actual.add(str(p.relative_to(self.directory)))
   elif p.is_dir():check(p==self.directory/'content','UNDECLARED_DIRECTORY')
   else:raise PrefixError('UNSAFE_SLICE_NODE')
  expected={'ROOT.json'}|{'content/'+sha+'.blob' for sha in self.seen}
  check(actual==expected,'UNDECLARED_OR_MISSING_SCIENTIFIC_CONTENT')
  self.report={'status':'SCIENTIFIC_REFERENCE_CLOSURE_PASS','primitive_states':len(self.foundation['primitive_states']),
   'primitive_moves':sum(len(s['moves']) for s in self.foundation['primitive_states']),
   'l1_triples':len(a['cid']),'j3_carriers':3,'j3_events':sum(len(c['events']) for c in self.carriers.values()),
   'j3_option_realizations':realizations,'abc_objects':len(self.objects),'abc_formations':len(self.formations),
   'component_occurrences':len(self.formations)*3,'external_port_incidences':len(self.formations)*5,
   'internal_bridges':len(self.formations)*2,'scientific_files':len(actual),'scientific_bytes':self.total+(self.directory/'ROOT.json').stat().st_size,
   'all_declared_scientific_dependencies_internal':True,'scientific_acceptance':'NOT_GRANTED_BY_READER'}
  self._verified=True;return dict(self.report)
 def _ready(self):check(self._verified,'VERIFY_BEFORE_LOOKUP')
 def primitive(self,sid):
  self._ready();integer(sid,0,len(self.foundation['primitive_states']));return parse(canonical_bytes(self.foundation['primitive_states'][sid]))
 def triple(self,tid):
  self._ready();a=self.foundation['arrays'];integer(tid,0,len(a['cid']))
  return {'tid':tid,'source_sids':a['tri'][3*tid:3*tid+3],'candidate_class_id':a['cid'][tid],'identity':'EXACT_TRIPLE_NOT_CANDIDATE_QUOTIENT'}
 def lookup_j3(self,tid):
  self._ready();matches=[c for c in self.carriers.values() if c['representative_tid']==tid]
  check(len(matches)==1,'J3_CARRIER_NOT_IN_EXPORTED_SCOPE');return parse(canonical_bytes(matches[0]))
 def lookup_abc(self,oid,*,catalog_sha256):
  self._ready();check(catalog_sha256==self.root['catalog_sha256'],'CATALOG_SCOPE_MISMATCH')
  check(oid in self.objects,'ABC_OBJECT_NOT_FOUND')
  rows=[]
  for formation in self.by_object[oid]:
   components=[]
   for comp in formation['components']:
    carrier=self.carriers[comp['j3_id']];event=self.events[comp['j3_id']][comp['event_id']]
    components.append({'role':comp['role'],'exact_carrier':{k:carrier[k] for k in ['j3_id','class_id','representative_tid','source_sids']},
     'event':event,'primitive_source_states':[self.foundation['primitive_states'][s] for s in carrier['source_sids']]})
   rows.append({'formation':formation,'components':components})
  return parse(canonical_bytes({'object':self.objects[oid],'formation_occurrences':rows,'wiring':RECIPE}))
