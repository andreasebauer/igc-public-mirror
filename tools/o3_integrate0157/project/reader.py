from pathlib import Path
import json,hashlib,gzip,zipfile,copy,collections
from . import frozen_audit as m,frozen_identity as identity
sha=lambda b:hashlib.sha256(b).hexdigest()
class CarrierReader:
 def __init__(self,path,archive_sha,root_sha):
  if sha(Path(path).read_bytes())!=archive_sha:raise ValueError('ARCHIVE_HASH')
  self.z=zipfile.ZipFile(path);raw=self.z.read('ROOT.json')
  if sha(raw)!=root_sha:raise ValueError('ROOT_HASH')
  self.root=json.loads(raw);self.closed=False;self.raw={};seen={'ROOT.json'}
  for n,h in self.root['reader_bindings'].items():
   if sha((Path(__file__).parent/n).read_bytes())!=h:raise ValueError('FROZEN_READER_BINDING')
  for item in self.root['content']:
   n='content/'+item['sha256']+'.blob';b=self.z.read(n)
   if sha(b)!=item['sha256'] or len(b)!=item['bytes']:raise ValueError('CONTENT_HASH')
   if item['name'] in self.raw:raise ValueError('DUPLICATE_CONTENT')
   self.raw[item['name']]=b;seen.add(n)
  if set(self.z.namelist())!=seen or len(seen)!=len(self.z.namelist()):raise ValueError('CONTENT_CLOSURE')
  expected={'panel'+str(r) for r in range(65)}|{'bank','spec','primitives','historical_graduation','contract','readiness','payloads','components','manifest'}
  if set(self.raw)!=expected:raise ValueError('SCOPE')
  self.bank=json.loads(self.raw['bank']);self.qstates={e['q_hash']:e for e in self.bank['entries']}
  if len(self.qstates)!=256 or len(self.bank['entries'])!=256:raise ValueError('QBANK_CLOSURE')
  self.panels={r:json.loads(gzip.decompress(self.raw['panel'+str(r)])) for r in range(65)}
  a=m.Audit.__new__(m.Audit);a.spec=json.loads(self.raw['spec']);a.bank=self.bank;a.PORTS=tuple(tuple(x) for x in a.spec['port_atoms']);idx={p:i for i,p in enumerate(a.PORTS)};a.B=[[False]*7 for _ in range(7)]
  for x,y in a.spec['ordered_compatible_port_pairs']:a.B[idx[tuple(x)]][idx[tuple(y)]]=True
  a.qmap={q:tuple((tuple(s[0]),tuple(s[1])) for s in e['sites']) for q,e in self.qstates.items()};a.panels={r:p['selected'] for r,p in self.panels.items()};self.audit=a
  v=json.loads(self.raw['readiness']);s,f,components,w,roll=a.full_sweep()
  if f or s!=v['summary'] or roll!=v['stored_key_roll_sha256'] or components!=json.loads(self.raw['components']):raise ValueError('FROZEN_INTEGRITY_SWEEP')
  self.records={};self.component_records=collections.defaultdict(list);links=0;direct=fallback=0;payloads=json.loads(self.raw['payloads'])
  for c in components:self.component_records[(c['r3'],c['exact_key'])].append(c)
  for r,rows in a.panels.items():
   self.records[r]={}
   for row in rows:
    h=row['exact_key']
    if h in self.records[r]:raise ValueError('DUPLICATE_IDENTITY')
    if payloads[f'{r}:{h}']!=sha(json.dumps({'entities':row['entities'],'edges':row['edges']},sort_keys=True,separators=(',',':')).encode()):raise ValueError('PAYLOAD_HASH')
    ps=row['parent_keys']
    if len(ps)!=len(set(ps)) or row['parent_count']!=len(ps) or (r and (not ps or not set(ps)<=self.records[r-1].keys())) or (r==0 and ps):raise ValueError('PARENT_CLOSURE')
    expected=None
    if r==0:expected='H000_'+identity.sha_repr(tuple(sorted(row['entities'])))[:24]
    elif r==1:
     x,i,pa,y,j,pb=row['edges'][0];eps=tuple(sorted(((row['entities'][x],a.qmap[row['entities'][x]][i],pa),(row['entities'][y],a.qmap[row['entities'][y]][j],pb)),key=repr));rem=list(row['entities']);rem.pop(y);rem.pop(x);expected='H001_'+identity.sha_repr((tuple(sorted(rem)),eps))[:24]
    else:
     ck=identity.exact_canonical_base_key(row,a.qmap)
     if ck is not None:expected=f'H{r:03d}_'+identity.sha_repr(('C',ck))[:24]
    if expected is not None:
     if expected!=h:raise ValueError('CANONICAL_IDENTITY')
     direct+=1
    else:fallback+=1
    self.records[r][h]=row;links+=len(ps)
  if links!=v['parent_links'] or sha(self.raw['bank'])!=v['bank_sha256']:raise ValueError('READINESS_BINDING')
  self.report={'carriers':s['states']-len(self.records[0]),'seed_carriers':len(self.records[0]),'parent_links':links,'components':len(components),'typed_edges':s['edges'],'bank_entries':len(self.qstates),'recomputed_keys':direct,'source_bound_fallback_keys':fallback,'generation_calls':0,'microscopic_O2_source_closed':False,'missing_O2_panels':76}
 def lookup(self,r,h):
  if self.closed:raise ValueError('READER_CLOSED')
  return {'rank':r,'exact_key':h,'record':copy.deepcopy(self.records[r][h]),'external_seed':r==0,'identity_scope':'frozen producer label and literal payload hash'}
 def parents(self,r,h):
  row=self.lookup(r,h)['record']
  if r==0:raise ValueError('EXTERNAL_SEED_ANCESTRY_EXCLUDED')
  return [self.lookup(r-1,p) for p in row['parent_keys']]
 def qstate(self,q):
  if self.closed:raise ValueError('READER_CLOSED')
  return {'q_hash':q,'state':copy.deepcopy(self.qstates[q]),'boundary':'anonymous local(p,u2) Q state; microscopic O2 incidence external'}
 def components(self,r,h):
  row=self.lookup(r,h)['record'];out=[]
  for c in self.component_records[(r,h)]:
   comp=c['component_entities'];remap={v:i for i,v in enumerate(comp)};edges=[[remap[v],i,a,remap[w],j,b] for v,i,a,w,j,b in row['edges'] if v in remap and w in remap]
   out.append({'diagnostic':copy.deepcopy(c),'entities':[row['entities'][v] for v in comp],'edges':edges,'source_entities':list(comp)})
  return out
 def content_bytes(self,n):
  if self.closed:raise ValueError('READER_CLOSED')
  return self.raw[n]
 def close(self):self.closed=True;self.z.close()
