from pathlib import Path
import json,hashlib,zipfile,gzip,copy
from . import frozen_in9 as m
sha=lambda b:hashlib.sha256(b).hexdigest()
class CarrierReader:
 def __init__(self,path,archive_sha,root_sha):
  if sha(Path(path).read_bytes())!=archive_sha:raise ValueError('ARCHIVE_HASH')
  self.z=zipfile.ZipFile(path);b=self.z.read('ROOT.json')
  if sha(b)!=root_sha:raise ValueError('ROOT_HASH')
  self.root=json.loads(b);self.closed=False;self.raw={};seen={'ROOT.json'}
  if sha(Path(m.__file__).read_bytes())!=self.root['canonicalizer_sha256']:raise ValueError('CANONICALIZER_HASH')
  for item in self.root['content']:
   n='content/'+item['sha256']+'.blob';b=self.z.read(n)
   if sha(b)!=item['sha256'] or len(b)!=item['bytes']:raise ValueError('CONTENT_HASH')
   if item['name'] in self.raw:raise ValueError('DUPLICATE_CONTENT_NAME')
   seen.add(n);self.raw[item['name']]=b
  if set(self.z.namelist())!=seen or len(self.z.namelist())!=len(seen):raise ValueError('CONTENT_CLOSURE')
  expected={n+str(d) for d in range(77,110) for n in ['panel','observer']}|{'contract','readiness','source_manifest','historical_result','historical_candidate'}
  if set(self.raw)!=expected:raise ValueError('SCOPE_CONTENT')
  self.records={};self.panels={};links=0
  for d in range(77,110):
   p=json.loads(gzip.decompress(self.raw['panel'+str(d)]));rows={}
   if p['d']!=d:raise ValueError('PANEL_DEPTH')
   for row in p['selected']:
    st=tuple(tuple(v) for v in row['states']);ed=tuple(tuple(v) for v in row['edges'])
    if row['d']!=d or not all(len(v)==16 and all(type(a)==int and a>=0 for a in v) for v in st):raise ValueError('STATE')
    if not all(0<=u<len(st) and 0<=v<len(st) and u!=v and 0<=a<7 and 0<=b<7 and m.bridge(m.PORTS[a],m.PORTS[b]) for u,a,v,b in ed):raise ValueError('TYPED_EDGE')
    if not m.capacity_valid(st,ed) or not all(sum(v[:7])==sum(v[7:])+2 for v in st):raise ValueError('CAPACITY_SIZE')
    if m.exact_key(st,ed)!=row['exact_key'] or m.role_info(st,ed)[0]!=row['role_hash']:raise ValueError('IDENTITY')
    if row['exact_key'] in rows:raise ValueError('DUPLICATE_IDENTITY')
    if d>77:
     for h in row['parents']:
      if h not in self.records[d-1]:raise ValueError('PARENT_CLOSURE')
      links+=1
    rows[row['exact_key']]=row
   self.records[d]=rows;self.panels[d]=p
  self.report={'carriers':sum(len(v) for d,v in self.records.items() if d>77),'seed_carriers':len(self.records[77]),'parent_links':links,'observer_sidecars':33,'generation_calls':0,'content_members':len(seen)-1}
 def lookup(self,d,h):
  if self.closed:raise ValueError('READER_CLOSED')
  return {'depth':d,'exact_key':h,'record':copy.deepcopy(self.records[d][h]),'external_seed':d==77}
 def parents(self,d,h):
  row=self.lookup(d,h)['record']
  if d==77:raise ValueError('EXTERNAL_SEED_ANCESTRY_EXCLUDED')
  return [self.lookup(d-1,p) for p in row['parents']]
 def content_bytes(self,name):
  if self.closed:raise ValueError('READER_CLOSED')
  return self.raw[name]
 def close(self):self.closed=True;self.z.close()
