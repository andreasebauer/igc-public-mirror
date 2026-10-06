from pathlib import Path
import json,hashlib,zipfile,ast,collections,copy
sha=lambda b:hashlib.sha256(b).hexdigest()
class PanelReader:
 def __init__(self,path,archive_sha,root_sha):
  raw=Path(path).read_bytes()
  if sha(raw)!=archive_sha:raise ValueError('ARCHIVE_HASH')
  self.z=zipfile.ZipFile(path);b=self.z.read('ROOT.json')
  if sha(b)!=root_sha:raise ValueError('ROOT_HASH')
  self.root=json.loads(b);self.closed=False;self.records={};self.edges={};self.panels={};seen={'ROOT.json'}
  for item in self.root['content']:
   n='content/'+item['sha256']+'.blob';b=self.z.read(n)
   if sha(b)!=item['sha256'] or len(b)!=item['bytes']:raise ValueError('CONTENT_HASH')
   seen.add(n);self.panels[item['name']]=json.loads(b)
  if set(self.z.namelist())!=seen or len(self.z.namelist())!=len(seen):raise ValueError('CONTENT_CLOSURE')
  self.primitives=[ast.literal_eval(x['record']) for x in self.panels['primitives']['primitive']]
  for L in range(38,65):
   p=self.panels['panel'+str(L)];rows={}
   for x in p['selected']:
    r=ast.literal_eval(x['record'])
    if sha(repr(r).encode())!=x['sha256'] or x['sha256'] in rows:raise ValueError('RECORD_HASH')
    rows[x['sha256']]=r
   self.records[L]=rows
   if L>38:self.edges[L]=p['edges']
  self.parents=collections.defaultdict(list);recipes={}
  for name in ['recipes39_44','recipes45_64']:
   for row in self.panels[name]:
    k=(row['level'],row['parent'],row['child'])
    if k in recipes:raise ValueError('DUPLICATE_RECIPE')
    recipes[k]=row['lawful_boundary_recipes']
  edgecount=recipecount=0;keys=set()
  for L,edges in self.edges.items():
   for ph,ch in edges:
    k=(L,ph,ch)
    if k in keys:raise ValueError('DUPLICATE_EDGE')
    keys.add(k);a=self.records[L-1][ph];c=self.records[L][ch];rs=recipes[k]
    if not rs:raise ValueError('UNEXPLAINED_EDGE')
    for i,u,v in rs:
     p=self.primitives[i];u=tuple(u);v=tuple(v)
     if u not in a[1] or v not in p[1] or not ((u[1]==0 or u[1]&v[0]) and (v[1]==0 or v[1]&u[0])):raise ValueError('BRIDGE')
     ports=collections.Counter(a[1])+collections.Counter(p[1]);ports.subtract([u,v])
     if ports!=collections.Counter(c[1]) or collections.Counter(a[3])+collections.Counter(p[3])!=collections.Counter(c[3]) or (a[0]|p[0],min(a[2],p[2]),int(a[4] and p[4]))!=(c[0],c[2],c[4]):raise ValueError('OUTPUT')
     recipecount+=1
    self.parents[(L,ch)].append(ph);edgecount+=1
  if keys!=set(recipes):raise ValueError('RECIPE_CLOSURE')
  self.report={'levels':[39,64],'seed_level':38,'seed_records':len(self.records[38]),'panel_occurrences':sum(len(v) for L,v in self.records.items() if L>38),'edges':edgecount,'diagnostic_boundary_recipes':recipecount,'content_members':len(seen)-1,'generation_calls':0}
 def lookup(self,L,h):
  if self.closed:raise ValueError('READER_CLOSED')
  return {'level':L,'boundary_sha256':h,'record':copy.deepcopy(self.records[L][h]),'parents':copy.deepcopy(self.parents[(L,h)]) if L>38 else [],'external_seed':L==38}
 def close(self):self.closed=True;self.z.close()
