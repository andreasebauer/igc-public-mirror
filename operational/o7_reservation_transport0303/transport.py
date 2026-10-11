import json,gzip,hashlib,os,sys,importlib.util,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
B=Path(__file__).parent;B.mkdir(exist_ok=True);W=Path('/workspace/scratch/87702f15343f/witness_emission');os.environ['OSCOUT_DATA_ROOT']=str(W/'fixture')
sp=importlib.util.spec_from_file_location('_transport_fixture',W/'fixture/02_CODE/o7_live_engine.py');engine=importlib.util.module_from_spec(sp);sys.modules[sp.name]=engine;sp.loader.exec_module(engine);engine.O6=engine.import_o6();parents=engine.load_parent_records()
sha=lambda b:hashlib.sha256(b).hexdigest();hashj=lambda o:sha(json.dumps(o,sort_keys=True,separators=(',',':')).encode())
p=Path('/workspace/scratch/2a87972b51af/continuation0231/BOOTSTRAP100_HISTORICAL.json.gz');assert sha(p.read_bytes())=='d9ba6b5e8e4d6465a2739bde77e381c60eb0fe3bdefc6becce1151b1195462f5'
d=json.load(gzip.open(p,'rt'));variants=[v for v in d['dag']['nodes'].values() if v['kind']=='O7'];del d
assert len(variants)==193
base={}
def flatten(owners):
 out={}
 def walk(o,p,x):
  if len(p)==5:out[(o,p)]=(tuple(x[0]),tuple(x[1]))
  else:
   for i,c in enumerate(x):walk(o,p+(i,),c)
 for o,x in enumerate(owners):walk(o,(),x)
 return out
def skin(flat,trees):
 def walk(o,p,x):
  if len(p)==5:
   pp,ff=flat[(o,p)];return sha(b'SITE|'+json.dumps([list(pp),list(ff)],separators=(',',':')).encode())
  tag='R6' if not p else 'O'+str(6-len(p));return sha((tag+'|'+'|'.join(sorted(walk(o,p+(i,),c) for i,c in enumerate(x)))).encode())
 return sha(('R7|'+'|'.join(sorted(walk(o,(),x) for o,x in enumerate(trees)))).encode())
def overlay(flat,counts):
 out={k:(pp,list(ff)) for k,(pp,ff) in flat.items()};takes=[]
 for t,cnt in enumerate(counts):
  rem=cnt
  for k in sorted(out):
   n=min(rem,out[k][1][t])
   if n:out[k][1][t]-=n;rem-=n;takes.append([k[0],list(k[1]),t,n])
   if not rem:break
  assert rem==0
 return {k:(v[0],tuple(v[1])) for k,v in out.items()},takes
for file in sorted(W.glob('WITNESS_*.json')):
 r=json.loads(file.read_text());row=r['source_row'];ctx=engine._profile_row_context(row,parents);oldtrees=[engine.materialize_owner(p.h6,row['edges'],i) for i,p in enumerate(ctx.parents)];newtrees=[engine.materialize_owner(p.h6,r['result']['edges'],i) for i,p in enumerate(ctx.parents)]
 old=flatten(oldtrees);new=flatten(newtrees);mapping={(a,tuple(p)):(b,tuple(q)) for a,p,b,q in r['result']['witness']['node_map'] if len(p)==5}
 assert {mapping[k]:v for k,v in old.items()}==new
 base[row['state_digest']]=(old,new,mapping,oldtrees,newtrees,sha(file.read_bytes()))
rows=[];successors=0;naive_differences=0
for v in variants:
 old,new,mapping,ot,nt,pin=base[v['source_id']];counts=v['reserve_counts'];source,takes=overlay(old,counts);target={mapping[k]:val for k,val in source.items()};oldskin=skin(source,ot);newskin=skin(target,nt);assert oldskin==newskin
 caps=[sum(f[t] for pp,f in source.values()) for t in range(7)];assert caps==[sum(f[t] for pp,f in target.values()) for t in range(7)]
 mappedtakes=[[mapping[(o,tuple(p))][0],list(mapping[(o,tuple(p))][1]),t,n] for o,p,t,n in takes]
 expected=dict(level=7,parents=[p.pid for p in engine._profile_row_context(json.loads(next(f for f in W.glob('WITNESS_*.json') if json.loads(f.read_text())['source_row']['state_digest']==v['source_id']).read_text())['source_row'],parents).parents],edges=None)
 succ=[]
 for t in range(7):
  if caps[t]<=0:succ.append({'type':t,'available':False});continue
  cnt=list(counts);cnt[t]+=1;s,st=overlay(old,cnt);tr={mapping[k]:val for k,val in s.items()};ss=skin(s,ot);assert ss==skin(tr,nt);assert [sum(f[j] for pp,f in s.values()) for j in range(7)]==[caps[j]-(j==t) for j in range(7)]
  naive,_=overlay(new,cnt);diff=skin(naive,nt)!=ss;naive_differences+=diff;successors+=1;succ.append({'type':t,'available':True,'skin':ss,'naive_reselection_skin_differs':diff,'transported_reservation_patch':[[mapping[(o,tuple(p))][0],list(mapping[(o,tuple(p))][1]),tt,n] for o,p,tt,n in st]})
 rows.append({'legacy_construction_digest':v['construction_digest'],'source_id':v['source_id'],'reserve_counts':counts,'witness_file_sha256':pin,'skin':oldskin,'caps':caps,'transported_patch':mappedtakes,'successors':succ})
print('PASS variants',len(rows),'successors',successors,'naive differences',naive_differences,flush=True)
(B/'TRANSPORT.json').write_text(json.dumps({'status':'PASS_EXPLICIT_O7_RESERVATION_TRANSPORT','bootstrap_sha256':sha(p.read_bytes()),'variants':len(rows),'available_successors':successors,'naive_reselection_different_skin_count':naive_differences,'rows':rows,'scope':'Explicit transported coordinate policy, not migrated producer activation or full G2 identity','observer_runs':0},indent=2)+'\n')
