"""Historical-witness descendant replay; append-only V3 map, no selection claim."""
import gzip,hashlib,json,resource,sys,shutil
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'decoder/source'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(x,s):
 if not x:raise RuntimeError(s)
def write(p,x):p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def merkle(tag,children):return hashlib.sha256((tag+'|'+'|'.join(sorted(children))).encode()).hexdigest()
def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));check(shutil.disk_usage(ROOT).free>=10*1024**3,'SPACE')
 out=ROOT/'descendant_step/results';reg=json.loads((out/'PREREGISTRATION.json').read_text())
 for n,pin in reg['inputs'].items():check(sha(ROOT/n)==pin,'INPUT:'+n)
 check(sha(Path(__file__))==reg['script_sha256'],'SCRIPT')
 from infinity_grid.isolated_runtime import verified_runtime
 verified_runtime()
 from infinity_grid.regime_scanner import LiftState
 from infinity_grid.maturation_parallel import _tupleify_tree
 with gzip.open(ROOT/'recovery/g1/BOOTSTRAP100_HISTORICAL.json.gz','rt') as f:boot=json.load(f)
 dag=boot['dag'];check(digest({k:v for k,v in dag.items() if k!='science_sha256'})==dag['science_sha256'],'DAG')
 overlays=json.loads((ROOT/'overlay_step/results/MAPPING.json').read_text());mapped={r['target']['old_reference']:r for r in overlays};states={};ref={};mapping=[];routes=[]
 from infinity_grid.frontier import _load_engine
 engine=_load_engine(next((ROOT/'overlay_step/kernel').iterdir()))
 for key,n in dag['nodes'].items():
  if n['kind']=='O7':
   r=mapped[key];x=r['result'];check(n['source_id']==r['target']['anchor_source_id'] and n['reserve_counts']==r['target']['reserve_counts'],'O7_BINDING')
   s=SimpleNamespace(construction_digest=x['new_construction'],skin=x['skin'],total_caps=tuple(x['total_caps']),level=7,reserve_counts=n['reserve_counts'])
   states[key]=s;ref[key]={'new':s.construction_digest,'skin':s.skin,'caps':list(s.total_caps)}
 def witness(key,path,t):
  node=dag['nodes'][key]
  if node['kind']=='O7':check(len(path)==2 and path[0]==t and 0<=path[1]<node['reserve_counts'][t],'O7_WITNESS');return
  check(len(path)==3 and path[2]==t and 0<=path[0]<len(node['children']),'LIFT_WITNESS');witness(node['children'][path[0]],path[1],t)
 for level in range(8,101):
  keys=sorted(k for k,n in dag['nodes'].items() if n['kind']=='LIFT' and n['level']==level)
  for key in keys:
   n=dag['nodes'][key];children=n['children'];oldpayload={'level':level,'children':children,'edges':n['top_edges_full'],'lane':n['lane'],'motif':n['motif_id']};check(digest(oldpayload)==key,'HISTORICAL_KEY')
   s=LiftState(engine,level,tuple(states[c] for c in children),_tupleify_tree(n['top_edges_full']),n['lane'],n['motif_id'])
   p={**oldpayload,'children':[ref[c]['new'] for c in children]};skin=merkle('R'+str(level),[ref[c]['skin'] for c in children]);caps=[sum(ref[c]['caps'][t] for c in children) for t in range(7)]
   check(s.construction_digest==digest(p) and s.skin==skin and list(s.total_caps)==caps,'NATIVE_REFERENCE')
   for e in n['top_edges_full']:
    u,v,a,b,wu,wv=e;check(0<=u<len(children) and 0<=v<len(children) and u!=v,'EDGE_OWNERS');witness(children[u],wu,a);witness(children[v],wv,b)
   for t in range(7):
    eligible=[i for i,c in enumerate(children) if ref[c]['caps'][t]>0]
    if eligible:
     old=min(eligible,key=lambda i:(children[i],i));new=min(eligible,key=lambda i:(ref[children[i]]['new'],i))
     if old!=new:routes.append({'historical_node':key,'level':level,'type':t,'old_owner':old,'v3_owner':new,'old_child':children[old],'v3_child_old_reference':children[new]})
   states[key]=s;ref[key]={'new':s.construction_digest,'skin':skin,'caps':caps}
   mapping.append({'old_reference':key,'level':level,'new_construction':s.construction_digest,'skin':skin,'total_caps':caps,'witness_sha256':digest(n['top_edges_full'])})
  print('LEVEL_PASS',level,len(keys),flush=True)
 write(out/'DESCENDANT_MAPPING.json',mapping);write(out/'RESERVATION_ORDER_DIVERGENCES.json',routes)
 write(out/'ROOT_MAPPING.json',[{'old_reference':k,**ref[k]} for k in dag['roots']])
 summary={'status':'PASS_HISTORICAL_WITNESS_REPLAY','o7_leaves':193,'lift_nodes':len(mapping),'levels':[8,100],'roots':len(dag['roots']),'reservation_order_divergences':len(routes),'first_divergence':routes[0] if routes else None,'matched_regeneration_completed':False,'full_p0_closed':False,'controller_unchanged':True}
 write(out/'SUMMARY.json',summary);print(summary,flush=True)
if __name__=='__main__':main()
