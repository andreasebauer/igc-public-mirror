"""Candidate-native source-bound paused checkpoint recovery, 193 O9 chains."""
import pathlib,json,hashlib,sys,importlib.util,resource,shutil,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'backbone160';sys.path[:0]=[str(ROOT/'decoder159'),str(ROOT)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x
def write(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n');q.replace(p)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--pause-after',type=int);args=parser.parse_args()
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));assert shutil.disk_usage(ROOT).free>=10*1024**3
 reg=json.loads((OUT/'REGISTRATION.json').read_text());regsha=sha(OUT/'REGISTRATION.json')
 for p,h in reg['pins'].items():assert sha(ROOT/p)==h,p
 from infinity_grid.isolated_runtime import verified_runtime
 runtime=verified_runtime()
 from infinity_grid.frontier import _load_engine
 from infinity_grid import regime_scanner as rs
 engine=_load_engine(next((ROOT/'kernel').iterdir()));parents=engine.load_parent_records();records=json.loads(next((ROOT/'material').rglob('07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json')).read_text())['records'];contexts={};occurrences={}
 def restore(w,policy):
  if w['level']==7:
   i=w['occurrence'];r=records[i]
   if i not in contexts:contexts[i]=engine._profile_row_context(r,parents);occurrences[id(contexts[i])]=i
   return rs.O7State(engine,contexts[i],tuples(r['edges']),tuple(w['counts']),r['state_digest'],r['lane'],execution_policy=policy)
  return rs.LiftState(engine,w['level'],tuple(restore(c,policy) for c in w['children']),tuples(w['edges']),w['lane'],w['motif'],execution_policy=policy)
 def observation(s):
  if s.level==7:return {'occurrence':occurrences[id(s.ctx)],'counts':list(s.reserve_counts),'skin':s.skin,'caps':list(s.total_caps)}
  return {'skin':s.skin,'caps':list(s.total_caps),'children':[observation(c) for c in s.children],'edges':[rs._edge_json(e) for e in s.top_edges_full]}
 def wire(s):
  if s.level==7:return {'level':7,'occurrence':occurrences[id(s.ctx)],'counts':list(s.reserve_counts)}
  return {'level':s.level,'lane':s.lane,'motif':s.motif_id,'children':[wire(c) for c in s.children],'edges':[rs._edge_json(e) for e in s.top_edges_full]}
 from infinity_grid import maturation_parallel as mp
 context=json.loads((ROOT/'worker159/CONTEXT.json').read_text())
 # Reconstruct the registered exact O8 parent panel through production DAG transport.
 eng,roots=mp._states_from_dag(context['state_dag']);engine=eng
 center=next(s for s in roots if s.construction_digest==context['center_digest']);pairs=tuples(context['pairs'])
 def physical(s):
  if s.level==7:return {'source':s.source_id,'counts':list(s.reserve_counts),'skin':s.skin,'caps':list(s.total_caps)}
  return {'skin':s.skin,'caps':list(s.total_caps),'children':[physical(c) for c in s.children],'edges':[rs._edge_json(e) for e in s.top_edges_full]}
 def norm(x):
  x=dict(x)
  if 'occurrence' in x:x['source']=records[x.pop('occurrence')]['state_digest']
  if 'children' in x:x['children']=[norm(c) for c in x['children']]
  return x
 builds=0;reused=0
 for name,edges in rs.BACKBONE_MOTIFS:
  p=OUT/(name+'.json');binding={'registration':regsha,'name':name,'center':center.construction_digest,'policy':rs.HISTORICAL_EXECUTION_POLICY}
  expected=json.loads((ROOT/'restored/panel_o9'/('compat_BACKBONE_'+name+'.json')).read_text())['result']
  if p.exists():
   row=json.loads(p.read_text());assert row['binding']==binding and digest(row['result'])==row['result_digest'];reused+=1
  else:
   n=1+max(max(e) for e in edges);s=rs._build_lift(engine,9,[center]*n,list(edges),pairs,'BACKBONE','BACKBONE:'+name,force_pair=(0,0));assert s is not None
   result={'physical':physical(s),'construction':s.construction_digest,'order_key':rs.execution_order_key(s)}
   row={'binding':binding,'result':result,'result_digest':digest(result)};write(p,row);builds+=1
  assert row['result']['physical']==norm(expected['physical'])
  assert row['result']['construction']==expected['state']['construction'] and row['result']['order_key']==expected['order_key']
  print('PASS',name,'built' if builds else 'reused',flush=True)
 write(OUT/('RESULT.json' if builds else 'RESUME_RESULT.json'),{'status':'PASS_DEV159_NATIVE_BACKBONES','builds':builds,'reused':reused,'roots':7,'production_dag_restore':True,'shared_service':False,'full_p0_closed':False,'controller_changed':False})
if __name__=='__main__':
 try:main()
 except BaseException as e:write(OUT/'STOPPED.json',{'error':type(e).__name__,'message':str(e)});raise
