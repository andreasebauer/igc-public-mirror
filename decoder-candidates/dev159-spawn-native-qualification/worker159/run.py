"""Candidate-native source-bound paused checkpoint recovery, 193 O9 chains."""
import pathlib,json,hashlib,sys,importlib.util,resource,shutil,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'worker159';sys.path[:0]=[str(ROOT/'decoder159'),str(ROOT)]
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
 import multiprocessing,statistics
 from worker159.worker import build
 inputroot=ROOT/'qualification_inputs';rows=[json.loads(p.read_text()) for p in (inputroot/'compatibility_o8/compat/recipes').glob('*.json')];by_motif={x['call']['motif_id']:x for x in rows}
 selected=json.loads((ROOT/'restored/panel_o8/compat_SELECTION.json').read_text())['selected'];prev=[]
 for motif in selected:
  row=by_motif[motif];children=[]
  for c in row['result']['children']:children.append(restore({'level':7,'occurrence':c['occurrence'],'counts':c['counts']},rs.HISTORICAL_EXECUTION_POLICY))
  st=rs.LiftState(engine,8,tuple(children),tuples(row['result']['edges']),row['call']['lane'],motif,execution_policy=rs.HISTORICAL_EXECUTION_POLICY);assert st.construction_digest==row['result']['state']['construction'];prev.append(st)
 prev=sorted(prev,key=rs.execution_order_key);median=statistics.median(sum(s.total_caps) for s in prev);center=min(prev,key=lambda s:(abs(sum(s.total_caps)-median),rs.execution_order_key(s)))
 assert center.motif_id==json.loads((ROOT/'restored/o9_qualification/CENTER.json').read_text())['compat_center']
 pairs=sorted(tuple(map(int,p)) for p in engine.O6.load_rules()[1]);assert len(pairs)==31
 recipes=mp.enumerate_candidate_recipes(rs.load_motif_library(),center_supports_twin=True,pairs_count=len(pairs));assert len(recipes)==193
 expected={x['recipe']['motif_id']:x for x in [json.loads(p.read_text()) for p in (ROOT/'restored/o9_qualification/recipes').glob('*.json')]}
 context={'schema_id':'IG_MATURATION_CANDIDATE_CONTEXT_V1','level':9,'pairs':[list(p) for p in pairs],'center_digest':center.construction_digest,'state_dag':mp.state_dag_wire(prev),'execution_policy':rs.HISTORICAL_EXECUTION_POLICY,'prev_execution_order_keys':[rs.execution_order_key(s) for s in prev]};write(OUT/'CONTEXT.json',context)
 def compare(got):
  want=expected[got['motif_id']];assert got['recipe']==want['recipe'];w=want['result']['compat'];assert got['construction']==w['construction'] and got['execution_order_key']==w['order_key'] and got['execution_policy']==rs.HISTORICAL_EXECUTION_POLICY
  def normalize(x):
   x=dict(x)
   if 'occurrence' in x:x['source']=records[x.pop('occurrence')]['state_digest']
   if 'children' in x:x['children']=[normalize(c) for c in x['children']]
   return x
  assert got['physical']==normalize(w['physical'])
 (OUT/'checkpoints').mkdir(exist_ok=True);missing=[];results={};reused=0
 for recipe in recipes:
  key=digest(recipe);p=OUT/'checkpoints'/(key+'.json')
  if p.exists():
   row=json.loads(p.read_text());assert row['registration']==regsha and digest(row['result'])==row['result_digest'];compare(row['result']);results[recipe['motif_id']]=row['result'];reused+=1
  else:missing.append(recipe)
 multiprocessing.set_executable(str(runtime/'python-bundled-host'))
 if missing:
  with multiprocessing.get_context('spawn').Pool(1,initializer=mp._candidate_initializer,initargs=({'context_path':str(OUT/'CONTEXT.json'),'context_sha256':mp.canonical_sha256(context)},)) as pool:
   for got in pool.imap(build,missing,chunksize=1):
    compare(got);key=digest(got['recipe']);write(OUT/'checkpoints'/(key+'.json'),{'registration':regsha,'result':got,'result_digest':digest(got)});results[got['motif_id']]=got;print('SPAWN_NATIVE_RAW',got['motif_id'],flush=True)
 descriptors=list(results.values());beam=mp._farthest_select_descriptors(descriptors,24,must_include_motif_ids={'TWIN:A','TWIN:B'})
 wanted=json.loads((ROOT/'restored/panel_o9/compat_SELECTION.json').read_text());assert [x['motif_id'] for x in beam]==wanted['selected'];assert [x['execution_order_key'] for x in beam]==wanted['selected_order_keys']
 # Rehashed context-policy mismatch must fail before reconstruction or native builds.
 bad=dict(context);bad['execution_policy']=rs.NATIVE_EXECUTION_POLICY;write(OUT/'BAD_CONTEXT.json',bad)
 try:mp._candidate_initializer({'context_path':str(OUT/'BAD_CONTEXT.json'),'context_sha256':mp.canonical_sha256(bad)})
 except RuntimeError:negative=True
 else:raise AssertionError('mismatched context policy accepted')
 write(OUT/'RESULT.json',{'status':'PASS_CONTEXT_SPAWN_RAW_O9_AND_BEAM','executed_raw_builds':len(missing),'reused_raw_checkpoints':reused,'raw_recipes':193,'selected':24,'actual_spawn_pool':True,'workers':1,'production_initializer':True,'authority_fixture':False,'full_physical_witness_identity_comparison':True,'context_policy_negative_rejected':negative,'shared_execution_service_qualified':False,'backbone_under_dev159_qualified':False,'full_p0_closed':False,'independent_host':False,'controller_changed':False})
if __name__=='__main__':
 try:main()
 except BaseException as e:write(OUT/'STOPPED.json',{'error':type(e).__name__,'message':str(e)});raise
