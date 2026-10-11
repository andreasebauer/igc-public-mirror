"""Candidate-native source-bound paused checkpoint recovery, 193 O9 chains."""
import pathlib,json,hashlib,sys,importlib.util,resource,shutil,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'serialization158';sys.path.insert(0,str(ROOT/'decoder158'))
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
 verified_runtime()
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
 from infinity_grid import materialized_discovery as md
 from unittest.mock import patch
 selection=json.loads((ROOT/'restored/panel_o9/compat_SELECTION.json').read_text())['selected']
 rows={row['recipe']['motif_id']:row['result']['compat'] for row in [json.loads(p.read_text()) for p in (ROOT/'restored/o9_qualification/recipes').glob('*.json')]}
 expected=[rows[m] for m in selection]
 for p in sorted((ROOT/'restored/panel_o9').glob('compat_BACKBONE_*.json')):
  row=json.loads(p.read_text())['result'];expected.append({'wire':row['wire'],'physical':row['physical'],'construction':row['state']['construction'],'order_key':row['order_key']})
 assert len(expected)==31
 states=[restore(x['wire'],rs.HISTORICAL_EXECUTION_POLICY) for x in expected]
 dag=mp.state_dag_wire(states);write(OUT/'HISTORICAL_POLICY_DAG.json',dag)
 bases=[restore({'level':7,'occurrence':i,'counts':[0]*7},rs.NATIVE_EXECUTION_POLICY) for i in range(len(records))]
 # Supply the exact recovered authority through the production reader's dependency;
 # neither state construction nor the DAG reconstruction algorithm is replaced.
 authority={'engine':engine,'base_states':bases}
 with patch.object(md,'_get_process_o7_runtime',return_value=authority):
  gotengine,got=mp._states_from_dag(json.loads(json.dumps(dag)));assert gotengine is engine and len(got)==31
  for st,want in zip(got,expected):assert observation(st)==want['physical'] and st.construction_digest==want['construction'] and rs.execution_order_key(st)==want['order_key'] and st.execution_policy==rs.HISTORICAL_EXECUTION_POLICY
  assert mp.state_dag_wire(got)==dag
  native=restore(expected[0]['wire'],rs.NATIVE_EXECUTION_POLICY);nd=mp.state_dag_wire([native]);_,nr=mp._states_from_dag(nd);assert nr[0].execution_policy==rs.NATIVE_EXECUTION_POLICY and nr[0].construction_digest==native.construction_digest
  negatives=[]
  import copy
  cases=[]
  bad=copy.deepcopy(dag);bad.pop('execution_policy');cases.append(('missing_root_policy',bad))
  bad=copy.deepcopy(dag);bad['execution_policy']='UNKNOWN';cases.append(('unknown_root_policy',bad))
  bad=copy.deepcopy(dag);next(iter(bad['nodes'].values())).pop('execution_policy');cases.append(('missing_node_policy',bad))
  bad=copy.deepcopy(dag);next(iter(bad['nodes'].values()))['execution_policy']=rs.NATIVE_EXECUTION_POLICY;cases.append(('mixed_node_policy',bad))
  bad=copy.deepcopy(dag);next(iter(bad['nodes'].values()))['execution_order_key']='0'*64;cases.append(('changed_order_key',bad))
  bad=copy.deepcopy(dag);bad['schema_id']='IG_MATURATION_STATE_DAG_V1';cases.append(('legacy_unbound_schema',bad))
  for name,bad in cases:
   raw=dict(bad);raw.pop('science_sha256',None);bad['science_sha256']=mp.canonical_sha256(raw)
   try:mp._states_from_dag(bad)
   except (RuntimeError,ValueError):negatives.append(name)
   else:raise AssertionError(name+' accepted')
  try:mp.state_dag_wire([states[0],native])
  except ValueError:negatives.append('mixed_root_policies')
  else:raise AssertionError('mixed roots accepted')
 result={'status':'PASS_EXPLICIT_POLICY_DAG_ROUNDTRIP','historical_policy_roots':31,'native_default_roots':1,'dag_nodes':len(dag['nodes']),'negative_cases_rejected':negatives,'same_host':True,'authority_dependency_fixture':True,'actual_spawn_worker_qualified':False,'native_raw_generation_qualified':False,'full_p0_closed':False,'controller_changed':False};write(OUT/'RESULT.json',result);print(result)
if __name__=='__main__':
 try:main()
 except BaseException as e:write(OUT/'STOPPED.json',{'error':type(e).__name__,'message':str(e)});raise
