"""Candidate-native source-bound paused checkpoint recovery, 193 O9 chains."""
import pathlib,json,hashlib,sys,importlib.util,resource,shutil,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'integration157';sys.path.insert(0,str(ROOT/'decoder157'))
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
 rows=sorted((ROOT/'restored/o9_qualification/recipes').glob('*.json'));assert len(rows)==193
 executed=reused=steps=0;checks=0;(OUT/'checkpoints').mkdir(exist_ok=True)
 for p in rows:
  row=json.loads(p.read_text());expected=row['result']['compat'];chain=json.loads((ROOT/'restored/o9_qualification/chains'/p.name).read_text());assert digest(chain['steps'])==chain['steps_digest']
  binding={'registration':regsha,'recipe':row['recipe'],'policy':rs.HISTORICAL_EXECUTION_POLICY,'candidate_scanner':sha(ROOT/'decoder157/infinity_grid/regime_scanner.py')};cp=OUT/'checkpoints'/p.name
  if cp.exists():
   saved=json.loads(cp.read_text());assert saved['binding']==binding and digest(saved['result'])==saved['result_digest'];reused+=1
  else:
   s=restore(expected['wire'],rs.HISTORICAL_EXECUTION_POLICY);assert s.construction_digest==expected['construction'] and rs.execution_order_key(s)==expected['order_key'] and observation(s)==expected['physical']
   if executed==0:
    default=restore(expected['wire'],rs.NATIVE_EXECUTION_POLICY);assert default.construction_digest==s.construction_digest and rs.execution_order_key(default)==default.construction_digest;checks+=1
    for badpolicy in ['UNKNOWN','']:
     try:restore(expected['wire'],badpolicy)
     except ValueError:checks+=1
     else:raise AssertionError('unknown policy accepted')
    children=list(s.children);children[0]=default.children[0]
    try:rs.LiftState(engine,s.level,tuple(children),s.top_edges_full,s.lane,s.motif_id,execution_policy=rs.HISTORICAL_EXECUTION_POLICY)
    except ValueError:checks+=1
    else:raise AssertionError('mixed policy accepted')
   got=[]
   for expectedstep in chain['steps']:
    t=expectedstep['type'];s,w=s.reserve_external(t)
    assert rs._path_json(w)==expectedstep['compat_witness'] and list(s.total_caps)==expectedstep['after_caps']
    assert s.construction_digest==expectedstep['compat_v3_key'] and rs.execution_order_key(s)==expectedstep['compat_order_key'];assert digest(observation(s))==expectedstep['compat_physical_digest'];got.append({'type':t,'witness':rs._path_json(w),'construction':s.construction_digest,'execution_order_key':rs.execution_order_key(s)});steps+=1
   assert wire(s)==chain['final_compat_wire'];result={'steps':got,'final_wire_digest':digest(wire(s)),'native_policy_preserved':s.execution_policy==rs.HISTORICAL_EXECUTION_POLICY};write(cp,{'binding':binding,'result':result,'result_digest':digest(result)});executed+=1
  print('NATIVE_CHAIN',row['recipe']['motif_id'],flush=True)
  if args.pause_after is not None and executed>=args.pause_after:break
 result={'status':'PAUSED' if args.pause_after is not None else 'PASS_CANDIDATE_NATIVE_POLICY_CHAIN_RECOVERY','executed_chains':executed,'reused_chains':reused,'new_reservation_steps':steps,'policy_checks':checks,'full_release_integration':False,'parallel_packet_policy_serialization_qualified':False,'native_raw_builds_qualified':False,'independent_host':False,'full_p0_closed':False,'controller_changed':False};write(OUT/'RESULT.json',result)
if __name__=='__main__':
 try:main()
 except BaseException as e:write(OUT/'STOPPED.json',{'error':type(e).__name__,'message':str(e)});raise
