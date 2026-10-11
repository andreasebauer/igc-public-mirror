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
 reg=json.loads((OUT/'CORRECTION_REGISTRATION.json').read_text());regsha=sha(OUT/'CORRECTION_REGISTRATION.json')
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
 import copy
 regraw=json.loads((OUT/'REGISTRATION.json').read_text());original=sha(OUT/'REGISTRATION.json');paths=sorted((OUT/'checkpoints').glob('*.json'));assert len(paths)==193
 before={p.name:sha(p) for p in paths};rows=[];states=[]
 expected={x['recipe']['motif_id']:x for x in [json.loads(p.read_text()) for p in (ROOT/'restored/o9_qualification/recipes').glob('*.json')]}
 for p in paths:
  row=json.loads(p.read_text());assert row['registration']==original and digest(row['result'])==row['result_digest'];got=row['result'];want=expected[got['motif_id']]['result']['compat'];assert got['construction']==want['construction'] and got['execution_order_key']==want['order_key']
  def normalize(x):
   x=dict(x)
   if 'occurrence' in x:x['source']=records[x.pop('occurrence')]['state_digest']
   if 'children' in x:x['children']=[normalize(c) for c in x['children']]
   return x
  assert got['physical']==normalize(want['physical']);rows.append(got);states.append(restore(want['wire'],rs.HISTORICAL_EXECUTION_POLICY))
 twins=[next(s for s in states if s.motif_id==m) for m in ['TWIN:A','TWIN:B']]
 object_beam=rs._farthest_select(states,24,twins);descriptor_beam=mp._farthest_select_descriptors(rows,24,must_include_motif_ids={'TWIN:A','TWIN:B'});selected=[s.motif_id for s in object_beam];assert selected==[r['motif_id'] for r in descriptor_beam]
 old=json.loads((ROOT/'restored/panel_o9/compat_SELECTION.json').read_text())['selected'];assert selected[:2]==['TWIN:A','TWIN:B'] and old[:2]==['TWIN:B','TWIN:A'] and selected[2:]==old[2:]
 write(OUT/'ORDERING_CORRECTION.json',{'reason':'prior qualification supplied twins in checkpoint filename order instead of frozen recipe order','old_prefix':old[:2],'correct_prefix':selected[:2],'remaining_22_unchanged':True,'membership_unchanged':set(old)==set(selected),'native_object_and_descriptor_order_equal':True,'source_algorithm_changed':False})
 context=json.loads((OUT/'CONTEXT.json').read_text());bad=dict(context);bad['execution_policy']=rs.NATIVE_EXECUTION_POLICY;write(OUT/'BAD_CONTEXT.json',bad)
 try:mp._candidate_initializer({'context_path':str(OUT/'BAD_CONTEXT.json'),'context_sha256':mp.canonical_sha256(bad)})
 except RuntimeError:negative=True
 else:raise AssertionError('mismatched context policy accepted')
 sample=copy.deepcopy(json.loads(paths[0].read_text()));sample['result']['execution_policy']='UNKNOWN'
 assert digest(sample['result'])!=sample['result_digest']
 assert before=={p.name:sha(p) for p in paths}
 write(OUT/'RESULT.json',{'status':'PASS_SPAWN_RAW_AND_CORRECTED_O9_BEAM','raw_recipes':193,'actual_spawn_native_builds':193,'new_builds_during_correction':0,'reused_verified_checkpoints':193,'checkpoint_bytes_unchanged':True,'selected':selected,'actual_production_initializer':True,'authority_fixture':False,'context_policy_negative_rejected':negative,'tamper_rejected':True,'prior_twin_prefix_correction':True,'shared_execution_service_qualified':False,'backbone_under_dev159_qualified':False,'independent_host':False,'full_p0_closed':False,'controller_changed':False})
 print('PASS_SPAWN_RAW_AND_CORRECTED_O9_BEAM',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:write(OUT/'CORRECTION_STOPPED.json',{'error':type(e).__name__,'message':str(e)});raise
