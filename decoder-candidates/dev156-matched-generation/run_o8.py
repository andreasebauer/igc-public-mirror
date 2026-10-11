"""Traced native O7 selection/O8 generation; durable missing-only recipe reuse."""
import hashlib,importlib.util,json,os,pathlib,resource,shutil,statistics,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'decoder156'))
OUT=ROOT/'matched_run'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(x,s):
 if not x:raise RuntimeError(s)
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
 with q.open('w') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 q.replace(p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def snap(s):return {'construction':s.construction_digest,'skin':s.skin,'caps':list(s.total_caps),'counts':list(s.reserve_counts) if hasattr(s,'reserve_counts') else None}
def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));check(shutil.disk_usage(ROOT).free>=10*1024**3,'SPACE')
 reg=json.loads((OUT/'REGISTRATION.json').read_text());regsha=sha(OUT/'REGISTRATION.json')
 for n,h in reg['pins'].items():check(sha(ROOT/n)==h,'PIN:'+n)
 from infinity_grid.isolated_runtime import verified_runtime
 verified_runtime()
 manifest=json.loads((ROOT/'candidate156/dev156/SOURCE_MANIFEST.json').read_text())['candidate_files']
 actual={p.relative_to(ROOT/'decoder156').as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (ROOT/'decoder156').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'};check(actual==manifest,'SOURCE')
 from infinity_grid.frontier import _load_engine
 from infinity_grid import regime_scanner as native
 engine=_load_engine(next((ROOT/'kernel').iterdir()));parents=engine.load_parent_records()
 spec0=importlib.util.spec_from_file_location('sealed_matched_historical',ROOT/'recovery/g1/decoder/regime_scanner.py');hist=importlib.util.module_from_spec(spec0);sys.modules[spec0.name]=hist;spec0.loader.exec_module(hist)
 records=json.loads(next((ROOT/'material').rglob('07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json')).read_text())['records'];pairs=[tuple(x) for x in json.loads((ROOT/'matched_protocol/ORDERED_OPERATORS.json').read_text())];motifs=native.load_motif_library();spec=native.load_regime_scanner_spec()
 check(len(records)==205,'COHORT');summaries={};base_refs={};executed=0;reused=0
 for arm,mod in [('historical',hist),('dev156',native)]:
  base=[];ctxindex={}
  for i,r in enumerate(records):
   ctx=engine._profile_row_context(r,parents);ctxindex[id(ctx)]=i;s=mod.O7State(engine,ctx,tuple(map(tuple,r['edges'])),(0,)*7,r['state_digest'],r['lane']);base.append(s)
   obs=snap(s)
   if arm=='historical':base_refs[i]=obs
   else:check(obs['skin']==base_refs[i]['skin'] and obs['caps']==base_refs[i]['caps'],'O7_RESOURCE:'+str(i))
   print('BASE',arm,i+1,flush=True)
  write(OUT/arm/'BASE.json',[{'index':i,**snap(s)} for i,s in enumerate(base)])
  original_select=mod._farthest_select;selection=[]
  def select(states,k,must_include=None):
   trace={};code=original_select.__code__
   def tr(frame,event,arg):
    if frame.f_code is code and event=='return':
     loc=frame.f_locals;trace.update({'pool':[s.construction_digest for s in loc.get('pool',[])],'vectors':loc.get('vec'),'normalized':loc.get('norm'),'chosen_indices':loc.get('chosen'),'k':k,'must_include':[s.construction_digest for s in must_include or []]})
    return tr
   sys.settrace(tr)
   try:result=original_select(states,k,must_include)
   finally:sys.settrace(None)
   trace['selected']=[s.construction_digest for s in result];selection.append(trace);return result
  mod._farthest_select=select
  selected=select(base,spec['panel']['beam']);write(OUT/arm/'O7_SELECTION.json',selection[-1])
  sortedprev=sorted(selected,key=lambda s:s.construction_digest);med=statistics.median(sum(s.total_caps) for s in sortedprev);center=min(sortedprev,key=lambda s:(abs(sum(s.total_caps)-med),s.construction_digest))
  write(OUT/arm/'CENTER.json',{'median':med,'center':snap(center),'ordered_parent_constructions':[s.construction_digest for s in sortedprev],'ordered_parent_occurrences':[ctxindex[id(s.ctx)] for s in sortedprev]})
  original_build=mod._build_lift;original_reserve=mod.O7State.reserve_external;active=None;raw=[]
  def reserve(s,t):
   before=snap(s);ns,w=original_reserve(s,t);after=snap(ns);expected=list(before['caps']);expected[t]-=1;check(after['caps']==expected,'RESERVATION_CAPACITY')
   check(list(w)==[t,before['counts'][t]],'RESERVATION_WITNESS')
   if active is not None:active.append({'before':before,'type':t,'witness':list(w),'after':after,'occurrence':ctxindex[id(s.ctx)]})
   return ns,w
  mod.O7State.reserve_external=reserve
  def build(engine0,level,owners,edges,pairs0,lane,motif_id,schedule_seed=0,force_pair=None):
   nonlocal active,executed,reused
   call={'level':level,'owners':[{'occurrence':ctxindex[id(s.ctx)],**snap(s)} for s in owners],'edges':edges,'operators':pairs0,'lane':lane,'motif_id':motif_id,'seed':schedule_seed,'force_pair':force_pair};key=digest(call);p=OUT/arm/'recipes'/(key+'.json')
   if p.exists():
    row=json.loads(p.read_text());check(row['registration']==regsha and row['call_digest']==key and digest(row['result'])==row['result_digest'],'CHECKPOINT')
    result=row['result'];reused+=1
    if result['state'] is None:state=None
    else:
     children=tuple(mod.O7State(engine,base[x['occurrence']].ctx,base[x['occurrence']].edges,tuple(x['counts']),base[x['occurrence']].source_id,base[x['occurrence']].lane) for x in result['children'])
     def tup(x):return tuple(tup(v) for v in x) if isinstance(x,list) else x
     state=mod.LiftState(engine,level,children,tup(result['edges']),lane,motif_id);check(snap(state)==result['state'],'RESTORED_RECIPE_STATE')
   else:
    active=[];state=original_build(engine0,level,owners,edges,pairs0,lane,motif_id,schedule_seed,force_pair);events=active;active=None
    sortededges=sorted(tuple(sorted(map(int,e))) for e in edges);choices=[];caps=[list(s.total_caps) for s in owners]
    for idx,(u,v) in enumerate(sortededges):
     candidates=[force_pair] if force_pair is not None else [pairs0[(idx+schedule_seed+j)%len(pairs0)] for j in range(len(pairs0))];chosen=None;tests=[]
     for a,b in candidates:
      ok=caps[u][a]>0 and caps[v][b]>0;tests.append({'pair':[a,b],'legal_capacity':ok})
      if ok:chosen=(a,b);break
     choices.append({'edge':[u,v],'tests':tests,'chosen':chosen})
     if chosen is None:break
     caps[u][chosen[0]]-=1;caps[v][chosen[1]]-=1
    if state is not None:
     check(len(events)==2*len(sortededges),'EVENT_CENSUS');check([list(c.total_caps) for c in state.children]==caps,'RAW_CAPS')
     for i,e in enumerate(state.top_edges_full):check(list(e[4])==events[2*i]['witness'] and list(e[5])==events[2*i+1]['witness'] and list(e[2:4])==list(choices[i]['chosen']),'RAW_EDGE_WITNESS')
    result={'state':snap(state) if state else None,'children':[{'occurrence':ctxindex[id(c.ctx)],'counts':list(c.reserve_counts)} for c in state.children] if state else None,'edges':mod._edge_json if False else [mod._edge_json(e) for e in state.top_edges_full] if state else None,'reservation_events':events,'operator_choices':choices,'failed_edge':choices[-1]['edge'] if state is None else None}
    row={'registration':regsha,'call_digest':key,'call':call,'result':result,'result_digest':digest(result)};write(p,row);executed+=1
   raw.append(row);print('RECIPE',arm,len(raw),lane,motif_id,'REUSE' if p.exists() and row['result']==result and False else 'DONE',flush=True);return state
  mod._build_lift=build
  try:beam,meta,backbone=mod._build_next_panel(engine,selected,8,pairs,motifs,spec)
  finally:mod._build_lift=original_build;mod.O7State.reserve_external=original_reserve;mod._farthest_select=original_select
  candidates=[r for r in raw if r['call']['lane']!='BACKBONE'];check(len(candidates) in (191,193),'RAW_RECIPE_CENSUS');check(len(candidates)<=spec['panel']['candidate_cap'],'NO_CAP_EXPECTATION')
  summary={'raw_recipes':len(candidates),'successes':sum(r['result']['state'] is not None for r in candidates),'failures':sum(r['result']['state'] is None for r in candidates),'raw_calls':[{ 'motif':r['call']['motif_id'],'call_digest':r['call_digest'],'result_digest':r['result_digest']} for r in raw],'o7_beam_occurrences':[ctxindex[id(s.ctx)] for s in selected],'o8_beam':[snap(s) for s in beam],'backbone':[snap(s) for s in backbone],'meta':meta,'selection_traces':selection,'center_occurrence':ctxindex[id(center.ctx)]}
  write(OUT/arm/'SUMMARY.json',summary);summaries[arm]=summary
 h,n=summaries['historical'],summaries['dev156'];same=h['o7_beam_occurrences']==n['o7_beam_occurrences'];center_same=h['center_occurrence']==n['center_occurrence']
 diagnosis={'status':'MATCHED_GENERATION_EXECUTED_DIFFERENCES_REQUIRE_REVIEW','identical_O7_resource_observations':205,'same_ordered_o7_beam_occurrences':same,'same_center_occurrence':center_same,'historical_o7_beam':h['o7_beam_occurrences'],'dev156_o7_beam':n['o7_beam_occurrences'],'historical_center':h['center_occurrence'],'dev156_center':n['center_occurrence'],'raw_recipe_counts':[h['raw_recipes'],n['raw_recipes']],'raw_success_counts':[h['successes'],n['successes']],'executed_recipes':executed,'reused_recipes':reused,'registered_source_unchanged':True,'full_p0_closed':False,'independent_host':False,'O9_dispatched':False,'historical_retained_DAG_match_not_yet_qualified':True,'next':'Diagnose first O7 ordering/selection difference before continuation.'}
 write(OUT/'RESULT.json',diagnosis);print('FINISHED',json.dumps(diagnosis),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(OUT/'STOPPED.json',{'error':type(e).__name__,'message':str(e),'full_p0_closed':False});raise
