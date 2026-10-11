"""Registered O9 selection and native forced backbone comparison; no O10."""
import hashlib,importlib.util,json,os,pathlib,resource,shutil,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder156'),str(ROOT/'compatibility_o8'),str(ROOT/'recursive_order'),str(ROOT/'o9_qualification')];OUT=ROOT/'panel_o9'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(name,x):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n');q.replace(p)
def snap(s):return {'construction':s.construction_digest,'skin':s.skin,'caps':list(s.total_caps)}
def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x
def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));assert shutil.disk_usage(ROOT).free>=10*1024**3
 reg=json.loads((OUT/'REGISTRATION.json').read_text());regsha=sha(OUT/'REGISTRATION.json')
 for p,h in reg['pins'].items():assert sha(ROOT/p)==h,p
 from infinity_grid.isolated_runtime import verified_runtime
 verified_runtime()
 from infinity_grid.frontier import _load_engine
 from infinity_grid import regime_scanner as native
 from ordering import historical_order_key
 from adapter import reserve_with_historical_order
 spec=importlib.util.spec_from_file_location('sealed_recursive_historical',ROOT/'recovery/g1/decoder/regime_scanner.py');hist=importlib.util.module_from_spec(spec);sys.modules[spec.name]=hist;spec.loader.exec_module(hist)
 engine=_load_engine(next((ROOT/'kernel').iterdir()));parents=engine.load_parent_records()
 records=json.loads(next((ROOT/'material').rglob('07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json')).read_text())['records']
 arms={};ctxs={}
 for arm,mod,directory in [('historical',hist,'matched_run2/historical'),('compat',native,'compatibility_o8/compat')]:
  raw=[json.loads(p.read_text()) for p in (ROOT/directory/'recipes').glob('*.json')];base={};states={}
  for row in raw:
   children=[]
   for c in row['result']['children']:
    i=c['occurrence']
    if i not in base:
     r=records[i];ctx=engine._profile_row_context(r,parents);ctxs[id(ctx)]=i;base[i]=mod.O7State(engine,ctx,tuples(r['edges']),(0,)*7,r['state_digest'],r['lane'])
    b=base[i];children.append(mod.O7State(engine,b.ctx,b.edges,tuple(c['counts']),b.source_id,b.lane))
   s=mod.LiftState(engine,8,tuple(children),tuples(row['result']['edges']),row['call']['lane'],row['call']['motif_id']);assert snap(s)=={k:v for k,v in row['result']['state'].items() if k!='counts'};states[s.motif_id]=s
  for p in sorted((ROOT/'panel_o8').glob(arm+'_BACKBONE_*.json')):
   row=json.loads(p.read_text());children=tuple(mod.O7State(engine,base[126].ctx,base[126].edges,tuple(c['counts']),base[126].source_id,base[126].lane) for c in row['result']['children'])
   s=mod.LiftState(engine,8,children,tuples(row['result']['edges']),'BACKBONE','BACKBONE:'+row['binding']['name']);assert snap(s)==row['result']['state'];states[s.motif_id]=s
  arms[arm]=states
  print('REHYDRATED',arm,len(states),flush=True)
 def observation(s):
  if s.level==7:return {'occurrence':ctxs[id(s.ctx)],'counts':list(s.reserve_counts),'skin':s.skin,'caps':list(s.total_caps)}
  return {'skin':s.skin,'caps':list(s.total_caps),'children':[observation(c) for c in s.children],'edges':[native._edge_json(e) for e in s.top_edges_full]}
 def wire(s):
  if s.level==7:return {'level':7,'occurrence':ctxs[id(s.ctx)],'counts':list(s.reserve_counts)}
  return {'level':s.level,'lane':s.lane,'motif':s.motif_id,'children':[wire(c) for c in s.children],'edges':[native._edge_json(e) for e in s.top_edges_full]}
 bases={a:{ctxs[id(c.ctx)]:c for st in states.values() for c in st.children} for a,states in arms.items()}
 def restore(w,arm,mod):
  if w['level']==7:
   b=bases[arm][w['occurrence']];return mod.O7State(engine,b.ctx,b.edges,tuple(w['counts']),b.source_id,b.lane)
  return mod.LiftState(engine,w['level'],tuple(restore(c,arm,mod) for c in w['children']),tuples(w['edges']),w['lane'],w['motif'])
 from policy import historical_reservation_policy
 selections={a:json.loads((ROOT/'panel_o8'/(a+'_SELECTION.json')).read_text())['selected'] for a in arms}
 selected={a:[arms[a][m] for m in selections[a]] for a in arms}
 ordered={a:sorted(ss,key=(lambda s:s.construction_digest) if a=='historical' else historical_order_key) for a,ss in selected.items()}
 medians={a:__import__('statistics').median(sum(s.total_caps) for s in ss) for a,ss in ordered.items()}
 centers={a:min(ss,key=lambda s:(abs(sum(s.total_caps)-medians[a]),s.construction_digest if a=='historical' else historical_order_key(s))) for a,ss in ordered.items()}
 assert medians['historical']==medians['compat'] and observation(centers['historical'])==observation(centers['compat'])
 pairs=tuples(json.loads((ROOT/'matched_protocol/ORDERED_OPERATORS.json').read_text()));motifs=native.load_motif_library()
 specmp=importlib.util.spec_from_file_location('infinity_grid.sealed_o9_recipes',ROOT/'recovery/g1/decoder/maturation_parallel.py');mp=importlib.util.module_from_spec(specmp);sys.modules[specmp.name]=mp;specmp.loader.exec_module(mp)
 recipes=mp.enumerate_candidate_recipes(motifs,center_supports_twin=centers['historical'].total_caps[0]>=6,pairs_count=len(pairs));assert recipes==json.loads((ROOT/'matched_protocol/ORDERED_RECIPES.json').read_text())['with_twins']
 write('CENTER.json',{'median':medians['historical'],'historical_center':centers['historical'].motif_id,'compat_center':centers['compat'].motif_id,'historical_ordered_parents':[s.motif_id for s in ordered['historical']],'compat_ordered_parents':[s.motif_id for s in ordered['compat']]})
 from ordering import select_with_historical_order
 rawrows=[json.loads(p.read_text()) for p in sorted((ROOT/'o9_qualification/recipes').glob('*.json'))];assert len(rawrows)==193
 panel=native.load_regime_scanner_spec()['panel'];summaries={};executed=0;reused=0
 for arm,mod in [('historical',hist),('compat',native)]:
  states=[restore(row['result'][arm]['wire'],arm,mod) for row in rawrows]
  for state,row in zip(states,rawrows):assert observation(state)==row['result'][arm]['physical'] and state.construction_digest==row['result'][arm]['construction']
  key=(lambda st:st.construction_digest) if arm=='historical' else historical_order_key
  twins=[st for st in states if st.lane=='TWIN'];assert len(twins)==2
  candidates=list(states);cap=int(panel['candidate_cap'])
  if len(candidates)>cap:
   candidates=sorted(candidates,key=key)[:cap]
   for t in twins:
    if all(key(x)!=key(t) for x in candidates):candidates[-1]=t
  selected=mod._farthest_select(candidates,int(panel['beam']),twins) if arm=='historical' else select_with_historical_order(mod,candidates,int(panel['beam']),twins)
  uniq={key(st):st for st in candidates}
  result={'raw':len(states),'cap':cap,'cap_active':len(states)>cap,'after_cap':len(candidates),'after_dedup':len(uniq),'selected':[st.motif_id for st in selected],'selected_order_keys':[key(st) for st in selected],'selected_native_ids':[st.construction_digest for st in selected],'dispositions':[{'motif':st.motif_id,'order_key':key(st),'native_id':st.construction_digest,'selected':st in selected,'kept_by_cap':st in candidates,'dedup_representative':uniq.get(key(st)) is st} for st in states]}
  write(arm+'_SELECTION.json',result);summaries[arm]=result;print('SELECTION',arm,len(selected),flush=True)
 assert summaries['historical']['selected']==summaries['compat']['selected'] and summaries['historical']['selected_order_keys']==summaries['compat']['selected_order_keys']
 backbone=[]
 for name,edges in native.BACKBONE_MOTIFS:
  rows={}
  for arm,mod in [('historical',hist),('compat',native)]:
   filename=arm+'_BACKBONE_'+name+'.json';cp=OUT/filename;binding={'registration':regsha,'arm':arm,'name':name,'edges':edges}
   if cp.exists():
    row=json.loads(cp.read_text());assert row['binding']==json.loads(json.dumps(binding)) and digest(row['result'])==row['result_digest'];reused+=1
   else:
    n=1+max(max(e) for e in edges)
    if arm=='compat':
     with historical_reservation_policy(native):st=mod._build_lift(engine,9,[centers[arm]]*n,list(edges),pairs,'BACKBONE','BACKBONE:'+name,force_pair=(0,0))
    else:st=mod._build_lift(engine,9,[centers[arm]]*n,list(edges),pairs,'BACKBONE','BACKBONE:'+name,force_pair=(0,0))
    assert st is not None
    result={'state':snap(st),'physical':observation(st),'wire':wire(st),'order_key':key(st) if arm=='compat' else st.construction_digest}
    row={'binding':binding,'result':result,'result_digest':digest(result)};write(filename,row);executed+=1
   rows[arm]=row['result']
  h,n=rows['historical'],rows['compat'];assert h['physical']==n['physical'] and h['wire']==n['wire'] and h['order_key']==n['order_key'];assert h['state']['construction']!=n['state']['construction'];backbone.append(name);print('BACKBONE',name,flush=True)
 write('RESULT.json',{'status':'PASS_O9_SELECTION_BACKBONE','raw_recipe_executions':0,'raw_candidates_per_arm':193,'beam':24,'backbone_motifs':backbone,'executed_backbone_builds':executed,'reused_backbone_checkpoints':reused,'mandatory_twins_retained':all(x in summaries['compat']['selected'] for x in ['TWIN:A','TWIN:B']),'exact_order_resources_nested_witnesses':True,'O10_dispatched':False,'full_p0_closed':False,'independent_host':False,'controller_changed':False})
if __name__=='__main__':
 try:main()
 except BaseException as e:write('STOPPED.json',{'error':type(e).__name__,'message':str(e)});raise
