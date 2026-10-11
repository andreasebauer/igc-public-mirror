"""Registered O8 disposition/beam and forced backbone comparison; no O9."""
import hashlib,importlib.util,json,os,pathlib,resource,shutil,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder156'),str(ROOT/'compatibility_o8'),str(ROOT/'recursive_order')];OUT=ROOT/'o9_qualification'
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
 executed=0;reused=0;chain_executed=0;chain_reused=0
 for recipe in recipes:
  key=digest(recipe);p=OUT/'recipes'/(key+'.json');p.parent.mkdir(exist_ok=True)
  if p.exists():
   row=json.loads(p.read_text());assert row['registration']==regsha and row['recipe']==recipe and digest(row['result'])==row['result_digest'];outputs={a:restore(row['result'][a]['wire'],a,hist if a=='historical' else native) for a in arms};reused+=1
  else:
   outputs={}
   for arm,mod in [('historical',hist),('compat',native)]:
    mp.rs=mod;mp.install_candidate_context(engine=engine,prev=ordered[arm],level=9,pairs=pairs,motifs=motifs,center=centers[arm])
    if arm=='compat':
     with historical_reservation_policy(native):outputs[arm]=mp._build_recipe_state(recipe)
    else:outputs[arm]=mp._build_recipe_state(recipe)
    mp.clear_candidate_context();assert outputs[arm] is not None
   h,n=outputs['historical'],outputs['compat'];assert observation(h)==observation(n) and h.construction_digest==historical_order_key(n)
   result={a:{'wire':wire(s),'physical':observation(s),'construction':s.construction_digest,'order_key':s.construction_digest if a=='historical' else historical_order_key(s)} for a,s in outputs.items()}
   row={'registration':regsha,'recipe':recipe,'result':result,'result_digest':digest(result)};write('recipes/'+p.name,row);executed+=1
  h,n=outputs['historical'],outputs['compat'];assert observation(h)==observation(n) and h.construction_digest==historical_order_key(n)
  cp=OUT/'chains'/(key+'.json');cp.parent.mkdir(exist_ok=True)
  if cp.exists():
   chain=json.loads(cp.read_text());assert chain['registration']==regsha and chain['recipe_digest']==key and digest(chain['steps'])==chain['steps_digest'];chain_reused+=1
  else:
   steps=[]
   with historical_reservation_policy(native):
    for t in reg['repeated_sequence']:
     before=list(h.total_caps);h,hw=h.reserve_external(t);n,nw=n.reserve_external(t);ho,no=observation(h),observation(n)
     assert ho==no and tuples(hw)==tuples(nw) and h.construction_digest==historical_order_key(n)
     expected=before[:];expected[t]-=1;assert list(h.total_caps)==expected
     steps.append({'type':t,'before_caps':before,'after_caps':list(h.total_caps),'historical_witness':native._path_json(hw),'compat_witness':native._path_json(nw),'historical_physical_digest':digest(ho),'compat_physical_digest':digest(no),'historical_output_key':h.construction_digest,'compat_order_key':historical_order_key(n),'compat_v3_key':n.construction_digest})
   chain={'registration':regsha,'recipe_digest':key,'steps':steps,'steps_digest':digest(steps),'final_historical_wire':wire(h),'final_compat_wire':wire(n)};write('chains/'+cp.name,chain);chain_executed+=1
  print('O9_RECIPE_CHAIN',recipe['motif_id'],flush=True)
 result={'status':'PASS_INTEGRATED_O9_RAW_AND_REPEATED_DEPTH2_RESERVATIONS','raw_recipes':len(recipes),'executed_recipe_pairs':executed,'reused_recipe_pairs':reused,'executed_chains':chain_executed,'reused_chains':chain_reused,'chain_steps':len(recipes)*len(reg['repeated_sequence']),'O9_dispatched':True,'O9_selection_and_backbone_dispatched':False,'two_lift_layers_exercised':True,'native_v3_identity_preserved':True,'scope':'Isolated process-local native Lift reserve policy; O9 raw candidates and fourteen repeated reservations per candidate; no release/controller integration.','independent_host':False,'full_p0_closed':False,'controller_changed':False};write('RESULT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:write('STOPPED.json',{'error':type(e).__name__,'message':str(e)});raise
