"""Registered O8 disposition/beam and forced backbone comparison; no O9."""
import hashlib,importlib.util,json,os,pathlib,resource,shutil,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder156'),str(ROOT/'compatibility_o8')];OUT=ROOT/'recursive_order'
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
 executed=0;reused=0;changed=0;rows=[]
 assert set(arms['historical'])==set(arms['compat']) and len(arms['compat'])==200
 for motif in sorted(arms['historical']):
  h,n=arms['historical'][motif],arms['compat'][motif];assert historical_order_key(n)==h.construction_digest and observation(h)==observation(n)
  for t in range(7):
   call={'motif':motif,'type':t,'historical_key':h.construction_digest,'v3_key':n.construction_digest};key=digest(call);p=OUT/'cases'/(key+'.json')
   if p.exists():
    row=json.loads(p.read_text());assert row['registration']==regsha and row['call']==call and digest(row['result'])==row['result_digest'];reused+=1
   else:
    hs,hw=h.reserve_external(t);ns,nw=reserve_with_historical_order(n,t,native);default,dw=n.reserve_external(t)
    ho,no=observation(hs),observation(ns);assert ho==no and tuples(hw)==tuples(nw) and historical_order_key(ns)==hs.construction_digest
    expected=list(h.total_caps);expected[t]-=1;assert list(hs.total_caps)==expected and list(ns.total_caps)==expected
    result={'historical_witness':native._path_json(hw),'compat_witness':native._path_json(nw),'default_v3_witness':native._path_json(dw),'default_v3_route_differs':tuples(dw)!=tuples(nw),'historical_output':ho,'compat_output':no,'historical_output_key':hs.construction_digest,'compat_order_key':historical_order_key(ns),'compat_v3_output_key':ns.construction_digest}
    row={'registration':regsha,'call':call,'result':result,'result_digest':digest(result)};p.parent.mkdir(exist_ok=True);write('cases/'+p.name,row);executed+=1
   changed+=row['result']['default_v3_route_differs'];rows.append(key)
  print('RESERVED',motif,7,flush=True)
 result={'status':'PASS_O8_RECURSIVE_ORDERING_COMPATIBILITY','states':200,'raw_candidates':193,'backbone_states':7,'endpoint_types':7,'cases':len(rows),'executed_cases':executed,'reused_cases':reused,'default_v3_route_differences':changed,'nested_witnesses_and_resources_match':True,'v3_identities_preserved':True,'scope':'One Lift layer over O7 leaves; native historical Lift reserve vs explicit compatibility router delegating native O7 reserve and native Lift construction. Deeper recursion not exercised.','O9_dispatched':False,'independent_host':False,'controller_changed':False,'full_p0_closed':False};write('RESULT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:write('STOPPED.json',{'error':type(e).__name__,'message':str(e)});raise
