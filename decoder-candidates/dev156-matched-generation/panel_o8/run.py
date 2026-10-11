"""Registered O8 disposition/beam and forced backbone comparison; no O9."""
import hashlib,importlib.util,json,os,pathlib,resource,shutil,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'decoder156'),str(ROOT/'compatibility_o8')];OUT=ROOT/'panel_o8'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(name,x):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n');q.replace(p)
def snap(s):return {'construction':s.construction_digest,'skin':s.skin,'caps':list(s.total_caps)}
def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x
def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));assert shutil.disk_usage(ROOT).free>=10*1024**3
 reg=json.loads((OUT/'REGISTRATION.json').read_text())
 for p,h in reg['pins'].items():assert sha(ROOT/p)==h,p
 from infinity_grid.isolated_runtime import verified_runtime
 verified_runtime()
 from infinity_grid.frontier import _load_engine
 from infinity_grid import regime_scanner as native
 from ordering import historical_order_key,select_with_historical_order
 spec=importlib.util.spec_from_file_location('sealed_panel_historical',ROOT/'recovery/g1/decoder/regime_scanner.py');hist=importlib.util.module_from_spec(spec);sys.modules[spec.name]=hist;spec.loader.exec_module(hist)
 engine=_load_engine(next((ROOT/'kernel').iterdir()));parents=engine.load_parent_records()
 records=json.loads(next((ROOT/'material').rglob('07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json')).read_text())['records']
 pairs=tuples(json.loads((ROOT/'matched_protocol/ORDERED_OPERATORS.json').read_text()));panel=native.load_regime_scanner_spec()['panel']
 summary={};centers={};allstates={}
 for arm,mod,directory in [('historical',hist,'matched_run2/historical'),('compat',native,'compatibility_o8/compat')]:
  raw=[json.loads(p.read_text()) for p in (ROOT/directory/'recipes').glob('*.json')];by_motif={r['call']['motif_id']:r for r in raw}
  recipes=json.loads((ROOT/'matched_protocol/ORDERED_RECIPES.json').read_text())['with_twins'];raw=[by_motif[r['motif_id']] for r in recipes];assert len(raw)==193
  indices=sorted({c['occurrence'] for r in raw for c in r['result']['children']});base={}
  for i in indices:
   record=records[i];ctx=engine._profile_row_context(record,parents);base[i]=mod.O7State(engine,ctx,tuples(record['edges']),(0,)*7,record['state_digest'],record['lane'])
  states=[]
  for row in raw:
   children=tuple(mod.O7State(engine,base[c['occurrence']].ctx,base[c['occurrence']].edges,tuple(c['counts']),base[c['occurrence']].source_id,base[c['occurrence']].lane) for c in row['result']['children'])
   state=mod.LiftState(engine,8,children,tuples(row['result']['edges']),row['call']['lane'],row['call']['motif_id']);assert snap(state)=={k:v for k,v in row['result']['state'].items() if k!='counts'};states.append(state)
  key=(lambda s:s.construction_digest) if arm=='historical' else historical_order_key
  twins=[s for s in states if s.lane=='TWIN'];assert len(twins)==2
  cap=int(panel['candidate_cap']);candidates=list(states)
  if len(candidates)>cap:
   candidates=sorted(candidates,key=key)[:cap]
   for t in twins:
    if all(key(x)!=key(t) for x in candidates):candidates[-1]=t
  selected=mod._farthest_select(candidates,int(panel['beam']),twins) if arm=='historical' else select_with_historical_order(mod,candidates,int(panel['beam']),twins)
  uniq={key(s):s for s in candidates}
  result={'raw':193,'cap':cap,'cap_active':len(states)>cap,'after_cap':len(candidates),'after_dedup':len(uniq),'twins':[s.motif_id for s in twins],'selected':[s.motif_id for s in selected],'selected_order_keys':[key(s) for s in selected],'selected_native_ids':[s.construction_digest for s in selected],'dispositions':[{'motif':s.motif_id,'order_key':key(s),'native_id':s.construction_digest,'selected':s in selected,'kept_by_cap':s in candidates,'dedup_representative':uniq.get(key(s)) is s} for s in states]}
  write(arm+'_SELECTION.json',result);summary[arm]=result;allstates[arm]=selected;centers[arm]=base[126]
  print('SELECTION',arm,len(uniq),[s.motif_id for s in selected],flush=True)
 assert summary['historical']['selected']==summary['compat']['selected']
 assert summary['historical']['selected_order_keys']==summary['compat']['selected_order_keys']
 # The historical slim scanner has no backbone driver. Explicit frozen motifs
 # are built through each arm's actual _build_lift; no full-driver restoration claim.
 backbone=[]
 for name,edges in native.BACKBONE_MOTIFS:
  rows={}
  for arm,mod in [('historical',hist),('compat',native)]:
   checkpoint=OUT/(arm+'_BACKBONE_'+name+'.json')
   binding={'registration':sha(OUT/'REGISTRATION.json'),'arm':arm,'name':name,'edges':edges}
   if checkpoint.exists():
    row=json.loads(checkpoint.read_text());assert row['binding']==json.loads(json.dumps(binding)) and digest(row['result'])==row['result_digest'];print('BACKBONE_REUSED',arm,name,flush=True)
   else:
    n=1+max(max(e) for e in edges);s=mod._build_lift(engine,8,[centers[arm]]*n,list(edges),pairs,'BACKBONE','BACKBONE:'+name,force_pair=(0,0));assert s is not None
    result={'state':snap(s),'order_key':historical_order_key(s) if arm=='compat' else s.construction_digest,'children':[{'skin':c.skin,'caps':list(c.total_caps),'counts':list(c.reserve_counts)} for c in s.children],'edges':[mod._edge_json(e) for e in s.top_edges_full]}
    row={'binding':binding,'result':result,'result_digest':digest(result)};write(checkpoint.name,row);print('BACKBONE_EXECUTED',arm,name,flush=True)
   rows[arm]=row['result']
  h,n=rows['historical'],rows['compat'];assert h['state']['skin']==n['state']['skin'] and h['state']['caps']==n['state']['caps'] and h['children']==n['children'] and h['edges']==n['edges'] and h['order_key']==n['order_key'];backbone.append(name)
 result={'status':'PASS_O8_CAP_DEDUP_BEAM_BACKBONE_COMPATIBILITY','raw':193,'cap_active':summary['compat']['cap_active'],'dedup':summary['compat']['after_dedup'],'beam':len(allstates['compat']),'mandatory_twins_retained':all(s in summary['compat']['selected'] for s in ['TWIN:A','TWIN:B']),'backbone_motifs':backbone,'exact_selection_order_and_backbone_resources_witnesses':True,'raw_recipe_executions':0,'O9_dispatched':False,'recursive_lift_reservation_qualified':False,'independent_host':False,'full_p0_closed':False,'controller_changed':False};write('RESULT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:write('STOPPED.json',{'error':type(e).__name__,'message':str(e)});raise
