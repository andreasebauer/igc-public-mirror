"""Fresh-process saved O9 root restoration. No candidate builders or reservations."""
import pathlib,json,hashlib,sys,importlib.util,resource,shutil,copy
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'cold_o9';sys.path[:0]=[str(ROOT/'decoder156'),str(ROOT/'restored/compatibility_o8')]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x
def write(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n');q.replace(p)
def main():
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));assert shutil.disk_usage(ROOT).free>=10*1024**3
 reg=json.loads((OUT/'REGISTRATION.json').read_text());regsha=sha(OUT/'REGISTRATION.json')
 for p,h in reg['pins'].items():assert sha(ROOT/p)==h,p
 from infinity_grid.isolated_runtime import verified_runtime
 runtime=verified_runtime()
 from infinity_grid.frontier import _load_engine
 from infinity_grid import regime_scanner as native
 from ordering import historical_order_key
 spec=importlib.util.spec_from_file_location('cold_historical',ROOT/'recovery/g1/decoder/regime_scanner.py');hist=importlib.util.module_from_spec(spec);sys.modules[spec.name]=hist;spec.loader.exec_module(hist)
 engine=_load_engine(next((ROOT/'kernel').iterdir()));parents=engine.load_parent_records();records=json.loads(next((ROOT/'material').rglob('07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json')).read_text())['records'];contexts={};occurrences={}
 def restore(w,mod):
  if w['level']==7:
   i=w['occurrence'];r=records[i]
   if i not in contexts:contexts[i]=engine._profile_row_context(r,parents);occurrences[id(contexts[i])]=i
   return mod.O7State(engine,contexts[i],tuples(r['edges']),tuple(w['counts']),r['state_digest'],r['lane'])
  return mod.LiftState(engine,w['level'],tuple(restore(c,mod) for c in w['children']),tuples(w['edges']),w['lane'],w['motif'])
 def observation(s):
  if s.level==7:return {'occurrence':occurrences[id(s.ctx)],'counts':list(s.reserve_counts),'skin':s.skin,'caps':list(s.total_caps)}
  return {'skin':s.skin,'caps':list(s.total_caps),'children':[observation(c) for c in s.children],'edges':[native._edge_json(e) for e in s.top_edges_full]}
 def endpoint(w,path,t):
  if w['level']==7:
   assert path[0]==t and 0<=path[1]<w['counts'][t];return 1
  index,local,kind=path;assert kind==t and 0<=index<len(w['children']);return endpoint(w['children'][index],local,t)
 def witness_check(w):
  if w['level']==7:return 0
  n=0
  for u,v,a,b,wu,wv in w['edges']:n+=endpoint(w['children'][u],wu,a)+endpoint(w['children'][v],wv,b)
  return n+sum(witness_check(c) for c in w['children'])
 rows={r['recipe']['motif_id']:r for r in [json.loads(p.read_text()) for p in (ROOT/'restored/o9_qualification/recipes').glob('*.json')]};jobs=[]
 for arm,mod in [('historical',hist),('compat',native)]:
  selection=json.loads((ROOT/'restored/panel_o9'/(arm+'_SELECTION.json')).read_text());assert len(selection['selected'])==24
  for motif,key,nativeid in zip(selection['selected'],selection['selected_order_keys'],selection['selected_native_ids']):
   row=rows[motif];assert digest(row['result'])==row['result_digest'];r=row['result'][arm];assert r['order_key']==key and r['construction']==nativeid;jobs.append((arm,mod,motif,r))
  for p in sorted((ROOT/'restored/panel_o9').glob(arm+'_BACKBONE_*.json')):
   row=json.loads(p.read_text());assert digest(row['result'])==row['result_digest'];r=row['result'];jobs.append((arm,mod,'BACKBONE:'+row['binding']['name'],{'wire':r['wire'],'physical':r['physical'],'construction':r['state']['construction'],'order_key':r['order_key']}))
 assert len(jobs)==62
 (OUT/'roots').mkdir(exist_ok=True);results={};executed=0;reused=0;events=0
 for arm,mod,motif,expected in jobs:
  ident=digest({'arm':arm,'motif':motif});p=OUT/'roots'/(ident+'.json');binding={'registration':regsha,'arm':arm,'motif':motif,'expected':digest(expected)}
  if p.exists():
   row=json.loads(p.read_text());assert row['binding']==binding and digest(row['result'])==row['result_digest'];result=row['result'];reused+=1
  else:
   st=restore(expected['wire'],mod);physical=observation(st);key=st.construction_digest if arm=='historical' else historical_order_key(st)
   assert physical==expected['physical'] and st.construction_digest==expected['construction'] and key==expected['order_key']
   count=witness_check(expected['wire']);result={'physical_digest':digest(physical),'construction':st.construction_digest,'order_key':key,'witness_endpoints_checked':count};write(p,{'binding':binding,'result':result,'result_digest':digest(result)});executed+=1
  results[(arm,motif)]=result;events+=result['witness_endpoints_checked'];print('COLD_ROOT',arm,motif,flush=True)
 for arm,mod,motif,expected in jobs:
  if arm!='historical':continue
  h,n=results[(arm,motif)],results[('compat',motif)];assert h['physical_digest']==n['physical_digest'] and h['order_key']==n['order_key'] and h['construction']!=n['construction']
 bad=copy.deepcopy(jobs[0][3]['wire']);bad['edges'][0][4][0]=len(bad['children'][bad['edges'][0][0]]['children'])
 try:witness_check(bad)
 except (AssertionError,IndexError):pass
 else:raise AssertionError('tamper accepted')
 write(OUT/'RESULT.json',{'status':'PASS_COLD_O9_ROOT_RESTORATION','roots':62,'selected_roots':48,'backbone_roots':14,'executed_restorations':executed,'reused_restorations':reused,'witness_endpoints_checked':events,'tamper_rejected':1,'runtime':str(runtime),'candidate_builds':0,'reservation_calls':0,'native_v3_ids_preserved':True,'independent_host':False,'full_p0_closed':False,'controller_changed':False,'scope':'Fresh workspace and fresh runtime; full saved root skin/capacity/counter/identity reconstruction and nested witness bound checks. Same host, same native skin implementations; no independent skin algorithm or full raw-generation closure claim.'})
if __name__=='__main__':
 try:main()
 except BaseException as e:write(OUT/'STOPPED.json',{'error':type(e).__name__,'message':str(e)});raise
