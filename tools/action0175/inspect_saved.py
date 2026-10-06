from pathlib import Path
import zipfile,io,json,hashlib,re
R=Path.cwd();B=R/'g_readiness0175';B.mkdir(exist_ok=True);sha=lambda b:hashlib.sha256(b).hexdigest()
archives=[];seen={};hits=[];errors=[]
def walk(z,chain,h,depth=0):
 if h in seen:archives.append({'chain':chain,'sha256':h,'duplicate_of':seen[h]});return
 seen[h]=chain;archives.append({'chain':chain,'sha256':h,'members':len(z.namelist())})
 for info in z.infolist():
  n=info.filename
  if info.is_dir() or '__MACOSX' in Path(n).parts:continue
  if n.lower().endswith('.zip'):
   if depth>=32:errors.append({'chain':chain+[n],'reason':'DEPTH_LIMIT'});continue
   try:
    raw=z.read(info)
    with zipfile.ZipFile(io.BytesIO(raw)) as inner:walk(inner,chain+[n],sha(raw),depth+1)
   except (zipfile.BadZipFile,RuntimeError) as e:errors.append({'chain':chain+[n],'reason':str(e)})
   continue
  context='/'.join(chain+[n]).lower();name=Path(n).name.lower()
  if not n.lower().endswith(('.json','.json.gz','.txt')) or not re.search(r'g1|r100|g_uplift|g2.*population|uplift.*(cohort|population)',context):continue
  if info.file_size>32*1024*1024:errors.append({'chain':chain+[n],'reason':'MEMBER_SIZE_LIMIT','bytes':info.file_size});continue
  raw=z.read(info);hit={'chain':chain+[n],'sha256':sha(raw),'bytes':len(raw)}
  if n.endswith('.json'):
   try:
    d=json.loads(raw);hit['keys']=list(d)[:32] if isinstance(d,dict) else None
    if isinstance(d,dict):
     for k in ['records','carriers','states','population','candidates','rows']:
      v=d.get(k)
      if isinstance(v,list):hit[k+'_count']=len(v);hit[k+'_row_keys']=list(v[0])[:30] if v and isinstance(v[0],dict) else None
   except ValueError as e:hit['parse_error']=str(e)
  p=B/'saved_candidates'/hit['sha256']/Path(n).name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);hit['recovered_path']=str(p.relative_to(R));hits.append(hit)
names=['07-scout.zip','08-O3.zip','09-Nodes.zip','11-Infinity_Grid_Formal_Foundation_v2.2_CLOSEOUT_CERTIFIED_COMPLETE_BUNDLE_2026-08-27-2-.zip','14-Infinity_Grid_O1_O2_O3_Mathematical_Integration_v1.2_FF2.2_COMPATIBILITY_COMPLETE_BUNDLE_2026-08-27-1-.zip','15-Theory-2.zip','16-Algebra-decoder-1-.zip','17-OScout2.zip','18-OScout-4-.zip','19-Chats-all-0830.zip','20-Infinity_Grid_Algebra_Decoder_v0.26.0_CLEAN_COMPLETE_BUNDLE_2026-08-30-1-.zip']
for name in names:
 p=R/'project_sources'/name
 with zipfile.ZipFile(p) as z:walk(z,[str(p.relative_to(R))],sha(p.read_bytes()))
 print('Scanned',name,len(seen),'archives',len(hits),'hits',flush=True)
out={'schema':'IG_BOUNDED_SAVED_G1_G2_SOURCE_SEARCH_V1','scope':'Eleven supplied science/scout/decoder/chat ZIPs,recursive depth32,filename/context-directed G1/R100/G_UPLIFT/G2 population patterns;no anonymous JSON census and no Drive-wide absence claim','archives':archives,'hits':hits,'errors':errors,'unique_archives':len(seen),'chain_occurrences':len(archives),'candidate_hits':len(hits),'generation_calls':0,'missing_O2_search_repeated':False}
(B/'SAVED_SOURCE_SEARCH.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'BOUNDED_SEARCH_COMPLETE','archives':len(seen),'hits':len(hits),'errors':len(errors)}))
