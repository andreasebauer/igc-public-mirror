from pathlib import Path
import zipfile,io,json,hashlib,re
R=Path.cwd();B=R/'scope_reconcile0171';B.mkdir(exist_ok=True);sha=lambda b:hashlib.sha256(b).hexdigest()
seen={};archives=[];hits=[];errors=[];recover=[]
source_names={'regime_scanner.py','g2_relation.py','uplift_r0.py','uplift_g3_s0.py','uplift_g4_s0.py','uplift_g5_r1.py','uplift_g6.py','uplift_g7.py','uplift_g8.py'}
def walk(z,chain,raw_sha,depth=0):
 if raw_sha in seen:archives.append({'chain':chain,'sha256':raw_sha,'duplicate_of':seen[raw_sha]});return
 seen[raw_sha]=chain;archives.append({'chain':chain,'sha256':raw_sha,'members':len(z.namelist())})
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
  context='/'.join(chain+[n]).lower();name=Path(n).name
  kind=None
  if name in source_names:kind='G_SOURCE_LEAD'
  elif name.endswith('.py') and re.search(r'(uplift|grade|frontier|cap37).*(g[678])|g[678].*(uplift|producer|frontier)',name.lower()):kind='HIGHER_G_SOURCE_LEAD'
  elif 'o7' in context and n.endswith('.json') and (re.search(r'(hom6|twin4)_m7_[2-6]',name.lower()) or any(s in n.lower() for s in ['survivor','profile','checkpoints/','checkpoint/'])):kind='O7_SAVED_DATA_CANDIDATE'
  if not kind:continue
  if info.file_size>32*1024*1024:errors.append({'chain':chain+[n],'reason':'CANDIDATE_MEMBER_SIZE_LIMIT','bytes':info.file_size});continue
  raw=z.read(info);h=sha(raw);hit={'kind':kind,'chain':chain+[n],'sha256':h,'bytes':len(raw)}
  if kind=='O7_SAVED_DATA_CANDIDATE':
   try:
    d=json.loads(raw);v=d.get('result',d) if isinstance(d,dict) else d;hit['json_keys']=list(v)[:25] if isinstance(v,dict) else None
    if isinstance(v,dict) and 'beam' in v:
     hit['beam_rows']=len(v['beam']);hit['literal_edges_present']=all('edges' in x for x in v['beam']);hit['saved_rows_rank_counts']={str(k):sum(len(x.get('edges',[]))==k for x in v['beam']) for k in sorted({len(x.get('edges',[])) for x in v['beam']})}
    if isinstance(v,dict) and any(k in v for k in ['profiles','components','survivors']):hit['profile_or_survivor_keys']=[k for k in ['profiles','components','survivors'] if k in v]
   except Exception as e:hit['parse_error']=str(e)
  dest=B/'recovered_candidates'/h/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);hit['recovered_path']=str(dest.relative_to(R));hits.append(hit)
paths=[R/'project_sources'/n for n in ['07-scout.zip','08-O3.zip','09-Nodes.zip','11-Infinity_Grid_Formal_Foundation_v2.2_CLOSEOUT_CERTIFIED_COMPLETE_BUNDLE_2026-08-27-2-.zip','14-Infinity_Grid_O1_O2_O3_Mathematical_Integration_v1.2_FF2.2_COMPATIBILITY_COMPLETE_BUNDLE_2026-08-27-1-.zip','15-Theory-2.zip','16-Algebra-decoder-1-.zip','17-OScout2.zip','18-OScout-4-.zip','19-Chats-all-0830.zip','20-Infinity_Grid_Algebra_Decoder_v0.26.0_CLEAN_COMPLETE_BUNDLE_2026-08-30-1-.zip']]
paths.extend(sorted((R/'engine/infinity_grid/resources/decoder').glob('O7_MATERIAL_ROOT*.zip')))
for p in paths:
 with zipfile.ZipFile(p) as z:walk(z,[str(p.relative_to(R))],sha(p.read_bytes()))
 print('Scanned',p.name,'unique archives',len(seen),'candidate hits',len(hits),flush=True)
result={'schema':'IG_O7_COVERAGE_G_SOURCE_ARCHIVE_SEARCH_V1','scope':'Eleven supplied science/scout/decoder/chat archives plus the exact O7 material-root ZIP named by current G1 source, recursively,depth<=32;named O7 checkpoint/profile/survivor and G producer patterns;hash-identical ZIPs deduplicated','archives':archives,'hits':hits,'errors':errors,'unique_archives':len(seen),'chain_occurrences':len(archives),'generation_calls':0,'missing_O2_search_repeated':False,'limits':'Filename-directed search is not proof of global absence from Drive or all possible anonymous JSON members.'};(B/'SOURCE_SEARCH.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',len(seen),'unique archives',len(hits),'hits',len(errors),'errors',flush=True)
