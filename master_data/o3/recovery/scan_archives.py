from pathlib import Path
import zipfile,io,json,hashlib,re,time
B=Path(__file__).resolve().parent;R=B.parent;L=R/'o3_readiness0153/sources/Infinity_Grid_SCOUT3_V1_0_FROZEN_LAUNCH_BUNDLE_2026-08-27';expected={int(n[1:4]):h for h,n in (x.split() for x in (L/'inputs/O2_SOURCE_PANEL_SHA256.txt').read_text().splitlines())};prior=json.load(open(R/'o3_readiness0153/READINESS_VALIDATION.json'));missing=set(prior['missing_k']);archives=[];hits=[];errors=[];start=time.time()
def walk(z,chain,depth=0):
 names=z.namelist();archives.append({'chain':chain,'members':len(names)})
 for n in names:
  if n.startswith('__MACOSX/') or '/__MACOSX/' in n:continue
  match=re.search(r'k(\d{3})_selected_carriers\.json\.gz$',n)
  if match:
   k=int(match.group(1));raw=z.read(n);h=hashlib.sha256(raw).hexdigest();ok=h==expected.get(k);hits.append({'k':k,'chain':chain+[n],'sha256':h,'expected_match':ok,'bytes':len(raw)})
   if ok and k in missing:(B/'recovered_panels'/f'k{k:03d}_selected_carriers.json.gz').write_bytes(raw)
  elif n.lower().endswith('.zip'):
   if depth>=32:errors.append({'chain':chain+[n],'reason':'DEPTH_LIMIT'});continue
   try:
    raw=z.read(n)
    with zipfile.ZipFile(io.BytesIO(raw)) as inner:walk(inner,chain+[n],depth+1)
   except (zipfile.BadZipFile,RuntimeError) as e:errors.append({'chain':chain+[n],'reason':str(e)})
 if len(archives)%20==0:print({'archives':len(archives),'hits':len(hits),'elapsed_seconds':round(time.time()-start)},flush=True)
paths=[R/'project_sources'/n for n in ['07-scout.zip','08-O3.zip','09-Nodes.zip','11-Infinity_Grid_Formal_Foundation_v2.2_CLOSEOUT_CERTIFIED_COMPLETE_BUNDLE_2026-08-27-2-.zip','14-Infinity_Grid_O1_O2_O3_Mathematical_Integration_v1.2_FF2.2_COMPATIBILITY_COMPLETE_BUNDLE_2026-08-27-1-.zip','15-Theory-2.zip','16-Algebra-decoder-1-.zip','17-OScout2.zip','18-OScout-4-.zip','20-Infinity_Grid_Algebra_Decoder_v0.26.0_CLEAN_COMPLETE_BUNDLE_2026-08-30-1-.zip']]
paths.append(R/'project_sources/19-Chats-all-0830.zip')
paths.extend(sorted((B/'library_sources').rglob('*.zip')))
for p in paths:
 if not p.exists():errors.append({'chain':[p.name],'reason':'MISSING_LOCAL_FILE'});continue
 with zipfile.ZipFile(p) as z:walk(z,[p.name])
 print({'outer_complete':p.name,'archives':len(archives),'hits':len(hits)},flush=True)
res={'schema':'IG_MISSING_O2_EXACT_PANEL_ARCHIVE_SEARCH_V1','search_scope':'Eleven available producer/foundation/O/scout/decoder/chat attachment archives and two saved Scout2-B restart bundles recursively, depth<=32; filenames matching kNNN_selected_carriers.json.gz; no generation','archives':archives,'hits':hits,'errors':errors,'new_missing_panel_matches':sorted({h['k'] for h in hits if h['k'] in missing and h['expected_match']}),'prior_available':53,'required':129,'missing_before':sorted(missing),'generator_calls':0};(B/'ARCHIVE_SEARCH.json').write_text(json.dumps(res,indent=2));print({'new_recovered':res['new_missing_panel_matches'],'archives':len(archives),'errors':len(errors)},flush=True)
