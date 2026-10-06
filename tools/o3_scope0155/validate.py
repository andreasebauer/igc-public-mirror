from pathlib import Path
import ast,collections,functools,itertools,math,hashlib,json,gzip,importlib.util
B=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
roots=list((B/'sources').glob('*/*'))
primary=next(p for p in roots if 'V1_3' in str(p))
checks=[]
for root in roots:
 m=root/'provenance/BUNDLE_MANIFEST.json'
 if not m.exists():continue
 entries=json.load(open(m))['files']
 for e in entries:
  p=root/e['path'];assert p.stat().st_size==e['bytes'] and sha(p.read_bytes())==e['sha256'],p
 checks.append({'source':str(root.relative_to(B)),'files':len(entries),'status':'PASS'})
code=primary/'code/o3_graduation_classification_audit_v1_3.py'
spec=importlib.util.spec_from_file_location('frozen_o3_audit',code);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
a=mod.Audit();assert not a.source_checks()
summary,failures,components,witness,roll=a.full_sweep();assert not failures,failures
bank=primary/'inputs/O2_SOURCE_BANK.json'
frozen=next(Path('o3_readiness0153/sources').rglob('O2_SOURCE_BANK.json'))
assert bank.read_bytes()==frozen.read_bytes()
# Load only pure original identity functions: no producer run or candidate generation.
orig=next((B/'sources').glob('*V1_1*/*/parent_v1_1_snapshot/code/scout3_o3_roadmap_v1_0.py'))
names={'sha_repr','_site_maps_cached','_site_maps','_entity_maps','base_automorphism_search_space','exact_canonical_base_key'}
parsed=ast.parse(orig.read_text());pure=ast.Module(body=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
ns=dict(collections=collections,functools=functools,itertools=itertools,math=math,hashlib=hashlib);exec(compile(pure,str(orig),'exec'),ns)
key_checks=collections.Counter();parent_links=0;panels=[];payloads={}
for r,rows in a.panels.items():
 p=primary/'inputs/source_panels'/f'r{r:03d}_selected_carriers.json.gz';raw=p.read_bytes()
 copies=[x for x in (B/'sources').rglob(p.name)];assert len(copies)==4
 assert all(x.read_bytes()==raw for x in copies)
 keys={h['exact_key'] for h in rows};assert len(keys)==len(rows)
 prev={h['exact_key'] for h in a.panels.get(r-1,[])}
 for h in rows:
  parents=h['parent_keys'];assert len(set(parents))==len(parents)==h['parent_count'];assert set(parents)<=prev
  if r:assert parents
  else:assert not parents
  parent_links+=len(parents)
  expected=None
  if r==0:expected='H000_'+ns['sha_repr'](tuple(sorted(h['entities'])))[:24]
  elif r==1:
   v,i,pa,w,j,pb=h['edges'][0];eps=tuple(sorted(((h['entities'][v],a.qmap[h['entities'][v]][i],pa),(h['entities'][w],a.qmap[h['entities'][w]][j],pb)),key=repr));rem=list(h['entities']);rem.pop(w);rem.pop(v);expected='H001_'+ns['sha_repr']((tuple(sorted(rem)),eps))[:24]
  else:
   ck=ns['exact_canonical_base_key'](h,a.qmap)
   if ck is not None:expected=f'H{r:03d}_'+ns['sha_repr'](('C',ck))[:24]
  if expected is not None:assert h['exact_key']==expected,(r,h['exact_key'],expected);key_checks['recomputed_original_key']+=1
  else:key_checks['frozen_label_only_symmetric_fallback']+=1
  payloads[f'{r}:{h["exact_key"]}']=sha(json.dumps({'entities':h['entities'],'edges':h['edges']},sort_keys=True,separators=(',',':')).encode())
 panels.append({'r':r,'carriers':len(rows),'sha256':sha(raw),'bytes':len(raw)})
out={'schema':'IG_SAVED_O3_EXTERNAL_QBANK_READINESS_V1','status':'PASS_WITH_EXPLICIT_EXTERNAL_QBANK_BOUNDARY','manifest_checks':checks,'panels':panels,'summary':summary,'failures':failures,'parent_links':parent_links,'key_checks':dict(key_checks),'stored_key_roll_sha256':roll,'bank_sha256':sha(bank.read_bytes()),'bank_entries':len(a.qmap),'component_rows':len(components),'topology_hidden_witness':witness,'generator_calls':0,'new_admissions':0,'master_release':'MASTER_DATA_V1_0143','scientific_slices':143,'missing_microscopic_O2_panels':76,'limits':['Q bank is frozen anonymous (p,u2) state, not full microscopic O2 incidence.','No reproduction of source-bank selection or missing O2 panels.','Historical graduation is an authority reference; this is a saved-data integrity sweep.','Symmetric fallback IDs remain source-bound labels; literal payload SHA is not a global isomorphism label.']}
(B/'READINESS_VALIDATION.json').write_text(json.dumps(out,indent=2));(B/'PAYLOAD_SHA256.json').write_text(json.dumps(payloads,indent=2));(B/'COMPONENT_ROWS.json').write_text(json.dumps(components,indent=2));print(json.dumps({k:out[k] for k in ['status','summary','parent_links','key_checks','bank_entries']}))
