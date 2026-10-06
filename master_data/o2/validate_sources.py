from pathlib import Path
import json,hashlib,gzip,importlib.util
B=Path(__file__).resolve().parent;R=B.parent;sha=lambda b:hashlib.sha256(b).hexdigest();sources=[]
for D in (B/'sources').iterdir():
 S=next(D.iterdir());checked=0
 for line in (S/'SHA256_MANIFEST.txt').read_text().splitlines():
  if line.strip():
   h,n=line.split(None,1);assert sha((S/n.strip()).read_bytes())==h;checked+=1
 bindings=[]
 for p in (S/'inputs').glob('*'):
  if p.name in ['OBSERVED_S15_RECORDS.json.gz','FROZEN_PRIMITIVES.json','MATURE_NODE_ALGEBRA_SPEC.json']:
   old=R/'node_bindings0144/source_bundle'/('spec' if p.name=='MATURE_NODE_ALGEBRA_SPEC.json' else 'inputs')/p.name;assert p.read_bytes()==old.read_bytes();bindings.append({'name':p.name,'sha256':sha(p.read_bytes()),'status':'BYTE_IDENTICAL_TO_ADMITTED_NODE_DEPENDENCY'})
 sources.append({'bundle':D.name,'manifest_files':checked,'bindings':bindings})
S=next(next(p for p in (B/'sources').iterdir() if 'IN11_' in p.name).iterdir());code=S/'code/in9_relational_longitudinal_scout.py';sp=importlib.util.spec_from_file_location('frozen_in9',code);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
allrows={};levels=[];refs=[];parent_links=0
for d in range(77,110):
 p=S/f'checkpoints/d{d:03d}_selected_carriers.json.gz';x=json.load(gzip.open(p,'rt'));assert x['d']==d;rows={}
 for row in x['selected']:
  st=tuple(tuple(v) for v in row['states']);ed=tuple(tuple(v) for v in row['edges']);assert row['d']==d and all(len(v)==16 and all(type(a)==int and a>=0 for a in v) for v in st);assert all(0<=u<len(st) and 0<=v<len(st) and u!=v and 0<=a<7 and 0<=b<7 and m.bridge(m.PORTS[a],m.PORTS[b]) for u,a,v,b in ed);assert m.capacity_valid(st,ed);assert all(sum(v[:7])==sum(v[7:])+2 for v in st)
  assert m.exact_key(st,ed)==row['exact_key'];rh,_=m.role_info(st,ed);assert rh==row['role_hash'];assert row['exact_key'] not in rows;rows[row['exact_key']]=row
 if d>77:
  for row in rows.values():
   for ph in row['parents']:
    assert ph in allrows[d-1],(d,ph);parent_links+=1
 allrows[d]=rows;levels.append({'d':d,'carriers':len(rows),'nodes':sum(len(r['states']) for r in rows.values()),'typed_edges':sum(len(r['edges']) for r in rows.values())});refs.append({'member':str(p.relative_to(S)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size})
a=json.load(open(S/'results/IN11_RESULT.json'));res={'schema':'IG_INTER_NODE_O2_READINESS_VALIDATION_V1','source_bundles':sources,'source_code_sha256':sha(code.read_bytes()),'checkpoint_panels':refs,'levels':levels,'seed_depth':77,'seed_carriers':len(allrows[77]),'scope_depths':[78,109],'scope_carriers':sum(len(allrows[d]) for d in range(78,110)),'parent_links_resolved':parent_links,'capacity_bridge_exact_key_role_and_size_invariants':'PASS_ALL_ROWS','historical_audit_science_sha256':a['science_sha256'],'historical_audit_reexecuted':False,'new_generation':False,'new_admission':False,'master_release_unchanged':'MASTER_DATA_V1_0142','next':'REGISTERED_O2_SELECTED_CARRIER_EXPORT_AND_READER_GATE','scope_limit':'Saved selected typed carriers and parent identifiers, external d77 seed; no full candidate census or unique microscopic occurrence lineage.'};(B/'READINESS_VALIDATION.json').write_text(json.dumps(res,indent=2));print({k:v for k,v in res.items() if k not in ['source_bundles','levels','checkpoint_panels']})
