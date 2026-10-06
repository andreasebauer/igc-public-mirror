from pathlib import Path
import json,gzip,hashlib,importlib.util,collections
B=Path(__file__).resolve().parent;R=B.parent;sha=lambda b:hashlib.sha256(b).hexdigest();L=B/'sources/Infinity_Grid_SCOUT3_V1_0_FROZEN_LAUNCH_BUNDLE_2026-08-27';checked=[]
fm=json.load(open(L/'provenance/FROZEN_MANIFEST.json'))
for n,h in fm['files'].items():assert sha((L/n).read_bytes())==h
checked.append({'manifest':str((L/'provenance/FROZEN_MANIFEST.json').relative_to(B)),'files':len(fm['files']),'status':'PASS'})
for p in (B/'sources').rglob('*MANIFEST*'):
 if p.name=='FROZEN_MANIFEST.json' or 'SNAPSHOT' in p.name:continue
 root=p.parent.parent if p.parent.name=='provenance' else p.parent
 count=0
 if p.suffix=='.txt':
  for line in p.read_text().splitlines():
   if not line.strip():continue
   h,n=line.split(None,1);n=n.strip().removeprefix('*');t=root/n
   if len(h)!=64:continue
   assert t.is_file(),(p,t);assert sha(t.read_bytes())==h,(p,n);count+=1
 elif p.suffix=='.json':
  obj=json.load(open(p));files=obj.get('files',[])
  if isinstance(files,list):
   for row in files:assert sha((root/row['path']).read_bytes())==row['sha256'];count+=1
  elif isinstance(files,dict):
   for n,h in files.items():assert sha((root/n).read_bytes())==h;count+=1
 if count:checked.append({'manifest':str(p.relative_to(B)),'files':count,'status':'PASS'})
g=json.load(open(L/'inputs/O2_GRADUATION_AUDIT_RESULT.json'));assert g['science_sha256']=='d138de3c6c906be3ab453c225ecce2e3fff5d3e4f91d220ed681ef5253a86bbb' and g['status']=='O2_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_V1_9'
G=next((B/'sources/Infinity_Grid_SCOUT2B_V1_9_O2_GRADUATION_AUDIT_BUNDLE_2026-08-27').rglob('O2_GRADUATION_AUDIT_RESULT.json'));assert G.read_bytes()==(L/'inputs/O2_GRADUATION_AUDIT_RESULT.json').read_bytes()
for name,old in [('FROZEN_PRIMITIVES.json',R/'node_bindings0144/source_bundle/inputs/FROZEN_PRIMITIVES.json'),('MATURE_NODE_ALGEBRA_SPEC.json',R/'node_bindings0144/source_bundle/spec/MATURE_NODE_ALGEBRA_SPEC.json')]:assert (L/'inputs'/name).read_bytes()==old.read_bytes()
expected={int(n[1:4]):h for h,n in (line.split() for line in (L/'inputs/O2_SOURCE_PANEL_SHA256.txt').read_text().splitlines())};assert set(expected)==set(range(129));actual={};duplicates=0
for p in (B/'sources').rglob('k*_selected_carriers.json.gz'):
 k=int(p.name[1:4]);assert sha(p.read_bytes())==expected[k],p
 if k in actual:assert p.read_bytes()==actual[k].read_bytes();duplicates+=1
 else:actual[k]=p
sp=importlib.util.spec_from_file_location('frozen_sc',L/'code/vendor/scout2_relational_longitudinal_v1_1.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
qrows=collections.defaultdict(list);allrows={};refs=[];count=links=missingparents=0
for k,p in sorted(actual.items()):
 x=json.load(gzip.open(p));assert x['k']==k;rows={}
 for row in x['selected']:
  st=tuple(tuple(v) for v in row['states']);ed=tuple(tuple(v) for v in row['edges']);assert row['k']==k and all(len(v)==16 and all(type(a)==int and a>=0 for a in v) for v in st)
  assert all(0<=u<len(st) and 0<=v<len(st) and u!=v and 0<=a<7 and 0<=b<7 and m.bridge(m.PORTS[a],m.PORTS[b]) for u,a,v,b in ed);assert m.capacity_valid(st,ed);assert all(sum(v[:7])==sum(v[7:])+2 for v in st)
  assert m.exact_key(st,ed)==row['exact_key'];assert m.role_info(st,ed)[0]==row['role_hash'];assert row['exact_key'] not in rows;rows[row['exact_key']]=row;U=m.usage(len(st),ed);q=tuple(sorted((tuple(s[:7]),tuple(U[i])) for i,s in enumerate(st)));qrows[sha(repr(q).encode())].append({'k':k,'exact_key':row['exact_key']});count+=1
 if k>0:
  for row in rows.values():
   if k-1 in allrows:
    for h in row['parents']:assert h in allrows[k-1];links+=1
   else:missingparents+=len(row['parents'])
 allrows[k]=rows;refs.append({'k':k,'sha256':expected[k],'bytes':p.stat().st_size,'path':str(p.relative_to(B)),'carriers':len(rows)})
# k0 is copied from the admitted d109 seed with source-declared k/parent metadata reset.
oldS=next(next(p for p in (R/'inter_node0150/sources').iterdir() if 'IN11_' in p.name).iterdir());old=json.load(gzip.open(oldS/'checkpoints/d109_selected_carriers.json.gz'));old={x['exact_key']:x for x in old['selected']};assert set(allrows[0])==set(old)
for h,x in allrows[0].items():assert x['states']==old[h]['states'] and x['edges']==old[h]['edges'] and x['role_hash']==old[h]['role_hash']
bank=json.load(open(L/'inputs/O2_SOURCE_BANK.json'));seed=json.load(open(L/'inputs/O3_SEED_PANEL.json'))
for x in [bank,seed]:
 a=dict(x);h=a.pop('science_sha256');assert sha(json.dumps(a,sort_keys=True,separators=(',',':')).encode())==h
assert bank['selected_Q']==len(bank['entries'])==256 and bank['o2_graduation_science_sha256']==g['science_sha256'];qmap={};coverage=[]
for e in bank['entries']:
 q=tuple(sorted((tuple(p),tuple(u)) for p,u in e['sites']));assert sha(repr(q).encode())==e['q_hash'];assert e['q_hash'] not in qmap;qmap[e['q_hash']]=e
 P=[sum(p[a] for p,u in q) for a in range(7)];U=[sum(u[a] for p,u in q) for a in range(7)];F=[a-b for a,b in zip(P,U)];assert all(a>=0 for a in F) and P==e['total_p'] and U==e['total_u'] and F==e['free'] and len(q)==e['n_sites'] and sum(U)//2-len(q)+1==e['beta_o2'];coverage.append({'q_hash':e['q_hash'],'available_witnesses':qrows[e['q_hash']]})
assert seed['count']==len(seed['entries'])==128 and seed['source_bank_science_sha256']==bank['science_sha256']
for e in seed['entries']:
 es=[qmap[h] for h in e['entity_q_hashes']];assert len(es)==e['n_entities'] and len(set(e['entity_q_hashes']))==e['distinct_q_types'] and sum(x['n_sites'] for x in es)==e['total_o1_sites'] and sum(x['beta_o2'] for x in es)==e['sum_o2_beta'] and [sum(x['free'][a] for x in es) for a in range(7)]==e['total_free']
missing=sorted(set(expected)-set(actual));res={'schema':'IG_O2_TO_O3_SOURCE_BOUND_READINESS_V1','status':'BLOCKED_MISSING_EXACT_PARENT_PANELS' if missing else 'SOURCE_BINDINGS_READY','manifests':checked,'expected_panel_count':129,'available_panel_count':len(actual),'missing_k':missing,'panel_refs':refs,'duplicate_panel_copies_byte_identical':duplicates,'available_carrier_occurrences_verified':count,'available_parent_links_resolved':links,'parent_links_crossing_missing_panel_gaps':missingparents,'k0_binding_to_admitted_d109':'FULL_STATES_EDGES_EXACT_KEYS_ROLES_MATCH','k0_carriers':len(allrows[0]),'graduation_record_byte_match_to_independent_saved_bundle':True,'graduation_science_sha256':g['science_sha256'],'graduation_reexecuted_or_recertified':False,'source_bank':{'selected_Q':256,'reported_full_source_unique_Q':bank['all_unique_Q'],'science_sha256':bank['science_sha256'],'saved_payload_hash_and_internal_fields':'PASS','selection_reexecuted':False,'entries_with_available_exact_panel_witness':sum(bool(x['available_witnesses']) for x in coverage),'original_selection_over129_panels_reproduced':False},'seed_panel':{'count':128,'science_sha256':seed['science_sha256'],'internal_bank_bindings':'PASS','regenerated':False},'new_generation':False,'new_admission':False,'master_release_unchanged':'MASTER_DATA_V1_0143','next':'RECOVER_MISSING_SCOUT2B_K_PANELS_BEFORE_FULL_SOURCE_BANK_REPRODUCTION','stopping_rule':'Missing exact source bytes block full readiness; do not repair by generation or infer full129-panel coverage from saved bank/observer outputs'};(B/'READINESS_VALIDATION.json').write_text(json.dumps(res,indent=2));(B/'AVAILABLE_Q_WITNESSES.json').write_text(json.dumps(coverage,indent=2));print({k:v for k,v in res.items() if k not in ['panel_refs','manifests']})
