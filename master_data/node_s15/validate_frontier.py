import pathlib,json,hashlib,ast,collections,shutil
B=pathlib.Path(__file__).resolve().parent;R=B.parent;S=next((R/'node_seed0146/original').iterdir());sha=lambda b:hashlib.sha256(b).hexdigest();checked=[]
for line in (S/'MANIFEST_SHA256.txt').read_text().splitlines():
 if not line.strip():continue
 h,n=line.split(None,1);p=S/n.strip();assert p.is_file() and sha(p.read_bytes())==h,n;checked.append(n)
l44=S/'inputs/SELECTED_RELATION_L44.json';old=R/'node_routes0145/lineage_source/evidence/stress/inputs/L44_SELECTED_RELATION.json';assert l44.read_bytes()==old.read_bytes()
state=S/'current_checkpoint/MATURATION_STATE.json';oldstate=R/'node_routes0145/lineage_source/evidence/stress/inputs/L39_L44_MATURATION_STATE.json';assert state.read_bytes()==oldstate.read_bytes()
assert (S/'inputs/primitive.pkl').read_bytes()==(R/'node_routes0145/lineage_source/evidence/stress/inputs/primitive.pkl').read_bytes()
canon=lambda r:(int(r[0]),tuple(sorted(tuple(p) for p in r[1])),int(r[2]),tuple(sorted(r[3] if isinstance(r[3],tuple) else (r[3],))),int(r[4]))
prim=[canon(ast.literal_eval(x['record'])) for x in json.load(open(R/'node_bindings0144/source_bundle/inputs/FROZEN_PRIMITIVES.json'))['primitive']]
l38=json.load(open(next((R/'node_frontier0147/l38_source').iterdir())/'evidence/SELECTED_RELATION.json'));parents={r['sha256']:canon(ast.literal_eval(r['record'])) for r in l38['selected']};levels=[];frontier=set();recipes=[]
for L in range(39,45):
 x=json.load(open(S/f'inputs/SELECTED_RELATION_L{L}.json'));children={}
 for row in x['selected']:
  rec=canon(ast.literal_eval(row['record']));assert sha(repr(rec).encode())==row['sha256'];assert row['sha256'] not in children;children[row['sha256']]=rec
 valid=missing=0
 for ph,ch in x['edges']:
  assert ch in children
  if parents is None:frontier.add(ph);continue
  if ph not in parents:missing+=1;continue
  a,c=parents[ph],children[ch];delta=collections.Counter(c[3]);delta.subtract(collections.Counter(a[3]));hits=[]
  for i,p in enumerate(prim):
   if delta!=collections.Counter(p[3]) or (a[0]|p[0],min(a[2],p[2]),int(a[4] and p[4]))!=(c[0],c[2],c[4]):continue
   pa=collections.Counter(a[1]);pp=collections.Counter(p[1]);removed=pa+pp;removed.subtract(collections.Counter(c[1]))
   for u in pa:
    for v in pp:
     if ((u[1]==0 or u[1]&v[0]) and (v[1]==0 or v[1]&u[0])) and removed==collections.Counter([u,v]):hits.append([i,list(u),list(v)])
  assert hits,(L,ph,ch);valid+=1;recipes.append({'level':L,'parent':ph,'child':ch,'lawful_boundary_recipes':hits})
 assert missing==0
 levels.append({'level':L,'records_verified':len(children),'edges':len(x['edges']),'lawful_edges_verified':valid,'frontier_edges':len(x['edges']) if parents is None else 0});parents=children

C=next((B/'l38_source').iterdir());manifest=json.load(open(C/'SHA256_MANIFEST.json'));checked38=[]
for n,h in manifest.items():
 assert sha((C/n).read_bytes())==h,n;checked38.append(n)
res={'schema':'IG_NODE_L38_FRONTIER_RESOLUTION_V1','L38_selected_records':len(l38['selected']),'L38_manifest_files_verified':len(checked38),'all1036_L39_parent_hashes_resolved':True,'levels':levels,'verified_edges':sum(x['lawful_edges_verified'] for x in levels),'remaining_frontier':'Earlier-than-L38 ancestry; external L38 seed is explicit','new_generation':False,'new_admission':False}
(B/'FRONTIER_VALIDATION.json').write_text(json.dumps(res,indent=2));(B/'EDGE_BOUNDARY_RECIPES.json').write_text(json.dumps(recipes,separators=(',',':')));print(res)
