import pathlib,zipfile,io,json,hashlib,ast,collections
B=pathlib.Path(__file__).resolve().parent;R=B.parent;sha=lambda b:hashlib.sha256(b).hexdigest();name='Infinity_Grid_WHOLE_NODE_GRAMMAR_CLOSURE_AUDIT_REPRO_BUNDLE_2026-08-26.zip';b=zipfile.ZipFile(R/'project_sources/09-Nodes.zip').read(name);assert sha(b)=='1ca64d203cf1d5b059733fe9d274514e1077c2d0a4b3dc6543a1df0fb7a0a947';z=zipfile.ZipFile(io.BytesIO(b));base='Infinity_Grid_WHOLE_NODE_GRAMMAR_CLOSURE_AUDIT_2026-08-26/';out=B/'lineage_source';out.mkdir(exist_ok=True);sources=[]
selected_names=[base+'evidence/stress/inputs/L44_SELECTED_RELATION.json']+[n for n in z.namelist() if '/levels/' in n and n.endswith('/evidence/SELECTED_RELATION.json')]+[base+'evidence/stress/code/run_level.py',base+'evidence/stress/inputs/L39_L44_MATURATION_STATE.json',base+'evidence/stress/inputs/primitive.pkl']
for n in selected_names:
 v=z.read(n);p=out/n[len(base):];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v);sources.append({'member':n,'sha256':sha(v),'bytes':len(v)})
canon=lambda r:(int(r[0]),tuple(sorted(tuple(p) for p in r[1])),int(r[2]),tuple(sorted(r[3] if isinstance(r[3],tuple) else (r[3],))),int(r[4]))
prim=[canon(ast.literal_eval(x['record'])) for x in json.load(open(R/'node_bindings0144/source_bundle/inputs/FROZEN_PRIMITIVES.json'))['primitive']];observed={x['sha256'] for x in json.load(__import__('gzip').open(R/'node_bindings0144/source_bundle/inputs/OBSERVED_S15_RECORDS.json.gz','rt'))['records']}
def load(n):
 x=json.loads(z.read(n));d={}
 for row in x['selected']:
  rec=canon(ast.literal_eval(row['record']));assert sha(repr(rec).encode())==row['sha256'];assert row['sha256'] not in d;d[row['sha256']]=rec
 return x,d
prev,parents=load(base+'evidence/stress/inputs/L44_SELECTED_RELATION.json');union=set();rows=[];witnesses=[]
for L in range(45,65):
 n=base+f'evidence/stress/levels/L{L}/evidence/SELECTED_RELATION.json';x,children=load(n);union.update(h for h,c in children.items() if c[0:1]==(15,) and c[2]==1 and c[4]==0);missing=invalid=valid=0
 for ph,ch in x['edges']:
  if ph not in parents or ch not in children:missing+=1;continue
  a,c=parents[ph],children[ch];found=[];ta=collections.Counter(a[3]);tc=collections.Counter(c[3]);delta=tc.copy();delta.subtract(ta)
  for i,p in enumerate(prim):
   if delta!=collections.Counter(p[3]) or (a[0]|p[0],min(a[2],p[2]),int(a[4] and p[4]))!=(c[0],c[2],c[4]):continue
   pa=collections.Counter(a[1]);pc=collections.Counter(p[1]);cc=collections.Counter(c[1]);removed=pa+pc;removed.subtract(cc)
   for u in pa:
    for v in pc:
     if not ((u[1]==0 or u[1]&v[0]) and (v[1]==0 or v[1]&u[0])):continue
     if removed==collections.Counter([u,v]):found.append([i,list(u),list(v)])
  if not found:invalid+=1
  else:valid+=1;witnesses.append({'level':L,'parent':ph,'child':ch,'lawful_boundary_recipes':found})
 rows.append({'level':L,'selected_records_verified':len(children),'edges':len(x['edges']),'missing_endpoints':missing,'lawfully_explained_edges':valid,'unexplained_edges':invalid});parents=children
res={'schema':'IG_RECOVERED_NODE_LINEAGE_VALIDATION_V1','archive_chain':['09-Nodes.zip',name],'inner_archive_sha256':sha(b),'source_members':sources,'levels':rows,'observed_records':len(observed),'observed_in_recovered_L45_L64':len(observed&union),'recovered_S15_union':len(union),'observed_missing_from_union':len(observed-union),'all_edge_endpoints_resolved':all(x['missing_endpoints']==0 for x in rows),'all_edges_have_lawful_boundary_recipe':all(x['unexplained_edges']==0 for x in rows),'scope':'Stored selected parent-child incidence L45-L64 with L44 boundary seed; lawful recipe recovery is diagnostic ambiguity-preserving, not original microscopic witness identity. Pre-L44 construction lineage remains outside this recovery.','new_generation':False,'new_admission':False}
(B/'LINEAGE_VALIDATION.json').write_text(json.dumps(res,indent=2));(B/'RECOVERED_EDGE_BOUNDARY_RECIPES.json').write_text(json.dumps(witnesses,separators=(',',':')));print(json.dumps({k:v for k,v in res.items() if k not in ['source_members','levels']},indent=2));print(rows)
