import json,os,sys,importlib.util,collections
from pathlib import Path
B=Path(__file__).parent;os.environ['OSCOUT_DATA_ROOT']=str(B/'fixture')
sp=importlib.util.spec_from_file_location('_independent_seed_fixture',B/'fixture/02_CODE/o7_live_engine.py');e=importlib.util.module_from_spec(sp);sys.modules[sp.name]=e;sp.loader.exec_module(e);e.O6=e.import_o6();parents=e.load_parent_records()
def verify(record):
 row=record['source_row'];ctx=e._profile_row_context(row,parents);w=record['result']['witness'];nodes={}
 def visit(o,path,value):
  nodes[(o,path)]=('leaf',tuple(value[0]),tuple(value[1])) if len(path)==5 else ('bag',len(path),len(value))
  if len(path)<5:
   for i,ch in enumerate(value):visit(o,path+(i,),ch)
 for i,p in enumerate(ctx.parents):visit(i,(),p.h6)
 pairs=[((a,tuple(p)),(b,tuple(q))) for a,p,b,q in w['node_map']];mapping=dict(pairs)
 assert len(mapping)==len(pairs)==len(nodes) and set(mapping)==set(nodes) and set(mapping.values())==set(nodes)
 for src,dst in mapping.items():
  assert nodes[src]==nodes[dst]
  if src[1]:assert mapping[(src[0],src[1][:-1])]==(dst[0],dst[1][:-1])
  else:assert w['owner_map'][src[0]]==dst[0] and ctx.parents[src[0]].exact_color==ctx.parents[dst[0]].exact_color
 def normalized(edge):
  a=(edge[0],tuple(edge[1:6]),edge[6]);b=(edge[7],tuple(edge[8:13]),edge[13]);return tuple(sorted((a,b)))
 mapped=[]
 for edge in row['edges']:
  a,p=mapping[(edge[0],tuple(edge[1:6]))];b,q=mapping[(edge[7],tuple(edge[8:13]))];mapped.append(tuple(sorted(((a,p,edge[6]),(b,q,edge[13])))))
 assert collections.Counter(mapped)==collections.Counter(normalized(x) for x in record['result']['edges'])
 return len(nodes)
out=[]
for p in sorted(B.glob('WITNESS_*.json')):
 r=json.loads(p.read_text());n=verify(r);bad=json.loads(p.read_text());bad['result']['witness']['node_map'][0][2]=(bad['result']['witness']['node_map'][0][2]+1)%len(bad['source_row']['parent_ids'])
 try:verify(bad)
 except (AssertionError,KeyError):pass
 else:raise AssertionError('FORGED_WITNESS_ACCEPTED')
 out.append({'file':p.name,'nodes_checked':n,'whole_tree_bijection':True,'parentage_and_resource_colors':True,'typed_edge_multiset_exact':True,'forged_witness_refused':True})
(B/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps({'status':'PASS','canonicalizer_imported':False,'seeds':out},indent=2)+'\n');print('Independent PASS',len(out))
