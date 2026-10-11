import json,gzip,hashlib,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
B=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
digest=lambda x:sha(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())
p=B/'input/BOOTSTRAP100_HISTORICAL.json.gz';assert sha(p.read_bytes())=='d9ba6b5e8e4d6465a2739bde77e381c60eb0fe3bdefc6becce1151b1195462f5'
d=json.load(gzip.open(p,'rt'));nodes=d['dag']['nodes'];ledger=json.loads(Path('/workspace/scratch/87702f15343f/reservation_transport/TRANSPORT.json').read_text());o7={r['legacy_construction_digest']:r for r in ledger['rows']};ordinals={}
for r in ledger['rows']:
 for t in range(7):
  seq=[]
  for owner,path,tt,n in r['transported_patch']:
   if tt==t:seq.extend([(owner,tuple(path))]*n)
  assert len(seq)==r['reserve_counts'][t]
  key=(r['source_id'],t);old=ordinals.get(key,[]);assert seq[:min(len(seq),len(old))]==old[:min(len(seq),len(old))]
  if len(seq)>len(old):ordinals[key]=seq
def resolve(ref,w,t):
 route=[]
 while nodes[ref]['kind']=='LIFT':
  assert len(w)==3 and w[2]==t;i=w[0];route.append(i);ref=nodes[ref]['children'][i];w=w[1]
 assert len(w)==2 and w[0]==t;ordinal=w[1];n=nodes[ref];assert 0<=ordinal<n['reserve_counts'][t]
 owner,path=ordinals[(n['source_id'],t)][ordinal]
 return {'child_occurrence_path':route,'O7_source_id':n['source_id'],'type':t,'historical_ordinal':ordinal,'corrected_owner':owner,'corrected_leaf_path':list(path)}
computed={};out=[];edgecount=0
for ref,n in sorted(nodes.items(),key=lambda kv:(kv[1].get('level',7),kv[0])):
 assert ref==n['construction_digest']
 if n['kind']=='O7':
  r=o7[ref];assert n['reserve_counts']==r['reserve_counts'];computed[ref]=(r['skin'],r['caps']);continue
 children=n['children'];assert all(c in computed for c in children)
 skin=sha(('R'+str(n['level'])+'|'+'|'.join(sorted(computed[c][0] for c in children))).encode());caps=[sum(computed[c][1][t] for c in children) for t in range(7)]
 payload={'level':n['level'],'children':children,'edges':n['top_edges_full'],'lane':n['lane'],'motif':n['motif_id']};assert digest(payload)==ref
 edges=[]
 for u,v,a,b,wu,wv in n['top_edges_full']:
  assert 0<=u<len(children) and 0<=v<len(children);left=resolve(children[u],wu,a);right=resolve(children[v],wv,b);edges.append({'owners':[u,v],'types':[a,b],'left':left,'right':right});edgecount+=1
 computed[ref]=(skin,caps);out.append({'legacy_construction_ref':ref,'level':n['level'],'children':children,'resource_skin':skin,'caps':caps,'transported_edges':edges})
population_path=B/'input/HISTORICAL_V1_INTERFACE_POPULATION.json';assert sha(population_path.read_bytes())=='6b7405f8e3374061f167cecd20225db4f54f22ad45ed8ac97cace64f1fe8f666';pop=json.loads(population_path.read_text())
rows=pop['interfaces'] if 'interfaces' in pop else pop['records']
for row in rows:
 skin,caps=computed[row['carrier_ref']];assert skin==row['boundary_resource_skin_sha256'] and caps==row['total_free_by_type']
assert len(rows)==193 and len(out)==16335
result={'status':'PASS_SAVED_LIFT_ENDPOINT_TRANSPORT_AND_INTERFACE_RESOURCES','LIFT_nodes':len(out),'top_edges':edgecount,'resolved_endpoints':2*edgecount,'interfaces_exact':len(rows),'scope':'Preserved child occurrences, transported O7 leaf endpoints, saved resource interfaces; not corrected global LIFT identity or new selection policy','observer_runs':0,'migration_activated':False,'nodes':out}
(B/'LIFT_TRANSPORT.json').write_text(json.dumps(result,separators=(',',':'))+'\n');print('PASS',len(out),edgecount,len(rows))
