from __future__ import annotations
import collections,hashlib,itertools,struct
PN=7
STATE_HASH_PREFIX=b'IG-E7-STATE-v1|'

def site_at(h6,path):
 a,j,k,v,i=path; return h6[a][j][k][v][i]

def edge_bytes(e): return struct.pack('>14I',*e)

def state_digest(edges): return hashlib.sha256(STATE_HASH_PREFIX+b''.join(edge_bytes(e) for e in edges)).hexdigest()

def eparts(e): return (e[0],tuple(e[1:6]),e[6],e[7],tuple(e[8:13]),e[13])

def make_edge(c,path,a,d,path2,b):
 if c<d: return (c,*path,a,d,*path2,b)
 return (d,*path2,b,c,*path,a)

def active_owner_permutations(ctx:LaneContext, active_owners):
 # Exact quotient of owner_permutations by permutations of inactive equal-colour owners.
 # Only images of owners incident to E7 edges can affect the canonical edge tuple.
 active_owners=set(active_owners); choices=[]; actclasses=[]
 for cls in ctx.color_classes:
  ac=tuple(x for x in cls if x in active_owners); actclasses.append((cls,ac))
  choices.append(list(itertools.permutations(cls,len(ac))) if ac else [tuple()])
 for combo in itertools.product(*choices):
  mp=list(range(ctx.n))
  for (cls,ac),targets in zip(actclasses,combo):
   for old,new in zip(ac,targets): mp[old]=new
  yield tuple(mp)

def canonicalize_edges(ctx:LaneContext, edges):
 # v0.1.3 execution-only optimization: compute current p/f only for active E7 endpoints.
 # This is algebraically identical to materializing every owner and then reading active sites,
 # because E7 writes only decrement the endpoint-type free counter at its incident sites.
 edges=tuple(tuple(x) for x in edges)
 if not edges: return tuple()
 active=set(); inc=collections.defaultdict(list); used=collections.Counter()
 for e in edges:
  c,p,a,d,q,b=eparts(e); n1=(c,p); n2=(d,q); active|={n1,n2}; inc[n1].append((a,n2,b)); inc[n2].append((b,n1,a)); used[(c,p,a)]+=1; used[(d,q,b)]+=1
 basecol={}
 for node in active:
  o,path=node; pp,ff0=site_at(ctx.parents[o].h6,path); ff=tuple(ff0[t]-used[(o,path,t)] for t in range(PN))
  if min(ff)<0: raise ValueError('E7 oversubscribed')
  basecol[node]=(ctx.parents[o].base_p2k[path],pp,ff)
 best=None
 for omap in active_owner_permutations(ctx,{n[0] for n in active}):
  sig={n:hashlib.sha256(repr(basecol[n]).encode()).hexdigest() for n in active}
  for _ in range(12):
   ns={}
   for n in active:
    desc=[]
    for ta,nb,tb in inc[n]: desc.append((ta,omap[nb[0]],basecol[nb],tb,sig[nb]))
    ns[n]=hashlib.sha256(repr((basecol[n],tuple(sorted(desc)))).encode()).hexdigest()
   if all(ns[n]==sig[n] for n in active): break
   oldpart=sorted(sorted(x for x in active if sig[x]==ss) for ss in set(sig.values()))
   newpart=sorted(sorted(x for x in active if ns[x]==ss) for ss in set(ns.values()))
   sig=ns
   if [len(x) for x in oldpart]==[len(x) for x in newpart] and len(oldpart)==len(newpart): break
  remap={}
  for o in range(ctx.n):
   groups=collections.defaultdict(list)
   for n in active:
    if n[0]==o: groups[ctx.parents[o].base_p2k[n[1]]].append(n)
   for bk,nodes in groups.items():
    def sk(n):
     desc=tuple(sorted((ta,omap[nb[0]],basecol[nb],tb,sig[nb]) for ta,nb,tb in inc[n]))
     return (sig[n],basecol[n][1:],desc,n[1])
    nodes=sorted(nodes,key=sk); targets=ctx.parents[o].base_k2p[bk][:len(nodes)]
    for n,t in zip(nodes,targets): remap[n]=t
  out=[]
  for e in edges:
   c,p,a,d,q,b=eparts(e); nc,nd=omap[c],omap[d]; np,nq=remap[(c,p)],remap[(d,q)]
   out.append(make_edge(nc,np,a,nd,nq,b))
  cand=tuple(sorted(out))
  if best is None or cand<best: best=cand
 return best

from pathlib import Path
from types import SimpleNamespace
import json
_CONTEXT=None

def context():
 global _CONTEXT
 if _CONTEXT is None:
  obj=json.loads(Path(__file__).with_name('alias_context.json').read_bytes())
  if obj['schema']!='IG_HET4_HISTORICAL_ALIAS_CONTEXT_V1':raise ValueError('ALIAS_CONTEXT_SCHEMA')
  def freeze(v):return tuple(freeze(x) for x in v) if isinstance(v,list) else v
  parents=[]
  for p in obj['parents']:
   k2p={k:tuple(tuple(x) for x in paths) for k,paths in p['base_groups']}
   parents.append(SimpleNamespace(exact_color=p['exact_color'],h6=freeze(p['h6']),base_k2p=k2p,base_p2k={x:k for k,paths in k2p.items() for x in paths}))
  _CONTEXT=SimpleNamespace(n=len(parents),parents=tuple(parents),color_classes=freeze(obj['color_classes']))
 return _CONTEXT

def old_edge_alias(edges):return state_digest(canonicalize_edges(context(),edges))
