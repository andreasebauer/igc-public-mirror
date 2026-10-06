#!/usr/bin/env python3
from __future__ import annotations
import json,csv,gzip,hashlib,itertools,collections,math,heapq,time,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; INP=ROOT/'08_INPUTS'; OUT=ROOT/'04_RESULTS'; PN=7

def jload(p): return json.load(open(p))
def jdump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def shaj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dist(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def canon_block(b): return tuple(sorted((tuple(p),tuple(f)) for p,f in b))
def canon_o3(o3): return tuple(sorted(canon_block(b) for b in o3))
def canon_o4(o4): return tuple(sorted(canon_o3(g) for g in o4))
def canon_r5(h5): return tuple(sorted(canon_o4(o4) for o4 in h5))

def load_rules():
 s=jload(INP/'MATURE_NODE_ALGEBRA_SPEC.json'); ports=tuple(tuple(x) for x in s['port_atoms']); idx={p:i for i,p in enumerate(ports)}
 return ports,tuple((idx[tuple(x)],idx[tuple(y)]) for x,y in s['ordered_compatible_port_pairs'])
def load_templates():
 rows=[]
 with open(INP/'MATURE_TEMPLATE_PETRI_TRANSITIONS.csv',newline='') as f:
  for r in csv.DictReader(f): rows.append({'index':int(r['template_index']),'source':int(r['source_port_index']),'dp':tuple(int(r[f'inc_{i}']) for i in range(7))})
 assert len(rows)==77; return rows

def proto_o4(p): return tuple(tuple(tuple((tuple(s[0]),tuple(s[1])) for s in block) for block in g) for g in p['ordered_resource_o3_carriers'])
def all_sites_proto(proto_ids,P):
 for c,pid in enumerate(proto_ids):
  o4=proto_o4(P[pid])
  for gi,g in enumerate(o4):
   for bi,b in enumerate(g):
    for si,(p,f) in enumerate(b): yield (c,gi,bi,si,p,f)
def action_universe(proto_ids,P,pairs):
 sites=list(all_sites_proto(proto_ids,P)); by=collections.defaultdict(list)
 for x in sites: by[x[0]].append(x)
 A=[]
 for c1 in range(len(proto_ids)):
  for c2 in range(c1+1,len(proto_ids)):
   for _,g1,b1,s1,p1,f1 in by[c1]:
    for _,g2,b2,s2,p2,f2 in by[c2]:
     for a,b in pairs:
      if f1[a]>0 and f2[b]>0: A.append((c1,g1,b1,s1,a,c2,g2,b2,s2,b))
 return tuple(A)
def usage(edges):
 U=collections.Counter()
 for c,g,b,s,a,d,G,B,S,q in edges: U[(c,g,b,s,a)]+=1; U[(d,G,B,S,q)]+=1
 return U
def base_free(proto_ids,P,c,g,b,s,a): return P[proto_ids[c]]['ordered_resource_o3_carriers'][g][b][s][1][a]
def valid_action(proto_ids,P,U,e):
 c,g,b,s,a,d,G,B,S,q=e
 return U[(c,g,b,s,a)]<base_free(proto_ids,P,c,g,b,s,a) and U[(d,G,B,S,q)]<base_free(proto_ids,P,d,G,B,S,q)
def add_edge(edges,e): return tuple(sorted(edges+(e,)))
def state_digest(edges): return hashlib.sha256(repr(edges).encode()).hexdigest()
def graph_components(K,edges):
 adj=[set() for _ in range(K)]
 for e in edges: adj[e[0]].add(e[5]); adj[e[5]].add(e[0])
 seen=set();out=[]
 for x in range(K):
  if x in seen: continue
  st=[x];seen.add(x);cc=[]
  while st:
   v=st.pop();cc.append(v)
   for w in adj[v]:
    if w not in seen: seen.add(w);st.append(w)
  out.append(tuple(sorted(cc)))
 return tuple(sorted(out))
def materialize(proto_ids,P,edges):
 U=usage(edges); out=[]
 for c,pid in enumerate(proto_ids):
  car=[]
  for gi,g in enumerate(P[pid]['ordered_resource_o3_carriers']):
   og=[]
   for bi,b in enumerate(g):
    row=[]
    for si,s in enumerate(b):
     p=tuple(s[0]); f=tuple(s[1][a]-U[(c,gi,bi,si,a)] for a in range(PN)); row.append((p,f))
    og.append(tuple(row))
   car.append(tuple(og))
  out.append(tuple(car))
 return tuple(out)
def state_feature(edges,proto_ids,P):
 K=len(proto_ids);U=usage(edges);comps=graph_components(K,edges);deg=[0]*K;paircnt=collections.Counter();types=set();eps=set();typeuse=[0]*7
 for e in edges:
  c,g,b,s,a,d,G,B,S,q=e;deg[c]+=1;deg[d]+=1;paircnt[(c,d)]+=1;types.add((min(a,q),max(a,q)));eps.add((c,g,b,s));eps.add((d,G,B,S));typeuse[a]+=1;typeuse[q]+=1
 tf=[]
 for c,pid in enumerate(proto_ids):
  z=0
  for gi,g in enumerate(P[pid]['ordered_resource_o3_carriers']):
   for bi,b in enumerate(g):
    for si,s in enumerate(b): z+=sum(s[1][a]-U[(c,gi,bi,si,a)] for a in range(PN))
  tf.append(z)
 beta=len(edges)-K+len(comps)
 return tuple([len(edges),len(comps),beta,len(types),len(eps),max(paircnt.values(),default=0),min(tf),max(tf),sum(tf)]+sorted(deg)+typeuse)
def farthest(states,proto_ids,P,cap):
 if len(states)<=cap: return sorted(states,key=state_digest)
 feats=[state_feature(e,proto_ids,P) for e in states];D=len(feats[0]);mins=[min(f[i] for f in feats) for i in range(D)];maxs=[max(f[i] for f in feats) for i in range(D)]
 nf=[tuple(0 if maxs[i]==mins[i] else (f[i]-mins[i])/(maxs[i]-mins[i]) for i in range(D)) for f in feats];cent=tuple(sum(x[i] for x in nf)/len(nf) for i in range(D));digs=[state_digest(e) for e in states]
 start=min(range(len(states)),key=lambda i:(dist(nf[i],cent),digs[i]));sel=[start];chosen={start}
 while len(sel)<cap:
  best=None
  for i in range(len(states)):
   if i in chosen: continue
   md=min(dist(nf[i],nf[j]) for j in sel)
   if best is None or md>best[0]+1e-15 or (abs(md-best[0])<=1e-15 and digs[i]<digs[best[1]]): best=(md,i)
  sel.append(best[1]);chosen.add(best[1])
 return [states[i] for i in sel]
def reservoir_successors(beam,proto_ids,P,A,cap):
 heap=[];keep={};enabled=0
 for edges in beam:
  U=usage(edges)
  for a in A:
   if not valid_action(proto_ids,P,U,a): continue
   enabled+=1;ne=add_edge(edges,a);d=state_digest(ne)
   if d in keep: continue
   di=int(d,16)
   if len(keep)<cap: keep[d]=ne;heapq.heappush(heap,(-di,d))
   else:
    maxd=heap[0][1]
    if d<maxd:
     _,old=heapq.heappop(heap);keep.pop(old,None);keep[d]=ne;heapq.heappush(heap,(-di,d))
 return list(keep.values()),enabled

def comp_invariants(proto_ids,P,edges,comp):
 S=set(comp);ee=tuple(e for e in edges if e[0] in S and e[5] in S);K=len(comp);m5=len(ee);N=sum(P[proto_ids[c]]['N'] for c in comp);U2=sum(P[proto_ids[c]]['U2'] for c in comp);m3=sum(P[proto_ids[c]]['m3'] for c in comp);m4=sum(P[proto_ids[c]]['m4'] for c in comp);d=sum(P[proto_ids[c]]['d'] for c in comp);r=U2+m3+m4+m5;beta5=m5-K+1
 cars=materialize(proto_ids,P,edges);F=sum(sum(f) for c in comp for g in cars[c] for b in g for p,f in b);Ptot=sum(sum(p) for c in comp for g in cars[c] for b in g for p,f in b);used=sum(sum(p[a]-f[a] for a in range(PN)) for c in comp for g in cars[c] for b in g for p,f in b)
 beta_flat=r-N+1;expected=sum(P[proto_ids[c]]['beta_flat4'] for c in comp)+beta5;deg=collections.Counter()
 for e in ee:deg[e[0]]+=1;deg[e[5]]+=1
 parity=[]
 for c in comp:
  load=sum(sum(p[a]-f[a] for a in range(PN)) for g in cars[c] for b in g for p,f in b);parity.append((c,load%2,deg[c]%2))
 ok=(used==2*r and beta5>=0 and beta_flat==expected and Ptot==d+2*N and F==Ptot-2*r and F==d+2-2*beta_flat and all(x==y for _,x,y in parity))
 return {'K4':K,'m5':m5,'N':N,'U2':U2,'m3':m3,'m4':m4,'d':d,'r':r,'beta5':beta5,'beta_flat':beta_flat,'expected_beta_flat':expected,'F':F,'P':Ptot,'direct_used':used,'parity':parity,'g':d+r,'ok':ok}

def mut(h,a,g,b,s,new):
 H=[[[list(block) for block in o3] for o3 in o4] for o4 in h];H[a][g][b][s]=new
 return tuple(tuple(tuple(tuple(block) for block in o3) for o3 in o4) for o4 in H)
def all_sites(h):
 for a,o4 in enumerate(h):
  for g,o3 in enumerate(o4):
   for b,block in enumerate(o3):
    for s,site in enumerate(block): yield (a,g,b,s,site)
def level(x,y):
 a,g,b,s,_=x;A,G,B,S,_=y
 if a!=A:return 'R5'
 if g!=G:return 'R4'
 if b!=B:return 'R3'
 if s!=S:return 'R2'
 return None
def resource_profile(h,templates,pairs):
 C=collections.Counter();sites=list(all_sites(h))
 for a,g,b,s,(p,f) in sites:
  u=tuple(p[i]-f[i] for i in range(PN))
  for t in templates:
   src=t['source'];q=tuple(p[i]+t['dp'][i] for i in range(PN))
   if p[src]<=0 or min(q)<0:continue
   mode=None
   if f[src]>0:mode='STRICT'
   elif all(q[i]>=u[i] for i in range(PN)):mode='CROSS'
   if mode is None:continue
   qf=tuple(q[i]-u[i] for i in range(PN));nh=mut(h,a,g,b,s,(q,qf));C[(('LIFT',mode,t['index']),canon_r5(nh))]+=1
 for i in range(len(sites)):
  for j in range(i+1,len(sites)):
   L=level(sites[i],sites[j]);
   if L is None:continue
   a,g,b,s,(p1,f1)=sites[i];A,G,B,S,(p2,f2)=sites[j]
   for x,y in pairs:
    if f1[x]<=0 or f2[y]<=0:continue
    nf1=list(f1);nf2=list(f2);nf1[x]-=1;nf2[y]-=1;nh=mut(h,a,g,b,s,(p1,tuple(nf1)));nh=mut(nh,A,G,B,S,(p2,tuple(nf2)));C[((L,min(x,y),max(x,y)),canon_r5(nh))]+=1
 payload=[(repr(k),v) for k,v in sorted(C.items(),key=lambda kv:repr(kv[0]))]
 return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest(),len(C),sum(C.values())
def permute_h5(h,seed):
 r=random.Random(seed); outer=list(range(len(h)));r.shuffle(outer);H=[]
 for aa in outer:
  o4=h[aa];ogs=list(range(len(o4)));r.shuffle(ogs);O=[]
  for gg in ogs:
   o3=o4[gg];bs=list(range(len(o3)));r.shuffle(bs);G=[]
   for bb in bs:
    block=o3[bb];ss=list(range(len(block)));r.shuffle(ss);G.append(tuple(block[k] for k in ss))
   O.append(tuple(G))
  H.append(tuple(O))
 return tuple(H)
def topology_sig(K,edges):
 adj=[set() for _ in range(K)];deg=[0]*K
 for u,v in edges:adj[u].add(v);adj[v].add(u);deg[u]+=1;deg[v]+=1
 tri=sum(1 for a in range(K) for b in range(a+1,K) for c in range(b+1,K) if b in adj[a] and c in adj[a] and c in adj[b])
 def cc(skip=None):
  seen=set();n=0
  for s in range(K):
   if s==skip or s in seen:continue
   n+=1;st=[s];seen.add(s)
   while st:
    x=st.pop()
    for y in adj[x]:
     if y!=skip and y not in seen:seen.add(y);st.append(y)
  return n
 base=cc();arts=sum(cc(x)>base for x in range(K));return {'degree_sorted':sorted(deg),'triangles':tri,'articulation_count':arts,'edge_count':len(edges)}
def first_valid_bridge(proto_ids,P,edges,pairs,left,right):
 U=usage(edges)
 for c1 in sorted(left):
  for c2 in sorted(right):
   if c1==c2:continue
   x,y=(c1,c2) if c1<c2 else (c2,c1)
   for gi,g in enumerate(P[proto_ids[x]]['ordered_resource_o3_carriers']):
    for bi,b in enumerate(g):
     for si,s in enumerate(b):
      for G,Gg in enumerate(P[proto_ids[y]]['ordered_resource_o3_carriers']):
       for B,Bb in enumerate(Gg):
        for S,t in enumerate(Bb):
         for a,q in pairs:
          e=(x,gi,bi,si,a,y,G,B,S,q)
          if valid_action(proto_ids,P,U,e):return e
 return None
def normalize_component(proto_ids,edges,comp):
 old=list(comp);mp={c:i for i,c in enumerate(old)};pids=tuple(proto_ids[c] for c in old);ee=tuple(sorted((mp[e[0]],e[1],e[2],e[3],e[4],mp[e[5]],e[6],e[7],e[8],e[9]) for e in edges if e[0] in mp and e[5] in mp));return pids,ee

def verify_selection():
 sel=jload(INP/'O4_PROTOTYPE_SELECTION.json');rows=[]
 with gzip.open(INP/'O4_PHASE1_ELIGIBLE_COMPONENT_FEATURES.csv.gz','rt',newline='') as f:
  for r in csv.DictReader(f):
   feat=[int(r[k]) for k in ['K3','m4','beta4','N','U2','m3','d','r4','beta_flat4','total_free','min_site_free','distinct_site_states']]+json.loads(r['free7']);rows.append((r['prototype_id'],feat))
 D=len(rows[0][1]);mins=[min(x[1][i] for x in rows) for i in range(D)];maxs=[max(x[1][i] for x in rows) for i in range(D)];nf={pid:tuple(0 if maxs[i]==mins[i] else (f[i]-mins[i])/(maxs[i]-mins[i]) for i in range(D)) for pid,f in rows};cent=tuple(sum(nf[p][i] for p,_ in rows)/len(rows) for i in range(D));first=min((p for p,_ in rows),key=lambda p:(dist(nf[p],cent),p));got=[first]
 while len(got)<6:
  best=None
  for p,_ in rows:
   if p in got:continue
   md=min(dist(nf[p],nf[q]) for q in got)
   if best is None or md>best[0]+1e-15 or (abs(md-best[0])<=1e-15 and p<best[1]):best=(md,p)
  got.append(best[1])
 return got==sel['selected_prototype_ids'],got

def main():
 t0=time.time();ports,pairs=load_rules();templates=load_templates();selected=jload(INP/'SELECTED_O4_PROTOTYPES.json')['selected'];P={p['prototype_id']:p for p in selected};pin=jload(INP/'PHASE0_PINNED_ACTUAL_O4_COMPONENT.json');pid='PHASE0_PINNED_ACTUAL_O4_COMPONENT';A=pin['accounting']
 P[pid]={'prototype_id':pid,'ordered_resource_o3_carriers':pin['ordered_resource_o3_carriers'],'K3':A['K3'],'N':A['N'],'U2':A['U2'],'m3':A['m3'],'m4':A['m4'],'d':A['d'],'r4':A['r'],'beta_flat4':A['beta_flat4'],'beta4':A['beta4']}
 spec=jload(ROOT/'01_SPEC/O5_PHASE1_FORMAL_SPEC.json');selok,got=verify_selection();fail=[];lanes={};saved=[];components=[]
 if not selok:fail.append({'kind':'prototype_selection','got':got})
 for lname,L in spec['lanes'].items():
  proto_ids=tuple(L['prototype_ids']) if lname=='HET4' else tuple([L['prototype_id']]*L['copies']);AU=action_universe(proto_ids,P,pairs);beam=[tuple()];summary=[{'lane':lname,'m5':0,'beam_states':1,'reservoir_states':1,'enabled_action_copies':len(AU),'nontrivial_components':0,'invariant_failures':0}];saved.append({'lane':lname,'m5':0,'proto_ids':proto_ids,'edges':[],'digest':state_digest(tuple())})
  for r in range(1,L['max_m5']+1):
   res,enabled=reservoir_successors(beam,proto_ids,P,AU,L['reservoir_cap']);beam=farthest(res,proto_ids,P,L['beam_cap']);ncomp=bad=0
   for edges in beam:
    for comp in graph_components(len(proto_ids),edges):
     if len(comp)<2:continue
     ncomp+=1;inv=comp_invariants(proto_ids,P,edges,comp)
     if not inv['ok']:
      bad+=1
      if len(fail)<20:fail.append({'kind':'component_invariant','lane':lname,'m5':r,'digest':state_digest(edges),'comp':comp,'inv':inv})
     components.append((lname,proto_ids,edges,comp))
    saved.append({'lane':lname,'m5':r,'proto_ids':proto_ids,'edges':[list(x) for x in edges],'digest':state_digest(edges)})
   summary.append({'lane':lname,'m5':r,'beam_states':len(beam),'reservoir_states':len(res),'enabled_action_copies':enabled,'nontrivial_components':ncomp,'invariant_failures':bad})
  lanes[lname]={'action_universe':len(AU),'final_beam':len(beam),'rank_summary':summary}
 # Hidden E5 topology collision on six exact copies of the actual Phase-0 O4.
 pids=tuple([pid]*6);base=proto_o4(P[pid]);# choose canonical type-0 site with maximum free capacity, tie by path
 cand=[]
 for g,o3 in enumerate(base):
  for b,block in enumerate(o3):
   for s,(p,f) in enumerate(block): cand.append((f[0],-g,-b,-s,g,b,s))
 _,_,_,_,gg,bb,ss=max(cand);path=(gg,bb,ss);assert base[gg][bb][ss][1][0]>=4
 GA=((0,4),(0,5),(1,2),(1,4),(2,4),(3,4));GB=((0,2),(0,3),(0,4),(0,5),(1,2),(1,3))
 def gedges(G): return tuple(sorted((u,path[0],path[1],path[2],0,v,path[0],path[1],path[2],0) for u,v in G))
 EA,EB=gedges(GA),gedges(GB);HA=materialize(pids,P,EA);HB=materialize(pids,P,EB);same=(canon_r5(HA)==canon_r5(HB));pa=resource_profile(HA,templates,pairs);pb=resource_profile(HB,templates,pairs);hidden={'same_R5_resource_skin':same,'exact_topology_distinct':topology_sig(6,GA)!=topology_sig(6,GB),'graph_A':[list(x) for x in GA],'graph_B':[list(x) for x in GB],'topology_A':topology_sig(6,GA),'topology_B':topology_sig(6,GB),'source_prototype':pid,'reservation_path':list(path),'reservation_type':0,'resource_profile_A':list(pa),'resource_profile_B':list(pb),'resource_profile_equal':pa==pb}
 if not (same and hidden['exact_topology_distinct'] and pa==pb): fail.append({'kind':'hidden_topology_collision','witness':hidden})
 # Hierarchical anonymity on hidden witness.
 relabel_mis=0
 for seed in range(8):
  if resource_profile(permute_h5(HA,2605000+seed),templates,pairs)!=pa:relabel_mis+=1
 if relabel_mis: fail.append({'kind':'hierarchical_relabeling','mismatches':relabel_mis})
 # External closure O5 x O4 on first 128 normalized connected components.
 ext_o4=ext_o5=extfail=0;norm=[]
 for lname,pids0,e0,c0 in sorted(components,key=lambda z:(z[0],state_digest(z[2]),z[3])):
  qids,qe=normalize_component(pids0,e0,c0);norm.append((qids,qe))
 for qids,qe in norm[:128]:
  npids=qids+(got[0],);br=first_valid_bridge(npids,P,qe,pairs,range(len(qids)),[len(qids)])
  if br is None:extfail+=1;continue
  ne=add_edge(qe,br);inv=comp_invariants(npids,P,ne,tuple(range(len(npids))))
  if inv['ok']:ext_o4+=1
  else:extfail+=1
 # O5 x O5 closure on up to 64 pairs.
 for k in range(min(64,len(norm)//2)):
  pA,eA=norm[2*k];pB,eB=norm[2*k+1];off=len(pA);pids2=pA+pB;eB2=tuple((e[0]+off,e[1],e[2],e[3],e[4],e[5]+off,e[6],e[7],e[8],e[9]) for e in eB);e=tuple(sorted(eA+eB2));br=first_valid_bridge(pids2,P,e,pairs,range(off),range(off,len(pids2)))
  if br is None:extfail+=1;continue
  ne=add_edge(e,br);inv=comp_invariants(pids2,P,ne,tuple(range(len(pids2))))
  if inv['ok']:ext_o5+=1
  else:extfail+=1
 if extfail:fail.append({'kind':'external_closure','failures':extfail})
 # Construction-order independence on 32 states with >=3 E5 edges.
 order_checks=order_fail=0
 candidates=[(tuple(s['proto_ids']),tuple(tuple(x) for x in s['edges'])) for s in saved if s['m5']>=3]
 for pids0,e0 in candidates[:32]:
  sample=e0[:min(4,len(e0))];finals=[]
  for perm in itertools.permutations(sample):
   e=tuple();ok=True
   for ac in perm:
    if not valid_action(pids0,P,usage(e),ac):ok=False;break
    e=add_edge(e,ac)
   if ok:finals.append(e)
  if finals:
   order_checks+=1
   if len(set(finals))!=1:order_fail+=1
 if order_fail:fail.append({'kind':'order_independence','failures':order_fail})
 # Phase-0 inherited separators.
 ph0=jload(INP/'O5_PHASE0_REFERENCE_AUDIT_RESULT.json');inherited=(ph0['status']=='PASS' and ph0['checks'].get('outer_o4_grouping_separator')==1 and ph0['checks'].get('typed_reservation_separator')==1 and ph0['failure_count']==0)
 if not inherited:fail.append({'kind':'phase0_inherited_witness'})
 jdump(OUT/'O5_PHASE1_GENERATED_SELECTED_STATES.json',{'states':saved})
 with open(OUT/'O5_PHASE1_GENERATION_SUMMARY.csv','w',newline='') as f:
  fields=['lane','m5','beam_states','reservoir_states','enabled_action_copies','nontrivial_components','invariant_failures'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for lr in lanes.values():
   for row in lr['rank_summary']:w.writerow({k:row.get(k,0) for k in fields})
 result={'program':'IG_V25_PHASE1_BOUNDED_O5_GENERATION_ADVERSARIAL_AUDIT','status':'PASS' if not fail else 'FAIL','graduation':'NOT_ATTEMPTED_IN_PHASE1','prototype_selection_verified':selok,'selected_prototypes':got,'lanes':lanes,'selected_state_records':len(saved),'connected_component_occurrences_checked':len(components),'hidden_topology_collision':hidden,'hierarchical_relabeling':{'checks':8,'mismatches':relabel_mis},'external_closure':{'O5xO4_passes':ext_o4,'O5xO5_passes':ext_o5,'failures':extfail},'order_independence':{'checks':order_checks,'failures':order_fail},'phase0_witnesses_reverified':inherited,'failure_count':len(fail),'failures':fail,'conclusion':'The Phase-0 exact O5 carrier and R5^0 skin survive the frozen bounded generated/adversarial programme. Same R5^0 can hide distinct connected E5 topology while preserving the frozen resource action profile. O5 remains ungraduated until a separate classification pass.' if not fail else 'Phase1 failed; do not proceed to graduation.'}
 science={k:result[k] for k in result if k not in ['science_sha256']};result['science_sha256']=shaj(science);jdump(OUT/'O5_PHASE1_AUDIT_RESULT.json',result);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not fail else 1
if __name__=='__main__':raise SystemExit(main())
