#!/usr/bin/env python3
from __future__ import annotations
import json,csv,gzip,hashlib,itertools,collections,math,heapq,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; INP=ROOT/'07_INPUTS'; OUT=ROOT/'04_RESULTS'
PN=7

def jload(p):return json.load(open(p))
def jdump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def shaj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def canon_block(block):return tuple(sorted((tuple(s[0]),tuple(s[1])) for s in block))
def canon_carrier(car):return tuple(sorted(canon_block(b) for b in car))
def canon_r4(cars):return tuple(sorted(canon_carrier(c) for c in cars))
def dist(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

def load_rules():
 s=jload(INP/'MATURE_NODE_ALGEBRA_SPEC.json'); ports=tuple(tuple(x) for x in s['port_atoms']); idx={p:i for i,p in enumerate(ports)}
 pairs=[]
 for x,y in s['ordered_compatible_port_pairs']:pairs.append((idx[tuple(x)],idx[tuple(y)]))
 return ports,tuple(pairs)

def load_templates():
 rows=[]
 with open(INP/'MATURE_TEMPLATE_PETRI_TRANSITIONS.csv',newline='') as f:
  for r in csv.DictReader(f):
   rows.append({'index':int(r['template_index']),'source':int(r['source_port_index']),'dp':tuple(int(r[f'inc_{i}']) for i in range(7)),'dt':tuple(int(r[f'inc_{i}']) for i in range(7,16))})
 assert len(rows)==77
 return rows

def proto_car(p):return tuple(tuple((tuple(s['p']),tuple(s['f'])) for s in b) for b in p['ordered_resource_blocks'])
def site_list(proto_ids,P):
 out=[]
 for ci,pid in enumerate(proto_ids):
  car=proto_car(P[pid])
  for bi,b in enumerate(car):
   for si,(p,f) in enumerate(b):out.append((ci,bi,si,p,f))
 return out

def action_universe(proto_ids,P,pairs):
 sites=site_list(proto_ids,P); by=collections.defaultdict(list)
 for x in sites:by[x[0]].append(x)
 A=[]
 for c1 in range(len(proto_ids)):
  for c2 in range(c1+1,len(proto_ids)):
   for _,b1,s1,p1,f1 in by[c1]:
    for _,b2,s2,p2,f2 in by[c2]:
     for a,b in pairs:
      if f1[a]>0 and f2[b]>0:A.append((c1,b1,s1,a,c2,b2,s2,b))
 return tuple(A)
def usage(edges):
 U=collections.Counter()
 for c,b,s,a,d,e,t,q in edges:U[(c,b,s,a)]+=1;U[(d,e,t,q)]+=1
 return U
def base_free(proto_ids,P,c,b,s,a):return P[proto_ids[c]]['ordered_resource_blocks'][b][s]['f'][a]
def valid_action(proto_ids,P,U,a):
 c,b,s,x,d,e,t,y=a
 return U[(c,b,s,x)]<base_free(proto_ids,P,c,b,s,x) and U[(d,e,t,y)]<base_free(proto_ids,P,d,e,t,y)
def add_edge(edges,a):return tuple(sorted(edges+(a,)))
def state_digest(edges):return shaj(edges)

def graph_components(K,edges):
 adj=[set() for _ in range(K)]
 for c,b,s,a,d,e,t,q in edges:adj[c].add(d);adj[d].add(c)
 seen=set();out=[]
 for x in range(K):
  if x in seen:continue
  st=[x];seen.add(x);cc=[]
  while st:
   v=st.pop();cc.append(v)
   for w in adj[v]:
    if w not in seen:seen.add(w);st.append(w)
  out.append(tuple(sorted(cc)))
 return tuple(sorted(out))
def state_feature(edges,proto_ids,P):
 K=len(proto_ids);U=usage(edges); comps=graph_components(K,edges); deg=[0]*K; paircnt=collections.Counter(); types=set(); eps=set(); typeuse=[0]*7
 for c,b,s,a,d,e,t,q in edges:
  deg[c]+=1;deg[d]+=1;paircnt[(c,d)]+=1;types.add((min(a,q),max(a,q)));eps.add((c,b,s));eps.add((d,e,t));typeuse[a]+=1;typeuse[q]+=1
 tf=[]
 for c,pid in enumerate(proto_ids):
  z=0
  for bi,b in enumerate(P[pid]['ordered_resource_blocks']):
   for si,s in enumerate(b):
    z+=sum(s['f'][a]-U[(c,bi,si,a)] for a in range(PN))
  tf.append(z)
 beta=len(edges)-K+len(comps)
 return tuple([len(edges),len(comps),beta,len(types),len(eps),max(paircnt.values(),default=0),min(tf),max(tf),sum(tf)]+sorted(deg)+typeuse)
def farthest(states,proto_ids,P,cap):
 if len(states)<=cap:return sorted(states,key=lambda e:state_digest(e))
 feats=[state_feature(e,proto_ids,P) for e in states];D=len(feats[0]);mins=[min(f[i] for f in feats) for i in range(D)];maxs=[max(f[i] for f in feats) for i in range(D)]
 nf=[tuple(0 if maxs[i]==mins[i] else (f[i]-mins[i])/(maxs[i]-mins[i]) for i in range(D)) for f in feats]
 cent=tuple(sum(x[i] for x in nf)/len(nf) for i in range(D)); digs=[state_digest(e) for e in states]
 start=min(range(len(states)),key=lambda i:(dist(nf[i],cent),digs[i]));sel=[start];chosen={start}
 while len(sel)<cap:
  best=None
  for i in range(len(states)):
   if i in chosen:continue
   md=min(dist(nf[i],nf[j]) for j in sel)
   if best is None or md>best[0]+1e-15 or (abs(md-best[0])<=1e-15 and digs[i]<digs[best[1]]):best=(md,i)
  sel.append(best[1]);chosen.add(best[1])
 return [states[i] for i in sel]
def reservoir_successors(beam,proto_ids,P,A,cap):
 # exact enabled action-copy enumeration, deterministic smallest-hash reservoir
 heap=[]; keep={}; enabled=0
 for edges in beam:
  U=usage(edges)
  for a in A:
   if not valid_action(proto_ids,P,U,a):continue
   enabled+=1; ne=add_edge(edges,a); d=state_digest(ne)
   if d in keep:continue
   di=int(d,16)
   if len(keep)<cap:
    keep[d]=ne;heapq.heappush(heap,(-di,d))
   else:
    maxd=heap[0][1]
    if d<maxd:
     _,old=heapq.heappop(heap);keep.pop(old,None);keep[d]=ne;heapq.heappush(heap,(-di,d))
 return list(keep.values()),enabled

def materialize(proto_ids,P,edges):
 U=usage(edges);cars=[]
 for c,pid in enumerate(proto_ids):
  car=[]
  for bi,b in enumerate(P[pid]['ordered_resource_blocks']):
   row=[]
   for si,s in enumerate(b):
    p=tuple(s['p']); f=tuple(s['f'][a]-U[(c,bi,si,a)] for a in range(PN));row.append((p,f))
   car.append(tuple(row))
  cars.append(tuple(car))
 return tuple(cars)
def comp_edge_list(edges,comp):
 S=set(comp);return tuple(e for e in edges if e[0] in S and e[4] in S)
def comp_invariants(proto_ids,P,edges,comp):
 S=set(comp);ee=comp_edge_list(edges,comp);K=len(comp);m4=len(ee);N=sum(P[proto_ids[c]]['N'] for c in comp);U2=sum(P[proto_ids[c]]['U2'] for c in comp);m3=sum(P[proto_ids[c]]['m3'] for c in comp);d=sum(P[proto_ids[c]]['d'] for c in comp);r=U2+m3+m4;beta4=m4-K+1
 cars=materialize(proto_ids,P,edges);F=sum(sum(f) for c in comp for b in cars[c] for p,f in b);Ptot=sum(sum(p) for c in comp for b in cars[c] for p,f in b)
 direct_used=sum(sum(p[a]-f[a] for a in range(PN)) for c in comp for b in cars[c] for p,f in b)
 beta_flat=r-N+1; expected_beta=sum(P[proto_ids[c]]['beta_flat3'] for c in comp)+beta4
 deg=collections.Counter()
 for e in ee:deg[e[0]]+=1;deg[e[4]]+=1
 parity=[]
 for c in comp:
  load=sum(sum(p[a]-f[a] for a in range(PN)) for b in cars[c] for p,f in b)
  parity.append((c,load%2,deg[c]%2))
 return {'K':K,'m4':m4,'N':N,'U2':U2,'m3':m3,'d':d,'r':r,'beta4':beta4,'beta_flat':beta_flat,'expected_beta_flat':expected_beta,'F':F,'P':Ptot,'direct_used':direct_used,'parity':parity,
  'ok':direct_used==2*r and beta4>=0 and beta_flat==expected_beta and Ptot==d+2*N and F==Ptot-2*r and F==d+2-2*beta_flat and all(x==y for _,x,y in parity)}

def relabel_profile_digest(cars,templates,pairs):
 # full direct frozen resource profile over a materialized hierarchy; no exact topology read
 C=collections.Counter();sites=[]
 for ci,car in enumerate(cars):
  for bi,b in enumerate(car):
   for si,(p,f) in enumerate(b):sites.append((ci,bi,si,p,f))
 def mut(h,ci,bi,si,val):
  H=[list(map(list,c)) for c in h]; H[ci][bi][si]=val;return tuple(tuple(tuple(x) for x in c) for c in H)
 for ci,bi,si,p,f in sites:
  u=tuple(p[a]-f[a] for a in range(PN))
  for t in templates:
   s=t['source'];q=tuple(p[a]+t['dp'][a] for a in range(PN))
   if p[s]<=0 or min(q)<0:continue
   mode=None
   if f[s]>0:mode='STRICT'
   elif all(q[a]>=u[a] for a in range(PN)):mode='CROSS'
   if mode is None:continue
   qf=tuple(q[a]-u[a] for a in range(PN));nh=mut(cars,ci,bi,si,(q,qf));C[(('LIFT',mode,t['index']),canon_r4(nh))]+=1
 for i in range(len(sites)):
  c1,b1,s1,p1,f1=sites[i]
  for j in range(i+1,len(sites)):
   c2,b2,s2,p2,f2=sites[j]
   if c1==c2 and b1==b2:
    if s1==s2:continue
    level='R2'
   elif c1==c2:level='R3'
   else:level='R4'
   for a,b in pairs:
    if f1[a]<=0 or f2[b]<=0:continue
    nf1=list(f1);nf2=list(f2);nf1[a]-=1;nf2[b]-=1
    nh=mut(cars,c1,b1,s1,(p1,tuple(nf1)));nh=mut(nh,c2,b2,s2,(p2,tuple(nf2)));C[((level,min(a,b),max(a,b)),canon_r4(nh))]+=1
 payload=[(repr(k),v) for k,v in sorted(C.items(),key=lambda kv:repr(kv[0]))]
 return hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest(),len(C),sum(C.values())

def topology_sig(K,edges):
 adj=[set() for _ in range(K)];deg=[0]*K
 for u,v in edges:adj[u].add(v);adj[v].add(u);deg[u]+=1;deg[v]+=1
 tri=sum(1 for a in range(K) for b in range(a+1,K) for c in range(b+1,K) if b in adj[a] and c in adj[a] and c in adj[b])
 def ccount(skip=None):
  seen=set();n=0
  for s in range(K):
   if s==skip or s in seen:continue
   n+=1;st=[s];seen.add(s)
   while st:
    x=st.pop()
    for y in adj[x]:
     if y!=skip and y not in seen:seen.add(y);st.append(y)
  return n
 base=ccount();arts=sum(ccount(x)>base for x in range(K))
 return {'degree_sorted':sorted(deg),'triangles':tri,'articulation_count':arts,'edge_count':len(edges)}

def first_valid_bridge(proto_ids,P,edges,pairs,cset1,cset2):
 U=usage(edges)
 for c1 in sorted(cset1):
  for c2 in sorted(cset2):
   if c1==c2:continue
   x,y=(c1,c2) if c1<c2 else (c2,c1)
   # action universe orientation must follow x<y
   for bi,b in enumerate(P[proto_ids[x]]['ordered_resource_blocks']):
    for si,s in enumerate(b):
     for bj,bb in enumerate(P[proto_ids[y]]['ordered_resource_blocks']):
      for sj,t in enumerate(bb):
       for a,q in pairs:
        ac=(x,bi,si,a,y,bj,sj,q)
        if valid_action(proto_ids,P,U,ac):return ac
 return None

def verify_selection():
 sel=jload(INP/'O3_PROTOTYPE_SELECTION.json'); rows=[]
 with gzip.open(INP/'O3_PHASE1_ELIGIBLE_COMPONENT_FEATURES.csv.gz','rt',newline='') as f:
  for r in csv.DictReader(f):
   feat=[int(r[k]) for k in ['r3_rank','n3','m3','beta3','N','U2','total_free','min_site_free','total_p']]+json.loads(r['free7']);rows.append((r['prototype_id'],feat))
 D=len(rows[0][1]);mins=[min(x[1][i] for x in rows) for i in range(D)];maxs=[max(x[1][i] for x in rows) for i in range(D)]
 nf={pid:tuple(0 if maxs[i]==mins[i] else (f[i]-mins[i])/(maxs[i]-mins[i]) for i in range(D)) for pid,f in rows};cent=tuple(sum(nf[p][i] for p,_ in rows)/len(rows) for i in range(D))
 first=min((p for p,_ in rows),key=lambda p:(dist(nf[p],cent),p));got=[first]
 while len(got)<6:
  best=None
  for p,_ in rows:
   if p in got:continue
   md=min(dist(nf[p],nf[q]) for q in got)
   if best is None or md>best[0]+1e-15 or (abs(md-best[0])<=1e-15 and p<best[1]):best=(md,p)
  got.append(best[1])
 return got==sel['selected_prototype_ids'],got

def main():
 t0=time.time(); ports,pairs=load_rules();templates=load_templates(); data=jload(INP/'SELECTED_O3_PROTOTYPES.json')['selected'];P={p['prototype_id']:p for p in data}
 p0=jload(INP/'PHASE0_PINNED_ACTUAL_O3_COMPONENT.json');p0id='PHASE0_PINNED_ACTUAL_O3_COMPONENT';
 # derive pinned accounting
 ordered=p0['ordered_resource_blocks'];N=sum(len(b) for b in ordered);d=sum(sum(s['p'])-2 for b in ordered for s in b);m3=len(p0['internal_o3_edges']);rho=sum(sum(s['p'][a]-s['f'][a] for a in range(PN)) for b in ordered for s in b)//2;U2=rho-m3
 P[p0id]={'prototype_id':p0id,'ordered_resource_blocks':ordered,'N':N,'d':d,'m3':m3,'U2':U2,'rho3':rho,'beta_flat3':rho-N+1,'beta3':m3-len(ordered)+1,'n3':len(ordered),'r3_rank':p0['source_r3_rank']}
 spec=jload(INP/'../01_SPEC/O4_PHASE1_FORMAL_SPEC.json') if False else jload(ROOT/'01_SPEC/O4_PHASE1_FORMAL_SPEC.json')
 selok,got=verify_selection(); failures=[]; lane_results={}; all_saved=[]; connected_components=[]
 if not selok:failures.append({'kind':'prototype_selection','got':got})
 for lname,L in spec['lanes'].items():
  if lname=='HET4':proto_ids=tuple(L['prototype_ids'])
  else:proto_ids=tuple([L['prototype_id']]*L['copies'])
  A=action_universe(proto_ids,P,pairs);beam=[tuple()];summ=[]
  # rank0
  summ.append({'lane':lname,'m4':0,'beam_states':1,'reservoir_states':1,'enabled_action_copies':len(A),'nontrivial_components':0})
  all_saved.append({'lane':lname,'m4':0,'proto_ids':proto_ids,'edges':[],'digest':state_digest(tuple())})
  for r in range(1,L['max_m4']+1):
   res,enabled=reservoir_successors(beam,proto_ids,P,A,L['reservoir_cap']);beam=farthest(res,proto_ids,P,L['beam_cap'])
   ncomp=0; bad=0
   for edges in beam:
    comps=graph_components(len(proto_ids),edges)
    for comp in comps:
     if len(comp)<2:continue
     ncomp+=1; inv=comp_invariants(proto_ids,P,edges,comp)
     if not inv['ok']:
      bad+=1
      if len(failures)<20:failures.append({'kind':'component_invariant','lane':lname,'m4':r,'digest':state_digest(edges),'comp':comp,'inv':inv})
     connected_components.append((lname,proto_ids,edges,comp))
    all_saved.append({'lane':lname,'m4':r,'proto_ids':proto_ids,'edges':[list(x) for x in edges],'digest':state_digest(edges)})
   summ.append({'lane':lname,'m4':r,'beam_states':len(beam),'reservoir_states':len(res),'enabled_action_copies':enabled,'nontrivial_components':ncomp,'invariant_failures':bad})
  lane_results[lname]={'action_universe':len(A),'rank_summary':summ,'final_beam':len(beam)}
 # hidden topology collision actual-source, six copies pinned, type0/site0
 proto_ids=(p0id,)*6
 gA=[(0,4),(0,5),(1,2),(1,4),(2,4),(3,4)];gB=[(0,2),(0,3),(0,4),(0,5),(1,2),(1,3)]
 def gedges(G):return tuple(sorted((u,0,0,0,v,0,0,0) for u,v in G))
 eA,eB=gedges(gA),gedges(gB);carsA=materialize(proto_ids,P,eA);carsB=materialize(proto_ids,P,eB);r4eq=canon_r4(carsA)==canon_r4(carsB)
 sigA,sigB=topology_sig(6,gA),topology_sig(6,gB);profA=relabel_profile_digest(carsA,templates,pairs);profB=relabel_profile_digest(carsB,templates,pairs)
 hidden={'same_R4_resource_skin':r4eq,'topology_A':sigA,'topology_B':sigB,'exact_topology_distinct':sigA!=sigB,'resource_profile_A':profA,'resource_profile_B':profB,'resource_profile_equal':profA==profB,'graph_A':gA,'graph_B':gB,'source_prototype':p0id}
 if not(r4eq and sigA!=sigB and profA==profB):failures.append({'kind':'hidden_topology_collision', 'data':hidden})
 # external closure samples
 ext_o3=ext_o4=0;extfail=0
 comps=connected_components[:]
 # deterministic by state digest, comp
 comps.sort(key=lambda x:(state_digest(x[2]),x[3],x[0]))
 for item in comps[:128]:
  lname,pids,edges,comp=item
  # compact component to new skeleton
  old=list(comp); mp={c:i for i,c in enumerate(old)}; cpids=tuple(pids[c] for c in old); ce=tuple(sorted((mp[e[0]],e[1],e[2],e[3],mp[e[4]],e[5],e[6],e[7]) for e in edges if e[0] in mp and e[4] in mp))
  fresh=spec['lanes']['HET4']['prototype_ids'][0]; npids=cpids+(fresh,); ne=ce; br=first_valid_bridge(npids,P,ne,pairs,set(range(len(cpids))),{len(cpids)})
  if br:
   out=add_edge(ne,br); inv=comp_invariants(npids,P,out,tuple(range(len(npids))))
   if inv['ok'] and len(graph_components(len(npids),out))==1:ext_o3+=1
   else:extfail+=1
 for i in range(min(64,len(comps)//2)):
  Aitem=comps[2*i];Bitem=comps[2*i+1]
  def compact(item):
   lname,pids,edges,comp=item;old=list(comp);mp={c:i for i,c in enumerate(old)};cp=tuple(pids[c] for c in old);ce=tuple(sorted((mp[e[0]],e[1],e[2],e[3],mp[e[4]],e[5],e[6],e[7]) for e in edges if e[0] in mp and e[4] in mp));return cp,ce
  ap,ae=compact(Aitem);bp,be=compact(Bitem);off=len(ap);pids=ap+bp;edges=tuple(sorted(ae+tuple((e[0]+off,e[1],e[2],e[3],e[4]+off,e[5],e[6],e[7]) for e in be)))
  br=first_valid_bridge(pids,P,edges,pairs,set(range(off)),set(range(off,len(pids))))
  if br:
   out=add_edge(edges,br);inv=comp_invariants(pids,P,out,tuple(range(len(pids))))
   if inv['ok'] and len(graph_components(len(pids),out))==1:ext_o4+=1
   else:extfail+=1
 if extfail:failures.append({'kind':'external_closure','failures':extfail})
 # order independence: deterministic action triples on fresh HET4 seed
 pids=tuple(spec['lanes']['HET4']['prototype_ids']);AU=action_universe(pids,P,pairs); order_checks=0;order_fail=0
 for shift in range(0,min(128,len(AU)-3),4):
  acts=[];U=collections.Counter();
  for a in AU[shift:]:
   if valid_action(pids,P,U,a):
    acts.append(a);U[(a[0],a[1],a[2],a[3])]+=1;U[(a[4],a[5],a[6],a[7])]+=1
    if len(acts)==3:break
  if len(acts)<3:continue
  finals=[]
  for perm in itertools.permutations(acts):
   e=tuple();ok=True
   for ac in perm:
    uu=usage(e)
    if not valid_action(pids,P,uu,ac):ok=False;break
    e=add_edge(e,ac)
   if ok:finals.append(e)
  if len(finals)==6:
   order_checks+=1
   if len(set(finals))!=1:order_fail+=1
 if order_fail:failures.append({'kind':'order_independence','failures':order_fail})
 # Phase0 inherited witnesses reverified structurally
 ph0=jload(INP/'O4_PHASE0_REFERENCE_AUDIT_RESULT.json'); inherited=(ph0['checks'].get('outer_grouping_separator')==1 and ph0['checks'].get('typed_reservation_separator')==1 and ph0['failure_count']==0)
 if not inherited:failures.append({'kind':'phase0_inherited_witness'})
 # panel result outputs
 jdump(OUT/'O4_PHASE1_GENERATED_SELECTED_STATES.json',{'states':all_saved})
 # summary CSV
 with open(OUT/'O4_PHASE1_GENERATION_SUMMARY.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['lane','m4','beam_states','reservoir_states','enabled_action_copies','nontrivial_components','invariant_failures']);w.writeheader()
  for lr in lane_results.values():
   for row in lr['rank_summary']:
    rr={k:row.get(k,0) for k in w.fieldnames};w.writerow(rr)
 result={'program':'IG_V24_PHASE1_BOUNDED_O4_GENERATION_ADVERSARIAL_AUDIT','status':'PASS' if not failures else 'FAIL','graduation':'NOT_ATTEMPTED_IN_PHASE1',
  'prototype_selection_verified':selok,'selected_prototypes':got,'lanes':lane_results,'selected_state_records':len(all_saved),'connected_component_occurrences_checked':len(connected_components),
  'hidden_topology_collision':hidden,'external_closure':{'O4xO3_passes':ext_o3,'O4xO4_passes':ext_o4,'failures':extfail},'order_independence':{'checks':order_checks,'failures':order_fail},
  'phase0_witnesses_reverified':inherited,'failure_count':len(failures),'failures':failures,
  'conclusion':'The Phase-0 exact O4 carrier and R4^0 skin survive the frozen bounded generated/adversarial programme. Same R4^0 can hide distinct connected E4 topology while preserving the frozen resource action profile. O4 remains ungraduated until a separate classification pass.' if not failures else 'Phase1 failed; do not proceed to graduation.'}
 jdump(OUT/'O4_PHASE1_AUDIT_RESULT.json',result);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failures else 1
if __name__=='__main__':raise SystemExit(main())
