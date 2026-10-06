#!/usr/bin/env python3
import argparse, ast, collections, gzip, hashlib, itertools, json, math, os, resource, statistics, sys, time
from pathlib import Path

PORTS=[(0,0),(1,0),(1,4),(1,24),(4,0),(8,0),(10,0)]
IDX={p:i for i,p in enumerate(PORTS)}
SELECTIVE={(IDX[(1,4)],IDX[(4,0)]),(IDX[(1,24)],IDX[(8,0)]),(IDX[(1,24)],IDX[(10,0)])}
LANES=('D','T','C','I','L')
PICK_CACHE={}
CANON_CACHE={}

def bridge(a,b):
    P1,M1=a; P2,M2=b
    return bool((M1==0 or (M1&P2)!=0) and (M2==0 or (M2&P1)!=0))

def sha_obj(x): return hashlib.sha256(repr(x).encode()).hexdigest()
def js_sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def popcount(x): return int(x).bit_count()
def support_mask(p):
    m=0
    for i,n in enumerate(p):
        if n>0:m|=1<<i
    return m

def edge_norm(u,a,v,b):
    if u<v: return (u,a,v,b)
    if v<u: return (v,b,u,a)
    raise ValueError('self edge forbidden')

def canon_edges(edges): return tuple(sorted(edge_norm(*e) for e in edges))

def usage(n,edges):
    U=[[0]*7 for _ in range(n)]
    for u,a,v,b in edges:
        U[u][a]+=1;U[v][b]+=1
    return tuple(tuple(x) for x in U)

def components(n,edges):
    adj=[set() for _ in range(n)]
    for u,a,v,b in edges: adj[u].add(v);adj[v].add(u)
    seen=set(); c=0
    for s in range(n):
        if s in seen: continue
        c+=1; stack=[s];seen.add(s)
        while stack:
            x=stack.pop()
            for y in adj[x]:
                if y not in seen: seen.add(y);stack.append(y)
    return c

def beta(n,edges): return len(edges)-n+components(n,edges)

def degs(n,edges):
    d=[0]*n
    for u,a,v,b in edges:d[u]+=1;d[v]+=1
    return tuple(sorted(d,reverse=True))

def selective_edge_count(edges):
    z=0
    for u,a,v,b in edges:
        if (a,b) in SELECTIVE or (b,a) in SELECTIVE:z+=1
    return z

def free_counts(states,edges):
    U=usage(len(states),edges)
    F=[]
    for st,u in zip(states,U):
        p=st[:7]; f=tuple(p[i]-u[i] for i in range(7));
        if min(f)<0: return None
        F.append(f)
    return tuple(F)

def capacity_valid(states,edges): return free_counts(states,edges) is not None

def selective_profile(states,edges):
    F=free_counts(states,edges); assert F is not None
    # Exact local free counts on the five selective-relevant port types; node order is current bookkeeping for parent-child comparison.
    sel_idx=[IDX[(1,4)],IDX[(4,0)],IDX[(1,24)],IDX[(8,0)],IDX[(10,0)]]
    return tuple(tuple(f[i] for i in sel_idx) for f in F)

def selective_opportunity_count(states,edges):
    F=free_counts(states,edges); assert F is not None
    tot=0
    for i in range(len(states)):
        for j in range(i+1,len(states)):
            for a,b in SELECTIVE:
                if F[i][a]>0 and F[j][b]>0: tot+=F[i][a]*F[j][b]
                if F[i][b]>0 and F[j][a]>0: tot+=F[i][b]*F[j][a]
    return tot

def tight_margin(states,edges):
    U=usage(len(states),edges); vals=[]
    for st,u in zip(states,U):
        p=st[:7]
        for i,x in enumerate(u):
            if x: vals.append(p[i]-x)
    return min(vals) if vals else min(min(st[:7]) for st in states)

def canonical_key(states,edges,role=False):
    n=len(states); F=free_counts(states,edges); assert F is not None
    if role:
        colors=tuple((support_mask(st[:7]),support_mask(F[i])) for i,st in enumerate(states))
    else:
        colors=tuple(tuple(st) for st in states)
    ck=(bool(role),colors,tuple(edges))
    if ck in CANON_CACHE: return CANON_CACHE[ck]
    # Any lexicographically minimal canonical form must list node colors in sorted order.
    # Therefore only permute nodes within equal-color classes; this is exact and avoids n! when colors differ.
    groups=collections.defaultdict(list)
    for i,c in enumerate(colors): groups[c].append(i)
    sorted_cols=sorted(groups)
    perms=[list(itertools.permutations(groups[c])) for c in sorted_cols]
    best_edges=None
    for choice in itertools.product(*perms):
        perm=tuple(i for block in choice for i in block)
        inv=[0]*n
        for new,old in enumerate(perm): inv[old]=new
        ee=tuple(sorted(edge_norm(inv[u],a,inv[v],b) for u,a,v,b in edges))
        if best_edges is None or ee<best_edges: best_edges=ee
    ans=(tuple(sorted(colors)),best_edges)
    if len(CANON_CACHE)>200000: CANON_CACHE.clear()
    CANON_CACHE[ck]=ans
    return ans

def role_info(states,edges):
    rk=canonical_key(states,edges,True); h=sha_obj(rk)
    return h,rk

def exact_key(states,edges): return sha_obj(canonical_key(states,edges,False))

def carrier_metrics(states,edges):
    F=free_counts(states,edges); U=usage(len(states),edges); n=len(states); b=beta(n,edges)
    future=selective_opportunity_count(states,edges)
    return {
      'n':n,'r':len(edges),'beta':b,'components':components(n,edges),'degrees':list(degs(n,edges)),
      'selective_edges':selective_edge_count(edges),'selective_opportunity_count':future,
      'tight_margin':tight_margin(states,edges),'total_free_ports':sum(sum(f) for f in F),
      'node_support_masks':[support_mask(st[:7]) for st in states],
      'node_free_support_masks':[support_mask(f) for f in F],
      'reservation_vectors':[list(u) for u in U],
      'free_counts':[list(f) for f in F],
    }

def feature_vec(c):
    m=c['metrics']; deg=m['degrees']+[0]*6
    return [m['n'],m['r'],m['beta'],m['selective_edges'],math.log1p(m['selective_opportunity_count']),m['tight_margin'],math.log1p(m['total_free_ports'])]+deg[:6]

def norm_features(pool):
    X=[feature_vec(c) for c in pool]
    if not X:return []
    cols=list(zip(*X)); lo=[min(z) for z in cols]; hi=[max(z) for z in cols]
    return [[0 if hi[j]==lo[j] else (x[j]-lo[j])/(hi[j]-lo[j]) for j in range(len(x))] for x in X]

def greedy_farthest(pool,cap):
    if len(pool)<=cap:return list(pool)
    X=norm_features(pool)
    # deterministic first: smallest exact key
    first=min(range(len(pool)),key=lambda i:pool[i]['exact_key'])
    chosen=[first]; mind=[sum((X[i][j]-X[first][j])**2 for j in range(len(X[i]))) for i in range(len(pool))]
    mind[first]=-1
    while len(chosen)<cap:
        k=max(range(len(pool)),key=lambda i:(mind[i],-i))
        chosen.append(k); mind[k]=-1
        for i in range(len(pool)):
            if mind[i]<0:continue
            d=sum((X[i][j]-X[k][j])**2 for j in range(len(X[i])))
            if d<mind[i]:mind[i]=d
    return [pool[i] for i in chosen]

def derive_templates(prims,tatoms):
    out=[]
    for pr in prims:
        p=pr['record']; pc=collections.Counter(p[1]); tc=collections.Counter(p[3])
        for si,source in enumerate(PORTS):
            for u,n in pc.items():
                if n<=0 or not bridge(source,u):continue
                dp=[0]*7;dp[si]-=1
                for z,c in pc.items():dp[IDX[z]]+=c
                dp[IDX[u]]-=1
                dt=[tc[t] for t in tatoms]
                out.append({'source':si,'primitive_port':IDX[u],'dp':tuple(dp),'dt':tuple(dt),'tid':pr['tid'],'rank':pr['rank'],'psha':pr['sha256']})
    # dedup exact update forms while preserving canonical smallest provenance
    d={}
    for x in out:
        k=(x['source'],x['primitive_port'],x['dp'],x['dt'])
        if k not in d or (x['tid'],x['rank'],x['psha'])<(d[k]['tid'],d[k]['rank'],d[k]['psha']): d[k]=x
    return sorted(d.values(),key=lambda x:(x['source'],x['primitive_port'],x['dp'],x['dt'],x['tid'],x['rank'],x['psha']))

def load_inputs(root):
    with gzip.open(root/'inputs/OBSERVED_S15_RECORDS.json.gz','rt') as f: raw=json.load(f)['records']
    spec=json.load(open(root/'inputs/MATURE_NODE_ALGEBRA_SPEC.json')); tatoms=tuple(spec['T_atoms'])
    recs=[]
    for x in raw:
        r=ast.literal_eval(x['record']); pc=collections.Counter(r[1]);tc=collections.Counter(r[3])
        st=tuple(pc[p] for p in PORTS)+tuple(tc[t] for t in tatoms)
        assert sum(st[:7])==sum(st[7:])+2
        recs.append({'sha':x['sha256'],'state':st,'support':support_mask(st[:7]),'ports':sum(st[:7])})
    primj=json.load(open(root/'inputs/FROZEN_PRIMITIVES.json'))['primitive']; prims=[]
    for p in primj:
        r=ast.literal_eval(p['record']); prims.append({**p,'record':r})
    templ=derive_templates(prims,tatoms); assert len(templ)==77, len(templ)
    return recs,spec,tatoms,templ

def pick_state(recs,need,mode,node_index=0):
    ck=(tuple(need),mode,int(node_index))
    if ck in PICK_CACHE: return PICK_CACHE[ck]
    cand=[]
    for x in recs:
        p=x['state'][:7]
        if all(p[i]>=need[i] for i in range(7)):
            slack=sum(p[i]-need[i] for i in range(7)); cand.append((x,slack,popcount(x['support'])))
    if not cand: raise RuntimeError('no witness state')
    if mode=='stress': key=lambda z:(z[1],z[0]['ports'],z[2],z[0]['sha'])
    elif mode=='generous': key=lambda z:(-z[1],-z[0]['ports'],-z[2],z[0]['sha'])
    elif mode=='sparse': key=lambda z:(z[2],z[1],z[0]['ports'],z[0]['sha'])
    else: key=lambda z:(z[0]['ports'],z[1],z[0]['sha'])
    arr=sorted(cand,key=key)
    if mode=='median': ans=arr[len(arr)//2][0]
    elif mode=='alt': ans=arr[(node_index*9973+17)%len(arr)][0]
    else: ans=arr[0][0]
    PICK_CACHE[ck]=ans
    return ans

def graph_shapes():
    return {
      2:[('edge',[(0,1)])],
      3:[('path',[(0,1),(1,2)]),('triangle',[(0,1),(1,2),(2,0)])],
      4:[('path',[(0,1),(1,2),(2,3)]),('star',[(0,1),(0,2),(0,3)]),('square',[(0,1),(1,2),(2,3),(3,0)]),('triangle_tail',[(0,1),(1,2),(2,0),(0,3)]),('beta2',[(0,1),(1,2),(2,3),(3,0),(0,2)])],
      5:[('path',[(0,1),(1,2),(2,3),(3,4)]),('star',[(0,1),(0,2),(0,3),(0,4)]),('C5',[(0,1),(1,2),(2,3),(3,4),(4,0)]),('figure8',[(0,1),(1,2),(2,0),(0,3),(3,4),(4,0)]),('C5_chord',[(0,1),(1,2),(2,3),(3,4),(4,0),(0,2)])],
      6:[('path',[(0,1),(1,2),(2,3),(3,4),(4,5)]),('star',[(0,1),(0,2),(0,3),(0,4),(0,5)]),('C6',[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)]),('two_triangles_bridge',[(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3)]),('C6_chords',[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3),(1,4)])]
    }

def edge_pattern(name, pairs):
    types=[]
    A=(IDX[(1,4)],IDX[(4,0)]);B=(IDX[(1,24)],IDX[(8,0)]);C=(IDX[(1,24)],IDX[(10,0)]);N=(0,0);P=(IDX[(1,0)],IDX[(10,0)])
    for k,(u,v) in enumerate(pairs):
        if name=='neutral': a,b=N
        elif name=='sel4': a,b=A
        elif name=='sel24_8': a,b=B
        elif name=='sel24_10': a,b=C
        elif name=='mixed': a,b=(A,B,C,P,N)[k%5]
        else: raise KeyError(name)
        types.append(edge_norm(u,a,v,b))
    return tuple(types)

def make_carrier(states,edges,meta=None):
    edges=canon_edges(edges); assert capacity_valid(states,edges)
    m=carrier_metrics(states,edges); rh,rk=role_info(states,edges); ek=exact_key(states,edges)
    return {'states':tuple(tuple(x) for x in states),'edges':edges,'exact_key':ek,'role_hash':rh,'role_repr':repr(rk),'metrics':m,'lanes':set(), 'parents':set(), 'parent_roles':set(), 'modulated':False, 'lift_tags':set(), **(meta or {})}

def serialize_carrier(c):
    return {**{k:v for k,v in c.items() if k not in ('lanes','parents','parent_roles','lift_tags')},'states':[list(x) for x in c['states']],'edges':[list(e) for e in c['edges']], 'lanes':sorted(c['lanes']),'parents':sorted(c['parents']),'parent_roles':sorted(c['parent_roles']),'lift_tags':sorted(c['lift_tags'])}

def deserialize_carrier(x):
    y=dict(x);y['states']=tuple(tuple(z) for z in x['states']);y['edges']=tuple(tuple(e) for e in x['edges']);
    for k in ('lanes','parents','parent_roles','lift_tags'):y[k]=set(x.get(k,[]))
    return y

def build_seeds(root,recs):
    seeds={}; relation_edges=[]
    # Deterministic bounded pilot: all frozen graph-shape families, four edge-typing regimes, three exact witness regimes.
    # Relation-rank coverage comes from different motif shapes/ranks; each seed also records its immediate connected prefix parent where one exists.
    modes=['stress','generous','alt']; pats=['neutral','sel4','sel24_8','mixed']
    for n,shs in graph_shapes().items():
        for sname,pairs in shs:
            for pname in pats:
                full=edge_pattern(pname,pairs); U=usage(n,full)
                for mode in modes:
                    sts=[pick_state(recs,U[i],mode,i)['state'] for i in range(n)]
                    c=make_carrier(sts,full,{'seed':f'{n}:{sname}:{pname}:{mode}:r{len(full)}','d':0})
                    if c['exact_key'] not in seeds: seeds[c['exact_key']]=c
                    if len(full)>1 and components(n,full[:-1])==1:
                        pcar=make_carrier(sts,full[:-1],{'d':0})
                        seeds[c['exact_key']]['parents'].add(pcar['exact_key']);seeds[c['exact_key']]['parent_roles'].add(pcar['role_hash']);relation_edges.append((pcar['exact_key'],c['exact_key']))
    pool=list(seeds.values()); select_lanes(pool,192)
    selected={c['exact_key']:c for c in pool if c['lanes']}
    out={'program':'IN9','d':0,'selected':[serialize_carrier(c) for c in sorted(selected.values(),key=lambda z:z['exact_key'])],'relation_seed_parent_edges':[list(x) for x in sorted(set(relation_edges))]}
    with gzip.open(root/'checkpoints/d000_selected_carriers.json.gz','wt') as f:json.dump(out,f,sort_keys=True)
    write_checkpoint_metrics(root,0,list(selected.values()),0.0,0)
    return len(selected)

def select_lanes(pool,cap):
    for c in pool:c['lanes']=set()
    # D
    for c in greedy_farthest(pool,min(cap,len(pool))):c['lanes'].add('D')
    # T
    for c in sorted(pool,key=lambda z:(z['metrics']['tight_margin'],z['exact_key']))[:cap]:c['lanes'].add('T')
    # C
    cyc=[c for c in pool if c['metrics']['beta']>=1]
    for c in sorted(cyc,key=lambda z:(-z['metrics']['beta'],-z['metrics']['selective_edges'],z['exact_key']))[:cap]:c['lanes'].add('C')
    # I
    ip=[c for c in pool if c.get('modulated')]
    for c in sorted(ip,key=lambda z:(-z['metrics']['selective_opportunity_count'],z['exact_key']))[:cap]:c['lanes'].add('I')
    # if no transition-modulated yet (d0), seed I from states with selective opportunity and tight margins
    if not ip:
        ip=[c for c in pool if c['metrics']['selective_opportunity_count']>0]
        for c in sorted(ip,key=lambda z:(z['metrics']['tight_margin'],-z['metrics']['selective_opportunity_count'],z['exact_key']))[:cap]:c['lanes'].add('I')
    # L
    lp=sorted(pool,key=lambda z:(-len(z['parents']),-len(z['parent_roles']),z['exact_key']))[:cap]
    for c in lp:c['lanes'].add('L')

def reduce_templates_for_node(st,res,templates):
    p=st[:7]; out={}
    for tm in templates:
        s=tm['source']
        if p[s]<=0:continue
        q=tuple(p[i]+tm['dp'][i] for i in range(7))
        if min(q)<0:continue
        strict=(p[s]-res[s]>0)
        cross=(not strict and all(q[i]>=res[i] for i in range(7)))
        if not strict and not cross:continue
        f=tuple(q[i]-res[i] for i in range(7))
        k=(support_mask(q),support_mask(f),s,'S' if strict else 'X',tuple(1 if q[i]>p[i] else -1 if q[i]<p[i] else 0 for i in range(7)))
        cand=(tm['tid'],tm['rank'],tm['primitive_port'],tm)
        if k not in out or cand[:3]<(out[k]['tid'],out[k]['rank'],out[k]['primitive_port']):out[k]=tm
    # Preserve all support/free-support outcomes, plus at most one neutral safe control and selective sources.
    vals=sorted(out.values(),key=lambda x:(x['source'],x['primitive_port'],x['dp'],x['dt'],x['tid']))
    if len(vals)<=24:return vals
    sel=[x for x in vals if x['source'] in (IDX[(1,4)],IDX[(1,24)],IDX[(4,0)],IDX[(8,0)],IDX[(10,0)])]
    neutral=[x for x in vals if x['source']==0][:2]
    rest=[x for x in vals if x not in sel and x not in neutral]
    return (sel+neutral+rest)[:24]

def generate_children(parent,templates):
    states=parent['states']; edges=parent['edges']; U=usage(len(states),edges); before=selective_profile(states,edges)
    children=[]
    for i,st in enumerate(states):
        for tm in reduce_templates_for_node(st,U[i],templates):
            p=st[:7];t=st[7:]; q=tuple(p[k]+tm['dp'][k] for k in range(7));tt=tuple(t[k]+tm['dt'][k] for k in range(9))
            if min(q)<0 or min(tt)<0:continue
            strict=(p[tm['source']]-U[i][tm['source']]>0)
            cross=(not strict and all(q[k]>=U[i][k] for k in range(7)))
            if not strict and not cross:continue
            ns=list(states);ns[i]=q+tt
            if not capacity_valid(ns,edges):continue
            after=selective_profile(ns,edges)
            c=make_carrier(ns,edges,{'d':parent.get('d',0)+1})
            c['parents'].add(parent['exact_key']);c['parent_roles'].add(parent['role_hash']);c['modulated']=(before!=after);c['lift_tags'].add('STRICT' if strict else 'CROSS_RANK')
            children.append(c)
    return children

def merge_children(cands):
    d={}
    for c in cands:
        k=c['exact_key']
        if k not in d:d[k]=c
        else:
            d[k]['parents']|=c['parents'];d[k]['parent_roles']|=c['parent_roles'];d[k]['modulated']=d[k]['modulated'] or c['modulated'];d[k]['lift_tags']|=c['lift_tags']
    return list(d.values())

def markers(panel):
    s=set()
    for c in panel:
        m=c['metrics']
        s.add('CYCLE' if m['beta']>=1 else 'FOREST')
        if m['beta']>1:s.add('BETA_GT1')
        if m['selective_edges']>0:s.add('SELECTIVE_EDGE')
        if m['selective_opportunity_count']>0:s.add('SELECTIVE_OPPORTUNITY')
        if m['tight_margin']<=0:s.add('TIGHT_CAPACITY')
        if len(c['parents'])>1:s.add('MULTIPARENT')
        if c.get('modulated'):s.add('INTERFACE_MODULATION')
    return sorted(s)

def write_checkpoint_metrics(root,d,panel,wall,tier):
    byrole=collections.defaultdict(list)
    for c in panel:byrole[c['role_hash']].append(c)
    dat={'d':d,'selected_count':len(panel),'role_count':len(byrole),'markers':markers(panel),'modulated_carriers':sum(c.get('modulated',False) for c in panel),'cyclic_carriers':sum(c['metrics']['beta']>=1 for c in panel),'multiparent_carriers':sum(len(c['parents'])>1 for c in panel),'wall_seconds':wall,'rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'tier':tier,'roles':{}}
    for r,xs in byrole.items():
        lane_counts={L:sum(L in c['lanes'] for c in xs) for L in LANES}
        dat['roles'][r]={'count':len(xs),'lanes':lane_counts,'exact_assignments':len({tuple(c['states']) for c in xs}),'same_role_parent':any(r in c['parent_roles'] for c in xs),'regenerated':any(r not in c['parent_roles'] and c['parent_roles'] for c in xs),'multiparent':any(len(c['parents'])>1 for c in xs),'selective':any(c['metrics']['selective_edges']>0 or c['metrics']['selective_opportunity_count']>0 for c in xs),'cyclic':any(c['metrics']['beta']>=1 for c in xs),'sample_role_repr':xs[0]['role_repr']}
    (root/f'checkpoints/d{d:03d}_metrics.json').write_text(json.dumps(dat,indent=2,sort_keys=True)+'\n')
    return dat

def load_panel(root,d):
    with gzip.open(root/f'checkpoints/d{d:03d}_selected_carriers.json.gz','rt') as f:x=json.load(f)
    return [deserialize_carrier(c) for c in x['selected']]

def save_panel(root,d,panel):
    out={'program':'IN9','d':d,'selected':[serialize_carrier(c) for c in sorted(panel,key=lambda z:z['exact_key'])]}
    tmp=root/f'checkpoints/d{d:03d}_selected_carriers.json.gz.tmp'
    with gzip.open(tmp,'wt') as f:json.dump(out,f,sort_keys=True)
    os.replace(tmp,root/f'checkpoints/d{d:03d}_selected_carriers.json.gz')

def run_d(root,d,templates,tier=0):
    t0=time.time(); prev=load_panel(root,d-1); cands=[]
    for p in prev:cands.extend(generate_children(p,templates))
    pool=merge_children(cands)
    cap=192 if tier<3 else 96
    select_lanes(pool,cap)
    selected=[c for c in pool if c['lanes']]
    save_panel(root,d,selected); wall=time.time()-t0
    m=write_checkpoint_metrics(root,d,selected,wall,tier)
    return m

def evaluate(root,current,window=16):
    if current<window-1:return None
    for start in range(max(0,current-window+1),current-window+2):
        ds=list(range(start,start+window)); mets=[json.load(open(root/f'checkpoints/d{d:03d}_metrics.json')) for d in ds]
        if len({tuple(m['markers']) for m in mets})!=1: continue
        roles=set.intersection(*(set(m['roles']) for m in mets)) if mets else set()
        cores=[]
        for r in sorted(roles):
            lane_full=[L for L in LANES if all(m['roles'][r]['lanes'].get(L,0)>0 for m in mets)]
            if len(lane_full)<2:continue
            direct=sum(mets[k]['roles'][r]['same_role_parent'] for k in range(1,window))
            if direct<12:continue
            regen=any(m['roles'][r]['regenerated'] for m in mets[1:])
            mp=sum(m['roles'][r]['multiparent'] for m in mets)
            if not regen and mp<4:continue
            if not any(m['roles'][r]['selective'] for m in mets):continue
            final8=set()
            for m in mets[-8:]: final8.add((m['roles'][r]['count'],m['roles'][r]['exact_assignments']))
            if sum(m['roles'][r]['exact_assignments'] for m in mets[-8:])<2:continue
            cores.append({'role_hash':r,'full_lanes':lane_full,'direct_persistence':direct,'regen':regen,'multiparent_checkpoints':mp,'cyclic':any(m['roles'][r]['cyclic'] for m in mets),'sample_role_repr':mets[-1]['roles'][r]['sample_role_repr']})
        if not cores:continue
        if not any(c['cyclic'] for c in cores):continue
        modcp=sum(m['modulated_carriers']>0 for m in mets)
        if modcp<4:continue
        if not any(any(m['roles'][c['role_hash']]['exact_assignments']>=2 for m in mets) for c in cores):continue
        return {'status':'RELATIONAL_MATURATION_CANDIDATE','window':[start,start+window-1],'cores':cores,'marker_repertoire':mets[0]['markers'],'modulation_checkpoints':modcp,'science_sha256':js_sha({'window':[start,start+window-1],'cores':cores,'markers':mets[0]['markers'],'modcp':modcp})}
    return None

def cmd_warmup(root):
    # integrity facts and small deterministic replay
    recs,spec,tatoms,templates=load_inputs(root); assert len(recs)==22885;assert len(templates)==77
    assert json.load(open(root/'inputs/IN6_RESULT.json'))['status']=='INTER_NODE_FINITE_CONTROL_GRAPH_GRAMMAR_EARNED'
    assert json.load(open(root/'inputs/IN8_RESULT.json'))['status']=='CROSS_LEVEL_CONNECTION_THREADING_EARNED_AS_RANK_STRUCTURE'
    # 18 unordered edge types
    und=set()
    for i,a in enumerate(PORTS):
        for j,b in enumerate(PORTS):
            if bridge(a,b):und.add(tuple(sorted((i,j))))
    assert len(und)==18
    out={'status':'PASS','observed_states':len(recs),'templates':len(templates),'edge_types':len(und),'sha256':js_sha([len(recs),len(templates),len(und)])}
    (root/'checkpoints/WARMUP.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,sort_keys=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--warmup',action='store_true');ap.add_argument('--seed',action='store_true');ap.add_argument('--run-one',type=int);ap.add_argument('--evaluate',type=int);ap.add_argument('--verify',type=int);ap.add_argument('--finalize',type=int);ap.add_argument('--tier',type=int,default=0);args=ap.parse_args();root=Path(args.root)
    if args.warmup:return cmd_warmup(root)
    recs,spec,tatoms,templates=load_inputs(root)
    if args.seed:
        n=build_seeds(root,recs);print(json.dumps({'status':'SEEDS_BUILT','selected':n},sort_keys=True));return
    if args.run_one is not None:
        m=run_d(root,args.run_one,templates,args.tier);print(json.dumps({'status':'D_DONE','d':args.run_one,'selected':m['selected_count'],'roles':m['role_count'],'wall':m['wall_seconds'],'rss_mb':m['rss_mb'],'markers':m['markers']},sort_keys=True));return
    if args.verify is not None:
        d=args.verify; panel=load_panel(root,d); old=json.load(open(root/f'checkpoints/d{d:03d}_metrics.json'));
        for c in panel:
            assert capacity_valid(c['states'],c['edges']); assert exact_key(c['states'],c['edges'])==c['exact_key']; assert role_info(c['states'],c['edges'])[0]==c['role_hash']
        chk={'status':'PASS','d':d,'selected':len(panel),'roles':len({c['role_hash'] for c in panel}),'markers':markers(panel),'panel_sha256':js_sha([serialize_carrier(c) for c in sorted(panel,key=lambda z:z['exact_key'])])}
        assert chk['selected']==old['selected_count'] and chk['roles']==old['role_count'] and chk['markers']==old['markers']; (root/f'checkpoints/d{d:03d}_terminal_replay.json').write_text(json.dumps(chk,indent=2,sort_keys=True)+'\n'); print(json.dumps(chk,sort_keys=True)); return
    if args.evaluate is not None:
        r=evaluate(root,args.evaluate)
        print(json.dumps(r or {'status':'NO_CANDIDATE','through':args.evaluate},sort_keys=True))
        if r:(root/'results/RELATIONAL_MATURATION_CANDIDATE.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
        return
    if args.finalize is not None:
        cur=args.finalize; mets=[json.load(open(root/f'checkpoints/d{d:03d}_metrics.json')) for d in range(cur+1) if (root/f'checkpoints/d{d:03d}_metrics.json').exists()]
        cand=json.load(open(root/'results/RELATIONAL_MATURATION_CANDIDATE.json')) if (root/'results/RELATIONAL_MATURATION_CANDIDATE.json').exists() else None
        rp=collections.defaultdict(lambda:{'checkpoints':[],'max_exact_assignments':0,'cyclic':False,'selective':False,'regen_checkpoints':[],'same_parent_checkpoints':[]})
        for m in mets:
            for r,z in m['roles'].items():
                q=rp[r];q['checkpoints'].append(m['d']);q['max_exact_assignments']=max(q['max_exact_assignments'],z['exact_assignments']);q['cyclic']=q['cyclic'] or z['cyclic'];q['selective']=q['selective'] or z['selective'];
                if z['regenerated']:q['regen_checkpoints'].append(m['d'])
                if z['same_role_parent']:q['same_parent_checkpoints'].append(m['d'])
        rolep={r:{**z,'first':min(z['checkpoints']),'last':max(z['checkpoints']),'count':len(z['checkpoints'])} for r,z in rp.items()}
        result={'program':'IN9_RELATIONAL_LONGITUDINAL_SCOUT_V1','through_d':cur,'status':cand['status'] if cand else ('UNRESOLVED' if cur<256 else 'FINITE_LOCAL_GRAMMAR_BUT_NO_RECURRENT_MOTIF_GRAMMAR'),'candidate':cand,'checkpoint_count':len(mets),'selected_counts':[m['selected_count'] for m in mets],'role_counts':[m['role_count'] for m in mets],'marker_repertoires':[m['markers'] for m in mets],'modulation_checkpoints':[m['d'] for m in mets if m['modulated_carriers']>0],'cyclic_checkpoints':[m['d'] for m in mets if m['cyclic_carriers']>0],'role_persistence':rolep,'scope':'Possibility-level relational Scout only; no geometry, physical time, actualization, propagation, psi, deletion or rewiring.'}
        result['science_sha256']=js_sha(result); (root/'results/IN9_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        with open(root/'results/COST_BY_CHECKPOINT.csv','w') as f:
            f.write('d,selected,roles,wall_seconds,rss_mb,tier,modulated,cyclic\n')
            for m in mets:f.write(f"{m['d']},{m['selected_count']},{m['role_count']},{m['wall_seconds']},{m['rss_mb']},{m['tier']},{m['modulated_carriers']},{m['cyclic_carriers']}\n")
        human=[f"IN9 status: {result['status']}",f"completed internal-lift checkpoints: 0..{cur}",f"science_sha256: {result['science_sha256']}"]
        if cand: human += [f"candidate window: {cand['window']}",f"core relational roles: {len(cand['cores'])}",'STOP: separate exact Relation-Grammar Closure Audit required before any graduation claim.']
        else: human += ['No maturation candidate under the frozen gate yet.']
        (root/'results/HUMAN_STATUS.txt').write_text('\n'.join(human)+'\n'); print(json.dumps({'status':result['status'],'through':cur,'science_sha256':result['science_sha256'],'cores':len(cand['cores']) if cand else 0},sort_keys=True)); return
if __name__=='__main__':main()
