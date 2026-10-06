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



# ===================== Scout2 V1.1 observer layer =====================
OBS_CANON_CACHE={}
EDGE_TYPES_UNORDERED=tuple(sorted({tuple(sorted((i,j))) for i,a in enumerate(PORTS) for j,b in enumerate(PORTS) if bridge(a,b)}))
EDGE_TYPE_INDEX={z:i for i,z in enumerate(EDGE_TYPES_UNORDERED)}
V11_STRESS_START=18
V11_WINDOW=32
V11_HORIZON=128


def canonical_colored_signature(colors,edges):
    """Exact node-renaming quotient for arbitrary immutable node colors plus typed incidence edges."""
    colors=tuple(colors); edges=tuple(edges); ck=(colors,edges)
    if ck in OBS_CANON_CACHE:return OBS_CANON_CACHE[ck]
    n=len(colors); groups=collections.defaultdict(list)
    for i,c in enumerate(colors):groups[c].append(i)
    sorted_cols=sorted(groups,key=repr); perms=[list(itertools.permutations(groups[c])) for c in sorted_cols]
    best=None
    for choice in itertools.product(*perms):
        perm=tuple(i for block in choice for i in block); inv=[0]*n
        for new,old in enumerate(perm):inv[old]=new
        ee=tuple(sorted(edge_norm(inv[u],a,inv[v],b) for u,a,v,b in edges))
        cand=(tuple(sorted(colors,key=repr)),ee)
        if best is None or repr(cand)<repr(best):best=cand
    if len(OBS_CANON_CACHE)>250000:OBS_CANON_CACHE.clear()
    OBS_CANON_CACHE[ck]=best
    return best


def node_action_masks(st,res,templates):
    p=st[:7]; strict=0; cross=0; atoms=[]
    for ti,tm in enumerate(templates):
        s=tm['source']
        if p[s]<=0:continue
        q=tuple(p[k]+tm['dp'][k] for k in range(7))
        if min(q)<0:continue
        is_strict=(p[s]-res[s]>0)
        is_cross=(not is_strict and all(q[k]>=res[k] for k in range(7)))
        if is_strict:
            strict|=1<<ti;atoms.append(f'IA:{ti}:S')
        elif is_cross:
            cross|=1<<ti;atoms.append(f'IA:{ti}:X')
    return strict,cross,atoms


def future_observer_signatures(c,templates):
    sts=c['states'];es=c['edges'];U=usage(len(sts),es);F=free_counts(sts,es);assert F is not None
    action_colors=[]; interface_colors=[]; capacity_colors=[]; grammar=set()
    for i,(st,u,f) in enumerate(zip(sts,U,F)):
        sm=support_mask(st[:7]);fsm=support_mask(f); strict,cross,atoms=node_action_masks(st,u,templates)
        action_colors.append((fsm,strict,cross)); interface_colors.append((fsm,));capacity_colors.append(tuple(st[:7]))
        grammar.add(f'NC:{sm}:{fsm}')
        grammar.update(atoms)
    for u,a,v,b in es:
        aa,bb=sorted((a,b));grammar.add(f'EA:{aa}:{bb}')
    relation_masks=[]
    for i in range(len(sts)):
        for j in range(i+1,len(sts)):
            mask=0
            for ai in range(7):
                if F[i][ai]<=0:continue
                for bj in range(7):
                    if F[j][bj]<=0 or not bridge(PORTS[ai],PORTS[bj]):continue
                    z=tuple(sorted((ai,bj)));mask|=1<<EDGE_TYPE_INDEX[z];grammar.add(f'RA:{z[0]}:{z[1]}')
            relation_masks.append(mask)
    fsig=sha_obj(canonical_colored_signature(action_colors,es))
    isig=sha_obj(canonical_colored_signature(interface_colors,es))
    psig=sha_obj(canonical_colored_signature(capacity_colors,es))
    # coarse deliberately lossy descriptor used as a kill-test comparator
    edge_ms=tuple(sorted(tuple(sorted((a,b))) for _,a,_,b in es))
    coarse=sha_obj((c['metrics']['n'],c['metrics']['r'],c['metrics']['beta'],tuple(c['metrics']['degrees']),c['metrics']['selective_edges'],edge_ms,tuple(sorted(sum(f[k] for f in F) for k in range(7)))))
    return {'future_support':fsig,'interface_support':isig,'port_capacity':psig,'coarse':coarse,'grammar':grammar,'relation_masks':tuple(relation_masks)}


def _partition_sufficiency(panel,sig_by_exact,keyfn):
    d=collections.defaultdict(set)
    for c in panel:d[keyfn(c)].add(sig_by_exact[c['exact_key']]['future_support'])
    bad={repr(k):sorted(v)[:4] for k,v in d.items() if len(v)>1}
    return {'classes':len(d),'ambiguous_classes':len(bad),'sufficient':not bad,'sample_ambiguities':dict(list(sorted(bad.items()))[:3])}


def observe_d(root,d,templates):
    panel=load_panel(root,d); t0=time.time(); sig={}
    grammar=set(f'MK:{m}' for m in markers(panel)); role_to_future=collections.defaultdict(set); future_info=collections.defaultdict(lambda:{'count':0,'roles':set(),'lanes':set(),'cyclic':False,'selective':False,'exact_keys':set(),'same_sig_parent':False,'regenerated':False,'multiparent':False})
    prev_map={}
    if d>0 and (root/f'checkpoints/d{d-1:03d}_observer_v11.json').exists():
        prev=json.load(open(root/f'checkpoints/d{d-1:03d}_observer_v11.json'));prev_map=prev.get('exact_to_future_support',{})
    for c in panel:
        z=future_observer_signatures(c,templates);sig[c['exact_key']]=z;grammar|=z['grammar'];role_to_future[c['role_hash']].add(z['future_support'])
        q=future_info[z['future_support']];q['count']+=1;q['roles'].add(c['role_hash']);q['lanes']|=c['lanes'];q['cyclic']|=(c['metrics']['beta']>=1);q['selective']|=(c['metrics']['selective_edges']>0 or c['metrics']['selective_opportunity_count']>0);q['exact_keys'].add(c['exact_key']);q['multiparent']|=(len(c['parents'])>1)
        ps={prev_map[p] for p in c['parents'] if p in prev_map}
        if z['future_support'] in ps:q['same_sig_parent']=True
        if ps and z['future_support'] not in ps:q['regenerated']=True
    # Cumulative vocab / role / action-signature history.
    seen_g=set();seen_r=set();seen_f=set(); absent_prev=set(); prior_roles=set();prev_roles=set()
    for k in range(0,d):
        op=root/f'checkpoints/d{k:03d}_observer_v11.json'
        if not op.exists():continue
        x=json.load(open(op));seen_g.update(x.get('grammar_elements',[]));seen_r.update(x.get('role_hashes',[]));seen_f.update(x.get('future_support_signatures',[]));prior_roles.update(x.get('role_hashes',[]))
        if k==d-1:prev_roles=set(x.get('role_hashes',[]))
    cur_roles=set(c['role_hash'] for c in panel);cur_f=set(z['future_support'] for z in sig.values())
    new_g=grammar-seen_g;new_r=cur_roles-seen_r;new_f=cur_f-seen_f; rebound=(cur_roles & prior_roles)-prev_roles
    envelope={'candidate_role_hash':None,'union_roles':[]}
    ep=root/'inputs/IN10_CLOSURE_ENVELOPE.json'
    if ep.exists():envelope=json.load(open(ep))
    env=set(envelope.get('union_roles',[]));env_cur=cur_roles&env
    # Prime-invariant / local sufficiency ladder for future-action support.
    p0=_partition_sufficiency(panel,sig,lambda c:sig[c['exact_key']]['coarse'])
    p1=_partition_sufficiency(panel,sig,lambda c:c['role_hash'])
    p2=_partition_sufficiency(panel,sig,lambda c:sig[c['exact_key']]['port_capacity'])
    p3=_partition_sufficiency(panel,sig,lambda c:c['exact_key'])
    exact_by_port=collections.defaultdict(set)
    for c in panel:exact_by_port[sig[c['exact_key']]['port_capacity']].add(c['exact_key'])
    multiT=sum(len(v)>1 for v in exact_by_port.values())
    fi={h:{'count':q['count'],'roles':sorted(q['roles']),'lanes':sorted(q['lanes']),'cyclic':q['cyclic'],'selective':q['selective'],'exact_assignments':len(q['exact_keys']),'same_sig_parent':q['same_sig_parent'],'regenerated':q['regenerated'],'multiparent':q['multiparent']} for h,q in sorted(future_info.items())}
    out={
      'program':'IN9_SCOUT2_V1_1_OBSERVER','d':d,'selected_count':len(panel),'role_count':len(cur_roles),'future_support_signature_count':len(cur_f),'interface_support_signature_count':len({z['interface_support'] for z in sig.values()}),'port_capacity_signature_count':len({z['port_capacity'] for z in sig.values()}),
      'grammar_element_count':len(grammar),'grammar_elements':sorted(grammar),'new_grammar_elements':sorted(new_g),'new_grammar_element_count':len(new_g),'role_hashes':sorted(cur_roles),'new_role_count':len(new_r),'new_role_hashes':sorted(new_r),'rebounded_role_count':len(rebound),'rebounded_role_hashes':sorted(rebound),'future_support_signatures':sorted(cur_f),'new_future_support_signature_count':len(new_f),'new_future_support_signatures':sorted(new_f),
      'closure_envelope':{'candidate_present':envelope.get('candidate_role_hash') in cur_roles if envelope.get('candidate_role_hash') else False,'envelope_roles_present':len(env_cur),'envelope_role_total':len(env),'present_hashes':sorted(env_cur)},
      'prime_invariant_sufficiency':{'P0_global_coarse':p0,'P1_scout_role':p1,'P2_exact_port_counters_plus_typed_incidence':p2,'P3_full_exact_carrier':p3,'port_signature_classes_with_multiple_full_exact_T_states':multiT,'interpretation':'P2 omits T but retains exact per-node port counters plus typed incidence. For currently earned move eligibility, T is not read; full T remains required for exact child identity.'},
      'future_signature_info':fi,'exact_to_future_support':{k:z['future_support'] for k,z in sorted(sig.items())},
      'compression':{'exact_to_scout_role':len(panel)/max(1,len(cur_roles)),'exact_to_future_support':len(panel)/max(1,len(cur_f)),'exact_to_interface_support':len(panel)/max(1,len({z['interface_support'] for z in sig.values()}))},
      'markers':markers(panel),'wall_seconds':time.time()-t0
    }
    out['observer_sha256']=js_sha({k:v for k,v in out.items() if k not in ('wall_seconds','observer_sha256')})
    tmp=root/f'checkpoints/d{d:03d}_observer_v11.json.tmp';tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');os.replace(tmp,root/f'checkpoints/d{d:03d}_observer_v11.json')
    return out


def retrospective_observe(root,through,templates):
    arr=[]
    for d in range(through+1):
        x=observe_d(root,d,templates);arr.append({'d':d,'roles':x['role_count'],'future':x['future_support_signature_count'],'new_grammar':x['new_grammar_element_count'],'new_roles':x['new_role_count'],'new_future':x['new_future_support_signature_count'],'env':x['closure_envelope']['envelope_roles_present']})
    (root/'results/V11_RETROSPECTIVE_BASELINE.json').write_text(json.dumps({'through':through,'checkpoints':arr,'status':'RETROSPECTIVE_OBSERVER_ONLY_NO_GENERATION_CHANGED'},indent=2,sort_keys=True)+'\n')
    return arr


def evaluate_v11(root,current,window=V11_WINDOW):
    # Prospective stress gate: only checkpoints generated after the V1.1 freeze may count.
    if current < V11_STRESS_START+window-1:return {'status':'V11_STRESS_WINDOW_INCOMPLETE','through':current,'first_eligible_end':V11_STRESS_START+window-1}
    start=current-window+1
    if start<V11_STRESS_START:return {'status':'V11_STRESS_WINDOW_INCOMPLETE','through':current,'first_eligible_end':V11_STRESS_START+window-1}
    obs=[json.load(open(root/f'checkpoints/d{d:03d}_observer_v11.json')) for d in range(start,current+1)]
    mets=[json.load(open(root/f'checkpoints/d{d:03d}_metrics.json')) for d in range(start,current+1)]
    # Grammar must have stopped requiring new local control atoms for the final half-window.
    no_new_grammar=all(x['new_grammar_element_count']==0 for x in obs[-16:])
    stable_markers=len({tuple(x['markers']) for x in obs})==1
    p2_sufficient=all(x['prime_invariant_sufficiency']['P2_exact_port_counters_plus_typed_incidence']['sufficient'] for x in obs)
    # Recurring future-action signatures are the behavioral cores, not merely coarse graph roles.
    fs=set.intersection(*(set(x['future_support_signatures']) for x in obs))
    cores=[]
    for h in sorted(fs):
        infos=[x['future_signature_info'][h] for x in obs]
        lane_full=[L for L in LANES if all(L in z['lanes'] for z in infos)]
        if len(lane_full)<2:continue
        direct=sum(z['same_sig_parent'] for z in infos[1:])
        if direct<24:continue
        regen=any(z['regenerated'] for z in infos[1:]);mp=sum(z['multiparent'] for z in infos)
        if not regen and mp<8:continue
        if not any(z['cyclic'] for z in infos):continue
        if not any(z['selective'] for z in infos):continue
        if max(z['exact_assignments'] for z in infos)<2:continue
        cores.append({'future_support_signature':h,'full_lanes':lane_full,'direct_persistence':direct,'regen':regen,'multiparent_checkpoints':mp,'cyclic':any(z['cyclic'] for z in infos),'selective':any(z['selective'] for z in infos),'max_exact_assignments':max(z['exact_assignments'] for z in infos)})
    modulation=sum(m['modulated_carriers']>0 for m in mets)
    envcp=sum(x['closure_envelope']['envelope_roles_present']>0 for x in obs)
    compression_ok=all(x['compression']['exact_to_future_support']>1.0 for x in obs[-16:])
    shock_depths=[x['d'] for x in obs if x['new_grammar_element_count']>0]
    result={'status':'NO_O2_MATURATION_CANDIDATE_V1_1','window':[start,current],'no_new_grammar_final16':no_new_grammar,'stable_markers':stable_markers,'P2_future_action_sufficiency':p2_sufficient,'behavioral_core_count':len(cores),'modulation_checkpoints':modulation,'closure_envelope_checkpoints':envcp,'compression_final16':compression_ok,'grammar_shock_depths':shock_depths,'cores':cores}
    if no_new_grammar and stable_markers and p2_sufficient and cores and modulation>=16 and envcp>=16 and compression_ok:
        result['status']='O2_MATURATION_CANDIDATE_V1_1';result['science_sha256']=js_sha(result)
        (root/'results/O2_MATURATION_CANDIDATE_V1_1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def finalize_v11(root,cur):
    obs=[json.load(open(root/f'checkpoints/d{d:03d}_observer_v11.json')) for d in range(cur+1) if (root/f'checkpoints/d{d:03d}_observer_v11.json').exists()]
    cand=json.load(open(root/'results/O2_MATURATION_CANDIDATE_V1_1.json')) if (root/'results/O2_MATURATION_CANDIDATE_V1_1.json').exists() else None
    last16=obs[-16:] if len(obs)>=16 else obs
    if cand:status=cand['status']
    elif cur>=V11_HORIZON:
        if any(x['new_grammar_element_count']>0 for x in last16):status='OPEN_ENDED_RELATIONAL_NOVELTY'
        elif len({tuple(x['markers']) for x in last16})>1:status='REGIME_REORGANIZATION'
        elif not all(x['prime_invariant_sufficiency']['P2_exact_port_counters_plus_typed_incidence']['sufficient'] for x in last16):status='NEW_CARRIER_VARIABLE_REQUIRED'
        else:status='FINITE_CONTROL_WITHOUT_O2_ENTITY_GRADUATION'
    else:status='UNRESOLVED_STRESS_IN_PROGRESS'
    out={'program':'SCOUT2_RELATIONAL_LONGITUDINAL_V1_1','through_d':cur,'status':status,'candidate':cand,'stress_start':V11_STRESS_START,'window':V11_WINDOW,'horizon':V11_HORIZON,'observer_checkpoints':len(obs),'role_counts':[x['role_count'] for x in obs],'future_support_signature_counts':[x['future_support_signature_count'] for x in obs],'grammar_element_counts':[x['grammar_element_count'] for x in obs],'new_grammar_depths':[x['d'] for x in obs if x['new_grammar_element_count']>0],'closure_envelope_counts':[x['closure_envelope']['envelope_roles_present'] for x in obs],'scope':'Possibility-level relational organization only; no geometry, physical time, actualization, propagation, psi, deletion or rewiring.'}
    out['science_sha256']=js_sha(out);(root/'results/SCOUT2_V1_1_RESULT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    return out
# =================== end Scout2 V1.1 observer layer ==================

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--warmup',action='store_true');ap.add_argument('--seed',action='store_true');ap.add_argument('--run-one',type=int);ap.add_argument('--run-one-v11',type=int);ap.add_argument('--evaluate',type=int);ap.add_argument('--verify',type=int);ap.add_argument('--finalize',type=int);ap.add_argument('--observe',type=int);ap.add_argument('--retrospective-observe',type=int);ap.add_argument('--evaluate-v11',type=int);ap.add_argument('--finalize-v11',type=int);ap.add_argument('--tier',type=int,default=0);args=ap.parse_args();root=Path(args.root)
    if args.warmup:return cmd_warmup(root)
    if args.evaluate_v11 is not None:
        x=evaluate_v11(root,args.evaluate_v11);print(json.dumps(x,sort_keys=True));return
    if args.finalize_v11 is not None:
        x=finalize_v11(root,args.finalize_v11);print(json.dumps({'status':x['status'],'through':args.finalize_v11,'science_sha256':x['science_sha256']},sort_keys=True));return
    recs,spec,tatoms,templates=load_inputs(root)
    if args.retrospective_observe is not None:
        arr=retrospective_observe(root,args.retrospective_observe,templates);print(json.dumps({'status':'V11_RETROSPECTIVE_OBSERVERS_DONE','through':args.retrospective_observe,'checkpoints':len(arr)},sort_keys=True));return
    if args.observe is not None:
        x=observe_d(root,args.observe,templates);print(json.dumps({'status':'V11_OBSERVER_DONE','d':args.observe,'roles':x['role_count'],'future_signatures':x['future_support_signature_count'],'new_grammar':x['new_grammar_element_count'],'envelope':x['closure_envelope']['envelope_roles_present'],'wall':x['wall_seconds']},sort_keys=True));return
    if args.run_one_v11 is not None:
        m=run_d(root,args.run_one_v11,templates,args.tier);o=observe_d(root,args.run_one_v11,templates);print(json.dumps({'status':'D_AND_V11_OBSERVER_DONE','d':args.run_one_v11,'selected':m['selected_count'],'roles':m['role_count'],'wall':m['wall_seconds'],'rss_mb':m['rss_mb'],'new_grammar':o['new_grammar_element_count'],'future_signatures':o['future_support_signature_count'],'envelope':o['closure_envelope']['envelope_roles_present']},sort_keys=True));return
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
