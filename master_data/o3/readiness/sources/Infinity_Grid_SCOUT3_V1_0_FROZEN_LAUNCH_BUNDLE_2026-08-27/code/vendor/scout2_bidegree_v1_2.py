#!/usr/bin/env python3
import argparse, ast, bisect, collections, gzip, hashlib, importlib.util, itertools, json, os, resource, sys, time
from pathlib import Path

ROOT_CODE=Path(__file__).resolve().parent

def load_mod(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

sc=load_mod(ROOT_CODE/'scout2_relational_longitudinal_v1_1.py','scv11')
PORTS=sc.PORTS; LANES=sc.LANES
K_WINDOW=32; K_HORIZON=128; CONTROL_PERIOD=8


def js_sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def load_json(path): return json.load(open(path))

def atomic_json(path,obj):
    path=Path(path); tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def load_input_panel(root,d):
    with gzip.open(root/f'inputs/d{d:03d}_selected_carriers.json.gz','rt') as f: x=json.load(f)
    return [sc.deserialize_carrier(c) for c in x['selected']]

def load_k_panel(root,k):
    with gzip.open(root/f'checkpoints/k{k:03d}_selected_carriers.json.gz','rt') as f:x=json.load(f)
    return [sc.deserialize_carrier(c) for c in x['selected']]

def save_k_panel(root,k,panel):
    out={'program':'SCOUT2_BIDEGREE_V1_2','k':k,'selected':[sc.serialize_carrier(c) for c in sorted(panel,key=lambda z:z['exact_key'])]}
    p=root/f'checkpoints/k{k:03d}_selected_carriers.json.gz'; tmp=Path(str(p)+'.tmp')
    with gzip.open(tmp,'wt') as f:json.dump(out,f,sort_keys=True)
    os.replace(tmp,p)

def seed_k0(root):
    panel=load_input_panel(root,109)
    out=[]
    for c in panel:
        c['parents']=set();c['parent_roles']=set();c['lift_tags']=set();c['modulated']=False;c['k']=0;c['anchor_d']=109
        out.append(c)
    save_k_panel(root,0,out)
    write_metrics(root,0,out,0.0,0,seed=True)
    return len(out)


# ===================== V1.2 implementation-only optimizations =====================
# These helpers preserve the frozen candidate enumeration, exact anonymous-node
# quotient, lane definitions and tie-breaking. They only avoid recomputing state
# that a one-edge relation addition cannot change.
BRIDGE_BJ=tuple(tuple(b for b in range(7) if sc.bridge(PORTS[a],PORTS[b])) for a in range(7))
SELECTIVE_PAIRS=tuple(sorted(sc.SELECTIVE))
SELECTIVE_RELEVANT=frozenset((sc.IDX[(1,4)],sc.IDX[(4,0)],sc.IDX[(1,24)],sc.IDX[(8,0)],sc.IDX[(10,0)]))
_EXACT_STATE_PLAN_CACHE={}
_NODE_ACTION_CACHE={}
_ORIG_NODE_ACTION_MASKS=sc.node_action_masks

def _node_action_masks_cached(st,res,templates):
    key=(st,res)
    z=_NODE_ACTION_CACHE.get(key)
    if z is None:
        a,b,atoms=_ORIG_NODE_ACTION_MASKS(st,res,templates)
        z=(a,b,tuple(atoms));_NODE_ACTION_CACHE[key]=z
    return z

# Observer semantics are unchanged; only repeated node-local exact action queries are memoized.
sc.node_action_masks=_node_action_masks_cached

def load_templates_light(root):
    spec=load_json(root/'inputs/MATURE_NODE_ALGEBRA_SPEC.json');tatoms=tuple(spec['T_atoms'])
    primj=load_json(root/'inputs/FROZEN_PRIMITIVES.json')['primitive'];prims=[]
    for q in primj:
        x=dict(q);x['record']=ast.literal_eval(q['record']);prims.append(x)
    templates=sc.derive_templates(prims,tatoms)
    if len(templates)!=77:raise AssertionError(len(templates))
    return spec,tatoms,templates

def _insert_sorted_edge(edges,e):
    j=bisect.bisect_left(edges,e)
    return edges[:j]+(e,)+edges[j:]

def _exact_state_plan(states):
    states=tuple(states)
    z=_EXACT_STATE_PLAN_CACHE.get(states)
    if z is not None:return z
    groups=collections.defaultdict(list)
    for i,c in enumerate(states):groups[c].append(i)
    sorted_cols=sorted(groups)
    blocks=[tuple(itertools.permutations(groups[c])) for c in sorted_cols]
    invs=[]; n=len(states)
    for choice in itertools.product(*blocks):
        perm=tuple(i for block in choice for i in block);inv=[0]*n
        for new,old in enumerate(perm):inv[old]=new
        invs.append(tuple(inv))
    z=(tuple(sorted(states)),tuple(invs))
    _EXACT_STATE_PLAN_CACHE[states]=z
    return z

def _relation_exact_plan(parent):
    sorted_colors,invs=_exact_state_plan(parent['states'])
    bases=[]
    for inv in invs:
        bases.append(tuple(sorted(sc.edge_norm(inv[u],a,inv[v],b) for u,a,v,b in parent['edges'])))
    return sorted_colors,invs,tuple(bases)

def _relation_child_exact_key(plan,i,ai,j,bj):
    sorted_colors,invs,bases=plan;best=None
    for inv,base in zip(invs,bases):
        e=sc.edge_norm(inv[i],ai,inv[j],bj)
        ee=_insert_sorted_edge(base,e)
        if best is None or ee<best:best=ee
    return sc.sha_obj((sorted_colors,best))

def _component_labels(n,edges):
    adj=[[] for _ in range(n)]
    for u,a,v,b in edges:adj[u].append(v);adj[v].append(u)
    lab=[-1]*n;c=0
    for s in range(n):
        if lab[s]>=0:continue
        lab[s]=c;stack=[s]
        while stack:
            x=stack.pop()
            for y in adj[x]:
                if lab[y]<0:lab[y]=c;stack.append(y)
        c+=1
    return tuple(lab),c

def _selective_opportunity_after(F,i,ai,j,bj):
    def f(n,p):
        return F[n][p]-(1 if (n==i and p==ai) or (n==j and p==bj) else 0)
    tot=0;n=len(F)
    for x in range(n):
        for y in range(x+1,n):
            for a,b in SELECTIVE_PAIRS:
                xa=f(x,a);xb=f(x,b);ya=f(y,a);yb=f(y,b)
                if xa>0 and yb>0:tot+=xa*yb
                if xb>0 and ya>0:tot+=xb*ya
    return tot

def _compact_relation_child(parent,ctx,i,ai,j,bj):
    F,U,rawdeg,labels,ncomp,plan=ctx
    e=sc.edge_norm(i,ai,j,bj);edges=_insert_sorted_edge(parent['edges'],e)
    ek=_relation_child_exact_key(plan,i,ai,j,bj)
    deg=list(rawdeg);deg[i]+=1;deg[j]+=1
    comps=ncomp if labels[i]==labels[j] else ncomp-1
    r=len(edges);n=len(parent['states'])
    mi=parent['states'][i][ai]-U[i][ai]-1; mj=parent['states'][j][bj]-U[j][bj]-1
    if parent['edges']:tight=min(parent['metrics']['tight_margin'],mi,mj)
    else:tight=min(mi,mj)
    seladd=1 if ((ai,bj) in sc.SELECTIVE or (bj,ai) in sc.SELECTIVE) else 0
    m={'n':n,'r':r,'beta':r-n+comps,'degrees':sorted(deg,reverse=True),
       'selective_edges':parent['metrics']['selective_edges']+seladd,
       'selective_opportunity_count':_selective_opportunity_after(F,i,ai,j,bj),
       'tight_margin':tight,'total_free_ports':parent['metrics']['total_free_ports']-2}
    return {'states':parent['states'],'edges':edges,'exact_key':ek,'metrics':m,'lanes':set(),
            'parents':{parent['exact_key']},'parent_roles':{parent['role_hash']},
            'modulated':(ai in SELECTIVE_RELEVANT or bj in SELECTIVE_RELEVANT),'lift_tags':set(),
            'k':parent.get('k',0)+1,'anchor_d':parent.get('anchor_d',109)}

def _relation_parent_context(parent):
    F=tuple(tuple(x) for x in parent['metrics']['free_counts'])
    U=tuple(tuple(x) for x in parent['metrics']['reservation_vectors'])
    deg=[0]*len(parent['states'])
    for u,a,v,b in parent['edges']:deg[u]+=1;deg[v]+=1
    labels,ncomp=_component_labels(len(parent['states']),parent['edges'])
    return F,U,tuple(deg),labels,ncomp,_relation_exact_plan(parent)

def greedy_farthest_exact_reduced(pool,cap):
    if len(pool)<=cap:return list(pool)
    X=sc.norm_features(pool)
    first=min(range(len(pool)),key=lambda i:pool[i]['exact_key'])
    earliest={}
    for i,x in enumerate(X):earliest.setdefault(tuple(x),i)
    if len(earliest)<cap:
        return sc.greedy_farthest(pool,cap)
    idxs=sorted(set(earliest.values())|{first})
    chosen=[first];chosen_set={first}
    mind={}
    xf=X[first]
    for i in idxs:
        d=sum((X[i][j]-xf[j])**2 for j in range(len(X[i])))
        if i!=first and tuple(X[i])!=tuple(xf) and d==0:
            return sc.greedy_farthest(pool,cap)
        mind[i]=d
    mind[first]=-1
    while len(chosen)<cap:
        avail=(i for i in idxs if i not in chosen_set)
        k=max(avail,key=lambda i:(mind[i],-i))
        chosen.append(k);chosen_set.add(k);mind[k]=-1
        xk=X[k]
        for i in idxs:
            if i in chosen_set:continue
            d=sum((X[i][j]-xk[j])**2 for j in range(len(X[i])))
            if tuple(X[i])!=tuple(xk) and d==0:
                return sc.greedy_farthest(pool,cap)
            if d<mind[i]:mind[i]=d
    return [pool[i] for i in chosen]

def select_lanes_fast(pool,cap):
    for c in pool:c['lanes']=set()
    for c in greedy_farthest_exact_reduced(pool,min(cap,len(pool))):c['lanes'].add('D')
    for c in sorted(pool,key=lambda z:(z['metrics']['tight_margin'],z['exact_key']))[:cap]:c['lanes'].add('T')
    cyc=[c for c in pool if c['metrics']['beta']>=1]
    for c in sorted(cyc,key=lambda z:(-z['metrics']['beta'],-z['metrics']['selective_edges'],z['exact_key']))[:cap]:c['lanes'].add('C')
    ip=[c for c in pool if c.get('modulated')]
    for c in sorted(ip,key=lambda z:(-z['metrics']['selective_opportunity_count'],z['exact_key']))[:cap]:c['lanes'].add('I')
    if not ip:
        ip=[c for c in pool if c['metrics']['selective_opportunity_count']>0]
        for c in sorted(ip,key=lambda z:(z['metrics']['tight_margin'],-z['metrics']['selective_opportunity_count'],z['exact_key']))[:cap]:c['lanes'].add('I')
    lp=sorted(pool,key=lambda z:(-len(z['parents']),-len(z['parent_roles']),z['exact_key']))[:cap]
    for c in lp:c['lanes'].add('L')

def _materialize_compact(c):
    full=sc.make_carrier(c['states'],c['edges'],{'k':c.get('k',0),'anchor_d':c.get('anchor_d',109)})
    if full['exact_key']!=c['exact_key']:
        raise AssertionError(('optimized exact-key mismatch',c['exact_key'],full['exact_key']))
    full['lanes']=set(c['lanes']);full['parents']=set(c['parents']);full['parent_roles']=set(c['parent_roles']);full['modulated']=bool(c['modulated']);full['lift_tags']=set(c.get('lift_tags',()))
    return full

def relation_additions(parent):
    # Frozen enumeration order, but with exact one-edge incremental bookkeeping.
    F=tuple(tuple(x) for x in parent['metrics']['free_counts']);n=len(parent['states']);ctx=_relation_parent_context(parent);children=[]
    for i in range(n):
        for j in range(i+1,n):
            for ai in range(7):
                if F[i][ai]<=0:continue
                for bj in BRIDGE_BJ[ai]:
                    if F[j][bj]<=0:continue
                    children.append(_compact_relation_child(parent,ctx,i,ai,j,bj))
    return children

def merge_children(cands):
    d={}
    for c in cands:
        k=c['exact_key']
        if k not in d:d[k]=c
        else:
            d[k]['parents']|=c['parents'];d[k]['parent_roles']|=c['parent_roles'];d[k]['modulated']=d[k]['modulated'] or c['modulated']
    return list(d.values())

def lane_cap(tier): return {0:192,1:160,2:128,3:96}[tier]

def possible_relation_count(c):
    F=c['metrics'].get('free_counts')
    if F is None:F=sc.free_counts(c['states'],c['edges'])
    n=len(c['states']);z=0
    for i in range(n):
        for j in range(i+1,n):
            for ai in range(7):
                if F[i][ai]<=0:continue
                for bj in BRIDGE_BJ[ai]:
                    if F[j][bj]>0:z+=1
    return z

def _cache_dir(root):
    p=root/'cache';p.mkdir(parents=True,exist_ok=True);return p

def _write_metrics_eval_cache(root,k,dat):
    q={'k':k,'modulated_carriers':dat['modulated_carriers'],'possible_relation_additions_total':dat['possible_relation_additions_total'],'selected_count':dat['selected_count']}
    atomic_json(_cache_dir(root)/f'k{k:03d}_metrics_eval.json',q)

def _write_observer_eval_cache(root,k,out):
    q={'k':k,'new_grammar_element_count':out['new_grammar_element_count'],'markers':out['markers'],
       'prime_invariant_sufficiency':{'P2_exact_port_counters_plus_typed_incidence':out['prime_invariant_sufficiency']['P2_exact_port_counters_plus_typed_incidence']},
       'future_support_signatures':out['future_support_signatures'],'future_signature_info':out['future_signature_info'],
       'compression':out['compression']}
    atomic_json(_cache_dir(root)/f'k{k:03d}_observer_eval.json',q)

def _observer_prior_unions(root,k):
    if k<=0:return set(),set(),set(),set()
    cp=_cache_dir(root)/f'k{k-1:03d}_observer_cumulative.json'
    if cp.exists():
        try:
            z=load_json(cp)
            if z.get('through_k')==k-1:
                prevobs=load_json(root/f'checkpoints/k{k-1:03d}_observer.json')
                if z.get('terminal_observer_sha256')==prevobs.get('observer_sha256'):
                    return set(z['grammar']),set(z['future']),set(z['roles']),set(z['current_roles'])
        except Exception:
            pass
    seen=set();seenf=set();seenr=set();prevroles=set();lastsha=None
    for qk in range(k):
        pp=root/f'checkpoints/k{qk:03d}_observer.json'
        if not pp.exists():continue
        x=load_json(pp);seen.update(x.get('grammar_elements',[]));seenf.update(x.get('future_support_signatures',[]));seenr.update(x.get('role_hashes',[]))
        if qk==k-1:prevroles=set(x.get('role_hashes',[]));lastsha=x.get('observer_sha256')
    if lastsha is not None:
        atomic_json(cp,{'through_k':k-1,'grammar':sorted(seen),'future':sorted(seenf),'roles':sorted(seenr),'current_roles':sorted(prevroles),'terminal_observer_sha256':lastsha})
    return seen,seenf,seenr,prevroles

def _write_observer_cumulative(root,k,out,prior):
    seen,seenf,seenr,_=prior
    atomic_json(_cache_dir(root)/f'k{k:03d}_observer_cumulative.json',{'through_k':k,
      'grammar':sorted(seen|set(out['grammar_elements'])),'future':sorted(seenf|set(out['future_support_signatures'])),
      'roles':sorted(seenr|set(out['role_hashes'])),'current_roles':out['role_hashes'],'terminal_observer_sha256':out['observer_sha256']})

def _load_observer_eval(root,k):
    p=_cache_dir(root)/f'k{k:03d}_observer_eval.json'
    return load_json(p) if p.exists() else load_json(root/f'checkpoints/k{k:03d}_observer.json')

def _load_metrics_eval(root,k):
    p=_cache_dir(root)/f'k{k:03d}_metrics_eval.json'
    return load_json(p) if p.exists() else load_json(root/f'checkpoints/k{k:03d}_metrics.json')

def write_metrics(root,k,panel,wall,tier,seed=False,pool_size=None):
    byrole=collections.defaultdict(list)
    for c in panel:byrole[c['role_hash']].append(c)
    poss=[possible_relation_count(c) for c in panel]
    dat={'program':'SCOUT2_BIDEGREE_V1_2','k':k,'seed':seed,'selected_count':len(panel),'pool_size':pool_size if pool_size is not None else len(panel),'role_count':len(byrole),'markers':sc.markers(panel),'modulated_carriers':sum(c.get('modulated',False) for c in panel),'cyclic_carriers':sum(c['metrics']['beta']>=1 for c in panel),'multiparent_carriers':sum(len(c['parents'])>1 for c in panel),'possible_relation_additions_total':sum(poss),'possible_relation_additions_minmax':[min(poss) if poss else 0,max(poss) if poss else 0],'wall_seconds':wall,'rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'tier':tier,'roles':{}}
    for r,xs in byrole.items():
        lane_counts={L:sum(L in c['lanes'] for c in xs) for L in LANES}
        dat['roles'][r]={'count':len(xs),'lanes':lane_counts,'exact_assignments':len({c['exact_key'] for c in xs}),'same_role_parent':any(r in c['parent_roles'] for c in xs),'regenerated':any(c['parent_roles'] and r not in c['parent_roles'] for c in xs),'multiparent':any(len(c['parents'])>1 for c in xs),'selective':any(c['metrics']['selective_edges']>0 or c['metrics']['selective_opportunity_count']>0 for c in xs),'cyclic':any(c['metrics']['beta']>=1 for c in xs),'sample_role_repr':xs[0]['role_repr']}
    atomic_json(root/f'checkpoints/k{k:03d}_metrics.json',dat);_write_metrics_eval_cache(root,k,dat);return dat

def run_k(root,k,tier):
    assert k>=1
    prev=load_k_panel(root,k-1);t0=time.time();d={}
    # Streaming exact dedup preserves the original first-occurrence representative and pool order.
    for p in prev:
        for c in relation_additions(p):
            ek=c['exact_key'];q=d.get(ek)
            if q is None:d[ek]=c
            else:
                q['parents']|=c['parents'];q['parent_roles']|=c['parent_roles'];q['modulated']=q['modulated'] or c['modulated']
    pool=list(d.values())
    if not pool:
        write_metrics(root,k,[],time.time()-t0,tier,pool_size=0);save_k_panel(root,k,[]);return {'status':'RELATION_CAPACITY_SATURATION','k':k,'selected':0,'pool':0}
    select_lanes_fast(pool,lane_cap(tier)); selected=[_materialize_compact(c) for c in pool if c['lanes']]
    save_k_panel(root,k,selected); m=write_metrics(root,k,selected,time.time()-t0,tier,pool_size=len(pool))
    return {'status':'K_DONE','k':k,'selected':m['selected_count'],'pool':len(pool),'roles':m['role_count'],'wall':m['wall_seconds'],'rss_mb':m['rss_mb'],'possible':m['possible_relation_additions_total']}

def part_suff(panel,sig,keyfn):
    d=collections.defaultdict(set)
    for c in panel:d[keyfn(c)].add(sig[c['exact_key']]['future_support'])
    bad=sum(len(v)>1 for v in d.values())
    return {'classes':len(d),'ambiguous_classes':bad,'sufficient':bad==0}

def observe_k(root,k,templates):
    panel=load_k_panel(root,k);t0=time.time();sig={}; grammar=set(f'MK:{m}' for m in sc.markers(panel));fi=collections.defaultdict(lambda:{'count':0,'lanes':set(),'cyclic':False,'selective':False,'exact_keys':set(),'same_sig_parent':False,'regenerated':False,'multiparent':False})
    prev_map={}
    if k>0 and (root/f'checkpoints/k{k-1:03d}_observer.json').exists():prev_map=load_json(root/f'checkpoints/k{k-1:03d}_observer.json').get('exact_to_future_support',{})
    for c in panel:
        z=sc.future_observer_signatures(c,templates);sig[c['exact_key']]=z;grammar|=z['grammar']
        q=fi[z['future_support']];q['count']+=1;q['lanes']|=c['lanes'];q['cyclic']|=c['metrics']['beta']>=1;q['selective']|=(c['metrics']['selective_edges']>0 or c['metrics']['selective_opportunity_count']>0);q['exact_keys'].add(c['exact_key']);q['multiparent']|=len(c['parents'])>1
        ps={prev_map[p] for p in c['parents'] if p in prev_map}
        if z['future_support'] in ps:q['same_sig_parent']=True
        if ps and z['future_support'] not in ps:q['regenerated']=True
    seen,seenf,seenr,prevroles=_observer_prior_unions(root,k);priorroles=set(seenr)
    curroles=set(c['role_hash'] for c in panel);curf=set(z['future_support'] for z in sig.values());newg=grammar-seen
    p0=part_suff(panel,sig,lambda c:sig[c['exact_key']]['coarse']);p1=part_suff(panel,sig,lambda c:c['role_hash']);p2=part_suff(panel,sig,lambda c:sig[c['exact_key']]['port_capacity'])
    info={h:{'count':q['count'],'lanes':sorted(q['lanes']),'cyclic':q['cyclic'],'selective':q['selective'],'exact_assignments':len(q['exact_keys']),'same_sig_parent':q['same_sig_parent'],'regenerated':q['regenerated'],'multiparent':q['multiparent']} for h,q in sorted(fi.items())}
    out={'program':'SCOUT2_BIDEGREE_V1_2_OBSERVER','k':k,'selected_count':len(panel),'role_count':len(curroles),'future_support_signature_count':len(curf),'interface_support_signature_count':len({z['interface_support'] for z in sig.values()}),'port_capacity_signature_count':len({z['port_capacity'] for z in sig.values()}),'grammar_element_count':len(grammar),'grammar_elements':sorted(grammar),'new_grammar_elements':sorted(newg),'new_grammar_element_count':len(newg),'role_hashes':sorted(curroles),'new_role_hashes':sorted(curroles-seenr),'rebounded_role_hashes':sorted((curroles&priorroles)-prevroles),'future_support_signatures':sorted(curf),'new_future_support_signatures':sorted(curf-seenf),'future_signature_info':info,'exact_to_future_support':{a:b['future_support'] for a,b in sorted(sig.items())},'prime_invariant_sufficiency':{'P0_global_coarse':p0,'P1_scout_role':p1,'P2_exact_port_counters_plus_typed_incidence':p2},'compression':{'exact_to_future_support':len(panel)/max(1,len(curf))},'markers':sc.markers(panel),'wall_seconds':time.time()-t0}
    out['observer_sha256']=js_sha({a:b for a,b in out.items() if a not in ('wall_seconds','observer_sha256')});atomic_json(root/f'checkpoints/k{k:03d}_observer.json',out);_write_observer_eval_cache(root,k,out);_write_observer_cumulative(root,k,out,(seen,seenf,seenr,prevroles));return out

def d_control(root,k,templates,cap=96):
    panel=load_k_panel(root,k); sample=sc.greedy_farthest(panel,min(cap,len(panel))); children={};bad=0
    for p in sample:
        sts=p['states'];es=p['edges'];U=sc.usage(len(sts),es)
        for node,st in enumerate(sts):
            for tm in templates:
                pp=st[:7];tt=st[7:];s=tm['source']
                if pp[s]<=0:continue
                q=tuple(pp[i]+tm['dp'][i] for i in range(7));t2=tuple(tt[i]+tm['dt'][i] for i in range(9))
                if min(q)<0 or min(t2)<0:continue
                strict=pp[s]-U[node][s]>0;cross=(not strict and all(q[i]>=U[node][i] for i in range(7)))
                if not strict and not cross:continue
                ns=list(sts);ns[node]=q+t2
                if not sc.capacity_valid(ns,es):bad+=1;continue
                c=sc.make_carrier(ns,es,{'k':k});children[c['exact_key']]=c
    arr=list(children.values()); grammar=set();sig={}
    for c in arr:
        z=sc.future_observer_signatures(c,templates);grammar|=z['grammar'];sig[c['exact_key']]=z
    p2=part_suff(arr,sig,lambda c:sig[c['exact_key']]['port_capacity']) if arr else {'classes':0,'ambiguous_classes':0,'sufficient':True}
    seen=set()
    for qk in range(0,k,CONTROL_PERIOD):
        p=root/f'checkpoints/k{qk:03d}_dcontrol.json'
        if p.exists():seen.update(load_json(p).get('grammar_elements',[]))
    out={'program':'SCOUT2_BIDEGREE_V1_2_DCONTROL','k':k,'sample_parents':len(sample),'exact_children':len(arr),'bad_capacity_or_invariant':bad,'future_support_signature_count':len({z['future_support'] for z in sig.values()}),'grammar_elements':sorted(grammar),'new_grammar_elements':sorted(grammar-seen),'new_grammar_element_count':len(grammar-seen),'P2_sufficient':p2['sufficient'],'science_sha256':js_sha({'k':k,'n':len(arr),'bad':bad,'grammar':sorted(grammar),'future':sorted({z['future_support'] for z in sig.values()}),'p2':p2['sufficient']})}
    atomic_json(root/f'checkpoints/k{k:03d}_dcontrol.json',out);return out

def anchor_baseline(root,templates):
    rows=[]; union=set()
    for d in (78,93,109):
        panel=load_input_panel(root,d); grammar=set();sig={}
        for c in panel:
            z=sc.future_observer_signatures(c,templates);grammar|=z['grammar'];sig[c['exact_key']]=z
        p2=part_suff(panel,sig,lambda c:sig[c['exact_key']]['port_capacity'])
        rows.append({'d':d,'selected':len(panel),'markers':sc.markers(panel),'roles':len({c['role_hash'] for c in panel}),'future_support_signatures':len({z['future_support'] for z in sig.values()}),'grammar_atoms':len(grammar),'P2_sufficient':p2['sufficient']});union|=grammar
    out={'status':'PASS' if all(x['P2_sufficient'] for x in rows) else 'FAIL','anchors':rows,'union_grammar_atoms':len(union),'science_sha256':js_sha(rows)};atomic_json(root/'results/ANCHOR_BASELINE.json',out);return out

def evaluate(root,k):
    if k<K_WINDOW:
        res={'status':'V12_WINDOW_INCOMPLETE','through_k':k,'first_eligible_end':K_WINDOW}
        atomic_json(root/f'checkpoints/k{k:03d}_evaluation.json',res)
        return res
    start=k-K_WINDOW+1; obs=[_load_observer_eval(root,x) for x in range(start,k+1)]; mets=[_load_metrics_eval(root,x) for x in range(start,k+1)]
    quiet=all(x['new_grammar_element_count']==0 for x in obs[-16:]);stable=len({tuple(x['markers']) for x in obs})==1;p2=all(x['prime_invariant_sufficiency']['P2_exact_port_counters_plus_typed_incidence']['sufficient'] for x in obs)
    fs=set.intersection(*(set(x['future_support_signatures']) for x in obs));cores=[]
    for h in sorted(fs):
        inf=[x['future_signature_info'][h] for x in obs];lanes=[L for L in LANES if all(L in z['lanes'] for z in inf)]
        if len(lanes)<2:continue
        direct=sum(z['same_sig_parent'] for z in inf[1:]);regen=any(z['regenerated'] for z in inf[1:]);mp=sum(z['multiparent'] for z in inf)
        if direct<24 or (not regen and mp<8) or not any(z['cyclic'] for z in inf) or not any(z['selective'] for z in inf) or max(z['exact_assignments'] for z in inf)<2:continue
        cores.append({'future_support_signature':h,'full_lanes':lanes,'direct_persistence':direct,'regen':regen,'multiparent_checkpoints':mp,'cyclic':True,'selective':True,'max_exact_assignments':max(z['exact_assignments'] for z in inf)})
    modulation=sum(m['modulated_carriers']>0 for m in mets);compression=all(x['compression']['exact_to_future_support']>1 for x in obs[-16:]);live=all(m['possible_relation_additions_total']>0 and m['selected_count']>0 for m in mets)
    ctrls=[]
    for qk in range(((start+CONTROL_PERIOD-1)//CONTROL_PERIOD)*CONTROL_PERIOD,k+1,CONTROL_PERIOD):
        p=root/f'checkpoints/k{qk:03d}_dcontrol.json'
        if p.exists():ctrls.append(load_json(p))
    control_ok=bool(ctrls) and all(x['bad_capacity_or_invariant']==0 and x['P2_sufficient'] for x in ctrls)
    final4=ctrls[-4:] if len(ctrls)>=4 else ctrls;control_quiet=(len(final4)>=4 and all(x['new_grammar_element_count']==0 for x in final4))
    res={'status':'NO_O2_RELATION_RANK_MATURATION_CANDIDATE_V1_2','window':[start,k],'no_new_main_grammar_final16':quiet,'stable_markers':stable,'P2_future_action_sufficiency':p2,'behavioral_core_count':len(cores),'modulation_checkpoints':modulation,'compression_final16':compression,'relation_addition_live':live,'d_controls_in_window':len(ctrls),'d_control_integrity':control_ok,'d_control_final4_quiet':control_quiet,'main_grammar_shock_ks':[x['k'] for x in obs if x['new_grammar_element_count']>0],'cores':cores}
    if quiet and stable and p2 and cores and modulation>=16 and compression and live and control_ok and control_quiet:
        res['status']='O2_RELATION_RANK_MATURATION_CANDIDATE_V1_2';res['science_sha256']=js_sha(res);atomic_json(root/'results/O2_RELATION_RANK_MATURATION_CANDIDATE_V1_2.json',res)
    atomic_json(root/f'checkpoints/k{k:03d}_evaluation.json',res);return res

def verify(root,k):
    panel=load_k_panel(root,k);m=load_json(root/f'checkpoints/k{k:03d}_metrics.json')
    chk={'status':'PASS','k':k,'selected':len(panel),'roles':len({c['role_hash'] for c in panel}),'markers':sc.markers(panel),'panel_sha256':js_sha([sc.serialize_carrier(c) for c in sorted(panel,key=lambda z:z['exact_key'])])}
    assert chk['selected']==m['selected_count'] and chk['roles']==m['role_count'] and chk['markers']==m['markers'];atomic_json(root/f'checkpoints/k{k:03d}_terminal_replay.json',chk);return chk

def finalize(root,k,status=None):
    obs=[load_json(root/f'checkpoints/k{x:03d}_observer.json') for x in range(k+1) if (root/f'checkpoints/k{x:03d}_observer.json').exists()]
    cand=load_json(root/'results/O2_RELATION_RANK_MATURATION_CANDIDATE_V1_2.json') if (root/'results/O2_RELATION_RANK_MATURATION_CANDIDATE_V1_2.json').exists() else None
    if status is None:
        if cand: status=cand['status']
        elif k>=K_HORIZON:
            last16=obs[-16:]
            if any(x['new_grammar_element_count']>0 for x in last16):status='OPEN_ENDED_RELATION_RANK_NOVELTY'
            elif len({tuple(x['markers']) for x in last16})>1:status='REGIME_REORGANIZATION'
            else:status='FINITE_CONTROL_WITHOUT_RELATION_RANK_ENTITY'
        else:status='UNRESOLVED_STRESS_IN_PROGRESS'
    out={'program':'SCOUT2_BIDEGREE_V1_2','through_k':k,'status':status,'candidate':cand,'window':K_WINDOW,'horizon':K_HORIZON,'new_grammar_ks':[x['k'] for x in obs if x['new_grammar_element_count']>0],'role_counts':[x['role_count'] for x in obs],'future_support_signature_counts':[x['future_support_signature_count'] for x in obs],'scope':'Possibility-level pre-geometric bidegree Scout only; no geometry, physical time, actualization, propagation, psi, deletion or rewiring.'};out['science_sha256']=js_sha(out);atomic_json(root/'results/SCOUT2_BIDEGREE_V1_2_RESULT.json',out);return out

def warmup(root):
    # prerequisite hashes and d109 terminal replay hard assertions
    required=['FROZEN_PRIMITIVES.json','MATURE_NODE_ALGEBRA_SPEC.json','OBSERVED_S15_RECORDS.json.gz','IN11_RESULT.json','O2_MATURATION_CANDIDATE_V1_1.json','d078_selected_carriers.json.gz','d093_selected_carriers.json.gz','d109_selected_carriers.json.gz','d109_terminal_replay.json']
    for x in required:
        if not (root/'inputs'/x).exists():raise FileNotFoundError(x)
    old=load_json(root/'inputs/d109_terminal_replay.json');panel=load_input_panel(root,109)
    replay={'selected':len(panel),'roles':len({c['role_hash'] for c in panel}),'markers':sc.markers(panel),'panel_sha256':js_sha([sc.serialize_carrier(c) for c in sorted(panel,key=lambda z:z['exact_key'])])}
    assert replay['selected']==old['selected'] and replay['roles']==old['roles'] and replay['markers']==old['markers'] and replay['panel_sha256']==old['panel_sha256']
    _,tatoms,templates=load_templates_light(root)
    assert len(templates)==77 and len(sc.EDGE_TYPES_UNORDERED)==18
    anchor=anchor_baseline(root,templates);assert anchor['status']=='PASS'
    out={'status':'PASS','d109_replay':replay,'templates':len(templates),'edge_types':len(sc.EDGE_TYPES_UNORDERED),'anchor_baseline_sha256':anchor['science_sha256']};atomic_json(root/'checkpoints/WARMUP.json',out);return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--warmup',action='store_true');ap.add_argument('--seed',action='store_true');ap.add_argument('--run-one',type=int);ap.add_argument('--observe',type=int);ap.add_argument('--d-control',type=int);ap.add_argument('--evaluate',type=int);ap.add_argument('--verify',type=int);ap.add_argument('--finalize',type=int);ap.add_argument('--final-status');ap.add_argument('--tier',type=int,default=0);a=ap.parse_args();root=Path(a.root)
    if a.warmup: print(json.dumps(warmup(root),sort_keys=True));return
    if a.seed:
        _,_,templates=load_templates_light(root);print(json.dumps({'status':'SEEDED_K0','selected':seed_k0(root)},sort_keys=True));o=observe_k(root,0,templates);print(json.dumps({'k':0,'future':o['future_support_signature_count'],'grammar':o['grammar_element_count']},sort_keys=True));d=d_control(root,0,templates);print(json.dumps({'dcontrol':0,'children':d['exact_children'],'new_grammar':d['new_grammar_element_count']},sort_keys=True));return
    if a.run_one is not None: print(json.dumps(run_k(root,a.run_one,a.tier),sort_keys=True));return
    if a.observe is not None:
        _,_,templates=load_templates_light(root);o=observe_k(root,a.observe,templates);print(json.dumps({'status':'OBSERVED','k':a.observe,'selected':o['selected_count'],'roles':o['role_count'],'future':o['future_support_signature_count'],'new_grammar':o['new_grammar_element_count'],'rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},sort_keys=True));return
    if a.d_control is not None:
        _,_,templates=load_templates_light(root);print(json.dumps(d_control(root,a.d_control,templates),sort_keys=True));return
    if a.evaluate is not None: print(json.dumps(evaluate(root,a.evaluate),sort_keys=True));return
    if a.verify is not None: print(json.dumps(verify(root,a.verify),sort_keys=True));return
    if a.finalize is not None: print(json.dumps(finalize(root,a.finalize,a.final_status),sort_keys=True));return
    raise SystemExit('No command')
if __name__=='__main__':main()
