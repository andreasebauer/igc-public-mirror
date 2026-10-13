from __future__ import annotations
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Hashable, Iterable
import json, resource, time
Node=Hashable

def _sorted(values:Iterable[Any])->list[Any]:
    try:return sorted(values)
    except TypeError:return sorted(values,key=repr)

def _maximum_bipartite_matching_size(adj:dict[int,list[int]],n_left:int,n_right:int)->int:
    pair_u=[-1]*n_left; pair_v=[-1]*n_right; dist=[0]*n_left; INF=10**18
    def bfs():
        q=deque(); found=False
        for u in range(n_left):
            if pair_u[u]==-1: dist[u]=0; q.append(u)
            else: dist[u]=INF
        while q:
            u=q.popleft()
            for v in adj.get(u,[]):
                mate=pair_v[v]
                if mate==-1: found=True
                elif dist[mate]==INF: dist[mate]=dist[u]+1; q.append(mate)
        return found
    def dfs(u):
        for v in adj.get(u,[]):
            mate=pair_v[v]
            if mate==-1 or (dist[mate]==dist[u]+1 and dfs(mate)):
                pair_u[u]=v; pair_v[v]=u; return True
        dist[u]=INF; return False
    matching=0
    while bfs():
        for u in range(n_left):
            if pair_u[u]==-1 and dfs(u): matching+=1
    return matching

def spectroscope_core(relation:dict[str,Any],level:int|None=None)->dict[str,Any]:
    edges=[tuple(x) for x in relation['edges']]; left=_sorted({a for a,_ in edges}); right=_sorted({b for _,b in edges}); l2r=defaultdict(set); r2l=defaultdict(set)
    for a,b in edges:l2r[a].add(b); r2l[b].add(a)
    seen=set(); comps=[]
    for start in [('L',x) for x in left]+[('R',x) for x in right]:
        if start in seen:continue
        stack=[start]; seen.add(start); nl=nr=ne=0
        while stack:
            side,x=stack.pop()
            if side=='L':
                nl+=1; ne+=len(l2r[x])
                for y in l2r[x]:
                    z=('R',y)
                    if z not in seen: seen.add(z); stack.append(z)
            else:
                nr+=1
                for y in r2l[x]:
                    z=('L',y)
                    if z not in seen:seen.add(z); stack.append(z)
        comps.append((nl,nr,ne))
    V=len(left)+len(right); E=len(edges); C=len(comps); cycle_rank=E-V+C
    colors={('L',x):('L',len(l2r[x])) for x in left}; colors.update({('R',x):('R',len(r2l[x])) for x in right}); rounds=[]
    def part(cs):
        d=defaultdict(set)
        for k,v in cs.items():d[repr(v)].add(k)
        return sorted((tuple(_sorted(v)) for v in d.values()),key=repr)
    for rnd in range(30):
        sig={}
        for x in left:sig[('L',x)]=(colors[('L',x)],tuple(_sorted(colors[('R',y)] for y in l2r[x])))
        for x in right:sig[('R',x)]=(colors[('R',x)],tuple(_sorted(colors[('L',y)] for y in r2l[x])))
        uniq={s:i for i,s in enumerate(sorted(set(sig.values()),key=repr))}; new={k:uniq[v] for k,v in sig.items()}; lc=len(set(new[('L',x)] for x in left)); rc=len(set(new[('R',x)] for x in right)); rounds.append({'round':rnd+1,'left_classes':lc,'right_classes':rc})
        if part(colors)==part(new):colors=new; break
        colors=new
    fiber_groups=defaultdict(list)
    for r in right:fiber_groups[tuple(_sorted(r2l[r]))].append(r)
    fibers=sorted(fiber_groups.keys(),key=lambda x:(len(x),repr(x))); fsets=[set(x) for x in fibers]; lookup={frozenset(s) for s in fsets}; by_size=defaultdict(list)
    for i,s in enumerate(fsets):by_size[len(s)].append(i)
    comparable=[]; sizes=sorted(by_size)
    for i,A in enumerate(fsets):
        for sz in sizes:
            if sz<=len(A):continue
            for j in by_size[sz]:
                if A<fsets[j]:comparable.append((i,j))
    covers=[]
    for i,j in comparable:
        A,B=fsets[i],fsets[j]; cover=True
        for sz in range(len(A)+1,len(B)):
            for k in by_size.get(sz,[]):
                K=fsets[k]
                if A<K<B:cover=False;break
            if not cover:break
        if cover:covers.append((i,j))
    preds=defaultdict(list)
    for i,j in covers:preds[j].append(i)
    dp=[1]*len(fsets)
    for j in sorted(range(len(fsets)),key=lambda q:len(fsets[q])):
        if preds[j]:dp[j]=1+max(dp[i] for i in preds[j])
    height=max(dp) if dp else 0; adj=defaultdict(list)
    for i,j in comparable:adj[i].append(j)
    for i in adj:adj[i].sort()
    matching=_maximum_bipartite_matching_size(adj,len(fsets),len(fsets)); width=len(fsets)-matching
    pairs=set()
    for rs in l2r.values():
        ss=_sorted(rs)
        for i in range(len(ss)):
            for j in range(i+1,len(ss)):pairs.add((ss[i],ss[j]))
    extent_cache={}; exactpair=enlarged=0; intent_hist=Counter(); extent_hist=Counter()
    for a,b in pairs:
        intent=tuple(_sorted(r2l[a]&r2l[b])); intent_hist[len(intent)]+=1
        if intent not in extent_cache:
            ext=None
            for p in intent:ext=set(l2r[p]) if ext is None else ext&l2r[p]
            extent_cache[intent]=ext or set()
        ext=extent_cache[intent]; extent_hist[len(ext)]+=1
        if ext=={a,b}:exactpair+=1
        elif len(ext)>2:enlarged+=1
    fiber_pairs=set()
    for a,b in pairs:
        A=tuple(_sorted(r2l[a])); B=tuple(_sorted(r2l[b]))
        if A!=B:fiber_pairs.add(tuple(sorted((A,B),key=repr)))
    inter=union=both=0
    for A0,B0 in fiber_pairs:
        A=set(A0); B=set(B0); ii=frozenset(A&B) in lookup; uu=frozenset(A|B) in lookup; inter+=int(ii); union+=int(uu); both+=int(ii and uu)
    left_twins=Counter(tuple(_sorted(l2r[x])) for x in left); right_twins=Counter(tuple(_sorted(r2l[x])) for x in right)
    result={'test_id':'SCOUT_SPECTROSCOPE_CHEAP_V2','level':level,'authoritative':False,'evidence_label':'SCOUT_OBSERVED','scope':{'left_nodes':len(left),'right_nodes':len(right),'incidences':E},'incidence':{'components':C,'component_shapes':sorted([{'left':a,'right':b,'edges':e} for a,b,e in comps],key=lambda z:-z['edges'])[:20],'abstract_cycle_rank':cycle_rank,'left_degree_histogram':dict(sorted(Counter(map(len,l2r.values())).items())),'right_degree_histogram':dict(sorted(Counter(map(len,r2l.values())).items()))},'refinement':{'rounds':rounds,'stable_left_classes':len(set(colors[('L',x)] for x in left)),'stable_right_classes':len(set(colors[('R',x)] for x in right)),'left_singletons':sum(v==1 for v in Counter(colors[('L',x)] for x in left).values()),'right_singletons':sum(v==1 for v in Counter(colors[('R',x)] for x in right).values()),'left_exact_twin_groups':sum(v>1 for v in left_twins.values()),'right_exact_twin_groups':sum(v>1 for v in right_twins.values())},'order':{'distinct_right_parent_fibers':len(fibers),'strict_inclusion_pairs':len(comparable),'cover_relations':len(covers),'height':height,'width':width,'fiber_size_histogram':dict(sorted(Counter(map(len,fibers)).items()))},'closure':{'right_pairs_with_common_left':len(pairs),'formal_pair_closure_exact_pair':exactpair,'formal_pair_closure_enlarged':enlarged,'intent_size_histogram':dict(sorted(intent_hist.items())),'extent_size_histogram':dict(sorted(extent_hist.items())),'overlapping_distinct_fiber_pairs':len(fiber_pairs),'intersection_realized':inter,'union_realized':union,'both_realized':both,'intersection_realized_fraction':inter/len(fiber_pairs) if fiber_pairs else 0.0,'union_realized_fraction':union/len(fiber_pairs) if fiber_pairs else 0.0},'recognition_summary':[],'prohibitions':['NO_FIRST_OCCURRENCE_CLAIM','NO_C_PROVED','NO_POPULATION_WIDE_NEGATIVE','NO_MECHANISM_PROMOTION','NO_GEOMETRY_PROMOTION','NO_TOPOLOGY_PROMOTION_FROM_CYCLE_RANK']}
    if comparable:result['recognition_summary'].append('NONTRIVIAL_PARENT_FIBER_INCLUSION_ORDER')
    if exactpair or enlarged:result['recognition_summary'].append('NONTRIVIAL_FORMAL_CONCEPT_STYLE_PAIR_CLOSURE')
    if result['refinement']['stable_right_classes']>0.8*len(right):result['recognition_summary'].append('HIGH_RELATIONAL_INDIVIDUATION_RIGHT_SIDE')
    if both:result['recognition_summary'].append('PARTIAL_MEET_JOIN_REALIZATION_IN_OBSERVED_FIBER_FAMILY')
    if cycle_rank>0:result['recognition_summary'].append('CYCLE_RICH_ABSTRACT_INCIDENCE_GRAPH_RECOGNITION_ONLY')
    result['recognition_summary'].append('INCLUSION_POSET_WIDTH_COMPUTED'); return result

def run_scout_spectroscope(input_path:Path,out_dir:Path,level:int|None=None):
    t0=time.perf_counter(); c0=time.process_time(); relation=json.loads(Path(input_path).read_text()); result=spectroscope_core(relation,level); result['cost']={'wall_seconds':time.perf_counter()-t0,'cpu_seconds':time.process_time()-c0,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}; out_dir.mkdir(parents=True,exist_ok=True); (out_dir/'SCOUT_SPECTROSCOPE_RESULT.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n'); return result

def scientific_projection(result):return {k:v for k,v in result.items() if k!='cost'}
