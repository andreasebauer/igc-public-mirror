from __future__ import annotations
import hashlib
import importlib.util
import itertools
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from functools import cached_property
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable
from infinity_grid.frontier import _apply_o7_overlay_skin
from infinity_grid.regime_scanner import _load_module

def _cbytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def _sha(obj: Any) -> str:
    return hashlib.sha256(_cbytes(obj)).hexdigest()

def load_regime_scanner_spec() -> dict:
    p = files("infinity_grid").joinpath("resources/decoder/O_REGIME_ADAPTIVE_SCANNER_SPEC_v1.json")
    return json.loads(p.read_text(encoding="utf-8"))

def load_motif_library() -> list[dict]:
    p = files("infinity_grid").joinpath("resources/decoder/O_REGIME_MOTIF_LIBRARY_v1.json")
    return json.loads(p.read_text(encoding="utf-8"))["motifs"]

def _graph_basic(n: int, pairs: Iterable[tuple[int, int]]) -> dict:
    pairs = [tuple(sorted(map(int, x))) for x in pairs]
    adj = [set() for _ in range(n)]
    mult = Counter(pairs)
    deg = [0] * n
    for u, v in pairs:
        adj[u].add(v); adj[v].add(u)
        deg[u] += 1; deg[v] += 1
    if n == 0:
        return {"degree": [], "beta": 0, "diameter": 0, "radius": 0, "triangles": 0, "articulations": 0, "bridges": 0, "distance_sum": 0, "shells": []}
    dist = []
    for s in range(n):
        d = [None] * n; d[s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[v] is None:
                    d[v] = d[u] + 1; q.append(v)
        dist.append(d)
    connected = all(x is not None for d in dist for x in d)
    if not connected:
        diameter = None; radius = None; distance_sum = None; shells = None
    else:
        diameter = max(max(d) for d in dist)
        radius = min(max(d) for d in dist)
        distance_sum = sum(dist[i][j] for i in range(n) for j in range(i+1, n))
        shells = sorted(tuple(Counter(d).get(k, 0) for k in range(max(d)+1)) for d in dist)
    triangles = sum(1 for a,b,c in itertools.combinations(range(n),3) if b in adj[a] and c in adj[a] and c in adj[b])

    def cc(removed_v=None, removed_e=None):
        aa = [set() for _ in range(n)]
        for idx,(u,v) in enumerate(pairs):
            if idx == removed_e: continue
            if removed_v is not None and (u == removed_v or v == removed_v): continue
            aa[u].add(v); aa[v].add(u)
        seen=set(); c=0
        for s in range(n):
            if s == removed_v or s in seen: continue
            c+=1; st=[s]; seen.add(s)
            while st:
                u=st.pop()
                for v in aa[u]:
                    if v not in seen: seen.add(v); st.append(v)
        return c
    basecc = cc()
    articulations = sum(cc(removed_v=v) > basecc for v in range(n)) if connected and n > 2 else 0
    bridges = sum(cc(removed_e=i) > basecc for i in range(len(pairs))) if connected else 0
    return {
        "degree": sorted(deg, reverse=True),
        "beta": len(pairs)-n+1 if connected else None,
        "diameter": diameter,
        "radius": radius,
        "triangles": triangles,
        "articulations": articulations,
        "bridges": bridges,
        "distance_sum": distance_sum,
        "shells": shells,
        "parallel_multiplicities": sorted(mult.values(), reverse=True),
        "connected": connected,
        "dist": dist,
    }

_O7_DATA_CACHE: dict[str, Any] = {}

_O7_SKIN_CACHE: dict[tuple[str, tuple[int,...]], str] = {}

_O7_OWNER_CAP_CACHE: dict[tuple[str, tuple[int,...]], list[tuple[int,...]]] = {}

@dataclass
class O7State:
    engine: Any
    ctx: Any
    edges: tuple
    reserve_counts: tuple[int,...] = (0,0,0,0,0,0,0)
    source_id: str = ""
    lane: str = "O7_AUTHORITY"
    level: int = 7

    @cached_property
    def base_key(self) -> str:
        return _sha({"source":self.source_id,"parents":list(self.ctx.source_order),"edges":[list(x) for x in self.edges]})

    @cached_property
    def base_data(self):
        if self.base_key not in _O7_DATA_CACHE:
            _O7_DATA_CACHE[self.base_key] = self.engine.current_owner_data(self.ctx, self.edges)
        return _O7_DATA_CACHE[self.base_key]

    def _materialize_counts(self):
        key=(self.base_key,tuple(map(int,self.reserve_counts)))
        if key in _O7_SKIN_CACHE and key in _O7_OWNER_CAP_CACHE:
            return _O7_SKIN_CACHE[key], _O7_OWNER_CAP_CACHE[key]
        need=list(map(int,self.reserve_counts))
        # Deterministic exact reservation overlay: for each endpoint type consume
        # the first canonical site occurrences in owner/path order. Multiple types
        # may hit the same leaf; modifications are combined before Merkle hashing.
        changes={}
        owner_caps=[]
        for oi,(_h,_root,_entries,_groups,byp) in enumerate(self.base_data):
            cap=[0]*7
            for path,(_p,f,_leaf,_ps) in byp.items():
                for t,x in enumerate(f): cap[t]+=int(x)
            owner_caps.append(cap)
        ordered=[]
        for oi,(_h,_root,_entries,_groups,byp) in enumerate(self.base_data):
            for path,(pp,ff,leaf,_ps) in sorted(byp.items(),key=lambda kv:tuple(kv[0])):
                ordered.append((oi,tuple(path),pp,tuple(ff),leaf))
        for t,cnt in enumerate(need):
            rem=cnt
            for oi,path,pp,ff,leaf in ordered:
                if rem<=0: break
                already=sum(v for (o,l,tt),v in changes.items() if o==oi and l is leaf and tt==t)
                avail=int(ff[t])-already
                if avail<=0: continue
                take=min(rem,avail); rem-=take
                changes[(oi,leaf,t)] = changes.get((oi,leaf,t),0)+take
                owner_caps[oi][t]-=take
            if rem: raise RuntimeError(f"O7 exact scanner overlay exceeds type-{t} free capacity by {rem}")
        sigs=[x[1].base_sig for x in self.base_data]
        byowner={}
        # combine all type changes on each leaf
        perleaf=defaultdict(lambda:[None,None,None])
        for oi,(_h,_root,_entries,_groups,byp) in enumerate(self.base_data):
            for path,(pp,ff,leaf,_ps) in byp.items():
                perleaf[(oi,leaf)] = [pp,list(ff),leaf]
        touched=set()
        for (oi,leaf,t),take in changes.items():
            pp,nf,_=perleaf[(oi,leaf)]; nf[t]-=take; touched.add((oi,leaf))
        for oi,leaf in touched:
            pp,nf,_=perleaf[(oi,leaf)]
            byowner.setdefault(oi,{})[leaf]=(pp,tuple(nf))
        for oi,mods in byowner.items():
            sigs[oi]=self.engine.O6._successor_merkle_sig(self.base_data[oi][1],mods)
        skin=self.engine.O6._digest_container("R7",sigs)
        owner_caps_t=[tuple(x) for x in owner_caps]
        _O7_SKIN_CACHE[key]=skin; _O7_OWNER_CAP_CACHE[key]=owner_caps_t
        return skin,owner_caps_t

    @cached_property
    def skin(self) -> str:
        return self._materialize_counts()[0]

    @cached_property
    def construction_digest(self) -> str:
        payload={"level":7,"parents":list(self.ctx.source_order),"edges":[list(x) for x in self.edges],"reserve_counts":list(self.reserve_counts),"source":self.source_id}
        return _sha(payload)

    @cached_property
    def top_pairs(self) -> list[tuple[int,int]]:
        return [(int(e[0]),int(e[7])) for e in self.edges]

    @cached_property
    def typed_edges(self) -> list[tuple[int,int,int,int]]:
        return [(int(e[0]),int(e[7]),int(e[6]),int(e[13])) for e in self.edges]

    @cached_property
    def owner_caps(self) -> list[tuple[int,...]]:
        return self._materialize_counts()[1]

    @cached_property
    def total_caps(self) -> tuple[int,...]:
        return tuple(sum(c[t] for c in self.owner_caps) for t in range(7))

    @cached_property
    def owner_colors(self) -> list[str]:
        # For base scanner states no external overlay is present. For embedded
        # states the aggregate exact skin is authoritative; per-owner colors are
        # used only as a symmetry diagnostic and can be derived from capacity vectors.
        if not any(self.reserve_counts):
            return [x[1].base_sig for x in self.base_data]
        return [_sha({"caps":list(c)}) for c in self.owner_caps]

    @cached_property
    def leaf_count(self) -> int:
        return sum(len(x[4]) for x in self.base_data)

    @cached_property
    def relation_count_total(self) -> int:
        return len(self.edges)

    def _first_endpoint(self, t: int, owner: int | None = None):
        # Used only for O7 top-level scanner actions, where reserve_counts is zero.
        if any(self.reserve_counts): raise RuntimeError("exact O7 top-action endpoint lookup with external overlay is outside scanner packet")
        cand=[]
        for oi,(_h,_root,_entries,_groups,byp) in enumerate(self.base_data):
            if owner is not None and oi != owner: continue
            for path,(_p,f,_leaf,_ps) in byp.items():
                if int(f[t])>0: cand.append((oi,tuple(path),t))
        return min(cand) if cand else None

    def reserve_external(self, t: int):
        if self.total_caps[t]<=0: raise RuntimeError(f"O7 no endpoint type {t}")
        cnt=list(self.reserve_counts); ordinal=cnt[t]; cnt[t]+=1
        return O7State(self.engine,self.ctx,self.edges,tuple(cnt),self.source_id,self.lane), (t,ordinal)

    def add_top_relation(self, u:int,v:int,a:int,b:int):
        eu=self._first_endpoint(a,u); ev=self._first_endpoint(b,v)
        if eu is None or ev is None: return None
        edge=self.engine.make_edge(u,eu[1],a,v,ev[1],b)
        try: edges=self.engine.canonicalize_edges(self.ctx, self.edges+(edge,))
        except Exception: return None
        return O7State(self.engine,self.ctx,edges,(0,0,0,0,0,0,0),self.source_id+"+A",self.lane)

@dataclass
class LiftState:
    engine: Any
    level: int
    children: tuple[Any,...]
    top_edges_full: tuple[tuple,...]
    lane: str
    motif_id: str

    @cached_property
    def skin(self) -> str:
        return self.engine.O6._digest_container(f"R{self.level}",[c.skin for c in self.children])

    @cached_property
    def construction_digest(self) -> str:
        payload={"level":self.level,"children":[c.construction_digest for c in self.children],"edges":[_edge_json(e) for e in self.top_edges_full],"lane":self.lane,"motif":self.motif_id}
        return _sha(payload)

    @cached_property
    def top_pairs(self) -> list[tuple[int,int]]:
        return [(int(e[0]),int(e[1])) for e in self.top_edges_full]

    @cached_property
    def typed_edges(self) -> list[tuple[int,int,int,int]]:
        return [(int(e[0]),int(e[1]),int(e[2]),int(e[3])) for e in self.top_edges_full]

    @cached_property
    def owner_caps(self) -> list[tuple[int,...]]:
        return [tuple(c.total_caps) for c in self.children]

    @cached_property
    def total_caps(self) -> tuple[int,...]:
        return tuple(sum(c.total_caps[t] for c in self.children) for t in range(7))

    @cached_property
    def owner_colors(self) -> list[str]:
        return [c.skin for c in self.children]

    @cached_property
    def leaf_count(self) -> int:
        return sum(c.leaf_count for c in self.children)

    @cached_property
    def relation_count_total(self) -> int:
        return sum(c.relation_count_total for c in self.children)+len(self.top_edges_full)

    def reserve_external(self,t:int):
        # Performance-only memoization. LiftState is treated as immutable after construction;
        # reserving a given endpoint type is therefore a pure deterministic function of this
        # state. Candidate generation repeatedly asks the same deep center/variant states for
        # identical reservations across the frozen motif catalogue. Caching the exact returned
        # child state/path removes repeated recursive descent without changing any scientific
        # field or digest. The cache is operational only and is never serialized or hashed.
        cache = self.__dict__.setdefault("_reserve_external_cache", {})
        key = int(t)
        if key in cache:
            return cache[key]
        cand=[]
        for i,c in enumerate(self.children):
            if c.total_caps[t]>0:
                cand.append((c.construction_digest,i))
        if not cand: raise RuntimeError(f"O{self.level} no endpoint type {t}")
        _d,i=min(cand)
        nc,local=self.children[i].reserve_external(t)
        ch=list(self.children);ch[i]=nc
        result = (LiftState(self.engine,self.level,tuple(ch),self.top_edges_full,self.lane,self.motif_id), (i,local,t))
        cache[key] = result
        return result

    def add_top_relation(self,u:int,v:int,a:int,b:int):
        if self.children[u].total_caps[a]<=0 or self.children[v].total_caps[b]<=0: return None
        cu,wu=self.children[u].reserve_external(a); cv,wv=self.children[v].reserve_external(b)
        ch=list(self.children);ch[u]=cu;ch[v]=cv
        edge=(u,v,a,b,wu,wv)
        return LiftState(self.engine,self.level,tuple(ch),self.top_edges_full+(edge,),self.lane,self.motif_id+"+A")

def _edge_json(e):
    u,v,a,b,wu,wv=e
    return [u,v,a,b,_path_json(wu),_path_json(wv)]

def _path_json(x):
    if isinstance(x, tuple): return [_path_json(v) for v in x]
    if isinstance(x, list): return [_path_json(v) for v in x]
    return x

def _build_lift(engine, level:int, owners:list[Any], motif_edges:list[tuple[int,int]], pairs:list[tuple[int,int]], lane:str, motif_id:str, schedule_seed:int=0, force_pair:tuple[int,int]|None=None):
    children=list(owners); full=[]
    for idx,(u,v) in enumerate(sorted(tuple(sorted(map(int,e))) for e in motif_edges)):
        candidates=[force_pair] if force_pair is not None else [pairs[(idx+schedule_seed+j)%len(pairs)] for j in range(len(pairs))]
        chosen=None
        for ab in candidates:
            if ab is None: continue
            a,b=map(int,ab)
            if children[u].total_caps[a]>0 and children[v].total_caps[b]>0:
                chosen=(a,b);break
        if chosen is None: return None
        a,b=chosen
        cu,wu=children[u].reserve_external(a);cv,wv=children[v].reserve_external(b)
        children[u]=cu;children[v]=cv
        full.append((u,v,a,b,wu,wv))
    return LiftState(engine,level,tuple(children),tuple(full),lane,motif_id)

def _state_pre_features(s) -> list[float]:
    gm=_graph_basic(len(s.owner_caps),s.top_pairs)
    deg=(gm["degree"]+[0]*6)[:6]
    caps=[math.log1p(x) for x in s.total_caps]
    support=sum(x>0 for x in s.total_caps)
    return [
        float(len(s.owner_caps)), float(len(s.top_pairs)), float(gm["beta"] or 0), float(gm["diameter"] or 0),
        float(gm["articulations"]),float(gm["triangles"]),float(max(gm["parallel_multiplicities"] or [0])),
        float(len(set(s.owner_colors))),float(support),math.log1p(s.leaf_count),math.log1p(s.relation_count_total),
        *map(float,deg),*caps,
    ]

def _farthest_select(states:list[Any], k:int, must_include:list[Any]|None=None) -> list[Any]:
    # Deduplicate by construction digest before selection.
    uniq={s.construction_digest:s for s in states}; pool=[uniq[x] for x in sorted(uniq)]
    if len(pool)<=k: return pool
    vec=[_state_pre_features(s) for s in pool]; dims=len(vec[0])
    mins=[min(v[j] for v in vec) for j in range(dims)];maxs=[max(v[j] for v in vec) for j in range(dims)]
    norm=[]
    for v in vec:
        norm.append([(v[j]-mins[j])/(maxs[j]-mins[j]) if maxs[j]>mins[j] else 0.0 for j in range(dims)])
    idx={s.construction_digest:i for i,s in enumerate(pool)}
    chosen=[]
    for s in must_include or []:
        i=idx.get(s.construction_digest)
        if i is not None and i not in chosen: chosen.append(i)
    if not chosen: chosen=[0]
    while len(chosen)<k:
        best=None
        for i in range(len(pool)):
            if i in chosen: continue
            dmin=min(sum((norm[i][j]-norm[c][j])**2 for j in range(dims))**0.5 for c in chosen)
            key=(dmin, ''.join(chr(255-ord(c)) if ord(c)<256 else c for c in pool[i].construction_digest))
            # explicit comparison for max distance, then lexicographically smallest digest
            if best is None or dmin>best[0]+1e-15 or (abs(dmin-best[0])<=1e-15 and pool[i].construction_digest<pool[best[1]].construction_digest):
                best=(dmin,i)
        chosen.append(best[1])
    return [pool[i] for i in chosen[:k]]
