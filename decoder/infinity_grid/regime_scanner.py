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

from .frontier import _apply_o7_overlay_skin

PHASE8_SEED_EXPECTED = "b28dc2c306f389a5191a9f512e6cbeeb03251e8b1234df187524f331ad3b0c9b"
GRAMMAR_EXPECTED = "5a2b34df6572a10648d46936da81228c606406d09b5b4c7e9603b624527d5a07"
GA = ((0, 1), (1, 2), (1, 3), (2, 4), (3, 5))
GB = ((0, 4), (0, 5), (1, 3), (2, 3), (3, 4))

# Frozen matched longitudinal packet. These motifs are not selected on outcomes;
# the same top-level structures are rebuilt at every synthetic O depth. They
# provide a clean depth-control lane beside the exploratory farthest-point beam.
BACKBONE_MOTIFS = (
    ("P4", ((0,1),(1,2),(2,3))),
    ("S4", ((0,1),(0,2),(0,3))),
    ("C4", ((0,1),(1,2),(2,3),(0,3))),
    ("K4", ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))),
    ("C5", ((0,1),(1,2),(2,3),(3,4),(0,4))),
    ("TWIN_A", GA),
    ("TWIN_B", GB),
)


def _cbytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(obj: Any) -> str:
    return hashlib.sha256(_cbytes(obj)).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {len(hits)}")
    return hits[0]


def load_regime_scanner_spec() -> dict:
    p = files("infinity_grid").joinpath("resources/decoder/O_REGIME_ADAPTIVE_SCANNER_SPEC_v1.json")
    return json.loads(p.read_text(encoding="utf-8"))


def load_motif_library() -> list[dict]:
    p = files("infinity_grid").joinpath("resources/decoder/O_REGIME_MOTIF_LIBRARY_v1.json")
    return json.loads(p.read_text(encoding="utf-8"))["motifs"]


def load_earned_regime_laws() -> dict:
    p = files("infinity_grid").joinpath("resources/decoder/O_REGIME_EARNED_LAW_REGISTRY_v1.json")
    d=json.loads(p.read_text(encoding="utf-8"))
    expected=d.get("registry_sha256")
    observed=_sha({k:v for k,v in d.items() if k!="registry_sha256"})
    if expected!=observed:
        raise RuntimeError(f"earned O-regime law registry hash mismatch: {expected} != {observed}")
    return d


def _extract_seed(seed_zip: Path, work: Path) -> tuple[Path, Path]:
    seed_zip = Path(seed_zip)
    got = _sha_file(seed_zip)
    if got != PHASE8_SEED_EXPECTED:
        raise RuntimeError(f"Phase8 seed SHA mismatch: {got}")
    seed_out = work / "phase8"
    with zipfile.ZipFile(seed_zip) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"Phase8 seed CRC failure at {bad}")
        zf.extractall(seed_out)
    roots = [p for p in seed_out.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError("ambiguous Phase8 seed root")
    phase8 = roots[0]
    o7_zip = phase8 / "replay" / "Infinity_Grid_O7_COMPACT_REPLAY_ROOT_v1_2026-08-29.zip"
    o7_out = work / "o7"
    with zipfile.ZipFile(o7_zip) as zf:
        bad = zf.testzip()
        if bad:
            raise RuntimeError(f"O7 compact root CRC failure at {bad}")
        zf.extractall(o7_out)
    roots = [p for p in o7_out.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError("ambiguous O7 compact root")
    return phase8, roots[0]


def _graph_basic_legacy(n: int, pairs: Iterable[tuple[int, int]]) -> dict:
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


def _graph_basic(n: int, pairs: Iterable[tuple[int, int]], *, backend=None) -> dict:
    from .graph_metrics_library import graph_basic
    return graph_basic(n, list(pairs), _graph_basic_legacy, backend=backend)


def _service_effect(n: int, pairs: list[tuple[int,int]], pair: tuple[int,int]) -> tuple[int,int,int,int,int,int]:
    pair = tuple(sorted(pair))
    before = _graph_basic(n, pairs)
    after = _graph_basic(n, pairs + [pair])
    u,v = pair
    pre = before["dist"][u][v]
    mult = Counter(tuple(sorted(x)) for x in pairs)
    return (
        int(pre),
        int(before["distance_sum"] - after["distance_sum"]),
        int(before["diameter"] - after["diameter"]),
        int(before["bridges"] - after["bridges"]),
        int(before["articulations"] - after["articulations"]),
        int(mult[pair] > 0),
    )


def _uncolored_graph_canon(n: int, pairs: Iterable[tuple[int,int]]) -> tuple:
    counts = Counter(tuple(sorted(map(int,e))) for e in pairs)
    best = None
    for perm in itertools.permutations(range(n)):
        ec = tuple(sorted((min(perm[u],perm[v]), max(perm[u],perm[v]), m) for (u,v),m in counts.items()))
        if best is None or ec < best: best = ec
    return best or tuple()


def _colored_typed_canon(n: int, colors: list[Any], edges: list[tuple[int,int,int,int]]) -> tuple:
    # Exact enough for scanner organization: owner colors + typed top multigraph.
    best = None
    for perm in itertools.permutations(range(n)):
        nc = [None] * n
        for old,new in enumerate(perm): nc[new] = colors[old]
        ee=[]
        for u,v,a,b in edges:
            nu,nv=perm[u],perm[v]
            if nu <= nv: ee.append((nu,nv,a,b))
            else: ee.append((nv,nu,b,a))
        cand=(tuple(nc),tuple(sorted(ee)))
        if best is None or cand < best: best = cand
    return best or (tuple(colors),tuple())


def _automorphisms_legacy(n: int, colors: list[Any], edges: list[tuple[int,int,int,int]]) -> list[tuple[int,...]]:
    """Return genuine automorphisms of the graph in its current labeling.

    Canonical-labelling witnesses are isomorphisms *to* a canonical
    representative and must not be treated as permutations acting on the
    original graph.
    """
    original = (tuple(colors), tuple(sorted(edges)))
    autos=[]
    for perm in itertools.permutations(range(n)):
        nc=[None]*n
        for old,new in enumerate(perm): nc[new]=colors[old]
        ee=[]
        for u,v,a,b in edges:
            nu,nv=perm[u],perm[v]
            if nu<=nv: ee.append((nu,nv,a,b))
            else: ee.append((nv,nu,b,a))
        if (tuple(nc),tuple(sorted(ee))) == original:
            autos.append(tuple(perm))
    return autos


def _automorphisms(n: int, colors: list[Any], edges: list[tuple[int,int,int,int]], *, backend=None) -> list[tuple[int,...]]:
    from .graph_library_backend import automorphisms
    return automorphisms(n, colors, edges, _automorphisms_legacy, backend=backend)


def _orbits_from_perms(n: int, perms: list[tuple[int,...]]) -> int:
    seen=set(); c=0
    for i in range(n):
        if i in seen: continue
        orb={p[i] for p in perms} if perms else {i}
        seen |= orb; c += 1
    return c


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


def _build_backbone(engine, level:int, center:Any, pairs:list[tuple[int,int]]) -> list[Any]:
    out=[]
    for name,edges0 in BACKBONE_MOTIFS:
        edges=[tuple(map(int,e)) for e in edges0]
        n=1+max(max(e) for e in edges)
        st=_build_lift(engine,level,[center]*n,edges,pairs,"BACKBONE",f"BACKBONE:{name}",force_pair=(0,0))
        if st is None:
            raise RuntimeError(f"matched backbone motif {name} failed at O{level}")
        out.append(st)
    return out


def _backbone_signature(states:list[Any], bridge_pairs:list[tuple[int,int]]) -> dict:
    rows=[]
    for st in sorted(states,key=lambda x:x.motif_id):
        a=_action_aggregate(st,bridge_pairs)
        gm=_graph_basic(len(st.owner_caps),st.top_pairs)
        fib=_factor_fiber(st)
        rows.append({
            "motif_id":st.motif_id,
            "owners":len(st.owner_caps),
            "top_edges":len(st.top_pairs),
            "topology_canon_sha256":_sha(_uncolored_graph_canon(len(st.owner_caps),st.top_pairs)),
            "degree":gm["degree"],"beta":gm["beta"],"diameter":gm["diameter"],"radius":gm["radius"],
            "articulations":gm["articulations"],"bridges":gm["bridges"],"triangles":gm["triangles"],
            "endpoint_support":tuple(sum(int(c[t]>0) for c in st.owner_caps) for t in range(7)),
            "type_pair_support":a["type_pair_support"],"owner_pair_support":a["owner_pair_support"],
            "legal_action_labels":a["legal_action_labels"],"action_orbits":a["action_orbits"],
            "service_classes":a["service_classes"],"service_signature_sha256":a["service_signature_sha256"],
            "automorphism_size":a["automorphism_size"],"owner_orbits":a["owner_orbits"],
            "factor_fiber_size":len(fib),
        })
    payload={"schema":"IG_O_REGIME_MATCHED_BACKBONE_V1","rows":rows}
    return {
        "status":"PASS",
        "signature_sha256":_sha(payload),
        "rows":rows,
        "raw_growth":{
            "median_leaf_count":_median([s.leaf_count for s in states]),
            "median_total_relations":_median([s.relation_count_total for s in states]),
            "median_total_free":_median([sum(s.total_caps) for s in states]),
        },
        "note":"Signature excludes exact skin hashes, level tags, Counter magnitudes and other pure size growth. The same frozen top-level motifs are rebuilt at every synthetic depth."
    }


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


def _motif_features(m:dict) -> tuple:
    gm=_graph_basic(int(m["n"]),[tuple(e) for e in m["edges"]])
    return (m["n"],m["m"],gm["beta"],gm["diameter"],gm["articulations"],gm["triangles"],tuple(gm["degree"]))


def _build_next_panel(engine, prev:list[Any], level:int, pairs:list[tuple[int,int]], motifs:list[dict], spec:dict) -> tuple[list[Any],dict,list[Any]]:
    prev=sorted(prev,key=lambda s:s.construction_digest)
    totals=[sum(s.total_caps) for s in prev]; med=statistics.median(totals)
    center=min(prev,key=lambda s:(abs(sum(s.total_caps)-med),s.construction_digest))
    diverse=prev
    cand=[]; failures=0
    for mi,m in enumerate(motifs):
        n=int(m["n"]); edges=[tuple(e) for e in m["edges"]]
        # HOM: full motif library.
        s=_build_lift(engine,level,[center]*n,edges,pairs,"HOM",f"HOM:{n}:{mi}",schedule_seed=mi%len(pairs))
        if s:cand.append(s)
        else:failures+=1
        # HET: every 4/5 motif, and a deterministic 1/6 sample of six-owner motifs.
        if n in (4,5) or (n==6 and mi%6==0):
            owners=[diverse[j%len(diverse)] for j in range(n)]
            s=_build_lift(engine,level,owners,edges,pairs,"HET",f"HET:{n}:{mi}",schedule_seed=(mi*3+1)%len(pairs))
            if s:cand.append(s)
            else:failures+=1
        # MIX for four-owner motifs.
        if n==4 and len(diverse)>=2:
            owners=[diverse[0],diverse[1],diverse[0],diverse[1]]
            s=_build_lift(engine,level,owners,edges,pairs,"MIX",f"MIX:4:{mi}",schedule_seed=(mi*5+2)%len(pairs))
            if s:cand.append(s)
            else:failures+=1
    # Exact same-resource degree-matched topology twins, independent of panel outcome.
    twin=[]
    if center.total_caps[0]>=6:
        A=_build_lift(engine,level,[center]*6,list(GA),pairs,"TWIN","TWIN:A",force_pair=(0,0))
        B=_build_lift(engine,level,[center]*6,list(GB),pairs,"TWIN","TWIN:B",force_pair=(0,0))
        if A and B: cand.extend([A,B]); twin=[A,B]
    # Candidate cap is applied by a deterministic pre-outcome digest cut only if necessary.
    cap=int(spec["panel"]["candidate_cap"])
    if len(cand)>cap:
        cand=sorted(cand,key=lambda s:s.construction_digest)[:cap]
        # keep twins even if outside cap
        for t in twin:
            if all(x.construction_digest!=t.construction_digest for x in cand): cand[-1]=t
    selected=_farthest_select(cand,int(spec["panel"]["beam"]),must_include=twin)
    backbone=_build_backbone(engine,level,center,pairs)
    return selected,{"candidates":len(cand),"build_failures":failures,"twin_constructed":len(twin)==2,"selected":len(selected),"matched_backbone_states":len(backbone)},backbone


def _action_aggregate(state, bridge_pairs:list[tuple[int,int]], *, automorphism_backend=None) -> dict:
    n=len(state.owner_caps); actions=[]; copies=0; type_pairs=set(); owner_pairs=set()
    # The graph-service effect depends only on the selected owner pair, not on
    # endpoint type or Counter magnitude. Compute it once per owner pair. This
    # both removes a large O(type-pairs) redundancy and prevents pure capacity
    # growth from masquerading as organizational novelty.
    gb=_graph_basic(n,state.top_pairs)
    pair_effect={}
    if gb["connected"]:
        for u in range(n):
            for v in range(u+1,n):
                pair_effect[(u,v)] = _service_effect(n,state.top_pairs,(u,v))
    structural_service=Counter()
    weighted_service=Counter()
    for u in range(n):
        for v in range(u+1,n):
            eff=pair_effect.get((u,v))
            pair_has_action=False
            for a,b in bridge_pairs:
                ca=int(state.owner_caps[u][a]);cb=int(state.owner_caps[v][b])
                if ca<=0 or cb<=0: continue
                pair_has_action=True
                c=ca*cb;copies+=c;actions.append((u,v,a,b,c));type_pairs.add((a,b))
                if eff is not None:
                    structural_service[(a,b,eff)] += 1
                    # Kept as a magnitude-sensitive diagnostic only. It is never
                    # used in the normalized organizational signature.
                    weighted_service[(a,b,eff,tuple(sorted((ca,cb))))] += 1
            if pair_has_action: owner_pairs.add((u,v))
    # Symmetry/action orbit diagnostic, using owner resource-support colors (not exact skins) to avoid spurious exact-color fragmentation.
    colors=[tuple(int(x>0) for x in c) for c in state.owner_caps]
    autos=_automorphisms(n,colors,state.typed_edges,backend=automorphism_backend)
    orbit_reps=set()
    for u,v,a,b,_c in actions:
        images=[]
        for p in autos or [tuple(range(n))]:
            uu,vv=p[u],p[v]
            if uu<=vv:images.append((uu,vv,a,b))
            else:images.append((vv,uu,b,a))
        orbit_reps.add(min(images))
    return {
        "legal_action_labels":len(actions),"total_action_copies":int(copies),"type_pair_support":len(type_pairs),"owner_pair_support":len(owner_pairs),
        "action_orbits":len(orbit_reps),
        "service_classes":len(structural_service),
        "service_signature_sha256":_sha([[repr(k),v] for k,v in sorted(structural_service.items(),key=lambda kv:repr(kv[0]))]),
        "resource_weighted_service_classes":len(weighted_service),
        "automorphism_size":len(autos),"owner_orbits":_orbits_from_perms(n,autos),
    }


def _factor_fiber(state) -> set[str]:
    # Reverse-topology factorization fingerprints. This is a reconnaissance relation,
    # not an admitted forward deletion operation.
    n=len(state.owner_caps); out=set()
    pairs=list(state.top_pairs)
    colors=[tuple(int(x>0) for x in c) for c in state.owner_caps]
    typed=list(state.typed_edges)
    for idx in range(len(pairs)):
        tp=typed[:idx]+typed[idx+1:]
        pp=pairs[:idx]+pairs[idx+1:]
        # record connected predecessor or two-component factorization alike
        gm=_graph_basic(n,pp)
        out.add(_sha({"connected":gm["connected"],"canon":_colored_typed_canon(n,colors,tp)}))
    return out


def _panel_overlap(fibers:list[set[str]]) -> dict:
    shared=0; inclusions=0; intersections=0; unions=0
    observed={frozenset(x) for x in fibers}
    for i in range(len(fibers)):
        for j in range(i+1,len(fibers)):
            A,B=fibers[i],fibers[j]
            if A & B: shared+=1
            if A < B or B < A: inclusions+=1
            if frozenset(A&B) in observed: intersections+=1
            if frozenset(A|B) in observed: unions+=1
    denom=max(1,len(fibers)*(len(fibers)-1)//2)
    return {"shared_fiber_pairs":shared,"strict_inclusions":inclusions,"intersection_realized_fraction":intersections/denom,"union_realized_fraction":unions/denom}


def _median(vals):
    return statistics.median(vals) if vals else None


def _p(vals,q):
    if not vals:return None
    a=sorted(vals);i=min(len(a)-1,max(0,math.ceil(q*len(a))-1));return a[i]


def _state_organizational_signature(state, action:dict) -> dict:
    gm=_graph_basic(len(state.owner_caps),state.top_pairs)
    return {
        "owners":len(state.owner_caps),"edges":len(state.top_pairs),"degree":gm["degree"],"beta":gm["beta"],"diameter":gm["diameter"],"articulations":gm["articulations"],"triangles":gm["triangles"],
        "colored_owner_resource_topology_sha256": _sha(_colored_typed_canon(
            len(state.owner_caps),
            [tuple(int(x>0) for x in c) for c in state.owner_caps],
            list(state.typed_edges),
        )),
        "type_pair_support":action["type_pair_support"],"owner_pair_support":action["owner_pair_support"],"action_orbits":action["action_orbits"],"service_classes":action["service_classes"],
        "automorphism_size":action["automorphism_size"],"owner_orbits":action["owner_orbits"]
    }


def _scan_level(states:list[Any], level:int, bridge_pairs:list[tuple[int,int]], grammar_hash:str) -> tuple[dict,str]:
    states=sorted(states,key=lambda s:s.construction_digest)
    actions=[_action_aggregate(s,bridge_pairs) for s in states]
    gms=[_graph_basic(len(s.owner_caps),s.top_pairs) for s in states]
    fibers=[_factor_fiber(s) for s in states]
    overlap=_panel_overlap(fibers)
    byskin=defaultdict(list)
    for s in states:byskin[s.skin].append(s)
    same_skin=[v for v in byskin.values() if len(v)>1]
    topo_collision=0
    for vals in same_skin:
        if len({_uncolored_graph_canon(len(s.owner_caps),s.top_pairs) for s in vals})>1:topo_collision+=1
    endpoint_support=[sum(x>0 for x in s.total_caps) for s in states]
    min_caps=[min(s.total_caps[t] for s in states) for t in range(7)]
    org=[_state_organizational_signature(s,a) for s,a in zip(states,actions)]
    org_hashes=sorted(_sha(x) for x in org)
    raw={
        "level":level,"states":len(states),"grammar_sha256":grammar_hash,
        "diversity":{
            "construction_states":len({s.construction_digest for s in states}),"resource_skins":len(byskin),"same_skin_groups":len(same_skin),"same_skin_topology_collision_groups":topo_collision,
            "topology_classes":len({_uncolored_graph_canon(len(s.owner_caps),s.top_pairs) for s in states}),"organizational_classes":len(set(org_hashes)),
            "owner_count_hist":dict(sorted(Counter(len(s.owner_caps) for s in states).items())),"edge_count_hist":dict(sorted(Counter(len(s.top_pairs) for s in states).items())),
            "endpoint_support_min":min(endpoint_support),"endpoint_support_max":max(endpoint_support),"min_total_free_by_type":min_caps,
            "leaf_count_min":min(s.leaf_count for s in states),"leaf_count_median":_median([s.leaf_count for s in states]),"leaf_count_max":max(s.leaf_count for s in states),
            "relation_count_total_median":_median([s.relation_count_total for s in states]),"relation_count_total_max":max(s.relation_count_total for s in states)
        },
        "branching":{
            "legal_action_labels_median":_median([a["legal_action_labels"] for a in actions]),"legal_action_labels_max":max(a["legal_action_labels"] for a in actions),
            "action_orbits_median":_median([a["action_orbits"] for a in actions]),"action_orbits_max":max(a["action_orbits"] for a in actions),
            "type_pair_support_min":min(a["type_pair_support"] for a in actions),"type_pair_support_max":max(a["type_pair_support"] for a in actions),
            "total_action_copies_log10_median":_median([math.log10(max(1,a["total_action_copies"])) for a in actions]),"total_action_copies_log10_p90":_p([math.log10(max(1,a["total_action_copies"])) for a in actions],.9),
            "service_classes_median":_median([a["service_classes"] for a in actions]),"service_classes_max":max(a["service_classes"] for a in actions)
        },
        "symmetry":{
            "automorphism_size_median":_median([a["automorphism_size"] for a in actions]),"automorphism_size_max":max(a["automorphism_size"] for a in actions),"owner_orbits_median":_median([a["owner_orbits"] for a in actions])
        },
        "overlap_gluing":overlap,
        "lineage":{
            "factor_fiber_median":_median([len(f) for f in fibers]),"factor_fiber_max":max(map(len,fibers)),"bridge_fraction_median":_median([g["bridges"]/max(1,len(s.top_pairs)) for g,s in zip(gms,states)]),
            "cycle_rank_median":_median([g["beta"] for g in gms])
        },
        "quotient_observer":{
            "same_skin_topology_hidden_present":topo_collision>0,"resource_future_equivalence_basis":"INHERITED_GRRL_THEOREM_NOT_RECOMPUTED_FULL_COUNTER_PROFILE",
            "organizational_to_skin_class_ratio":len(set(org_hashes))/max(1,len(byskin))
        },
        "topology_services":{
            "diameter_hist":dict(sorted(Counter(g["diameter"] for g in gms).items())),"articulation_hist":dict(sorted(Counter(g["articulations"] for g in gms).items())),"beta_hist":dict(sorted(Counter(g["beta"] for g in gms).items())),
            "service_signature_classes":len({a["service_signature_sha256"] for a in actions}),"access_bottleneck_service_present":any(a["service_classes"]>1 for a in actions)
        },
        "obstruction_relief":{
            "all_bridge_types_supported_everywhere":all(a["type_pair_support"]==len(set(bridge_pairs)) for a in actions),
            "all_owner_pairs_have_some_action":all(a["owner_pair_support"]==len(s.owner_caps)*(len(s.owner_caps)-1)//2 for a,s in zip(actions,states)),
            "zero_action_states":sum(a["legal_action_labels"]==0 for a in actions)
        },
        "raw_growth":{
            "median_total_free":_median([sum(s.total_caps) for s in states]),"median_leaf_count":_median([s.leaf_count for s in states]),"median_total_relations":_median([s.relation_count_total for s in states])
        },
    }
    # Normalized organizational signature deliberately excludes magnitude-only growth
    # (absolute capacities, leaf count, relation count, Counter copy totals).
    normalized={
        "grammar":grammar_hash,
        "owner_count_hist":raw["diversity"]["owner_count_hist"],"edge_count_hist":raw["diversity"]["edge_count_hist"],
        "topology_classes":raw["diversity"]["topology_classes"],"organizational_classes":raw["diversity"]["organizational_classes"],
        "endpoint_support_min":raw["diversity"]["endpoint_support_min"],"endpoint_support_max":raw["diversity"]["endpoint_support_max"],
        "branching":{k:raw["branching"][k] for k in ["legal_action_labels_median","legal_action_labels_max","action_orbits_median","action_orbits_max","type_pair_support_min","type_pair_support_max","service_classes_median","service_classes_max"]},
        "symmetry":raw["symmetry"],
        "overlap_gluing":{k:raw["overlap_gluing"][k] for k in ["shared_fiber_pairs","strict_inclusions","intersection_realized_fraction","union_realized_fraction"]},
        "lineage":{k:raw["lineage"][k] for k in ["factor_fiber_median","factor_fiber_max","bridge_fraction_median","cycle_rank_median"]},
        "quotient":{k:raw["quotient_observer"][k] for k in ["same_skin_topology_hidden_present","organizational_to_skin_class_ratio"]},
        "topology_services":raw["topology_services"],
        "obstruction_relief":raw["obstruction_relief"],
    }
    nh=_sha(normalized);raw["normalized_signature"]=normalized;raw["normalized_signature_sha256"]=nh
    return raw,nh


def _classify_genealogy(prev:dict|None, cur:dict) -> dict:
    if prev is None:return {k:"SCOUT_NEW" for k in ["grammar","diversity","branching","symmetry","overlap_gluing","lineage","quotient","topology_services","obstruction_relief","representation"]}
    out={}
    out["grammar"]="PERSISTS" if prev["grammar_sha256"]==cur["grammar_sha256"] else "REORGANIZES"
    # diversity: exact depth grows while normalized classes may hold or change
    if cur["diversity"]["organizational_classes"]>prev["diversity"]["organizational_classes"]:out["diversity"]="EXPANDS"
    elif cur["diversity"]["organizational_classes"]<prev["diversity"]["organizational_classes"]:out["diversity"]="CLOSES"
    else:out["diversity"]="PERSISTS"
    for lane,key in [("branching","branching"),("symmetry","symmetry"),("overlap_gluing","overlap_gluing"),("lineage","lineage"),("quotient","quotient_observer"),("topology_services","topology_services")]:
        out[lane]="PERSISTS" if _sha(prev[key])==_sha(cur[key]) else "REORGANIZES"
    po,co=prev["obstruction_relief"],cur["obstruction_relief"]
    if (not po["all_bridge_types_supported_everywhere"] and co["all_bridge_types_supported_everywhere"]) or (po["zero_action_states"]>0 and co["zero_action_states"]==0):out["obstruction_relief"]="DESTROYS_OR_RELIEVES"
    else:out["obstruction_relief"]="PERSISTS" if _sha(po)==_sha(co) else "REORGANIZES"
    out["representation"]="PERSISTS" if prev["normalized_signature_sha256"]==cur["normalized_signature_sha256"] else "REFINES"
    return out


def _changed_families(genealogy:dict) -> list[str]:
    return [k for k,v in genealogy.items() if v in {"REFINES","CLOSES","REORGANIZES","COMBINES","DESTROYS_OR_RELIEVES","SCOUT_NEW"} and k not in {"grammar","representation"}]


def _calibration(phase8:Path, scanner_levels:dict[int,dict], generated_levels:dict[int,list[Any]]) -> dict:
    grad=json.loads((phase8/"authority"/"O7_GRADUATION_RESULT.json").read_text())
    o8=json.loads((phase8/"authority"/"O8_THEOREM_ACCELERATED_GRADUATION_RESULT.json").read_text())
    o9=json.loads((phase8/"authority"/"O9_BP0_RESULT.json").read_text())
    imm=json.loads((phase8/"graduation_compact"/"07_INPUT_SNAPSHOTS"/"O7_IMMUTABLE_SURVIVORS.json").read_text())["records"]
    imm_dig={r["state_digest"] for r in imm}
    o7_overlap=sum(getattr(s,"source_id","") in imm_dig for s in generated_levels[7])
    return {
        "status":"PASS" if (
            grad.get("status")=="O7_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_V2_7_PHASE2"
            and o8.get("status")=="O8_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_THEOREM_ACCELERATED_V2_8"
            and o9.get("outcome")=="O9_EXISTS_GRRL_REPEAT_STOP"
            and o9.get("normalized_grammar",{}).get("candidate_sha256")==GRAMMAR_EXPECTED
            and (set(scanner_levels)>= {7,8,9} or max(scanner_levels)<9)
        ) else "FAIL",
        "O7_authority":grad.get("science_sha256"),"O8_authority":o8.get("science_sha256"),"O9_authority":o9.get("science_sha256"),
        "O7_selected_historical_overlap":o7_overlap,"O7_selected_states":len(generated_levels[7]),
        "O8_generated_scanner_states":len(generated_levels.get(8, [])),"O9_generated_scanner_states":len(generated_levels.get(9, [])),
        "grammar_O7_O9":[scanner_levels[i]["grammar_sha256"] for i in (7,8,9) if i in scanner_levels],
        "note":"O7 scanner seed is selected directly from immutable graduated O7 records. O8/O9 scanner panels are exact synthetic fixed-lift instances calibrated against their authority statuses; they are not replacement graduation populations."
    }


def run_o_regime_scanner(phase8_seed:Path, output:Path, max_level:int|None=None, keep_work:bool=False, honor_earned_laws:bool=True) -> dict:
    spec=load_regime_scanner_spec(); motifs=load_motif_library(); earned_registry=load_earned_regime_laws() if honor_earned_laws else {"laws":[]} ; output=Path(output);output.mkdir(parents=True,exist_ok=True)
    earned_laws={x.get("law_id"):x for x in earned_registry.get("laws",[]) if x.get("status")=="EARNED"}
    max_level=int(max_level or spec["stopping"]["max_level"])
    work=Path(tempfile.mkdtemp(prefix="ig_o_regime_",dir=str(output)))
    try:
        phase8,o7root=_extract_seed(Path(phase8_seed),work)
        os.environ["OSCOUT_DATA_ROOT"]=str(o7root.resolve())
        engine=_load_module("ig_regime_o7_engine",o7root/"02_CODE"/"o7_live_engine.py");engine.O6=engine.import_o6()
        parent_map=engine.load_parent_records()
        _,bpairs=engine.O6.load_rules(); bridge_pairs=sorted(tuple(map(int,x)) for x in bpairs)
        # Authority grammar gate.
        o8auth=json.loads((phase8/"authority"/"O8_BP0_RESULT.json").read_text());o9auth=json.loads((phase8/"authority"/"O9_BP0_RESULT.json").read_text())
        if o8auth.get("normalized_grammar",{}).get("candidate_sha256")!=GRAMMAR_EXPECTED or o9auth.get("normalized_grammar",{}).get("candidate_sha256")!=GRAMMAR_EXPECTED:
            raise RuntimeError("authority grammar hash mismatch")
        # Select O7 scanner panel directly from immutable graduated records.
        records=json.loads((phase8/"graduation_compact"/"07_INPUT_SNAPSHOTS"/"O7_IMMUTABLE_SURVIVORS.json").read_text())["records"]
        base=[]
        for r in records:
            ctx=engine._profile_row_context(r,parent_map);edges=tuple(tuple(x) for x in r["edges"])
            base.append(O7State(engine,ctx,edges,(0,0,0,0,0,0,0),r["state_digest"],r["lane"]))
        selected=_farthest_select(base,int(spec["panel"]["beam"]))
        levels:dict[int,list[Any]]={7:selected}; backbones:dict[int,list[Any]]={}; level_results={}; build_meta={7:{"candidates":len(base),"selected":len(selected),"source":"IMMUTABLE_O7_AUTHORITY","matched_backbone_states":0}}
        pending_shock=None; stop=None
        prev_result=None
        for level in range(7,max_level+1):
            if level>7:
                selected,bm,backbone=_build_next_panel(engine,levels[level-1],level,bridge_pairs,motifs,spec);levels[level]=selected;backbones[level]=backbone;build_meta[level]=bm
            scan,nh=_scan_level(levels[level],level,bridge_pairs,GRAMMAR_EXPECTED)
            scan["matched_backbone"] = _backbone_signature(backbones[level],bridge_pairs) if level in backbones else {"status":"HISTORICAL_O7_ANCHOR_NOT_SYNTHETIC_BACKBONE","signature_sha256":None}
            inherited_service=earned_laws.get("O_DEPTH_ERASED_RELATION_ADD_SERVICE_V1")
            if inherited_service and scan["matched_backbone"].get("signature_sha256")==inherited_service.get("discovery_backbone_signature_sha256"):
                scan["matched_backbone"]["inherited_earned_law"]="O_DEPTH_ERASED_RELATION_ADD_SERVICE_V1"
            genealogy=_classify_genealogy(prev_result,scan);scan["genealogy"]=genealogy;scan["build_meta"]=build_meta[level]
            changed=_changed_families(genealogy);scan["structural_shock_changed_families"]=changed
            level_results[level]=scan
            # Post-calibration stop logic only.
            if level>=10:
                # Exploratory-beam lane shocks are reconnaissance only. Beam membership is
                # allowed to change with depth, so persistence on the next exploratory beam
                # cannot promote a group raise. Record the event and require a separate matched
                # longitudinal audit (common exact recipes / fixed matched panels) before any
                # structural-shock raise can become authoritative. This rule was added after the
                # O12/O13 audit showed the earlier O12 candidate was a panel-selection artifact.
                if len(changed)>=int(spec["promotion"]["structural_shock"]["minimum_changed_families"]):
                    scan.setdefault("exploratory_shock_events",[]).append({
                        "level":level,
                        "families":changed,
                        "status":"MATCHED_LONGITUDINAL_AUDIT_REQUIRED",
                        "promotion_blocked":True,
                    })
                # Matched longitudinal representation window. The exploratory beam is
                # deliberately allowed to change membership; promotion therefore uses
                # the frozen matched backbone, while requiring the broad categorical
                # guards to remain inside the same grammar/service regime.
                W=int(spec["promotion"]["representation_transition"]["W"])
                if level>=8+W-1:
                    ids=list(range(level-W+1,level+1))
                    win=[level_results[i] for i in ids]
                    bsha=[x["matched_backbone"].get("signature_sha256") for x in win]
                    same_backbone=(None not in bsha and len(set(bsha))==1)
                    grow=all(win[j]["matched_backbone"]["raw_growth"]["median_leaf_count"]>win[j-1]["matched_backbone"]["raw_growth"]["median_leaf_count"] for j in range(1,len(win)))
                    exactdiff=len({tuple(sorted(s.construction_digest for s in backbones[i])) for i in ids})==W
                    broad_guard=all(
                        x["grammar_sha256"]==GRAMMAR_EXPECTED
                        and x["obstruction_relief"]["all_bridge_types_supported_everywhere"]
                        and x["obstruction_relief"]["all_owner_pairs_have_some_action"]
                        and x["topology_services"]["access_bottleneck_service_present"]
                        for x in win
                    )
                    if same_backbone and grow and exactdiff and broad_guard:
                        inherited=earned_laws.get("O_DEPTH_ERASED_RELATION_ADD_SERVICE_V1")
                        if inherited and inherited.get("discovery_backbone_signature_sha256")==bsha[-1]:
                            # This representation transition has already been independently
                            # validated and promoted to an earned observer-relative law. Keep
                            # scanning instead of rediscovering it at every deeper window.
                            scan.setdefault("inherited_law_events",[]).append({"law_id":"O_DEPTH_ERASED_RELATION_ADD_SERVICE_V1","window":ids,"disposition":"SUPPRESSED_AS_ALREADY_EARNED"})
                        else:
                            stop={"classification":"META_SERVICE_REPRESENTATION_TRANSITION_CANDIDATE","window":ids,"matched_backbone_signature_sha256":bsha[-1],"reason":"the preregistered matched depth-erased action/service packet is byte-identical over W consecutive synthetic O levels while exact constructions and interior complexity strictly grow; the exploratory panel remains in the same grammar/resource/service regime"};break
            prev_result=scan
        if stop is None:
            stop={"classification":"UNRESOLVED_WITHIN_O32_NOT_NEGATIVE" if max_level>=32 else "UNRESOLVED_WITHIN_REQUESTED_BUDGET_NOT_NEGATIVE","through":max(level_results),"reason":"no preregistered confirmed raise/representation/plateau trigger fired within budget"}
        cal=_calibration(phase8,level_results,levels)
        if cal["status"]!="PASS":raise RuntimeError("O7/O8/O9 scanner calibration failed")
        result={
            "schema":"IG_O_REGIME_ADAPTIVE_SCANNER_RESULT_V1","date":"2026-08-30","status":"PASS","scanner_spec_sha256":_sha(spec),"motif_library_sha256":_sha({"motifs":motifs}),
            "phase8_seed_sha256":_sha_file(Path(phase8_seed)),"calibration":cal,"levels_executed":sorted(level_results),"stop":stop,
            "honor_earned_laws":bool(honor_earned_laws),"earned_laws_honored":sorted(earned_laws),
            "level_summaries":{str(k):v for k,v in level_results.items()},
            "scientific_interpretation":_interpret_stop(stop),
            "nonclaims":spec["forbidden_claims"],
            "next":_next_for_stop(stop),
        }
        result["science_sha256"]=_sha({k:v for k,v in result.items() if k!="science_sha256"})
        (output/"O_REGIME_ADAPTIVE_SCANNER_RESULT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        (output/"O_REGIME_LEVEL_SERIES.json").write_text(json.dumps({"schema":"IG_O_REGIME_LEVEL_SERIES_V1","rows":[_compact_row(level_results[k]) for k in sorted(level_results)]},indent=2,sort_keys=True)+"\n",encoding="utf-8")
        return result
    finally:
        if keep_work:
            (output/"WORKDIR.txt").write_text(str(work)+"\n")
        else:
            shutil.rmtree(work,ignore_errors=True)


def _compact_row(x:dict)->dict:
    return {
        "level":x["level"],"normalized_signature_sha256":x["normalized_signature_sha256"],"genealogy":x.get("genealogy"),
        "states":x["states"],"topology_classes":x["diversity"]["topology_classes"],"organizational_classes":x["diversity"]["organizational_classes"],"resource_skins":x["diversity"]["resource_skins"],
        "median_leaf_count":x["raw_growth"]["median_leaf_count"],"median_total_free":x["raw_growth"]["median_total_free"],"median_total_relations":x["raw_growth"]["median_total_relations"],
        "action_orbits_median":x["branching"]["action_orbits_median"],"service_classes_median":x["branching"]["service_classes_median"],"automorphism_size_median":x["symmetry"]["automorphism_size_median"],
        "matched_backbone_signature_sha256":x.get("matched_backbone",{}).get("signature_sha256"),
        "same_skin_topology_hidden_present":x["quotient_observer"]["same_skin_topology_hidden_present"],"all_bridge_types_supported_everywhere":x["obstruction_relief"]["all_bridge_types_supported_everywhere"],
        "structural_shock_changed_families":x.get("structural_shock_changed_families",[])
    }


def _interpret_stop(stop:dict)->str:
    c=stop.get("classification")
    if c=="META_SERVICE_REPRESENTATION_TRANSITION_CANDIDATE":
        return "The full bounded organizational scanner, not merely the local GRRL grammar, has entered a repeated depth-erased service regime across the frozen W-level window while exact nested carrier complexity continues growing. This earns a targeted meta-representation/formal-raise investigation, not yet a new algebraic carrier."
    if c=="O_REGIME_RAISE_CANDIDATE":
        return "Several independent established organizational families changed together and the post-change organization persisted one further level. This is a preregistered structural-shock raise candidate requiring exact targeted formalization."
    if "UNRESOLVED" in str(c):
        return "The scanner found no preregistered group-raise trigger within the bounded run. This is not evidence of eternal closure."
    return "See stop record."


def _next_for_stop(stop:dict)->str:
    c=stop.get("classification")
    if c in {"META_SERVICE_REPRESENTATION_TRANSITION_CANDIDATE","O_REGIME_RAISE_CANDIDATE","O_REGIME_FULL_SCANNER_PLATEAU_CANDIDATE"}:
        return "FREEZE_TARGETED_GROUP_RAISE_FORMALIZATION; do not continue blind O-level generation until the candidate is tested."
    return "REVIEW_RESOURCE_BUDGET_OR_ADAPTER; no negative theorem."
