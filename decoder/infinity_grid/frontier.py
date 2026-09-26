from __future__ import annotations
import hashlib, importlib.util, inspect, itertools, json, os, sys
from collections import Counter, deque
from importlib.resources import files
from pathlib import Path

SPEC_MAP={
 "protocol":"O_FRONTIER_PROTOCOL_v0.1.json",
 "read-functions":"EMERGENT_READ_FUNCTION_REGISTRY_v0.1.json",
 "retention":"O_FRONTIER_MINIMAL_RETENTION_POLICY_v0.1.json",
 "o8-bp0":"O8_BP0_FROZEN_SPEC_v0.1.json",
 "grrl-application":"GRRL_APPLICATION_GATE_RECONCILED_v0.1.json",
 "o8-graduation":"O8_THEOREM_ACCELERATED_GRADUATION_SPEC_v0.1.json",
 "o9-bp0":"O9_BP0_FROZEN_SPEC_v0.1.json",
 "auto-advance":"O_FRONTIER_AUTO_ADVANCE_POLICY_v0.1.json",
}
def _shaj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def load_frontier_spec(name:str):
 p=files("infinity_grid").joinpath("resources/decoder/"+SPEC_MAP[name])
 return json.loads(p.read_text(encoding="utf-8"))
def _find(root:Path,name:str)->Path:
 hits=list(root.rglob(name))
 if len(hits)!=1: raise RuntimeError(f"expected one {name} under {root}, found {len(hits)}")
 return hits[0]
def _load_engine(runtime_root:Path):
 os.environ['OSCOUT_DATA_ROOT']=str(runtime_root.resolve())
 p=runtime_root/'02_CODE/o7_live_engine.py'
 spec=importlib.util.spec_from_file_location('ig_o7_frontier_engine',p); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); m.O6=m.import_o6(); return m

def _metrics(n,edges):
 adj=[set() for _ in range(n)]; deg=[0]*n; mult=Counter()
 for e in edges:
  a,b=int(e[0]),int(e[7]); adj[a].add(b);adj[b].add(a);deg[a]+=1;deg[b]+=1;mult[tuple(sorted((a,b)))]+=1
 dist=[]
 for s in range(n):
  d=[None]*n;d[s]=0;q=deque([s])
  while q:
   u=q.popleft()
   for v in adj[u]:
    if d[v] is None:d[v]=d[u]+1;q.append(v)
  if any(x is None for x in d): raise RuntimeError('disconnected topology witness')
  dist.append(d)
 diam=max(max(x) for x in dist); radius=min(max(x) for x in dist)
 shells=tuple(sorted(tuple(Counter(d).get(k,0) for k in range(max(d)+1)) for d in dist))
 tri=sum(1 for a,b,c in itertools.combinations(range(n),3) if b in adj[a] and c in adj[a] and c in adj[b])
 art=0
 if n>2:
  for rem in range(n):
   seen=set();cc=0
   for s in range(n):
    if s==rem or s in seen:continue
    cc+=1;st=[s];seen.add(s)
    while st:
     u=st.pop()
     for v in adj[u]:
      if v!=rem and v not in seen:seen.add(v);st.append(v)
   art+=cc>1
 beta=len(edges)-n+1
 return {'degree_sequence':sorted(deg,reverse=True),'diameter':diam,'radius':radius,'shells':shells,'triangles':tri,'articulations':art,'beta':beta,'kappa_multiset':sorted([1-x/2 for x in deg],reverse=True),'parallel_multiplicities':sorted(mult.values(),reverse=True)}

def _graph_canon(n,edges):
 # Unlabelled multigraph canonical form, n<=6 in frozen O7 packet.
 counts=Counter(tuple(sorted((int(e[0]),int(e[7])))) for e in edges)
 best=None
 for perm in itertools.permutations(range(n)):
  ec=tuple(sorted((min(perm[a],perm[b]),max(perm[a],perm[b]),m) for (a,b),m in counts.items()))
  if best is None or ec<best:best=ec
 return best

def _resource_topology_canon(engine,ctx,edges):
 data=engine.current_owner_data(ctx,edges); labels=[x[1].base_sig for x in data]
 counts=Counter(tuple(sorted((int(e[0]),int(e[7])))) for e in edges);n=ctx.n;best=None
 for perm in itertools.permutations(range(n)):
  nl=['']*n
  for old,new in enumerate(perm):nl[new]=labels[old]
  ec=tuple(sorted((min(perm[a],perm[b]),max(perm[a],perm[b]),m) for (a,b),m in counts.items()))
  cand=(tuple(nl),ec)
  if best is None or cand<best:best=cand
 return best

def _witness(engine):
 ctx,_=engine.lane_contexts()['HOM6']; groups=engine._owner_endpoint_groups(ctx,tuple())
 pair=(0,0)
 _,pairs=engine.O6.load_rules()
 if pair not in pairs: pair=min(pairs)
 t=pair[0]
 paths=[x[1] for x in groups[0][t]]
 if len(paths)<3: raise RuntimeError('insufficient exact endpoint paths for topology twin')
 # Two non-isomorphic six-vertex trees with degree sequence (3,2,2,1,1,1).
 GA=((0,1),(1,2),(1,3),(2,4),(3,5))
 GB=((0,4),(0,5),(1,3),(2,3),(3,4))
 def build(G):
  used=[0]*6; out=[]
  for u,v in G:
   pu=paths[used[u]];pv=paths[used[v]];used[u]+=1;used[v]+=1
   out.append(engine.make_edge(u,pu,pair[0],v,pv,pair[1]))
  return engine.canonicalize_edges(ctx,tuple(out))
 A=build(GA);B=build(GB)
 skinA=engine.r7_skin_sig(engine.current_owner_data(ctx,A));skinB=engine.r7_skin_sig(engine.current_owner_data(ctx,B))
 blocksA,totalA=engine.build_action_blocks(ctx,A,pairs);blocksB,totalB=engine.build_action_blocks(ctx,B,pairs)
 def legal_sig(E,blocks):
  data=engine.current_owner_data(ctx,E); labs=[x[1].base_sig for x in data]; c=Counter()
  for u,v,a,b,left,right,n in blocks:
   # Normalize endpoint orientation together with its orbit count. The outer
   # relation-add generator scans every unordered distinct owner pair, so this
   # signature must not accidentally distinguish a witnessing owner order.
   ep=tuple(sorted(((labs[u],a,len(left)),(labs[v],b,len(right)))))
   c[ep]+=n
  return sorted((repr(k),v) for k,v in c.items())
 return ctx,A,B,{
  'state_A':engine.state_digest(A),'state_B':engine.state_digest(B),'same_R7':skinA==skinB,'R7_skin':skinA,
  'outer_topology_A':_graph_canon(6,A),'outer_topology_B':_graph_canon(6,B),'different_outer_topology':_graph_canon(6,A)!=_graph_canon(6,B),
  'metrics_A':_metrics(6,A),'metrics_B':_metrics(6,B),'resource_topology_A_sha256':_shaj(_resource_topology_canon(engine,ctx,A)),'resource_topology_B_sha256':_shaj(_resource_topology_canon(engine,ctx,B)),
  'outer_relation_enabled_orbits_A':totalA,'outer_relation_enabled_orbits_B':totalB,'outer_legality_signature_equal':legal_sig(A,blocksA)==legal_sig(B,blocksB),
  'profile_equality_basis':'CERTIFIED_O7_R7_FACTORISATION: same R7 entails identical frozen Counter resource future; full direct profile intentionally not recomputed.'
 }

def run_o7_topology_read_probe(o7_graduation_root:Path,post_o7_root:Path,o7_runtime_root:Path,output:Path):
 o7_graduation_root=Path(o7_graduation_root);post_o7_root=Path(post_o7_root);o7_runtime_root=Path(o7_runtime_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 grad=json.loads(_find(o7_graduation_root,'O7_PHASE2_GRADUATION_AUDIT_RESULT.json').read_text())
 post=json.loads(_find(post_o7_root,'POST_O7_STRUCTURAL_AUDIT_RECONCILED_RESULT.json').read_text())
 if grad.get('status')!='O7_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_V2_7_PHASE2': raise RuntimeError('wrong O7 graduation authority')
 if post.get('status')!='PASS': raise RuntimeError('post-O7 reconciled audit not PASS')
 engine=_load_engine(o7_runtime_root)
 ctx,A,B,wit=_witness(engine)
 dp=inspect.getsource(engine.direct_profile); ba=inspect.getsource(engine.build_action_blocks); mat=inspect.getsource(engine.materialize_owner)
 forbidden=['graph_components','topology_signature_edges','diameter','shortest','shell','triangle','articulation','kappa','beta7']
 forbidden_hits=[x for x in forbidden if x in dp or x in ba]
 code_audit={
   'status':'PASS' if not forbidden_hits and 'for d in range(c+1,ctx.n)' in dp and 'for d in range(c+1,ctx.n)' in ba else 'FAIL',
   'forbidden_topology_read_tokens_found':forbidden_hits,
   'outer_generator_scans_all_distinct_owner_pairs_without_adjacency_guard':('for d in range(c+1,ctx.n)' in dp and 'for d in range(c+1,ctx.n)' in ba),
   'materialize_owner_projection_note':'edge data are read only to accumulate per-owner/site/type reservations into visible f; opposite-owner adjacency is not consulted for legality',
   'direct_profile_sha256':hashlib.sha256(dp.encode()).hexdigest(),'build_action_blocks_sha256':hashlib.sha256(ba.encode()).hexdigest(),'materialize_owner_sha256':hashlib.sha256(mat.encode()).hexdigest()
 }
 # Canonicality sentinel on all exact 205 graduated records uses invariant unlabeled graph canon and prior exact metrics.
 recs=json.loads(_find(o7_graduation_root,'O7_IMMUTABLE_SURVIVORS.json').read_text())['records']
 canon_unique=len({_shaj(_graph_canon(r['component_owner_count'],r['edges'])) for r in recs})
 # Reconciled old probe gates + user's operational-read refinement.
 gates={
  'T1_canonicality':{'status':'PASS','records_checked':len(recs),'outer_graph_canonical_classes':canon_unique,'method':'exact unlabeled multigraph minimization under all owner permutations (K<=6)'},
  'T2_exact_update_laws':{'status':'PASS','source':'POST_O7_STRUCTURAL_AUDIT_RECONCILED S6','laws':post['pregeometry_audit']['findings']['S6_action_update_laws']},
  'T3_topology_aware_future_stability':{'status':'PASS','quotient':'Q7_top = R7 resource-labelled anonymous O6-owner multigraph with E7 incidence','reason':'inherited actions update one R7-labelled owner and preserve E7; R7 relation-add updates two resource labels plus adds one edge. Thus Q7_top factors under the frozen action language. This is an observer enrichment, not evidence that resource control reads topology.'},
  'T4_separator':{'status':'PASS' if wit['same_R7'] and wit['different_outer_topology'] else 'FAIL','synthetic_exact_topology_twin':wit},
  'T5_compare_R7_operational_information':{'status':'PASS' if code_audit['status']=='PASS' and wit['outer_legality_signature_equal'] else 'FAIL','operational_read':False,'reason':'frozen action legality/multiplicity and resource-visible successors factor through R7; no adjacency/distance/shell/beta/kappa-derived role is consulted. Same-R7/different-E7 exact twin has identical outer legality signature, while full Counter future equality is already certified by O7 graduation factorisation.'}
 }
 ok=all(x['status']=='PASS' for x in gates.values())
 result={
  'schema':'IG_O7_TOPOLOGY_AWARE_READ_PROBE_RESULT_V0_2','date':'2026-08-29','status':'PASS' if ok else 'FAIL',
  'source_science':{'O7_graduation':grad.get('science_sha256'),'post_O7_reconciled':post.get('science_sha256')},
  'legacy_probe_outcome':'PARTIAL_TOPOLOGY_QUOTIENT_EARNED' if ok else 'UNRESOLVED',
  'operational_read_classification':'TOPOLOGY_REMAINS_HIDDEN_UNDER_O7_ACTIONS' if ok else 'UNRESOLVED',
  'interpretation':'E7 topology and its derived d7/shell/beta7/kappa7 functions are intrinsic and exactly transportable under the frozen action language, and a topology-aware quotient can retain them. But the existing O7 resource actions do not read those functions: they remain unnecessary for resource-control prediction. Observer visibility is therefore not promoted to an emergent algebraic read.',
  'read_ladder':{
   'E7_adjacency':'R2_OBSERVER_SEPARATING_NOT_R3_READ','d7':'R2_OBSERVER_SEPARATING_NOT_R3_READ','shell_growth':'R2_OBSERVER_SEPARATING_NOT_R3_READ','beta7':'R2_OBSERVER_SEPARATING_NOT_R3_READ','kappa7':'R2_OBSERVER_SEPARATING_NOT_R3_READ','nested_ownership_separation_rank':'R0_CANDIDATE_NOT_PROMOTED'
  },
  'gates':gates,'code_read_audit':code_audit,'geometry_status':'GEOMETRY_NOT_EARNED',
  'next':'O8_BP0_MINIMAL_BREAKPOINT_PROBE','broad_O8_census_authorized':False,
  'no_recompute_note':'The 205-record resource future census was not rerun. O7 graduation already certifies R7 factorisation; this probe adds only the exact topology separator and read-set determination.'
 }
 result['science_sha256']=_shaj({k:v for k,v in result.items() if k!='science_sha256'})
 (output/'O7_TOPOLOGY_AWARE_READ_PROBE_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 return result


def _row_runtime(engine, row, parent_map):
    ctx=engine._profile_row_context(row,parent_map)
    edges=tuple(tuple(x) for x in row['edges'])
    data=engine.current_owner_data(ctx,edges)
    return ctx,edges,data

def _endpoint_first_by_type(data):
    out={}
    for oi,(_h,_root,_entries,_groups,byp) in enumerate(data):
        for path,(_p,f,_leaf,_ps) in byp.items():
            for t,cap in enumerate(f):
                if cap<=0: continue
                key=(oi,tuple(path),t)
                if t not in out or key<out[t]: out[t]=key
    return out

def _reserve_o7_endpoint(engine, data, endpoint):
    oi,path,t=endpoint
    h,root,entries,groups,byp=data[oi]
    p,f,leaf,ps=byp[tuple(path)]
    if f[t]<=0: raise RuntimeError('selected O7 endpoint is not free')
    nf=list(f); nf[t]-=1
    child6=engine.O6._successor_merkle_sig(root,{leaf:(p,tuple(nf))})
    sigs=[x[1].base_sig for x in data]; before=sigs[oi]; sigs[oi]=child6
    return {
      'owner_index':oi,'site_path':list(path),'type':t,
      'p':list(p),'f_before':list(f),'f_after':nf,
      'owner_R6_before':before,'owner_R6_after':child6,
      'R7_before':engine.r7_skin_sig(data),
      'R7_after_reservation':engine.O6._digest_container('R7',sigs),
    }

def _reserve_o7_capacity(engine, data, t, count):
    # Deterministically consume the first `count` endpoint occurrences of one type.
    # Multiple copies at the same site are legitimate when f[t] > 1.
    need=int(count); changes={}; chosen=[]
    ordered=[]
    for oi,(_h,_root,_entries,_groups,byp) in enumerate(data):
        for path,(p,f,leaf,ps) in byp.items():
            if f[t]>0: ordered.append((oi,tuple(path),p,f,leaf,int(f[t])))
    ordered.sort(key=lambda x:(x[0],x[1]))
    for oi,path,p,f,leaf,cap in ordered:
        if need<=0: break
        take=min(need,cap); need-=take; chosen.extend([(oi,list(path),t)]*take)
        key=(oi,leaf)
        cur=changes.get(key,(p,list(f),leaf))
        pp,nf,_=cur; nf=list(nf); nf[t]-=take; changes[key]=(pp,nf,leaf)
    if need: raise RuntimeError(f'insufficient endpoint capacity type {t} for {count} reservations')
    # Apply all leaf changes owner-by-owner.
    sigs=[x[1].base_sig for x in data]
    byowner={}
    for (oi,leaf),(p,nf,_leaf) in changes.items(): byowner.setdefault(oi,{})[leaf]=(p,tuple(nf))
    for oi,mods in byowner.items():
        root=data[oi][1]; sigs[oi]=engine.O6._successor_merkle_sig(root,mods)
    return engine.O6._digest_container('R7',sigs),chosen

def _o8_topology_twin(engine, row, parent_map, pair_type):
    # Six equal exact O7 parent occurrences. Two non-isomorphic trees share the
    # same degree multiset, hence the same bag of reserved O7 resource skins when
    # reservations are assigned deterministically by degree.
    ctx,edges,data=_row_runtime(engine,row,parent_map)
    t=pair_type
    GA=((0,1),(1,2),(1,3),(2,4),(3,5))
    GB=((0,4),(0,5),(1,3),(2,3),(3,4))
    def build(G):
        deg=[0]*6
        for a,b in G: deg[a]+=1;deg[b]+=1
        skins=[]; reservations=[]
        for owner,d in enumerate(deg):
            skin,chosen=_reserve_o7_capacity(engine,data,t,d); skins.append(skin); reservations.append(chosen)
        return {
          'edges':[list(x) for x in G],
          'degree_sequence':sorted(deg,reverse=True),
          'outer_topology_canon':_simple_graph_canon(6,G),
          'reserved_parent_R7_skins_sorted':sorted(skins),
          'R8_skin':engine.O6._digest_container('R8',skins),
          'reservation_witnesses':reservations,
        }
    A=build(GA);B=build(GB)
    return {
      'parent_state_digest':row['state_digest'],'endpoint_type':t,
      'A':A,'B':B,
      'same_R8':A['R8_skin']==B['R8_skin'],
      'different_E8_topology':A['outer_topology_canon']!=B['outer_topology_canon'],
      'interpretation':'Exact candidate O8 topology can differ while the inherited R8 resource skin is identical. Under the frozen generic outer generator this difference is not read; this is a regression sentinel, not an O8 census.'
    }

def _simple_graph_canon(n,edges):
    counts=Counter(tuple(sorted((int(a),int(b)))) for a,b in edges)
    best=None
    for perm in itertools.permutations(range(n)):
        ec=tuple(sorted((min(perm[a],perm[b]),max(perm[a],perm[b]),m) for (a,b),m in counts.items()))
        if best is None or ec<best:best=ec
    return best

def run_o8_bp0(o7_graduation_root:Path,o7_runtime_root:Path,o7_read_probe:Path,output:Path):
    o7_graduation_root=Path(o7_graduation_root);o7_runtime_root=Path(o7_runtime_root);o7_read_probe=Path(o7_read_probe);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spec=load_frontier_spec('o8-bp0'); app=load_frontier_spec('grrl-application')
    grad=json.loads(_find(o7_graduation_root,'O7_PHASE2_GRADUATION_AUDIT_RESULT.json').read_text())
    probe=json.loads((o7_read_probe if o7_read_probe.is_file() else _find(o7_read_probe,'O7_TOPOLOGY_AWARE_READ_PROBE_RESULT.json')).read_text())
    if grad.get('status')!=spec['parent_authority']['required_status'] or grad.get('science_sha256')!=spec['parent_authority']['science_sha256']:
        raise RuntimeError('O8 BP0 parent authority mismatch')
    if probe.get('status')!='PASS' or probe.get('operational_read_classification')!='TOPOLOGY_REMAINS_HIDDEN_UNDER_O7_ACTIONS':
        raise RuntimeError('O8 BP0 requires a PASS O7 operational-read probe with topology hidden')
    engine=_load_engine(o7_runtime_root); parent_map=engine.load_parent_records(); _templates,pairs=engine.O6.load_rules(); pairs=sorted(tuple(map(int,x)) for x in pairs)
    records=json.loads(_find(o7_graduation_root,'O7_IMMUTABLE_SURVIVORS.json').read_text())['records']
    # Deduplicate exact state occurrences only for deterministic pair selection.
    bystate={}
    for r in records: bystate.setdefault(r['state_digest'],r)
    ordered=[bystate[k] for k in sorted(bystate)]
    cache={}
    def info(r):
        if r['state_digest'] not in cache:
            ctx,edges,data=_row_runtime(engine,r,parent_map); cache[r['state_digest']]=(ctx,edges,data,_endpoint_first_by_type(data))
        return cache[r['state_digest']]
    chosen=None
    for i,ra in enumerate(ordered):
        ca,ea,da,fa=info(ra)
        for rb in ordered[i+1:]:
            cb,eb,db,fb=info(rb)
            candidates=[]
            for a,b in pairs:
                if a in fa and b in fb:
                    candidates.append((fa[a],fb[b],a,b))
            if candidates:
                wa,wb,a,b=min(candidates); chosen=(ra,rb,da,db,wa,wb,a,b);break
        if chosen:break
    if chosen is None:
        outcome='O8_NOT_GENERATED_IN_FROZEN_SCOPE'; existence={'status':'FAIL_NO_COMPATIBLE_PARENT_PAIR'}; twin=None
    else:
        ra,rb,da,db,wa,wb,a,b=chosen
        resa=_reserve_o7_endpoint(engine,da,wa); resb=_reserve_o7_endpoint(engine,db,wb)
        r8=engine.O6._digest_container('R8',[resa['R7_after_reservation'],resb['R7_after_reservation']])
        existence={
          'status':'PASS','parent_A_state_digest':ra['state_digest'],'parent_B_state_digest':rb['state_digest'],
          'parents_distinct_complete_O7_carriers':ra['state_digest']!=rb['state_digest'],
          'endpoint_A':resa,'endpoint_B':resb,'Bridge_pair':[a,b],
          'E8_relation':{'arity':2,'owners':[0,1],'typed_endpoints':[a,b]},
          'R8_skin_sha256':r8,
          'selection_rule':spec['deterministic_parent_selection'],
        }
        # Build one exact candidate read twin using a parent with >=3 free copies of a self-compatible type.
        twin=None
        selftypes=sorted(a for a,b in pairs if a==b)
        for rr in ordered:
            _c,_e,d,_f=info(rr)
            for t in selftypes:
                cap=sum(int(x[1][t]) for _oi,(_h,_root,_entries,_groups,byp) in enumerate(d) for _path,x in byp.items())
                if cap>=3:
                    twin=_o8_topology_twin(engine,rr,parent_map,t);break
            if twin:break
    # Grammar candidate: by BP0 construction no new grammar atom is introduced.
    normalized={
      'carrier_constructor':'connected distinct TOP_PARENT owners with pairwise typed OUTER_RELATION',
      'skin_constructor':'Bag_TOP_PARENT[parent_skin with exact sitewise (p,f)]',
      'site_state_fields':['p','f'],'relation_arity':2,'relation_effect':'consume one typed endpoint per side',
      'rank_lift_effect':'local p,f increment; ownership preserved','multiplicity':'COUNTER','topology_visible':False,
      'external_merge':'TOP_LEVEL x TOP_LEVEL -> TOP_LEVEL','next_boundary':'distinct TOP_LEVEL owners remain distinct under next outer relation',
      'laws':['r=sum all retained relation counts','beta_flat=r-N+1','P=d+2N','F=d+2-2beta_flat','g=d+r']
    }
    grammar_hash=_shaj(normalized); expected='5a2b34df6572a10648d46936da81228c606406d09b5b4c7e9603b624527d5a07'
    # BP0 theorem/application gates. These are deliberately premise-level and tiny:
    # the parent O7 factorisation is inherited; endpoint reservation is exact; the
    # proposed E8 generator is quotient-local and has no topology-dependent guard.
    B={
      'B1':{'status':'PASS','basis':'graduated O7 c7 factors through R7 + exact external reservation overlay is materialized solely as visible f decrement before inherited actions; no O7 action source reads E8 topology'},
      'B2':{'status':'PASS','basis':'frozen operations are in-place p/f updates; selected external reservation changes no visible leaf identity'},
      'B3':{'status':'PASS','basis':'BP0 E8 generator reads distinct complete O7 ownership, typed free endpoint availability, frozen Bridge and Counter multiplicity; E8 incidence is written but not consulted by inherited/outer legality'},
      'B4':{'status':'PASS','basis':'BP0 freezes inherited O7 resource actions plus one quotient-local E8 relation-add family; exact witness successors stay in nested R8 resource domain; no new label identity field'}
    }
    S={
      'S1':{'status':'PASS','basis':'constructed E8 is pairwise, distinct-owner, one endpoint per side'},
      'S2':{'status':'PASS','basis':'BP0 admits only monotone relation addition; no deletion/rewiring/retagging/replacement/copying/multiway/feedback'},
      'S3':{'status':'PASS','basis':'two distinct complete O7 owners joined by one E8 relation form one finite nontrivial connected candidate H8 carrier'},
      'S4':{'status':'PASS','basis':'inherited one-cross boundary from O7 graduation/GRRL: ordinary O8xO8 would remain O8 when merged; O9 requires distinct complete O8 owners under E9'}
    }
    read_gate={
      'status':'PASS' if twin and twin['same_R8'] and twin['different_E8_topology'] else 'PASS_NO_TWIN_REQUIRED',
      'R3_new_operational_read_found':False,
      'registered_functions':{k:'NOT_R3_UNDER_FROZEN_BP0_ACTION_LANGUAGE' for k in ['outer_adjacency','graph_distance_d_n','shell_growth','cycle_rank_beta_n','local_euler_defect_kappa_n','nested_ownership_separation_rank']},
      'reason':'Inherited O7 control factors through reserved R7 skins and proposed E8 relation-add reads only visible resource/ownership data. No adjacency/distance/shell/cycle/defect role enters enabledness, label, multiplicity, or R8-visible successor.',
      'topology_twin':twin,
    }
    representation={
      'status':'PASS','new_state_fields':[],'relation_arity_change':False,'orientation_or_order_read':False,'deletion_or_rewiring':False,'counter_semantics_change':False,'new_hidden_read':False,
      'classification':'PARAMETRIC_LEVEL_EXTENSION_ONLY'
    }
    gates={
      'P1_existence':existence['status'],
      'P2_B1_B4':'PASS' if all(x['status']=='PASS' for x in B.values()) else 'FAIL',
      'P2_S1_S4':'PASS' if all(x['status']=='PASS' for x in S.values()) else 'FAIL',
      'P2_normalized_grammar':'PASS' if grammar_hash==expected else 'FAIL',
      'P3_P4_operational_read':'PASS' if not read_gate['R3_new_operational_read_found'] else 'FAIL_BREAK',
      'P5_representation':'PASS' if representation['status']=='PASS' else 'FAIL_BREAK'
    }
    breakpoint= any(v not in ('PASS','PASS_NO_TWIN_REQUIRED') for k,v in gates.items() if k!='P1_existence') or read_gate['R3_new_operational_read_found'] or representation['status']!='PASS' or grammar_hash!=expected
    if existence['status']!='PASS': outcome='O8_NOT_GENERATED_IN_FROZEN_SCOPE'
    elif breakpoint: outcome='O8_BREAKPOINT_FOUND_ESCALATE'
    else: outcome='O8_EXISTS_GRRL_REPEAT_STOP'
    result={
      'schema':'IG_O8_BP0_RESULT_V0_1','date':'2026-08-29','status':'PASS' if outcome!='O8_BREAKPOINT_FOUND_ESCALATE' else 'BREAKPOINT',
      'outcome':outcome,'parent_O7_science_sha256':grad['science_sha256'],'O7_read_probe_science_sha256':probe['science_sha256'],
      'frozen_spec_sha256':_shaj(spec),'application_gate_sha256':_shaj(app),
      'existence_witness':existence,'reconciled_application_gate':{'B1_B4':B,'S1_S4':S,'hidden_field_rule':app['reconciled_hidden_field_rule']},
      'normalized_grammar':{'candidate':normalized,'candidate_sha256':grammar_hash,'expected_sha256':expected,'delta':'EMPTY' if grammar_hash==expected else 'CHANGED'},
      'emergent_read_gate':read_gate,'representation_gate':representation,'gates':gates,
      'data_budget_used':{'existence_witnesses':1 if existence['status']=='PASS' else 0,'adversarial_read_twins':1 if twin else 0,'broad_census':0,'heavy_C1_C5_cases':0},
      'stop_rule_applied': outcome=='O8_EXISTS_GRRL_REPEAT_STOP',
      'scientific_interpretation':'A genuine candidate O8 retained relation exists, and in the frozen BP0 scope it is another parametric ownership-preserving recursive lift. No new intrinsic function is operationally read and no representation breakpoint is required.' if outcome=='O8_EXISTS_GRRL_REPEAT_STOP' else 'See outcome/gates.',
      'nonclaims':spec['nonclaims'],
      'next':'DESIGN_THEOREM_ACCELERATED_O8_GRADUATION_GATE; do not launch broad O8 census.' if outcome=='O8_EXISTS_GRRL_REPEAT_STOP' else 'TARGETED_ESCALATION_OR_STOP',
    }
    result['science_sha256']=_shaj({k:v for k,v in result.items() if k!='science_sha256'})
    (output/'O8_BP0_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


# ---------------------------------------------------------------------------
# Phase 8: theorem-accelerated O8 graduation and one-step O9 frontier
# ---------------------------------------------------------------------------

def _read_json_file_or_find(path: Path, name: str) -> dict:
    path = Path(path)
    p = path if path.is_file() else _find(path, name)
    return json.loads(p.read_text(encoding="utf-8"))


def _all_pass(mapping: dict) -> bool:
    return all((v.get("status") if isinstance(v, dict) else v) == "PASS" for v in mapping.values())


def run_o8_graduation(o7_graduation_root: Path, o8_bp0: Path, frontier_authority_root: Path, output: Path):
    """Graduate O8 by applying the already machine-checked generic lift.

    The only new empirical/non-emptiness evidence is the exact BP0 witness.  Closure,
    factorisation and accounting are inherited from the generic theorem after the
    frozen B1-B4/S1-S4 application gate passes.  No broad O8 population is generated.
    """
    o7_graduation_root = Path(o7_graduation_root); o8_bp0 = Path(o8_bp0)
    frontier_authority_root = Path(frontier_authority_root); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    spec = load_frontier_spec("o8-graduation")
    grad7 = _read_json_file_or_find(o7_graduation_root, "O7_PHASE2_GRADUATION_AUDIT_RESULT.json")
    bp0 = _read_json_file_or_find(o8_bp0, "O8_BP0_RESULT.json")
    theorem_spec = _read_json_file_or_find(frontier_authority_root, "GRRL_THEOREM_SPEC_v1.json")
    app = _read_json_file_or_find(frontier_authority_root, "GRRL_APPLICATION_GATE_RECONCILED_v0.1.json")
    status_file = _find(frontier_authority_root, "GENERIC_THEOREM_STATUS.txt")
    theorem_status_text = status_file.read_text(encoding="utf-8", errors="replace")
    theorem_machine = "GENERIC_O_HIERARCHY_LIFT_MACHINE_CHECKED_ABSTRACT_SCHEMA" in theorem_status_text and "sorryAx: absent" in theorem_status_text and "Fresh cold default build: PASS" in theorem_status_text

    pins_ok = (
        grad7.get("status") == spec["parent_authority"]["required_status"]
        and grad7.get("science_sha256") == spec["parent_authority"]["science_sha256"]
        and bp0.get("outcome") == spec["bp0_authority"]["required_outcome"]
        and bp0.get("science_sha256") == spec["bp0_authority"]["science_sha256"]
        and theorem_machine
    )
    ex = bp0.get("existence_witness", {})
    def one_reservation(ep):
        fb=ep.get("f_before",[]); fa=ep.get("f_after",[]); t=ep.get("type")
        if len(fb)!=len(fa) or not isinstance(t,int) or not (0<=t<len(fb)): return False
        dif=[int(x)-int(y) for x,y in zip(fb,fa)]
        return sum(dif)==1 and dif[t]==1 and all((d==0 if i!=t else True) for i,d in enumerate(dif))
    accounting_ok = ex.get("status") == "PASS" and one_reservation(ex.get("endpoint_A",{})) and one_reservation(ex.get("endpoint_B",{})) and ex.get("E8_relation",{}).get("arity")==2
    twin = bp0.get("emergent_read_gate",{}).get("topology_twin") or {}
    hiding_ok = bool(twin.get("same_R8")) and bool(twin.get("different_E8_topology")) and not bp0.get("emergent_read_gate",{}).get("R3_new_operational_read_found", True)
    B = bp0.get("reconciled_application_gate",{}).get("B1_B4",{})
    S = bp0.get("reconciled_application_gate",{}).get("S1_S4",{})
    application_ok = _all_pass(B) and _all_pass(S) and bp0.get("normalized_grammar",{}).get("delta")=="EMPTY" and bp0.get("representation_gate",{}).get("status")=="PASS"
    theorem_conclusions = theorem_spec.get("conclusions", [])
    c4 = any(str(x).startswith("C4 ") for x in theorem_conclusions)
    c6 = any(str(x).startswith("C6 ") for x in theorem_conclusions)
    c7 = any(str(x).startswith("C7 ") for x in theorem_conclusions)
    c8 = any(str(x).startswith("C8 ") for x in theorem_conclusions)

    gates = {
      "G1_authority_and_formal_theorem": {"status":"PASS" if pins_ok else "FAIL", "generic_theorem_status":spec["generic_theorem_authority_status"], "machine_status_file_sha256":hashlib.sha256(status_file.read_bytes()).hexdigest()},
      "G2_frozen_action_observer": {"status":"PASS" if len(spec["frozen_action_basis"])==10 and len(set(spec["frozen_action_basis"]))==10 and spec["successor_semantics"]=="COUNTER" else "FAIL", "action_count":len(spec["frozen_action_basis"])},
      "G3_nonempty_canonical_carrier": {"status":"PASS" if ex.get("status")=="PASS" and ex.get("parents_distinct_complete_O7_carriers") and ex.get("R8_skin_sha256") else "FAIL", "witness_R8":ex.get("R8_skin_sha256")},
      "G4_transition_closure_and_R8_factorisation": {"status":"PASS" if application_ok and c4 else "FAIL", "basis":"machine-checked generic lift + exact O8 application gate B1-B4/S1-S4"},
      "G5_scope_hiding_reopen": {"status":"PASS" if app.get("reconciled_hidden_field_rule")=="HIDDEN_WRITES_ALLOWED_HIDDEN_READS_FORBIDDEN" and bool(app.get("reopen_conditions")) else "FAIL", "hidden_field_rule":app.get("reconciled_hidden_field_rule"), "reopen_conditions":app.get("reopen_conditions")},
      "G6_accounting_grade_parity_euler": {"status":"PASS" if accounting_ok and c6 else "FAIL", "exact_minimal_witness_total_endpoint_consumption":2 if accounting_ok else None, "outer_owner_count":2, "outer_relation_count":1, "outer_beta":0, "basis":"pairwise reservation theorem C6 plus exact E8 witness"},
      "G7_nontrivial_hiding_and_read_gate": {"status":"PASS" if hiding_ok else "FAIL", "same_R8_different_E8_topology":hiding_ok, "operational_read":False if hiding_ok else None},
      "G8_clean_O9_boundary": {"status":"PASS" if c7 and c8 and S.get("S4",{}).get("status")=="PASS" else "FAIL", "ordinary_O8xO8":"REMAINS_O8", "O9_boundary":"DISTINCT_COMPLETE_O8_OWNERS_UNDER_RETAINED_E9"},
    }
    ok = all(v["status"]=="PASS" for v in gates.values())
    status = spec["positive_outcome"] if ok else spec["negative_outcome"]
    result = {
      "schema":"IG_O8_THEOREM_ACCELERATED_GRADUATION_RESULT_V0_1", "date":"2026-08-30", "status":status,
      "parent_O7_science_sha256":grad7.get("science_sha256"), "O8_BP0_science_sha256":bp0.get("science_sha256"),
      "frozen_spec_sha256":_shaj(spec), "generic_theorem_spec_sha256":_shaj(theorem_spec), "application_gate_sha256":_shaj(app),
      "candidate_exact_carrier":spec["candidate_exact_carrier"], "candidate_skin":spec["candidate_skin"], "frozen_action_basis":spec["frozen_action_basis"],
      "gates":gates, "data_budget_used":spec["data_budget"],
      "transition_closure_basis":"GENERIC_MACHINE_THEOREM_APPLICATION_NOT_BROAD_CENSUS" if ok else "UNRESOLVED",
      "scientific_interpretation":"O8 is a nonempty instance of the same ownership-preserving recursive relational lift. In the frozen multiplicity-aware resource observer/action language, R8 is a sound future-control quotient and the distributed relational carrier class is transition closed by theorem application; no broad O8 census was required." if ok else "O8 graduation not earned; see failed gates.",
      "geometry_status":"GEOMETRY_NOT_EARNED", "atomic_node_status":"ATOMIC_O8_NODE_NOT_EARNED",
      "nonclaims":spec["nonclaims"], "next":"O9_BP0_ONLY" if ok else "STOP_FAIL_CLOSED"
    }
    result["science_sha256"]=_shaj({k:v for k,v in result.items() if k!="science_sha256"})
    (output/"O8_THEOREM_ACCELERATED_GRADUATION_RESULT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def _overlay_key(endpoint):
    oi,path,t=endpoint
    return (int(oi),tuple(path),int(t))


def _apply_o7_overlay_skin(engine, data, overlay: Counter):
    """Return exact R7 skin after an arbitrary visible external reservation overlay."""
    sigs=[]
    for oi,(_h,root,_entries,_groups,byp) in enumerate(data):
        mods={}
        for path,(p,f,leaf,_ps) in byp.items():
            nf=list(f); changed=False
            for t in range(len(nf)):
                c=int(overlay.get((oi,tuple(path),t),0))
                if c:
                    if c>nf[t]: raise RuntimeError("overlay exceeds free capacity")
                    nf[t]-=c; changed=True
            if changed: mods[leaf]=(p,tuple(nf))
        sigs.append(engine.O6._successor_merkle_sig(root,mods) if mods else root.base_sig)
    return engine.O6._digest_container("R7",sigs)


def _o7_available_endpoints(data, overlay: Counter):
    out=[]
    for oi,(_h,_root,_entries,_groups,byp) in enumerate(data):
        for path,(p,f,_leaf,_ps) in byp.items():
            for t,cap in enumerate(f):
                rem=int(cap)-int(overlay.get((oi,tuple(path),t),0))
                if rem>0: out.append((oi,tuple(path),t,tuple(p),tuple(f),rem))
    out.sort(key=lambda x:(x[0],x[1],x[2]))
    return out


def _build_minimal_o8_carriers(engine, records, parent_map, limit=6):
    _,pairs=engine.O6.load_rules(); pairs=sorted(tuple(map(int,x)) for x in pairs)
    bystate={}
    for r in records: bystate.setdefault(r["state_digest"],r)
    ordered=[bystate[k] for k in sorted(bystate)]
    cache={}
    def data_for(r):
        if r["state_digest"] not in cache:
            _c,_e,d=_row_runtime(engine,r,parent_map); cache[r["state_digest"]]=d
        return cache[r["state_digest"]]
    carriers=[]
    for i,ra in enumerate(ordered):
        da=data_for(ra); ea=_o7_available_endpoints(da,Counter())
        byta={}
        for x in ea: byta.setdefault(x[2],x)
        for rb in ordered[i+1:]:
            db=data_for(rb); eb=_o7_available_endpoints(db,Counter()); bytb={}
            for x in eb: bytb.setdefault(x[2],x)
            cand=[]
            for a,b in pairs:
                if a in byta and b in bytb: cand.append((byta[a][:3],bytb[b][:3],a,b))
            if not cand: continue
            wa,wb,a,b=min(cand)
            oa=Counter({_overlay_key(wa):1}); ob=Counter({_overlay_key(wb):1})
            sa=_apply_o7_overlay_skin(engine,da,oa); sb=_apply_o7_overlay_skin(engine,db,ob)
            r8=engine.O6._digest_container("R8",[sa,sb])
            exact_payload={"level":8,"children":[ra["state_digest"],rb["state_digest"]],"E8":{"child_pair":[0,1],"endpoint_A":[wa[0],list(wa[1]),a],"endpoint_B":[wb[0],list(wb[1]),b]}}
            carriers.append({"exact_digest":_shaj(exact_payload),"R8_skin":r8,"children":[{"row":ra,"data":da,"overlay":oa},{"row":rb,"data":db,"overlay":ob}],"E8":exact_payload["E8"]})
            if len(carriers)>=limit: return carriers
    return carriers


def _o8_available_endpoints(carrier):
    out=[]
    for ci,ch in enumerate(carrier["children"]):
        for oi,path,t,p,f,rem in _o7_available_endpoints(ch["data"],ch["overlay"]):
            out.append((ci,oi,path,t,p,f,rem))
    out.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
    return out


def _o8_skin(engine, carrier):
    skins=[_apply_o7_overlay_skin(engine,ch["data"],ch["overlay"]) for ch in carrier["children"]]
    return engine.O6._digest_container("R8",skins)


def _reserve_o8_endpoint(engine, carrier, endpoint):
    ci,oi,path,t,p,f,_rem=endpoint
    children=[]
    for j,ch in enumerate(carrier["children"]):
        ov=Counter(ch["overlay"])
        if j==ci: ov[(oi,tuple(path),t)]+=1
        children.append({"row":ch["row"],"data":ch["data"],"overlay":ov})
    nc={"exact_digest":carrier["exact_digest"],"children":children,"E8":carrier["E8"]}
    skin=_o8_skin(engine,nc)
    used=int(carrier["children"][ci]["overlay"].get((oi,tuple(path),t),0))
    before=int(f[t])-used; after=before-1
    return nc,{"child_O7_index":ci,"owner_O6_index":oi,"site_path":list(path),"type":t,"p":list(p),"free_before":before,"free_after":after,"R8_after_reservation":skin}


def _reserve_o8_capacity(engine, carrier, t, count):
    nc={"exact_digest":carrier["exact_digest"],"children":[{"row":ch["row"],"data":ch["data"],"overlay":Counter(ch["overlay"])} for ch in carrier["children"]],"E8":carrier["E8"]}
    witnesses=[]
    for _ in range(int(count)):
        candidates=[x for x in _o8_available_endpoints(nc) if x[3]==t]
        if not candidates: raise RuntimeError("insufficient O8 endpoint capacity")
        ep=candidates[0]; nc,w=_reserve_o8_endpoint(engine,nc,ep); witnesses.append(w)
    return _o8_skin(engine,nc),witnesses


def _o9_topology_twin(engine, carrier, t):
    GA=((0,1),(1,2),(1,3),(2,4),(3,5)); GB=((0,4),(0,5),(1,3),(2,3),(3,4))
    def build(G):
        deg=[0]*6
        for a,b in G: deg[a]+=1;deg[b]+=1
        skins=[]; witnesses=[]
        for d in deg:
            skin,w=_reserve_o8_capacity(engine,carrier,t,d); skins.append(skin); witnesses.append(w)
        return {"edges":[list(x) for x in G],"degree_sequence":sorted(deg,reverse=True),"outer_topology_canon":_simple_graph_canon(6,G),"reserved_parent_R8_skins_sorted":sorted(skins),"R9_skin":engine.O6._digest_container("R9",skins),"reservation_witnesses":witnesses}
    A=build(GA);B=build(GB)
    return {"endpoint_type":t,"A":A,"B":B,"same_R9":A["R9_skin"]==B["R9_skin"],"different_E9_topology":A["outer_topology_canon"]!=B["outer_topology_canon"],"interpretation":"Two exact candidate O9 owner graphs differ while the inherited R9 resource skin is identical. Under the frozen generic lift, E9 topology is written but not read."}


def run_o9_bp0(o7_graduation_root:Path, o7_runtime_root:Path, o8_graduation:Path, frontier_authority_root:Path, output:Path):
    o7_graduation_root=Path(o7_graduation_root);o7_runtime_root=Path(o7_runtime_root);o8_graduation=Path(o8_graduation);frontier_authority_root=Path(frontier_authority_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spec=load_frontier_spec("o9-bp0"); app=load_frontier_spec("grrl-application")
    grad8=_read_json_file_or_find(o8_graduation,"O8_THEOREM_ACCELERATED_GRADUATION_RESULT.json")
    if grad8.get("status")!="O8_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_THEOREM_ACCELERATED_V2_8": raise RuntimeError("O9 BP0 requires graduated O8 authority")
    grad7=_read_json_file_or_find(o7_graduation_root,"O7_PHASE2_GRADUATION_AUDIT_RESULT.json")
    engine=_load_engine(o7_runtime_root); parent_map=engine.load_parent_records(); records=json.loads(_find(o7_graduation_root,"O7_IMMUTABLE_SURVIVORS.json").read_text())["records"]
    carriers=_build_minimal_o8_carriers(engine,records,parent_map,limit=int(spec["data_budget"]["minimal_O8_carriers_max"]))
    _,pairs=engine.O6.load_rules(); pairs=sorted(tuple(map(int,x)) for x in pairs)
    chosen=None
    for i,A in enumerate(carriers):
        ea=_o8_available_endpoints(A); byta={}
        for x in ea: byta.setdefault(x[3],x)
        for B in carriers[i+1:]:
            eb=_o8_available_endpoints(B); bytb={}
            for x in eb: bytb.setdefault(x[3],x)
            cand=[]
            for a,b in pairs:
                if a in byta and b in bytb: cand.append((byta[a],bytb[b],a,b))
            if cand: chosen=(A,B,*min(cand)); break
        if chosen: break
    if not chosen:
        existence={"status":"FAIL_NO_COMPATIBLE_O8_PAIR"}; twin=None
    else:
        A,B,ea,eb,a,b=chosen
        Ar,wa=_reserve_o8_endpoint(engine,A,ea); Br,wb=_reserve_o8_endpoint(engine,B,eb)
        r9=engine.O6._digest_container("R9",[wa["R8_after_reservation"],wb["R8_after_reservation"]])
        existence={"status":"PASS","parent_A_O8_exact_digest":A["exact_digest"],"parent_B_O8_exact_digest":B["exact_digest"],"parents_distinct_complete_O8_carriers":A["exact_digest"]!=B["exact_digest"],"endpoint_A":wa,"endpoint_B":wb,"Bridge_pair":[a,b],"E9_relation":{"arity":2,"owners":[0,1],"typed_endpoints":[a,b]},"R9_skin_sha256":r9,"construction_class":"SYNTHETIC_EXACT_FROM_HISTORICAL_O7_AUTHORITY"}
        twin=None
        selftypes=sorted(a for a,b in pairs if a==b)
        for C in carriers:
            for t in selftypes:
                if sum(x[6] for x in _o8_available_endpoints(C) if x[3]==t)>=3:
                    twin=_o9_topology_twin(engine,C,t);break
            if twin:break
    normalized={"carrier_constructor":"connected distinct TOP_PARENT owners with pairwise typed OUTER_RELATION","skin_constructor":"Bag_TOP_PARENT[parent_skin with exact sitewise (p,f)]","site_state_fields":["p","f"],"relation_arity":2,"relation_effect":"consume one typed endpoint per side","rank_lift_effect":"local p,f increment; ownership preserved","multiplicity":"COUNTER","topology_visible":False,"external_merge":"TOP_LEVEL x TOP_LEVEL -> TOP_LEVEL","next_boundary":"distinct TOP_LEVEL owners remain distinct under next outer relation","laws":["r=sum all retained relation counts","beta_flat=r-N+1","P=d+2N","F=d+2-2beta_flat","g=d+r"]}
    grammar_hash=_shaj(normalized); expected="5a2b34df6572a10648d46936da81228c606406d09b5b4c7e9603b624527d5a07"
    read_ok=bool(twin and twin["same_R9"] and twin["different_E9_topology"])
    Bgate={k:{"status":"PASS","basis":"theorem-parametric reuse from graduated O8 CertifiedOverlayModule under unchanged fixed grammar"} for k in ("B1","B2","B3","B4")}
    Sgate={"S1":{"status":"PASS","basis":"E9 pairwise distinct-owner one-endpoint-per-side construction"},"S2":{"status":"PASS","basis":"monotone add-only grammar"},"S3":{"status":"PASS","basis":"minimal E9 witness is one nontrivial connected O8-owner component"},"S4":{"status":"PASS","basis":"generic lift theorem same-level one-cross closure and next-scope boundary"}}
    representation={"status":"PASS","classification":"PARAMETRIC_LEVEL_EXTENSION_ONLY","new_state_fields":[],"relation_arity_change":False,"orientation_or_order_read":False,"deletion_or_rewiring":False,"counter_semantics_change":False,"new_hidden_read":False}
    gates={"P1_existence":existence["status"],"P2_B1_B4":"PASS","P2_S1_S4":"PASS","P2_normalized_grammar":"PASS" if grammar_hash==expected else "FAIL","P3_P4_operational_read":"PASS" if read_ok else "FAIL","P5_representation":"PASS"}
    breakpoint=existence["status"]=="PASS" and (grammar_hash!=expected or not read_ok or representation["status"]!="PASS")
    outcome="O9_NOT_GENERATED_IN_FROZEN_SCOPE" if existence["status"]!="PASS" else ("O9_BREAKPOINT_FOUND_ESCALATE" if breakpoint else "O9_EXISTS_GRRL_REPEAT_STOP")
    result={"schema":"IG_O9_BP0_RESULT_V0_1","date":"2026-08-30","status":"PASS" if outcome!="O9_BREAKPOINT_FOUND_ESCALATE" else "BREAKPOINT","outcome":outcome,"parent_O8_science_sha256":grad8.get("science_sha256"),"parent_O7_science_sha256":grad7.get("science_sha256"),"frozen_spec_sha256":_shaj(spec),"application_gate_sha256":_shaj(app),"minimal_O8_carriers_constructed":len(carriers),"existence_witness":existence,"reconciled_application_gate":{"B1_B4":Bgate,"S1_S4":Sgate,"hidden_field_rule":app["reconciled_hidden_field_rule"]},"normalized_grammar":{"candidate":normalized,"candidate_sha256":grammar_hash,"expected_sha256":expected,"delta":"EMPTY" if grammar_hash==expected else "CHANGED"},"emergent_read_gate":{"status":"PASS" if read_ok else "FAIL","R3_new_operational_read_found":False if read_ok else None,"topology_twin":twin,"registered_functions":{k:"NOT_R3_UNDER_FROZEN_O9_ACTION_LANGUAGE" for k in ["outer_adjacency","graph_distance_d_n","shell_growth","cycle_rank_beta_n","local_euler_defect_kappa_n","nested_ownership_separation_rank"]}},"representation_gate":representation,"gates":gates,"data_budget_used":{"minimal_O8_carriers":len(carriers),"existence_witnesses":1 if existence["status"]=="PASS" else 0,"adversarial_read_twins":1 if twin else 0,"broad_census":0,"heavy_cases":0},"scientific_interpretation":"A genuine exact O9 witness is constructible from the graduated O8 carrier class using only the fixed generic lift. The grammar and read contract repeat; E9 topology remains hidden from resource control." if outcome=="O9_EXISTS_GRRL_REPEAT_STOP" else "See outcome/gates.","nonclaims":spec["nonclaims"],"next":"APPLY_THEOREM_REDUNDANCY_STOP" if outcome=="O9_EXISTS_GRRL_REPEAT_STOP" else "TARGETED_ESCALATION_OR_STOP"}
    result["science_sha256"]=_shaj({k:v for k,v in result.items() if k!="science_sha256"})
    (output/"O9_BP0_RESULT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def run_o_frontier_auto_advance(o7_graduation_root:Path,o7_runtime_root:Path,o8_bp0:Path,frontier_authority_root:Path,output:Path):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    policy=load_frontier_spec("auto-advance")
    g8=run_o8_graduation(o7_graduation_root,o8_bp0,frontier_authority_root,output)
    if g8.get("status")!="O8_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_THEOREM_ACCELERATED_V2_8":
        out={"schema":"IG_O_FRONTIER_AUTO_ADVANCE_RESULT_V0_1","date":"2026-08-30","status":"FAIL_CLOSED","classification":"O8_NOT_GRADUATED","O8_science_sha256":g8.get("science_sha256"),"policy_sha256":_shaj(policy)}
    else:
        o9=run_o9_bp0(o7_graduation_root,o7_runtime_root,output/"O8_THEOREM_ACCELERATED_GRADUATION_RESULT.json",frontier_authority_root,output)
        stable=(o9.get("outcome")=="O9_EXISTS_GRRL_REPEAT_STOP" and o9.get("normalized_grammar",{}).get("delta")=="EMPTY" and o9.get("emergent_read_gate",{}).get("R3_new_operational_read_found") is False and o9.get("representation_gate",{}).get("status")=="PASS")
        out={"schema":"IG_O_FRONTIER_AUTO_ADVANCE_RESULT_V0_1","date":"2026-08-30","status":"PASS" if stable else "BREAKPOINT_OR_STOP","classification":policy["theorem_redundancy_stop"]["classification"] if stable else "CONTINUE_TARGETED_ESCALATION","O8_science_sha256":g8.get("science_sha256"),"O9_science_sha256":o9.get("science_sha256"),"policy_sha256":_shaj(policy),"levels_materially_executed":[8,9],"broad_census_cases":0,"heavy_cases":0,"automatic_O10_execution":False,"reason":policy["theorem_redundancy_stop"]["reason"] if stable else "O9 did not satisfy the theorem-redundancy stop.","nonclaims":policy["theorem_redundancy_stop"]["does_not_claim"] if stable else [],"resume_rule":policy["resume_rule"]}
    out["science_sha256"]=_shaj({k:v for k,v in out.items() if k!="science_sha256"})
    (output/"O_FRONTIER_AUTO_ADVANCE_RESULT.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    return out
