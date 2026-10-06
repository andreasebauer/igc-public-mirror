#!/usr/bin/env python3
import argparse, collections, copy, gzip, hashlib, importlib.util, itertools, json, math, os, resource, sys, time, functools
from pathlib import Path
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism as iso

HERE=Path(__file__).resolve().parent
VENDOR=HERE/'vendor'
ROOT=HERE.parent
EXPECTED_O2_GRAD='d138de3c6c906be3ab453c225ecce2e3fff5d3e4f91d220ed681ef5253a86bbb'
EXPECTED_BANK='806a934eb34f67a8a5ec797f088f353ba0d2fa908d056d3214ef29b32cf42ef0'
EXPECTED_SEED='e836f092fd3f4ae3dbbefaacb640156dbecd9b3d4b389a25c650bc30ca9f7061'
HORIZON=64
D2_CADENCE=8
D2_PANEL_CAP=8
LANES=('D','T','C','I','L')
BOOTSTRAP_LANE_CAP=32
DEFAULT_LANE_CAP=16
DEFAULT_ACTION_CAP=32
PORT_N=7

def load_mod(p,n):
    s=importlib.util.spec_from_file_location(n,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v12=load_mod(VENDOR/'scout2_bidegree_v1_2.py','scout3_v12')
sc=v12.sc
PORTS=sc.PORTS
NODE_MATCH=iso.categorical_node_match('color',None)

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_json(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha_repr(x):return hashlib.sha256(repr(x).encode()).hexdigest()
def loadj(p):
    with open(p) as f:return json.load(f)
def writej_atomic(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def writegz_atomic(path,obj):
    path=Path(path);tmp=Path(str(path)+'.tmp')
    with gzip.open(tmp,'wt',compresslevel=6) as f:json.dump(obj,f,sort_keys=True,separators=(',',':'))
    os.replace(tmp,path)

def load_inputs(root=ROOT):
    grad=loadj(root/'inputs/O2_GRADUATION_AUDIT_RESULT.json');bank=loadj(root/'inputs/O2_SOURCE_BANK.json');seed=loadj(root/'inputs/O3_SEED_PANEL.json')
    if grad.get('science_sha256')!=EXPECTED_O2_GRAD or grad.get('status')!='O2_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_V1_9':raise SystemExit('INTEGRITY_FAIL_O2_GRADUATION')
    if bank.get('science_sha256')!=EXPECTED_BANK:raise SystemExit('INTEGRITY_FAIL_O2_SOURCE_BANK')
    if seed.get('science_sha256')!=EXPECTED_SEED:raise SystemExit('INTEGRITY_FAIL_O3_SEED_PANEL')
    qmap={e['q_hash']:[(tuple(s[0]),tuple(s[1])) for s in e['sites']] for e in bank['entries']}
    meta={e['q_hash']:e for e in bank['entries']}
    _,_,templates=v12.load_templates_light(root)
    if len(templates)!=77:raise SystemExit('INTEGRITY_FAIL_TEMPLATE_COUNT')
    return grad,bank,seed,qmap,meta,templates

def expand(h,qmap):return [qmap[q] for q in h['entities']]
def r3_usage(h,qmap):
    ents=expand(h,qmap);U=[[[0]*PORT_N for _ in e] for e in ents]
    for v,i,a,w,j,b in h['edges']:
        U[v][i][a]+=1;U[w][j][b]+=1
    return U

def free_counts(entities,U):
    F=[]
    for v,e in enumerate(entities):
        ff=[]
        for i,(p,u2) in enumerate(e):ff.append(tuple(p[a]-u2[a]-U[v][i][a] for a in range(PORT_N)))
        F.append(ff)
    return F

def action_iter(h,qmap,entities_override=None):
    ents=entities_override if entities_override is not None else expand(h,qmap); U=r3_usage(h,qmap) if entities_override is None else r3_usage_from_edges(h,ents); F=free_counts(ents,U)
    for v in range(len(ents)):
        for w in range(v+1,len(ents)):
            for i in range(len(ents[v])):
                for j in range(len(ents[w])):
                    for a in range(PORT_N):
                        if F[v][i][a]<=0:continue
                        for b in range(PORT_N):
                            if F[w][j][b]<=0 or not sc.bridge(PORTS[a],PORTS[b]):continue
                            yield (v,i,a,w,j,b)

def r3_usage_from_edges(h,ents):
    U=[[[0]*PORT_N for _ in e] for e in ents]
    for v,i,a,w,j,b in h['edges']:
        if v>=len(ents) or w>=len(ents) or i>=len(ents[v]) or j>=len(ents[w]):raise ValueError('bad edge endpoint')
        U[v][i][a]+=1;U[w][j][b]+=1
    return U

def _entity_component_labels(h):
    n=len(h['entities']);par=list(range(n))
    def find(x):
        while par[x]!=x:par[x]=par[par[x]];x=par[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:par[b]=a
    for v,i,a,w,j,b in h['edges']:union(v,w)
    return [find(i) for i in range(n)]

def select_actions(h,actions,qmap,meta,cap):
    # Scout-level bounded deterministic action panel. All legal actions are enumerated;
    # only this pre-outcome subset is materialized into exact children.
    if len(actions)<=cap:return actions
    ents=expand(h,qmap);U=r3_usage(h,qmap);F=free_counts(ents,U);comp=_entity_component_labels(h);pair=collections.Counter((v,w) for v,i,a,w,j,b in h['edges'])
    acts=sorted(actions)
    def rt(x):return tuple(sorted((x[2],x[5])))
    required=[];seen=set()
    # Every available typed relation gets a representative.
    for a in acts:
        k=rt(a)
        if k not in seen:required.append(a);seen.add(k)
    # Structural stress controls, deterministic lexicographic first hits.
    cats=[lambda x:comp[x[0]]!=comp[x[3]], lambda x:comp[x[0]]==comp[x[3]], lambda x:pair[(x[0],x[3])]>0, lambda x:h['entities'][x[0]]==h['entities'][x[3]]]
    for pred in cats:
        required.extend([a for a in acts if pred(a)][:8])
    required=sorted(set(required))
    if len(required)>=cap:return required[:cap]
    def feat(x):
        v,i,a,w,j,b=x;fv=F[v][i];fw=F[w][j];mv=meta[h['entities'][v]];mw=meta[h['entities'][w]]
        return [a,b,fv[a],fw[b],sum(fv),sum(fw),sum(sum(z) for z in F[v]),sum(sum(z) for z in F[w]),int(comp[v]==comp[w]),pair[(v,w)],int(h['entities'][v]==h['entities'][w]),mv['n_sites'],mw['n_sites'],mv['beta_o2'],mw['beta_o2']]
    X=np.asarray([feat(a) for a in acts],float);lo=X.min(0);hi=X.max(0);den=hi-lo;den[den==0]=1;X=(X-lo)/den;pos={a:i for i,a in enumerate(acts)}
    sel=[pos[a] for a in required];chosen=np.zeros(len(acts),bool);chosen[sel]=1;mind=np.full(len(acts),np.inf)
    for i in sel:mind=np.minimum(mind,((X-X[i])**2).sum(1))
    while len(sel)<cap:
        z=mind.copy();z[chosen]=-1;mx=z.max();cand=np.where(np.isclose(z,mx,rtol=0,atol=1e-15))[0];i=min(cand,key=lambda j:acts[j]);sel.append(int(i));chosen[i]=1;mind=np.minimum(mind,((X-X[i])**2).sum(1))
    return sorted(acts[i] for i in sel)

def add_edge(h,e):
    z={'entities':list(h['entities']),'edges':[list(x) for x in h['edges']]+[list(e)],'r3':int(h['r3'])+1}
    z['edges']=sorted(z['edges'])
    return z

def build_graph(h,qmap):
    G=nx.Graph();nid=0; ent_nodes=[]; site_nodes=[]
    ents=expand(h,qmap)
    for v,e in enumerate(ents):
        en=nid;nid+=1;G.add_node(en,color='E');ent_nodes.append(en);sn=[]
        for p,u2 in e:
            s=nid;nid+=1;G.add_node(s,color='S:'+sha_repr((p,u2)));G.add_edge(en,s);sn.append(s)
        site_nodes.append(sn)
    for rel in h['edges']:
        v,i,a,w,j,b=rel;rn=nid;nid+=1;G.add_node(rn,color='R')
        pa=nid;nid+=1;pb=nid;nid+=1;G.add_node(pa,color=f'P:{a}');G.add_node(pb,color=f'P:{b}')
        G.add_edge(site_nodes[v][i],pa);G.add_edge(pa,rn);G.add_edge(rn,pb);G.add_edge(pb,site_nodes[w][j])
    return G

def coarse_invariant(h,qmap):
    ents=expand(h,qmap);U=r3_usage(h,qmap); qtypes=sorted(h['entities']);pair=collections.Counter();rt=collections.Counter();deg=[0]*len(ents)
    for v,i,a,w,j,b in h['edges']:
        pair[(min(v,w),max(v,w))]+=1;rt[tuple(sorted((a,b)))]+=1;deg[v]+=1;deg[w]+=1
    # entity-local invariant independent of entity labels
    ecol=[]
    for v,e in enumerate(ents):
        sites=[]
        for i,(p,u2) in enumerate(e):sites.append((p,u2,tuple(U[v][i])))
        ecol.append(tuple(sorted(sites)))
    return (len(ents),sum(len(e) for e in ents),len(h['edges']),tuple(sorted(ecol)),tuple(sorted(pair.values())),tuple(sorted(rt.items())),tuple(sorted(deg)))
def refinement_signature(h,qmap,iterations=4):
    ents=expand(h,qmap);U=r3_usage(h,qmap)
    base=[[sha_repr((p,u2,tuple(U[v][i]))) for i,(p,u2) in enumerate(e)] for v,e in enumerate(ents)]
    inc=[[[] for _ in e] for e in ents]
    for v,i,a,w,j,b in h['edges']:
        inc[v][i].append((a,b,w,j));inc[w][j].append((b,a,v,i))
    site=[list(x) for x in base]
    ent=[sha_repr(tuple(sorted(site[v]))) for v in range(len(ents))]
    for _ in range(iterations):
        ns=[]
        for v,e in enumerate(ents):
            row=[]
            for i in range(len(e)):
                rel=tuple(sorted((a,b,ent[w],site[w][j]) for a,b,w,j in inc[v][i]))
                row.append(sha_repr((base[v][i],ent[v],rel)))
            ns.append(row)
        site=ns;ent=[sha_repr(tuple(sorted(site[v]))) for v in range(len(ents))]
    rels=[]
    for v,i,a,w,j,b in h['edges']:
        x=(ent[v],site[v][i],a);y=(ent[w],site[w][j],b);rels.append(tuple(sorted((x,y))))
    return sha_repr((tuple(sorted(ent)),tuple(sorted(rels))))

def graph_fingerprint(h,qmap):
    G=build_graph(h,qmap);wl=nx.weisfeiler_lehman_graph_hash(G,node_attr='color',iterations=5,digest_size=20)
    return coarse_invariant(h,qmap),wl,G

@functools.lru_cache(maxsize=None)
def _site_maps_cached(qhash, sites_tuple):
    sites=list(sites_tuple); groups=collections.defaultdict(list)
    for i,c in enumerate(sites):groups[c].append(i)
    group_opts=[]
    for inds in [groups[k] for k in sorted(groups,key=repr)]:
        if len(inds)<=1:group_opts.append([tuple(inds)])
        else:group_opts.append(list(itertools.permutations(inds)))
    out=[]
    for choice in itertools.product(*group_opts):
        mp=list(range(len(sites)))
        for inds,perm in zip([groups[k] for k in sorted(groups,key=repr)],choice):
            # old indices are assigned to fixed canonical positions inds via perm ordering
            for newpos,oldidx in zip(inds,perm):mp[oldidx]=newpos
        out.append(tuple(mp))
    return tuple(out)

def _site_maps(qhash,qmap):
    return _site_maps_cached(qhash,tuple(qmap[qhash]))

def _entity_maps(entities):
    groups=collections.defaultdict(list)
    for i,q in enumerate(entities):groups[q].append(i)
    ordered=[groups[k] for k in sorted(groups)]
    opts=[list(itertools.permutations(g)) if len(g)>1 else [tuple(g)] for g in ordered]
    for choice in itertools.product(*opts):
        mp=list(range(len(entities)))
        for inds,perm in zip(ordered,choice):
            for newpos,oldidx in zip(inds,perm):mp[oldidx]=newpos
        yield tuple(mp)

def base_automorphism_search_space(h,qmap):
    ec=collections.Counter(h['entities']);space=math.prod(math.factorial(x) for x in ec.values())
    for q in h['entities']:space*=len(_site_maps(q,qmap))
    return space

def exact_canonical_base_key(h,qmap,threshold=8192):
    # Exhaust the full automorphism group of the immutable O2 base forest.
    # This is an exact canonical label whenever the group is small enough.
    if base_automorphism_search_space(h,qmap)>threshold:return None
    smopts=[_site_maps(q,qmap) for q in h['entities']]
    best=None
    for em in _entity_maps(tuple(h['entities'])):
        for sms in itertools.product(*smopts):
            zz=[]
            for v,i,a,w,j,b in h['edges']:
                nv,nw=em[v],em[w];ni=sms[v][i];nj=sms[w][j]
                if nv<nw:zz.append((nv,ni,a,nw,nj,b))
                else:zz.append((nw,nj,b,nv,ni,a))
            key=(tuple(h['entities']),tuple(sorted(zz)))
            if best is None or key<best:best=key
    return best

def dedup_children_r1(cands,qmap):
    # Exact closed form for one O3 edge on an r3=0 seed forest.
    # With no prior O3 incidence, sites sharing the same Q-local (p,u2) color are automorphic.
    reps={};lin={}
    for pk,act,ch in cands:
        v,i,a,w,j,b=act; qv=ch['entities'][v];qw=ch['entities'][w];cv=qmap[qv][i];cw=qmap[qw][j]
        eps=tuple(sorted(((qv,cv,a),(qw,cw,b)),key=repr)); rem=list(ch['entities']);rem.pop(max(v,w));rem.pop(min(v,w));desc=(tuple(sorted(rem)),eps)
        if desc not in reps:
            reps[desc]=ch;lin[desc]={'parents':set(),'actions':set()}
        lin[desc]['parents'].add(pk);lin[desc]['actions'].add(tuple(act))
    out=[]
    for desc in sorted(reps,key=repr):
        h=reps[desc];h['exact_key']='H001_'+sha_repr(desc)[:24];h['parent_keys']=sorted(lin[desc]['parents']);h['parent_count']=len(lin[desc]['parents']);h['generation_action_count']=len(lin[desc]['actions']);out.append(h)
    return out

def dedup_children(cands,qmap,r):
    if r==1:
        return dedup_children_r1(cands,qmap)
    # Most carriers have a small immutable-base automorphism group and receive an exact
    # canonical label directly. Highly symmetric cases fall back to invariant buckets +
    # full colored-graph isomorphism, so no equality is approximated.
    direct={};direct_lin={};sym_buckets=collections.defaultdict(list);sym_reps=[];sym_lin=[];sym_graph=[];sym_slots=[]
    for pk,act,ch in cands:
        ck=exact_canonical_base_key(ch,qmap)
        if ck is not None:
            if ck not in direct:
                direct[ck]=ch;direct_lin[ck]={'parents':set(),'actions':set()}
            direct_lin[ck]['parents'].add(pk);direct_lin[ck]['actions'].add(tuple(act));continue
        inv=coarse_invariant(ch,qmap);ci=sha_repr(inv);ri=refinement_signature(ch,qmap);bk=(ci,ri);found=None
        if sym_buckets[bk]:
            G=build_graph(ch,qmap)
            for ridx in sym_buckets[bk]:
                if sym_graph[ridx] is None:sym_graph[ridx]=build_graph(sym_reps[ridx],qmap)
                if nx.is_isomorphic(G,sym_graph[ridx],node_match=NODE_MATCH):found=ridx;break
        if found is None:
            ridx=len(sym_reps);slot=len(sym_buckets[bk]);sym_buckets[bk].append(ridx);sym_reps.append(ch);sym_graph.append(None);sym_slots.append((ci,ri,slot));sym_lin.append({'parents':{pk},'actions':{tuple(act)}})
        else:
            sym_lin[found]['parents'].add(pk);sym_lin[found]['actions'].add(tuple(act))
    out=[]
    for ck in sorted(direct,key=repr):
        h=direct[ck];h['exact_key']=f'H{r:03d}_'+sha_repr(('C',ck))[:24];h['parent_keys']=sorted(direct_lin[ck]['parents']);h['parent_count']=len(direct_lin[ck]['parents']);h['generation_action_count']=len(direct_lin[ck]['actions']);out.append(h)
    for i,h in enumerate(sym_reps):
        h['exact_key']=f'H{r:03d}_'+sha_repr(('I',sym_slots[i]))[:24];h['parent_keys']=sorted(sym_lin[i]['parents']);h['parent_count']=len(sym_lin[i]['parents']);h['generation_action_count']=len(sym_lin[i]['actions']);out.append(h)
    return sorted(out,key=lambda h:h['exact_key'])

def entity_components(h):
    n=len(h['entities']);par=list(range(n))
    def find(x):
        while par[x]!=x:par[x]=par[par[x]];x=par[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:par[b]=a
    for v,i,a,w,j,b in h['edges']:union(v,w)
    return len({find(i) for i in range(n)})
def metrics(h,qmap):
    ents=expand(h,qmap);U=r3_usage(h,qmap);F=free_counts(ents,U);n=len(ents);m=len(h['edges']);comp=entity_components(h);beta=m-n+comp
    deg=[0]*n;pair=collections.Counter();usedrt=set();r3type=[0]*7
    for v,i,a,w,j,b in h['edges']:
        deg[v]+=1;deg[w]+=1;pair[(v,w)]+=1;usedrt.add(tuple(sorted((a,b))));r3type[a]+=1;r3type[b]+=1
    # exact opportunity count without materializing every child action
    siteavail=[[0]*7 for _ in ents]
    for v,e in enumerate(ents):
        for i in range(len(e)):
            for a in range(7):
                if F[v][i][a]>0: siteavail[v][a]+=1
    availrt=set(); possible=0
    for v in range(n):
        for w in range(v+1,n):
            for a in range(7):
                if siteavail[v][a]==0:continue
                for b in range(7):
                    if siteavail[w][b]==0 or not sc.bridge(PORTS[a],PORTS[b]):continue
                    availrt.add(tuple(sorted((a,b)))); possible += siteavail[v][a]*siteavail[w][b]
    free7=[0]*7
    sitefree=[]
    for v,e in enumerate(ents):
        for i in range(len(e)):
            sitefree.append(sum(F[v][i]));
            for a in range(7):free7[a]+=F[v][i][a]
    return {
      'n_entities':n,'n_o1_sites':sum(len(e) for e in ents),'distinct_q_types':len(set(h['entities'])),'m3':m,'components':comp,'beta3':beta,'connected':comp==1,
      'max_degree':max(deg) if deg else 0,'degree_sorted':sorted(deg),'parallel_max':max(pair.values()) if pair else 0,'parallel':any(x>1 for x in pair.values()),
      'active_entities':sum(d>0 for d in deg),'used_relation_types':sorted([list(x) for x in usedrt]),'available_relation_types':sorted([list(x) for x in availrt]),
      'available_relation_type_count':len(availrt),'possible_relation_additions':possible,'total_free':sum(free7),'free7':free7,'min_site_free':min(sitefree) if sitefree else 0,
      'r3_usage7':r3type,'interface_modulation':any(sum(x)>0 for ev in U for x in ev),'selective':0<len(availrt)<18,
      'parent_count':int(h.get('parent_count',0)),'generation_action_count':int(h.get('generation_action_count',0)),
    }

def site_state_colors(h,qmap,entities_override=None):
    ents=entities_override if entities_override is not None else expand(h,qmap);U=r3_usage(h,qmap) if entities_override is None else r3_usage_from_edges(h,ents)
    return [[(tuple(p),tuple(u2),tuple(U[v][i])) for i,(p,u2) in enumerate(e)] for v,e in enumerate(ents)]
def h1_support(h,qmap,entities_override=None):
    states=site_state_colors(h,qmap,entities_override); ents=entities_override if entities_override is not None else expand(h,qmap)
    ecolors=[sha_repr(tuple(sorted(ss))) for ss in states];acts=set()
    for v,i,a,w,j,b in action_iter(h,qmap,entities_override):
        sa=sha_repr(states[v][i]);sb=sha_repr(states[w][j]);x=(ecolors[v],sa,a);y=(ecolors[w],sb,b);acts.add(tuple(sorted((x,y))))
    return sha_repr(tuple(sorted(acts))),len(acts)
def resource_signature(h,qmap):
    states=site_state_colors(h,qmap);ec=tuple(sorted(tuple(sorted(s)) for s in states));return sha_repr(ec)

def lane_feature(m):
    ds=m['degree_sorted'];degmean=sum(ds)/len(ds) if ds else 0;degvar=sum((x-degmean)**2 for x in ds)/len(ds) if ds else 0
    return [m['n_entities'],m['n_o1_sites'],m['distinct_q_types'],m['components'],m['beta3'],m['total_free'],m['possible_relation_additions'],m['available_relation_type_count'],m['max_degree'],degvar,m['parallel_max'],m['active_entities']]+m['free7']+m['r3_usage7']
def farthest_indices(rows,cap):
    if len(rows)<=cap:return list(range(len(rows)))
    X=np.asarray([lane_feature(r['metrics']) for r in rows],float);lo=X.min(0);hi=X.max(0);den=hi-lo;den[den==0]=1;X=(X-lo)/den
    # exact_key order fixes tie-breaking
    order=sorted(range(len(rows)),key=lambda i:rows[i]['exact_key']); first=order[0];sel=[first];chosen=np.zeros(len(rows),bool);chosen[first]=1;mind=((X-X[first])**2).sum(1)
    while len(sel)<cap:
        z=mind.copy();z[chosen]=-1;mx=z.max();cands=np.where(np.isclose(z,mx,rtol=0,atol=1e-15))[0];i=min(cands,key=lambda j:rows[j]['exact_key']);sel.append(int(i));chosen[i]=1;mind=np.minimum(mind,((X-X[i])**2).sum(1))
    return sorted(sel,key=lambda i:rows[i]['exact_key'])
def select_lanes(reps,qmap,cap):
    # Two-stage exact Scout selection: compute only cheap, current-state pre-outcome metrics
    # on the full exact pool; expensive H1/resource observers are computed only after lane union.
    rows=[]
    for h in reps:
        h['metrics']=metrics(h,qmap); rows.append(h)
    bykey={h['exact_key']:h for h in rows}; lanes={}
    lanes['D']=[rows[i]['exact_key'] for i in farthest_indices(rows,min(cap,len(rows)))]
    lanes['T']=[h['exact_key'] for h in sorted(rows,key=lambda x:(x['metrics']['total_free'],x['metrics']['min_site_free'],x['metrics']['possible_relation_additions'],x['exact_key']))[:cap]]
    lanes['C']=[h['exact_key'] for h in sorted(rows,key=lambda x:(-x['metrics']['beta3'],-int(x['metrics']['connected']),-x['metrics']['parallel_max'],x['exact_key']))[:cap]]
    lanes['I']=[h['exact_key'] for h in sorted(rows,key=lambda x:(-x['metrics']['available_relation_type_count'],-x['metrics']['possible_relation_additions'],x['exact_key']))[:cap]]
    lanes['L']=[h['exact_key'] for h in sorted(rows,key=lambda x:(-x['parent_count'],-x['generation_action_count'],x['exact_key']))[:cap]]
    tags=collections.defaultdict(list)
    for l,ks in lanes.items():
        for k in ks:tags[k].append(l)
    selected=[]
    for k in sorted(tags):
        h=bykey[k];h['lane_tags']=sorted(tags[k]);h1,nact=h1_support(h,qmap);h['h1']=h1;h['h1_support_count']=nact;h['resource_signature']=resource_signature(h,qmap);selected.append(h)
    return selected,lanes

def grammar_observer(selected,qmap,prev=None):
    main=set();rs=set();re=set();markers=set();h1g=collections.defaultdict(list)
    for h in selected:
        m=h['metrics']
        for a,b in m['used_relation_types']:main.add(f'RU:{a}-{b}')
        for a,b in m['available_relation_types']:main.add(f'RA:{a}-{b}')
        if m['connected']:markers.add('CONNECTED')
        if m['beta3']>0:markers.add('CYCLE')
        if m['parallel']:markers.add('PARALLEL')
        if h['parent_count']>1:markers.add('MULTIPARENT')
        if m['selective']:markers.add('SELECTIVE')
        if m['interface_modulation']:markers.add('INTERFACE_MODULATION')
        markers.add('HOMOGENEOUS_Q' if m['distinct_q_types']==1 else 'HETEROGENEOUS_Q')
        states=site_state_colors(h,qmap);ents=expand(h,qmap);U=r3_usage(h,qmap);F=free_counts(ents,U)
        for v,e in enumerate(ents):
            for i,(p,u2) in enumerate(e):
                pm=sum((1<<a) for a in range(7) if p[a]>0);um=sum((1<<a) for a in range(7) if u2[a]>0);rm=sum((1<<a) for a in range(7) if U[v][i][a]>0);fm=sum((1<<a) for a in range(7) if F[v][i][a]>0)
                rs.add(f'RS:{pm}:{um}:{rm}:{fm}');re.add(sha_repr((p,u2,tuple(U[v][i]))))
        h1g[h['h1']].append(h)
    prevmain=set(prev.get('cumulative_main_atoms',[])) if prev else set();prevrs=set(prev.get('cumulative_resource_support_atoms',[])) if prev else set();prevre=set(prev.get('cumulative_resource_exact_atoms',[])) if prev else set()
    summ={}
    for sig,arr in h1g.items():
        lanes=sorted({x for h in arr for x in h['lane_tags']});summ[sig]={'exact_count':len(arr),'lane_types':lanes,'cyclic_any':any(h['metrics']['beta3']>0 for h in arr),'selective_any':any(h['metrics']['selective'] for h in arr),'multiparent_any':any(h['parent_count']>1 for h in arr),'exact_keys':[h['exact_key'] for h in arr]}
    return {
      'main_atoms':sorted(main),'new_main_atoms':sorted(main-prevmain),'cumulative_main_atoms':sorted(prevmain|main),
      'resource_support_atoms':sorted(rs),'new_resource_support_atoms':sorted(rs-prevrs),'cumulative_resource_support_atoms':sorted(prevrs|rs),
      'resource_exact_count':len(re),'new_resource_exact_count':len(re-prevre),'cumulative_resource_exact_atoms':sorted(prevre|re),
      'marker_repertoire':sorted(markers),'h1_class_count':len(h1g),'h1_compression':len(selected)/len(h1g) if h1g else 1.0,'h1_classes':summ,
      'connected_count':sum(h['metrics']['connected'] for h in selected),'cyclic_count':sum(h['metrics']['beta3']>0 for h in selected),'possible_relation_additions_total':sum(h['metrics']['possible_relation_additions'] for h in selected),
    }

def d2_control(selected,qmap,templates,prev_d2=None):
    # Deterministic diversity panel; exhaust all local O2 actions exactly.
    # The sidecar records whether each action changes the local external availability mask
    # analytically, avoiding an unnecessary full O3 action re-enumeration per d2 child.
    idx=farthest_indices(selected,min(D2_PANEL_CAP,len(selected)));panel=[selected[i] for i in idx]
    atoms=set();rank_actions=0;int_actions=0;capacity_fail=0;support_changed=0;support_unchanged=0
    def mask(f):return sum((1<<a) for a,x in enumerate(f) if x>0)
    for h in panel:
        ents=expand(h,qmap);U3=r3_usage(h,qmap)
        for v,e in enumerate(ents):
            for i,(p,u2) in enumerate(e):
                total=tuple(u2[a]+U3[v][i][a] for a in range(7));before=[p[a]-total[a] for a in range(7)];bm=mask(before)
                if any(x<0 for x in before):capacity_fail+=1
                for ti,tm in enumerate(templates):
                    src=tm['source']
                    if p[src]<=0:continue
                    np=tuple(p[a]+tm['dp'][a] for a in range(7))
                    if min(np)<0:continue
                    strict=before[src]>0;cross=(not strict and all(np[a]>=total[a] for a in range(7)))
                    if not strict and not cross:continue
                    rank_actions+=1;atoms.add(f'LIFT:{ti}:{"S" if strict else "X"}')
                    am=mask([np[a]-total[a] for a in range(7)])
                    if am!=bm:support_changed+=1
                    else:support_unchanged+=1
            # Exact grouped count of all internal O2 relation-add actions.
            # Legality depends only on whether a port type is free; support-mask change
            # depends only on whether the consumed copy was the last free copy.
            # Grouping sites by (availability-mask, singleton-mask) is therefore exact
            # and avoids the quadratic site-pair loop on large distributed Q carriers.
            gcount=collections.Counter()
            for i,(p0,u0) in enumerate(e):
                total=[u0[a]+U3[v][i][a] for a in range(7)]
                ff=[p0[a]-total[a] for a in range(7)]
                av=mask(ff); one=sum((1<<a) for a,x in enumerate(ff) if x==1)
                gcount[(av,one)]+=1
            gkeys=sorted(gcount)
            for gi,g1 in enumerate(gkeys):
                av1,one1=g1;n1=gcount[g1]
                for gj in range(gi,len(gkeys)):
                    g2=gkeys[gj];av2,one2=g2;n2=gcount[g2]
                    pairs=n1*n2 if gi!=gj else n1*(n1-1)//2
                    if pairs<=0:continue
                    per=0;chg=0
                    for a in range(7):
                        if not (av1>>a)&1:continue
                        for b in range(7):
                            if not (av2>>b)&1 or not sc.bridge(PORTS[a],PORTS[b]):continue
                            per+=1;atoms.add(f'INTREL:{min(a,b)}-{max(a,b)}')
                            if ((one1>>a)&1) or ((one2>>b)&1):chg+=1
                    int_actions += pairs*per
                    support_changed += pairs*chg
                    support_unchanged += pairs*(per-chg)
    prev=set(prev_d2.get('cumulative_d2_atoms',[])) if prev_d2 else set()
    return {'panel_size':len(panel),'rank_lift_actions':rank_actions,'internal_relation_add_actions':int_actions,'capacity_failures':capacity_fail,'external_support_changed_actions':support_changed,'external_support_unchanged_actions':support_unchanged,'d2_atoms':sorted(atoms),'new_d2_atoms':sorted(atoms-prev),'cumulative_d2_atoms':sorted(prev|atoms),'pass':capacity_fail==0}

def bootstrap(root):
    grad,bank,seed,qmap,meta,templates=load_inputs(root);reps=[]
    for i,e in enumerate(seed['entries']):
        h={'entities':sorted(e['entity_q_hashes']),'edges':[],'r3':0,'exact_key':'H000_'+sha_repr(tuple(sorted(e['entity_q_hashes'])))[:24],'parent_keys':[],'parent_count':0,'generation_action_count':0}
        reps.append(h)
    # zero-edge seed was already exact-deduped by multiset in source input.
    selected,lanes=select_lanes(reps,qmap,BOOTSTRAP_LANE_CAP)
    obs=grammar_observer(selected,qmap,None);d2=d2_control(selected,qmap,templates,None)
    out={'r3':0,'selected':selected,'lane_cap':BOOTSTRAP_LANE_CAP,'lane_members':lanes,'seed_science_sha256':seed['science_sha256']}
    writegz_atomic(root/'checkpoints/r000_selected_carriers.json.gz',out);writej_atomic(root/'checkpoints/r000_observer.json',obs);writej_atomic(root/'checkpoints/r000_d2control.json',d2)
    met={'r3':0,'pool_exact':len(reps),'selected_exact':len(selected),'lane_cap':BOOTSTRAP_LANE_CAP,'maxrss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':0.0,'possible_relation_additions_total':obs['possible_relation_additions_total']};writej_atomic(root/'checkpoints/r000_metrics.json',met)
    writej_atomic(root/'checkpoints/r000_evaluation.json',{'r3':0,'status':'NOT_ELIGIBLE','reason':'need 32 new r3 checkpoints'})
    return met

def load_panel(root,r):
    with gzip.open(root/'checkpoints'/f'r{r:03d}_selected_carriers.json.gz','rt') as f:return json.load(f)['selected']
def load_obs(root,r):return loadj(root/'checkpoints'/f'r{r:03d}_observer.json')
def latest_d2_before(root,r):
    for x in range(r,-1,-1):
        p=root/'checkpoints'/f'r{x:03d}_d2control.json'
        if p.exists():return loadj(p)
    return None

def evaluate(root,r):
    if r<32:return {'r3':r,'status':'NOT_ELIGIBLE','reason':'need 32 new r3 checkpoints'}
    obs=[load_obs(root,x) for x in range(r-31,r+1)]; final16=obs[-16:]
    # recurrence candidate H1 across all 32 checkpoints
    common=set(obs[0]['h1_classes'])
    for o in obs[1:]:common &= set(o['h1_classes'])
    quals=[]
    # load panels for direct parent H1 matching only if common exists
    if common:
        panels={x:load_panel(root,x) for x in range(r-31,r+1)};h1maps={x:{h['exact_key']:h['h1'] for h in panels[x]} for x in panels}
        for sig in sorted(common):
            lane_types=set();direct=0;multi_cp=0;lineage_cp=0;cyc=False;sel=False;multi_exact=False
            for x in range(r-31,r+1):
                cc=obs[x-(r-31)]['h1_classes'][sig];lane_types.update(cc['lane_types']);cyc|=cc['cyclic_any'];sel|=cc['selective_any'];multi_exact|=cc['exact_count']>=2;multi_cp+=int(cc['multiparent_any']);lineage_cp+=int(cc['multiparent_any'])
                if x>r-31:
                    for h in panels[x]:
                        if h['h1']!=sig:continue
                        if any(h1maps[x-1].get(pk)==sig for pk in h.get('parent_keys',[])):direct+=1;break
            if len(lane_types)>=2 and direct>=24 and (multi_cp>=8 or lineage_cp>=8) and cyc and sel and multi_exact:
                quals.append({'h1':sig,'lane_types':sorted(lane_types),'direct_transition_checkpoints':direct,'multiparent_checkpoints':multi_cp})
    # d2 controls inside window
    d2s=[]
    for x in range(r-31,r+1):
        p=root/'checkpoints'/f'r{x:03d}_d2control.json'
        if p.exists():d2s.append(loadj(p))
    crit={
      'A_no_new_main_final16':all(len(o['new_main_atoms'])==0 for o in final16),
      'B_markers_stable32':all(o['marker_repertoire']==obs[0]['marker_repertoire'] for o in obs),
      'D_to_G_recurrent_entity':bool(quals),
      'H_interface_modulation_16':sum('INTERFACE_MODULATION' in o['marker_repertoire'] for o in obs)>=16,
      'I_compression_final16':all(o['h1_compression']>1.0 for o in final16),
      'J_relation_add_live':obs[-1]['possible_relation_additions_total']>0,
      'K_d2_integrity_quiet':bool(d2s) and all(d['pass'] for d in d2s) and len(d2s)>=4 and all(len(d['new_d2_atoms'])==0 for d in d2s[-4:]),
    }
    status='O3_SCOUT_TRIGGER_V1_0' if all(crit.values()) else 'NO_O3_SCOUT_TRIGGER_V1_0'
    return {'r3':r,'status':status,'criteria':crit,'qualifying_h1':quals[:10],'window':[r-31,r]}

def step(root,r,cap,action_cap):
    t0=time.time();grad,bank,seed,qmap,meta,templates=load_inputs(root);prev=load_panel(root,r-1);cands=[];raw=0;inspected=0
    for h in sorted(prev,key=lambda x:x['exact_key']):
        acts=list(action_iter(h,qmap));raw+=len(acts);pick=select_actions(h,acts,qmap,meta,action_cap);inspected+=len(pick)
        for act in pick:cands.append((h['exact_key'],act,add_edge(h,act)))
    if not cands:
        met={'r3':r,'legal_actions_total':0,'inspected_actions':0,'pool_exact':0,'selected_exact':0,'lane_cap':cap,'wall_seconds':time.time()-t0,'maxrss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};writej_atomic(root/'checkpoints'/f'r{r:03d}_metrics.json',met);writej_atomic(root/'checkpoints'/f'r{r:03d}_evaluation.json',{'r3':r,'status':'RELATION_CAPACITY_SATURATION'});return met
    reps=dedup_children(cands,qmap,r);selected,lanes=select_lanes(reps,qmap,cap);prevobs=load_obs(root,r-1);obs=grammar_observer(selected,qmap,prevobs)
    # direct lineage diagnostic
    prevh1={h['exact_key']:h['h1'] for h in prev};direct=0
    for h in selected:
        if any(prevh1.get(pk)==h['h1'] for pk in h.get('parent_keys',[])):direct+=1
    obs['direct_same_h1_selected']=direct
    out={'r3':r,'selected':selected,'lane_cap':cap,'lane_members':lanes};writegz_atomic(root/'checkpoints'/f'r{r:03d}_selected_carriers.json.gz',out);writej_atomic(root/'checkpoints'/f'r{r:03d}_observer.json',obs)
    if r%D2_CADENCE==0:
        d2=d2_control(selected,qmap,templates,latest_d2_before(root,r-1));writej_atomic(root/'checkpoints'/f'r{r:03d}_d2control.json',d2)
    ev=evaluate(root,r);writej_atomic(root/'checkpoints'/f'r{r:03d}_evaluation.json',ev)
    met={'r3':r,'legal_actions_total':raw,'inspected_actions':inspected,'pool_exact':len(reps),'selected_exact':len(selected),'lane_cap':cap,'wall_seconds':time.time()-t0,'maxrss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'possible_relation_additions_total':obs['possible_relation_additions_total'],'new_main_atoms':len(obs['new_main_atoms']),'new_resource_support_atoms':len(obs['new_resource_support_atoms']),'new_resource_exact_count':obs['new_resource_exact_count'],'h1_classes':obs['h1_class_count'],'h1_compression':obs['h1_compression'],'connected_count':obs['connected_count'],'cyclic_count':obs['cyclic_count'],'evaluation_status':ev['status']};writej_atomic(root/'checkpoints'/f'r{r:03d}_metrics.json',met)
    return met

def finalize(root):
    done=[]
    for r in range(HORIZON+1):
        p=root/'checkpoints'/f'r{r:03d}_metrics.json'
        if p.exists():done.append(loadj(p))
    last=done[-1]['r3'] if done else -1;ev=loadj(root/'checkpoints'/f'r{last:03d}_evaluation.json') if last>=0 else {}
    status=ev.get('status','INCOMPLETE')
    if last==HORIZON and status=='NO_O3_SCOUT_TRIGGER_V1_0':
        o=load_obs(root,last);status='OPEN_ENDED_O3_RELATIONAL_NOVELTY' if any(load_obs(root,x)['new_main_atoms'] for x in range(max(1,last-15),last+1)) else 'FINITE_CONTROL_WITHOUT_O3_SCOUT_TRIGGER_V1_0'
    result={'program':'SCOUT3_V1_0_O3_DISTRIBUTED_CARRIER_ROADMAP','status':status,'last_r3':last,'o2_graduation_science_sha256':EXPECTED_O2_GRAD,'o2_source_bank_science_sha256':EXPECTED_BANK,'seed_panel_science_sha256':EXPECTED_SEED,'metrics':done,'final_evaluation':ev,'scope':'Possibility-level pre-geometric O3 relations among distinct graduated O2 Q entities. r3 is construction rank, not time. Edges are typed incidences, not spatial connections. No geometry/axis/distance/dimension/actualization claim.'}
    result['science_sha256']=sha_json(result);writej_atomic(root/'results/SCOUT3_V1_0_RESULT.json',result);(root/'results/STATUS.txt').write_text(status+'\n');return result

def synthetic_test(root):
    grad,bank,seed,qmap,meta,templates=load_inputs(root);qhs=sorted(qmap)[:3];h={'entities':qhs,'edges':[],'r3':0,'exact_key':'S'};acts=list(action_iter(h,qmap));assert acts
    c=add_edge(h,acts[0]);G=build_graph(c,qmap);assert entity_components(c)==2
    # Relabel entities + corresponding edge endpoints: exact graph iso must hold.
    perm={0:2,1:0,2:1};rr={'entities':[None]*3,'edges':[],'r3':c['r3']}
    for old,new in perm.items():rr['entities'][new]=c['entities'][old]
    for v,i,a,w,j,b in c['edges']:
        nv,nw=perm[v],perm[w]
        if nv<nw:rr['edges'].append([nv,i,a,nw,j,b])
        else:rr['edges'].append([nw,j,b,nv,i,a])
    rr['edges']=sorted(rr['edges']);assert nx.is_isomorphic(build_graph(c,qmap),build_graph(rr,qmap),node_match=NODE_MATCH)
    # Capacity use increments by exactly one endpoint each side.
    U=r3_usage(c,qmap);assert sum(sum(x) for e in U for x in e)==2
    # H1 must not read exact edge serialization directly: it is computed from current interface states/action support.
    sig,_=h1_support(c,qmap);assert isinstance(sig,str) and len(sig)==64
    return {'pass':True,'actions_seed':len(acts),'templates':len(templates)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['synthetic','bootstrap','step','finalize']);ap.add_argument('--root',default=str(ROOT));ap.add_argument('--r3',type=int);ap.add_argument('--cap',type=int,default=DEFAULT_LANE_CAP);ap.add_argument('--action-cap',type=int,default=DEFAULT_ACTION_CAP);args=ap.parse_args();root=Path(args.root)
    if args.command=='synthetic':print(json.dumps(synthetic_test(root),sort_keys=True));return
    if args.command=='bootstrap':print(json.dumps(bootstrap(root),sort_keys=True));return
    if args.command=='step':
        if args.r3 is None or args.r3<1 or args.r3>HORIZON:
            raise SystemExit('bad r3')
        print(json.dumps(step(root,args.r3,args.cap,args.action_cap),sort_keys=True)); return
    if args.command=='finalize':
        print(json.dumps(finalize(root),sort_keys=True)); return
if __name__=='__main__':main()
