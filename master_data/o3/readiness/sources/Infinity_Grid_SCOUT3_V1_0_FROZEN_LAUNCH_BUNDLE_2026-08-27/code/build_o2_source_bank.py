#!/usr/bin/env python3
import gzip, json, glob, hashlib, math, os, sys, multiprocessing as mp, numpy as np
from collections import defaultdict, Counter
from pathlib import Path

PARENT=Path('/mnt/data/IN12_SCOUT2B_V1_2_OPT_FRESH_2026-08-27')
OUT=Path('/mnt/data/SCOUT3_V1_0_O3_ROADMAP_2026-08-27/inputs/O2_SOURCE_BANK.json')
CAP=256
LANE_REQ=24
EXPECTED_O2_SCIENCE='d138de3c6c906be3ab453c225ecce2e3fff5d3e4f91d220ed681ef5253a86bbb'
GRAD=Path('/mnt/data/SCOUT3_V1_0_O3_ROADMAP_2026-08-27/inputs/O2_GRADUATION_AUDIT_RESULT.json')

def sha_repr(x): return hashlib.sha256(repr(x).encode()).hexdigest()
def q_of(c):
    n=len(c['states']); u=[[0]*7 for _ in range(n)]
    for i,a,j,b in c['edges']:
        u[int(i)][int(a)]+=1; u[int(j)][int(b)]+=1
    return tuple(sorted((tuple(int(x) for x in s[:7]),tuple(int(x) for x in u[i])) for i,s in enumerate(c['states'])))
def qfeat(q, meta):
    P=[0]*7;U=[0]*7;free7=[0]*7;sitefree=[]
    for p,u in q:
        sf=0
        for a in range(7):
            P[a]+=p[a];U[a]+=u[a]; free7[a]+=p[a]-u[a]; sf+=p[a]-u[a]
        sitefree.append(sf)
    totalu=sum(U); n=len(q); beta=totalu//2-n+1
    totalfree=sum(free7); maxsf=max(sitefree) if sitefree else 0
    uniq=len(set(q)); repeats=n-uniq
    conc=0.0 if totalfree==0 else maxsf/totalfree
    return [n,beta,totalfree,uniq,repeats,conc,meta['occurrences'],meta['span']]+P+U+free7

def farthest(items, required, cap, feats):
    # Vectorized deterministic greedy farthest-point; items ordered by SHA for tie-breaking.
    items=sorted(items,key=sha_repr)
    if len(items)<=cap:return items
    X=np.asarray([feats[x] for x in items],dtype=np.float64)
    lo=X.min(axis=0); hi=X.max(axis=0); den=hi-lo; den[den==0]=1.0; X=(X-lo)/den
    pos={x:i for i,x in enumerate(items)}
    reqidx=sorted({pos[x] for x in required if x in pos})
    if len(reqidx)>=cap:return [items[i] for i in reqidx[:cap]]
    selected=list(reqidx)
    if not selected:selected=[0]
    chosen=np.zeros(len(items),dtype=bool); chosen[selected]=True
    mind=np.full(len(items),np.inf,dtype=np.float64)
    for i in selected:
        d=((X-X[i])**2).sum(axis=1); mind=np.minimum(mind,d)
    while len(selected)<cap:
        m=mind.copy();m[chosen]=-1.0; idx=int(np.argmax(m))
        selected.append(idx);chosen[idx]=True
        d=((X-X[idx])**2).sum(axis=1);mind=np.minimum(mind,d)
    return [items[i] for i in sorted(selected)]

def _panel_q_rows(path):
    with gzip.open(path,'rt') as f:d=json.load(f)
    k=int(d['k']); return k, [(q_of(c),c['exact_key']) for c in d['selected']]


def main():
    g=json.load(open(GRAD))
    if g.get('science_sha256')!=EXPECTED_O2_SCIENCE or g.get('status')!='O2_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_V1_9':
        raise SystemExit('O2 graduation integrity fail')
    info={}
    panel_paths=sorted(glob.glob(str(PARENT/'checkpoints/k*_selected_carriers.json.gz')))
    if len(panel_paths)!=129:raise SystemExit(f'expected 129 panels, got {len(panel_paths)}')
    def merge_panel(item):
        k, rows=item
        for q, exact_key in rows:
            rec=info.setdefault(q,{'first_k':k,'last_k':k,'occurrences':0,'exact_keys':set()})
            rec['first_k']=min(rec['first_k'],k);rec['last_k']=max(rec['last_k'],k);rec['occurrences']+=1;rec['exact_keys'].add(exact_key)
    with mp.Pool(processes=min(8,os.cpu_count() or 1)) as pool:
        for item in pool.imap_unordered(_panel_q_rows,panel_paths,chunksize=1): merge_panel(item)
    for q,r in info.items():
        r['span']=r['last_k']-r['first_k']+1;r['distinct_exact']=len(r['exact_keys']);del r['exact_keys']
    qs=list(info); feats={q:qfeat(q,info[q]) for q in qs}
    # Frozen pre-outcome source lanes.
    lanes={}
    lanes['BETA_HIGH']=sorted(qs,key=lambda q:(-feats[q][1],sha_repr(q)))[:LANE_REQ]
    lanes['FREE_HIGH']=sorted(qs,key=lambda q:(-feats[q][2],sha_repr(q)))[:LANE_REQ]
    lanes['TIGHT_LOW_FREE']=sorted(qs,key=lambda q:(feats[q][2],sha_repr(q)))[:LANE_REQ]
    lanes['HIDDEN_MULTIPLICITY']=sorted(qs,key=lambda q:(-info[q]['occurrences'],-info[q]['distinct_exact'],sha_repr(q)))[:LANE_REQ]
    lanes['REPEATED_SITE_SYMMETRY']=sorted(qs,key=lambda q:(-(len(q)-len(set(q))),-info[q]['occurrences'],sha_repr(q)))[:LANE_REQ]
    # broad long-span stable-presence stress without using any Scout3 outcome.
    lanes['LONG_SPAN']=sorted(qs,key=lambda q:(-info[q]['span'],-info[q]['occurrences'],sha_repr(q)))[:LANE_REQ]
    required=[]
    for arr in lanes.values():required.extend(arr)
    selected=farthest(qs,required,CAP,feats)
    tags=defaultdict(list)
    for name,arr in lanes.items():
        s=set(arr)
        for q in selected:
            if q in s:tags[q].append(name)
    entries=[]
    for idx,q in enumerate(selected):
        P=[0]*7;U=[0]*7;F=[0]*7
        for p,u in q:
            for a in range(7):P[a]+=p[a];U[a]+=u[a];F[a]+=p[a]-u[a]
        entries.append({
            'qid':f'Q{idx:03d}','q_hash':sha_repr(q),'sites':[[list(p),list(u)] for p,u in q],
            'n_sites':len(q),'beta_o2':sum(U)//2-len(q)+1,'total_p':P,'total_u':U,'free':F,
            'first_k':info[q]['first_k'],'last_k':info[q]['last_k'],'span':info[q]['span'],
            'occurrences':info[q]['occurrences'],'distinct_exact_occurrences':info[q]['distinct_exact'],'source_lane_tags':sorted(tags[q]),
        })
    out={
      'program':'SCOUT3_V1_0_O2_SOURCE_BANK','source_parent':'Scout2-B V1.2 selected k0..k128 panels','o2_graduation_science_sha256':EXPECTED_O2_SCIENCE,
      'all_unique_Q':len(qs),'selected_Q':len(entries),'selection_cap':CAP,'lane_required_each':LANE_REQ,
      'lanes':{k:[sha_repr(q) for q in v] for k,v in lanes.items()},'entries':entries,
    }
    payload=json.dumps(out,sort_keys=True,separators=(',',':')).encode();out['science_sha256']=hashlib.sha256(payload).hexdigest()
    OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'all_unique_Q':len(qs),'selected':len(entries),'science_sha256':out['science_sha256'],'n_site_counts':dict(Counter(e['n_sites'] for e in entries))},indent=2))
if __name__=='__main__':main()
