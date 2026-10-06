#!/usr/bin/env python3
import json, hashlib, math
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/data/SCOUT3_V1_0_O3_ROADMAP_2026-08-27')
BANK=ROOT/'inputs/O2_SOURCE_BANK.json'; OUT=ROOT/'inputs/O3_SEED_PANEL.json'
PER_N=32

def h(x):return hashlib.sha256(repr(x).encode()).hexdigest()
def qvec(e):
    return np.asarray([e['n_sites'],e['beta_o2'],sum(e['free']),e['occurrences'],e['span'],len(e['source_lane_tags'])]+e['free']+e['total_p']+e['total_u'],dtype=float)
def seedfeat(seed, qmap):
    es=[qmap[q] for q in seed]; X=np.stack([qvec(e) for e in es])
    return np.concatenate(([len(seed),len(set(seed)),sum(e['n_sites'] for e in es),sum(e['beta_o2'] for e in es)],X.sum(axis=0),X.mean(axis=0),X.std(axis=0)))
def farthest(cands,required,cap,qmap):
    cands=sorted(set(cands),key=h); req=set(required)
    X=np.stack([seedfeat(c,qmap) for c in cands]);lo=X.min(0);hi=X.max(0);den=hi-lo;den[den==0]=1;X=(X-lo)/den
    pos={c:i for i,c in enumerate(cands)};sel=sorted({pos[c] for c in req if c in pos})
    if len(sel)>=cap:return [cands[i] for i in sel[:cap]]
    if not sel:sel=[0]
    chosen=np.zeros(len(cands),bool);chosen[sel]=1;mind=np.full(len(cands),np.inf)
    for i in sel:mind=np.minimum(mind,((X-X[i])**2).sum(1))
    while len(sel)<cap:
        z=mind.copy();z[chosen]=-1;i=int(np.argmax(z));sel.append(i);chosen[i]=1;mind=np.minimum(mind,((X-X[i])**2).sum(1))
    return [cands[i] for i in sorted(sel)]
def main():
    b=json.load(open(BANK)); entries=sorted(b['entries'],key=lambda e:e['q_hash']); qmap={e['q_hash']:e for e in entries}; qs=[e['q_hash'] for e in entries]; N=len(qs)
    seeds=[]
    for n in range(3,7):
        cands=[];req=[]
        # Homogeneous/symmetric candidates are mandatory controls.
        for j in range(0,N,max(1,N//32)):
            s=tuple([qs[j]]*n);cands.append(s)
            if len(req)<8:req.append(s)
        # Deterministic modular candidate lattice, no random draw.
        for step in (1,7,17,43,97):
            for a in range(N):
                inds=tuple(sorted((a+step*t+((t*t)*(step+3)))%N for t in range(n)))
                s=tuple(sorted(qs[i] for i in inds));cands.append(s)
        hom=sorted(set(req),key=h)[:8]
        non=[c for c in set(cands) if len(set(c))==n and c not in set(hom)]
        chosen=hom+farthest(non,[],PER_N-len(hom),qmap);seeds.extend(chosen)
    seeds=sorted(set(seeds),key=h)
    out=[]
    for i,s in enumerate(seeds):
        es=[qmap[q] for q in s]
        out.append({'hid':f'H0_{i:03d}','entity_q_hashes':list(s),'n_entities':len(s),'distinct_q_types':len(set(s)),'total_o1_sites':sum(e['n_sites'] for e in es),'sum_o2_beta':sum(e['beta_o2'] for e in es),'total_free':[sum(e['free'][a] for e in es) for a in range(7)],'homogeneous':len(set(s))==1})
    obj={'program':'SCOUT3_V1_0_O3_SEED_PANEL','source_bank_science_sha256':b['science_sha256'],'selection':'32 deterministic farthest-point seed multisets for each n_entities=3,4,5,6 from a fixed modular candidate lattice; 8 homogeneous candidates per n are mandatory controls.','count':len(out),'entries':out}
    obj['science_sha256']=hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest();OUT.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    from collections import Counter
    print(json.dumps({'count':len(out),'n':dict(Counter(x['n_entities'] for x in out)),'homogeneous':sum(x['homogeneous'] for x in out),'science_sha256':obj['science_sha256']},indent=2))
if __name__=='__main__':main()
