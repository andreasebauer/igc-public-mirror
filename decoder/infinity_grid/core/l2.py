from __future__ import annotations

import hashlib
from typing import Protocol, Sequence, Any


class L2TableView(Protocol):
    tri: Sequence[int]
    cid: Sequence[int]
    oc: Sequence[int]
    ot: Sequence[int]
    os: Sequence[int]
    om: Sequence[int]
    on: Sequence[int]
    rk: Sequence[int]
    lookup: dict
    rep: dict
    def triple(self, tid: int) -> tuple[int, int, int]: ...
    def at(self, a: Sequence[int], s: int, i: int) -> int: ...


def need_satisfied(q: int, missing: int, supply: int) -> bool:
    return int(q) == 0 or (int(missing) & int(supply)) != 0


def destination_class(T: L2TableView, states, idxs):
    dest = tuple(sorted(T.at(T.ot, s, i) for s, i in zip(states, idxs)))
    return T.cid[T.lookup[dest]]


def ordered_states(T: L2TableView, tid: int, roots):
    t = list(T.triple(tid)); rest = [i for i in range(3) if i not in roots]
    return tuple(t[i] for i in list(roots) + rest)


def j3(T: L2TableView, tid: int, r0: int = 0, r1: int = 1, r2: int = 2):
    states = ordered_states(T, tid, (r0, r1, r2)); a, b, c = states; out = set()
    for ia in range(T.oc[a]):
      for ib in range(T.oc[b]):
       for ic in range(T.oc[c]):
        ss = (T.at(T.os,a,ia),T.at(T.os,b,ib),T.at(T.os,c,ic))
        mm = (T.at(T.om,a,ia),T.at(T.om,b,ib),T.at(T.om,c,ic))
        qq = (T.at(T.on,a,ia),T.at(T.on,b,ib),T.at(T.on,c,ic))
        c0=need_satisfied(qq[0],mm[0],ss[1]|ss[2]); c1=need_satisfied(qq[1],mm[1],ss[0]|ss[2]); c2=need_satisfied(qq[2],mm[2],ss[0]|ss[1])
        M0=0 if c0 else mm[0]; M1=0 if c1 else mm[1]; M2=0 if c2 else mm[2]
        if (not c0) and (qq[0]==0 or mm[0]==0): continue
        if (not c1) and (qq[1]==0 or mm[1]==0): continue
        if (not c2) and (qq[2]==0 or mm[2]==0): continue
        out.add((ss[0]|ss[1]|ss[2],((ss[0],M0),(ss[1],M1),(ss[2],M2)),min(T.at(T.rk,a,ia),T.at(T.rk,b,ib),T.at(T.rk,c,ic)),destination_class(T,states,(ia,ib,ic)),int(ia==ib==ic==0)))
    return frozenset(out)


def bridge_records(a, sa: int, b, sb: int) -> bool:
    PA,MA=a[1][sa]; PB,MB=b[1][sb]
    return ((MA==0 or (MA&PB)!=0) and (MB==0 or (MB&PA)!=0))


def composite_abc(T: L2TableView, ta: int, tb: int, tc: int):
    EA=j3(T,ta); EB=j3(T,tb); EC=j3(T,tc); out=set(); left=right=joined=0
    for b in EB:
        LA=[a for a in EA if bridge_records(a,0,b,0)]; LC=[c for c in EC if bridge_records(b,1,c,0)]
        left+=len(LA); right+=len(LC)
        for a in LA:
            for c in LC:
                joined+=1
                out.add((a[0]|b[0]|c[0],(a[1][1],a[1][2],b[1][2],c[1][1],c[1][2]),min(a[2],b[2],c[2]),(a[3],b[3],c[3]),int(a[4] and b[4] and c[4])))
    return frozenset(out),{'J3_A_events':len(EA),'J3_B_events':len(EB),'J3_C_events':len(EC),'left_compatible_event_links':left,'right_compatible_event_links':right,'internally_compatible_three_event_paths':joined}


def relation_digests(records):
    sorted_records=sorted(records)
    a=hashlib.sha256(repr(sorted_records).encode('utf-8')).hexdigest()
    h=hashlib.sha256()
    for r in sorted(records,key=repr): h.update(repr(r).encode('utf-8')); h.update(b'\n')
    return a,h.hexdigest()
