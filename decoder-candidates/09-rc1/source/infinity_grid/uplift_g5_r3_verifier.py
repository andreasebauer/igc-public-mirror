from __future__ import annotations

"""Independent verification for G5:R3.

This verifier intentionally does not import uplift_g5_r3 or typed_tree_messages.
It rebuilds a separate rooted endpoint-typed tree oracle and independently
checks the certified result contract plus a deterministic fresh continuation
panel.
"""

from collections import deque
from itertools import product
from typing import Any, Mapping, Sequence

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .uplift_g3_r1 import _generate_tree_shapes


class G5R3VerificationError(RuntimeError):
    pass


def _freeze(v: Any) -> tuple[Any,...]:
    if v is None:return ("null",)
    if isinstance(v,bool):return ("bool",bool(v))
    if isinstance(v,int):return ("int",int(v))
    if isinstance(v,float):return ("float",v.hex())
    if isinstance(v,str):return ("str",v)
    if isinstance(v,bytes):return ("bytes",v.hex())
    if isinstance(v,Mapping):return ("map",tuple(sorted((_freeze(k),_freeze(x)) for k,x in v.items())))
    if isinstance(v,(list,tuple)):return ("seq",tuple(_freeze(x) for x in v))
    raise G5R3VerificationError(type(v).__name__)


def _adj(n:int,edges,keys,ops):
    if len(edges)!=n-1 or len(keys)!=n or len(ops)!=n-1:raise G5R3VerificationError("arity")
    a=[[] for _ in range(n)]
    for e,op in zip(edges,ops):
        u,v=map(int,e);x,y=op
        a[u].append((v,_freeze((x,y))));a[v].append((u,_freeze((y,x))))
    q=deque([0]);seen={0}
    while q:
        v=q.popleft()
        for u,_ in a[v]:
            if u not in seen:seen.add(u);q.append(u)
    if len(seen)!=n:raise G5R3VerificationError("disconnected")
    return a,tuple(_freeze(x) for x in keys)


def _rooted(n:int,edges,keys,ops,root:int):
    a,k=_adj(n,edges,keys,ops)
    def f(v,p):return ("V",k[v],tuple(sorted((lab,f(u,v)) for u,lab in a[v] if u!=p)))
    return f(int(root),-1)


def _unrooted(n:int,edges,keys,ops):return min(_rooted(n,edges,keys,ops,r) for r in range(n))


def _relabel(t:DecoratedG4Tree,p:Sequence[int])->DecoratedG4Tree:
    p=list(map(int,p));colors=[None]*t.n
    for i,c in enumerate(t.H_classes):colors[p[i]]=c
    return DecoratedG4Tree(t.n,tuple((p[u],p[v]) for u,v in t.edges),tuple(colors),t.edge_operators)


def _future(ad:G4AcceptedAdapter,t:DecoratedG4Tree,word):
    cur={ad.unrooted_canon(t):t}
    for h,op in word:
        nxt={}
        for x in cur.values():
            for r in range(x.n):
                for y in ad.graft_relation(x,r,new_H_class=str(h),operator=tuple(map(int,op))):nxt.setdefault(ad.unrooted_canon(y),y)
        cur=nxt
    return tuple(sorted(cur))


def verify(primary:Mapping[str,Any])->dict[str,Any]:
    failures=[]
    if primary.get("schema_id")!="IG_G5_R3_GENERIC_ENDPOINT_TYPED_MESSAGE_CONGRUENCE_RESULT_V1" or primary.get("status")!="PASS":failures.append("PRIMARY_SCHEMA_STATUS")
    if primary.get("g5_graduation_preserved") is not True or primary.get("promotion") is not False:failures.append("FIREWALL")
    if primary.get("finite_future_congruence_earned") is not True:failures.append("FUTURE_CONGRUENCE_FLAG")
    th=primary.get("generic_message_theorem",{})
    tids={x.get("theorem_id") for x in th.get("theorems",[])}
    required={f"R3-T{i}-" for i in range(1,8)}
    for prefix in required:
        if not any(str(t).startswith(prefix) for t in tids):failures.append("MISSING_"+prefix)
    pan=primary.get("exhaustive_support_panel",{})
    if pan.get("status")!="PASS" or pan.get("decorated_parent_count")!=950 or pan.get("rooted_canon_case_count")!=3698 or pan.get("future_word_check_count")!=2850:failures.append("SUPPORT_COUNTS")
    sent=primary.get("all_operator_sentinel",{})
    if sent.get("status")!="PASS" or sent.get("operator_count")!=31 or sent.get("failures"):failures.append("OPERATOR_SENTINEL")
    bridge=primary.get("r2_bridge",{})
    if bridge.get("status")!="PASS" or bridge.get("fresh_n8_first_predictive_depth")!=6 or bridge.get("endpoint_n5_first_predictive_depth")!=4:failures.append("R2_BRIDGE")

    ad=G4AcceptedAdapter();shapes=_generate_tree_shapes(4);pal=((0,0),(0,1),(1,0));oracle_cases=0;future_cases=0
    # Full independent rooted/unrooted oracle through n=4, but one fixed
    # continuation word only, so this verifier is algorithmically separate and
    # cheaper than the primary support panel.
    for n in range(1,5):
        for edges in shapes[n].values():
            for colors in product(("C","D"),repeat=n):
                ops_iter=[tuple()] if n==1 else product(pal,repeat=n-1)
                for ops in ops_iter:
                    t=DecoratedG4Tree(n,tuple(edges),tuple(colors),tuple(ops));keys=[ad._vertex_key(x) for x in colors]
                    iu=_unrooted(n,t.edges,keys,t.edge_operators)
                    if iu!=ad.unrooted_canon(t):failures.append(f"INDEPENDENT_UNROOTED_N{n}");break
                    for r in range(n):
                        oracle_cases+=1
                        if _rooted(n,t.edges,keys,t.edge_operators,r)!=ad.rooted_canon(t,r):failures.append(f"INDEPENDENT_ROOTED_N{n}");break
                    rel=_relabel(t,tuple(reversed(range(n))))
                    if _future(ad,t,(("C",(0,0)),("D",(1,0))))!=_future(ad,rel,(("C",(0,0)),("D",(1,0)))):failures.append(f"FUTURE_RELABEL_N{n}");break
                    future_cases+=1
                if failures and failures[-1].startswith(("INDEPENDENT_","FUTURE_")):break
            if failures and failures[-1].startswith(("INDEPENDENT_","FUTURE_")):break
        if failures and failures[-1].startswith(("INDEPENDENT_","FUTURE_")):break
    if oracle_cases!=3698:failures.append(f"ORACLE_CASE_COUNT:{oracle_cases}")
    if future_cases!=950:failures.append(f"FUTURE_CASE_COUNT:{future_cases}")
    out={"schema_id":"IG_G5_R3_INDEPENDENT_VERIFICATION_V1","status":"PASS" if not failures else "FAIL","primary_science_sha256":primary.get("science_sha256"),"independent_rooted_oracle_case_count":oracle_cases,"independent_two_step_future_relabel_case_count":future_cases,"all_31_operator_primary_sentinel_bound":sent.get("science_sha256"),"structural_equality_only":True,"failures":failures[:50]}
    out["verification_sha256"]=canonical_sha256(out)
    if failures:raise G5R3VerificationError("verification failed: "+",".join(failures[:10]))
    return out
