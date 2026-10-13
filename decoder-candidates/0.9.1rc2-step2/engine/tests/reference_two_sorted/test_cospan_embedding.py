#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ig_two_sorted_tree_wiring import (
    LabeledRelation, Row, TreeEdge, TreeWiring,
    substitute_box,
)
from ig_treematch_cospan import (
    treematch_to_cospan,
    cospan_to_treematch,
    substitute_cospan_pushout,
)

ROOT = Path(__file__).parents[1]
OUT = ROOT / "05_RESULTS" / "TREEMATCH_COSPAN_EMBEDDING_COMPAT_TESTS.json"
checks=[]

def ck(name, cond, detail=None):
    if not cond:
        raise AssertionError(name)
    checks.append({"test":name,"status":"PASS","detail":detail or {}})

# A nontrivial four-box host.
def rel(prefix, n):
    return LabeledRelation.from_rows(
        tuple(f"{prefix}{i}" for i in range(n)),
        (f"{prefix}T",),
        [Row(1, tuple((0,0) for _ in range(n)), 2, (1,), True)],
    )

rels=(rel("a",3),rel("b",2),rel("c",3),rel("d",2))
w=TreeWiring((
    TreeEdge(0,"a0",1,"b0"),
    TreeEdge(0,"a1",2,"c0"),
    TreeEdge(2,"c1",3,"d0"),
))
C=treematch_to_cospan(rels,w)
wr=cospan_to_treematch(rels,C)
ck("faithful_inverse_fixed_tree", treematch_to_cospan(rels,wr).canonical_key()==C.canonical_key(), {"fibres":len(C.fibres())})

# Random trees: build each new box by one edge to an earlier box using unique ports.
rng=random.Random(20260827)
random_cases=0
for case in range(100):
    n=rng.randint(1,7)
    rs=tuple(rel(f"r{case}_{b}_", max(2,n+1)) for b in range(n))
    used=[set() for _ in range(n)]
    edges=[]
    for b in range(1,n):
        parent=rng.randrange(b)
        pslot=next(s for s in rs[parent].port_slots if s not in used[parent])
        bslot=next(s for s in rs[b].port_slots if s not in used[b])
        used[parent].add(pslot); used[b].add(bslot)
        edges.append(TreeEdge(parent,pslot,b,bslot))
    tw=TreeWiring(tuple(edges))
    c=treematch_to_cospan(rs,tw)
    back=cospan_to_treematch(rs,c)
    if treematch_to_cospan(rs,back).canonical_key()!=c.canonical_key():
        raise AssertionError((case,"faithfulness"))
    random_cases+=1
ck("faithful_inverse_random_trees", True, {"cases":random_cases})

# Direct cable-pushout substitution equals the cospan image of flattened
# TreeMatch substitution.
H0=rel("h0_",2)
H1=rel("q_",3)
H2=rel("h2_",2)
host_relations=(H0,H1,H2)
host=TreeWiring((
    TreeEdge(0,"h0_0",1,"q_0"),
    TreeEdge(1,"q_1",2,"h2_0"),
))
I0=rel("i0_",2)
I1=rel("i1_",3)
inner_relations=(I0,I1)
inner=TreeWiring((TreeEdge(0,"i0_0",1,"i1_0"),))
# Inner exposed halfports are (0,i0_1),(1,i1_1),(1,i1_2).
mapping={"q_0":(0,"i0_1"),"q_1":(1,"i1_1"),"q_2":(1,"i1_2")}
flat_rels,flat_w=substitute_box(host_relations,host,1,inner_relations,inner,mapping)
expected=treematch_to_cospan(flat_rels,flat_w)
pushed=substitute_cospan_pushout(host_relations,host,1,inner_relations,inner,mapping)
ck("cospan_pushout_equals_flattened_substitution", pushed.canonical_key()==expected.canonical_key(), {"fibres":len(expected.fibres())})

result={
    "suite":"IG_V2.1_TREEMATCH_COSPAN_EMBEDDING",
    "status":"PASS",
    "tests":checks,
    "summary":{"tests_passed":len(checks),"random_faithfulness_cases":random_cases},
    "scope":"restricted pairwise connected acyclic TreeMatch image only",
}
OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
