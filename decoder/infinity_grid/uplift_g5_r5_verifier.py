from __future__ import annotations

"""Independent G5:R5 verifier using a separately implemented manual graft/canon path."""
from collections import defaultdict
from itertools import product
from typing import Any, Mapping

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .uplift_g3_r1 import _generate_tree_shapes
from .uplift_g5_r3 import independent_unrooted_canon

class G5R5VerificationError(RuntimeError): pass


def _ican(ad:G4AcceptedAdapter,t:DecoratedG4Tree):
    return independent_unrooted_canon(t.n,t.edges,[ad._vertex_key(x) for x in t.H_classes],t.edge_operators)


def _manual_rel(ad:G4AcceptedAdapter,t:DecoratedG4Tree,new_H:str,op):
    op=tuple(map(int,op)); local=[list(map(int,ad._rows[k]['caps7'])) for k in t.H_classes]
    for (u,v),(a,b) in zip(t.edges,t.edge_operators): local[u][a]-=1;local[v][b]-=1
    nc=list(map(int,ad._rows[new_H]['caps7'])); out=set()
    for r in range(t.n):
        if local[r][op[0]]<=0 or nc[op[1]]<=0: continue
        ch=DecoratedG4Tree(t.n+1,t.edges+((r,t.n),),t.H_classes+(new_H,),t.edge_operators+(op,))
        out.add(_ican(ad,ch))
    return tuple(sorted(out))


def _sig(ad,t): return tuple((h,tuple(op),_manual_rel(ad,t,h,op)) for h in ('C','D') for op in ad.operator_basis())

def _q(pub): return (tuple(map(int,pub['caps7'])),tuple(sorted((str(k),int(v)) for k,v in pub['H_class_bag'].items())))


def _panel_primary_n4_n6(ad):
    shapes=_generate_tree_shapes(6);exact={}
    for n in range(4,7):
        for edges in shapes[n].values():
            for colors in product(('C','D'),repeat=n):
                t=DecoratedG4Tree(n,tuple(edges),tuple(colors),tuple((0,0) for _ in range(n-1)));pub=ad.public_read(t)
                if pub.get('legal'): exact.setdefault(_ican(ad,t),t)
    owners=defaultdict(dict);coll=0
    for p,t in exact.items():
        q=_q(ad.public_read(t));s=_sig(ad,t)
        if s in owners[q] and owners[q][s]!=p: coll+=1
        owners[q][s]=p
    return len(exact),coll


def _panel_endpoint_sample(ad):
    shapes=_generate_tree_shapes(3);ops=tuple(ad.operator_basis()); sel=(ops[0],ops[len(ops)//2],ops[-1]);exact={}
    for edges in shapes[3].values():
        for colors in product(('C','D'),repeat=3):
            for eo in product(sel,repeat=2):
                t=DecoratedG4Tree(3,tuple(edges),tuple(colors),tuple(eo));pub=ad.public_read(t)
                if pub.get('legal'): exact.setdefault(_ican(ad,t),t)
    owners=defaultdict(dict);coll=0
    for p,t in exact.items():
        q=_q(ad.public_read(t));s=_sig(ad,t)
        if s in owners[q] and owners[q][s]!=p: coll+=1
        owners[q][s]=p
    return len(exact),coll


def verify(primary:Mapping[str,Any])->dict[str,Any]:
    failures=[]
    if primary.get('schema_id')!='IG_G5_R5_MARKER_FREE_ONE_STEP_SEPARATION_RESULT_V1' or primary.get('status')!='PASS': failures.append('PRIMARY_SCHEMA_STATUS')
    if primary.get('promotion') is not False or primary.get('g5_graduation_preserved') is not True: failures.append('FIREWALL')
    ad=G4AcceptedAdapter(); n,c=_panel_primary_n4_n6(ad); n2,c2=_panel_endpoint_sample(ad)
    primary_inj=bool(primary.get('marker_free_one_step_separation_earned_on_frozen_scope'))
    # Any independent collision invalidates a primary injective claim. If primary reports a collision,
    # independently verify the supplied witness exactly below.
    if primary_inj and (c or c2): failures.append('INDEPENDENT_COLLISION_UNDER_INJECTIVE_PRIMARY')
    witness=primary.get('first_collision_witness')
    witness_verified=None
    if witness is not None:
        def mk(x): return DecoratedG4Tree(int(x['n']),tuple(tuple(e) for e in x['edges']),tuple(x['H_classes']),tuple(tuple(o) for o in x['edge_operators']))
        a=mk(witness['parent_a']);b=mk(witness['parent_b'])
        witness_verified=(_ican(ad,a)!=_ican(ad,b) and _q(ad.public_read(a))==_q(ad.public_read(b)) and _sig(ad,a)==_sig(ad,b))
        if not witness_verified: failures.append('COLLISION_WITNESS_NOT_REPRODUCED')
    out={'schema_id':'IG_G5_R5_INDEPENDENT_VERIFICATION_V1','status':'PASS' if not failures else 'FAIL','primary_science_sha256':primary.get('science_sha256'),'independent_n4_n6_exact_parent_count':n,'independent_n4_n6_collision_count':c,'independent_endpoint_sample_exact_parent_count':n2,'independent_endpoint_sample_collision_count':c2,'primary_collision_witness_verified':witness_verified,'manual_graft_path_used':True,'independent_unrooted_canon_used':True,'structural_equality_only':True,'failures':failures}
    out['verification_sha256']=canonical_sha256(out)
    if failures: raise G5R5VerificationError('verification failed: '+','.join(failures))
    return out
