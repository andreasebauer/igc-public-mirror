from __future__ import annotations
"""Independent G5:R6 verifier with manual fixed-payload graft/canon path."""
from collections import defaultdict
from itertools import product
from typing import Any,Mapping
from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter,DecoratedG4Tree
from .uplift_g3_r1 import _generate_tree_shapes
from .uplift_g5_r3 import independent_unrooted_canon
class G5R6VerificationError(RuntimeError):pass

def _ican(ad,t):return independent_unrooted_canon(t.n,t.edges,[ad._vertex_key(x) for x in t.H_classes],t.edge_operators)
def _q(pub):return (tuple(map(int,pub['caps7'])),tuple(sorted((str(k),int(v)) for k,v in pub['H_class_bag'].items())))
def _manual_rel(ad,t):
    op=(0,0);local=[list(map(int,ad._rows[k]['caps7'])) for k in t.H_classes]
    for (u,v),(a,b) in zip(t.edges,t.edge_operators):local[u][a]-=1;local[v][b]-=1
    nc=list(map(int,ad._rows['C']['caps7']));out=set()
    for r in range(t.n):
        if local[r][0]<=0 or nc[0]<=0:continue
        ch=DecoratedG4Tree(t.n+1,t.edges+((r,t.n),),t.H_classes+('C',),t.edge_operators+(op,));out.add(_ican(ad,ch))
    return tuple(sorted(out))

def _primary_n9(ad):
    sh=_generate_tree_shapes(9)[9];exact={}
    for edges in sh.values():
      for colors in product(('C','D'),repeat=9):
        if 'C' not in colors or 'D' not in colors:continue
        t=DecoratedG4Tree(9,tuple(edges),tuple(colors),tuple((0,0) for _ in range(8)));pub=ad.public_read(t)
        if pub.get('legal'):exact.setdefault(_ican(ad,t),t)
    d=defaultdict(dict);c=0
    for p,t in exact.items():
      q=_q(ad.public_read(t));rel=_manual_rel(ad,t)
      if rel in d[q] and d[q][rel]!=p:c+=1
      else:d[q][rel]=p
    return len(exact),c

def _endpoint_sample(ad):
    sh=_generate_tree_shapes(5)[5];ops=((0,0),(0,1),(1,0));exact={}
    for eo in product(ops,repeat=4):
      if (0,0) not in eo or all(x==(0,0) for x in eo):continue
      for edges in sh.values():
       for colors in product(('C','D'),repeat=5):
        if 'C' not in colors or 'D' not in colors:continue
        t=DecoratedG4Tree(5,tuple(edges),tuple(colors),tuple(eo));pub=ad.public_read(t)
        if pub.get('legal'):exact.setdefault(_ican(ad,t),t)
    d=defaultdict(dict);c=0
    for p,t in exact.items():
      q=_q(ad.public_read(t));rel=_manual_rel(ad,t)
      if rel in d[q] and d[q][rel]!=p:c+=1
      else:d[q][rel]=p
    return len(exact),c

def verify(primary:Mapping[str,Any])->dict[str,Any]:
    f=[]
    if primary.get('schema_id')!='IG_G5_R6_SINGLE_PAYLOAD_MARKER_FREE_SEPARATION_RESULT_V1' or primary.get('status')!='PASS':f.append('PRIMARY_SCHEMA_STATUS')
    if primary.get('promotion') is not False or primary.get('g5_graduation_preserved') is not True:f.append('FIREWALL')
    ad=G4AcceptedAdapter();n,c=_primary_n9(ad);n2,c2=_endpoint_sample(ad);inj=bool(primary.get('single_payload_marker_free_separation_earned_on_frozen_scope'))
    if inj and (c or c2):f.append('INDEPENDENT_COLLISION_UNDER_INJECTIVE_PRIMARY')
    w=primary.get('first_collision_witness');wv=None
    if w is not None:
      def mk(x):return DecoratedG4Tree(int(x['n']),tuple(tuple(e) for e in x['edges']),tuple(x['H_classes']),tuple(tuple(o) for o in x['edge_operators']))
      a=mk(w['parent_a']);b=mk(w['parent_b']);wv=(_ican(ad,a)!=_ican(ad,b) and _q(ad.public_read(a))==_q(ad.public_read(b)) and _manual_rel(ad,a)==_manual_rel(ad,b))
      if not wv:f.append('COLLISION_WITNESS_NOT_REPRODUCED')
    out={'schema_id':'IG_G5_R6_INDEPENDENT_VERIFICATION_V1','status':'PASS' if not f else 'FAIL','primary_science_sha256':primary.get('science_sha256'),'independent_n9_exact_parent_count':n,'independent_n9_collision_count':c,'independent_endpoint_subset_exact_parent_count':n2,'independent_endpoint_subset_collision_count':c2,'primary_collision_witness_verified':wv,'manual_single_payload_graft_path_used':True,'independent_unrooted_canon_used':True,'structural_equality_only':True,'failures':f};out['verification_sha256']=canonical_sha256(out)
    if f:raise G5R6VerificationError('verification failed: '+','.join(f))
    return out
