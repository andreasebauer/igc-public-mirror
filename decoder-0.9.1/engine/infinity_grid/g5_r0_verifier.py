from __future__ import annotations
import argparse, json, shutil, tempfile
from pathlib import Path
from itertools import product
import networkx as nx

from .canon import canonical_sha256, write_json_atomic
from .uplift_campaign import UpliftCampaignEngine
from .uplift_g5_r0 import run_g5_r0_recon, compare_cold_replay
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree


def _load(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def _prufer_edges(n,seq):
    if n==1:return []
    if n==2:return [(0,1)]
    deg=[1]*n
    for x in seq:deg[x]+=1
    out=[]
    for x in seq:
        leaf=next(i for i,d in enumerate(deg) if d==1)
        out.append((leaf,x));deg[leaf]-=1;deg[x]-=1
    rem=[i for i,d in enumerate(deg) if d==1];out.append((rem[0],rem[1]));return out

def _independent_shape_check(max_n=7):
    ad=G4AcceptedAdapter(); expected=[1,1,1,2,3,6,11]; rows=[]; failures=[]
    for n in range(1,max_n+1):
        reps=[]; seqs=[tuple()] if n<=2 else product(range(n),repeat=n-2)
        labelled=0
        for seq in seqs:
            e=_prufer_edges(n,tuple(seq)); labelled+=1
            g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from(e)
            if not any(nx.is_isomorphic(g,h) for h,_ in reps): reps.append((g,e))
        pubs=[]
        for _g,e in reps:
            t=DecoratedG4Tree(n,tuple(e),tuple(['C']*n),tuple([(0,0)]*len(e)))
            q=ad.public_read(t); pubs.append({k:q.get(k) for k in ('legal','descriptor','caps7','H_class_bag')})
        cnt=len(reps); ok=(cnt==expected[n-1] and all(x.get('legal') for x in pubs) and len({canonical_sha256(x) for x in pubs})==1)
        if not ok: failures.append({'n':n,'observed_count':cnt,'expected_count':expected[n-1]})
        rows.append({'n':n,'labelled_pruefer_tree_count':labelled,'networkx_exact_isomorphism_class_count':cnt,'expected':expected[n-1],'all_legal':all(x.get('legal') for x in pubs),'unique_public_state_count':len({canonical_sha256(x) for x in pubs})})
    out={'schema_id':'IG_G5_R0_INDEPENDENT_SHAPE_CHECK_V1','status':'PASS' if not failures else 'FAIL','method':'INDEPENDENT_NETWORKX_EXACT_GRAPH_ISOMORPHISM_GROUPING_OF_PRUEFER_TREES','rows':rows,'failures':failures,'verification_only_dependency':'networkx'}
    out['science_sha256']=canonical_sha256(out);return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primary',required=True);ap.add_argument('--graduation-decision',required=True);ap.add_argument('--s6-verification',required=True);ap.add_argument('--s6-primary',required=True);ap.add_argument('--out-dir',required=True);ns=ap.parse_args()
    out=Path(ns.out_dir);out.mkdir(parents=True,exist_ok=True)
    primary=_load(ns.primary); grad=_load(ns.graduation_decision); ver=_load(ns.s6_verification); s6=_load(ns.s6_primary)
    coldroot=Path(tempfile.mkdtemp(prefix='g5_r0_cold_'))
    try:
      with UpliftCampaignEngine(coldroot,requested_workers=1) as eng:
        cold=run_g5_r0_recon(engine=eng,graduation_decision=grad,independent_verification=ver,primary_s6=s6)
        cold['source_sha256']=primary.get('source_sha256');cold['source_version']=primary.get('source_version');cold['registry_sha256']=eng.registry['registry_sha256']
      comp=compare_cold_replay(primary,cold); ishape=_independent_shape_check(7)
      report={'schema_id':'IG_G5_R0_INDEPENDENT_VERIFICATION_V1','status':'PASS' if comp['status']=='PASS' and ishape['status']=='PASS' else 'FAIL','comparison':comp,'independent_shape_check':ishape,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'authority_effect':'NONE','limitations':['G5_R0_ONLY','NON_PROMOTING','NO_PHYSICAL_GEOMETRY','NETWORKX_USED_ONLY_AS_INDEPENDENT_BOUNDED_VERIFIER']}
      report['verification_sha256']=canonical_sha256(report)
      write_json_atomic(out/'G5_R0_COLD_RESULT.json',cold);write_json_atomic(out/'G5_R0_COLD_COMPARISON.json',comp);write_json_atomic(out/'G5_R0_INDEPENDENT_SHAPE_CHECK.json',ishape);write_json_atomic(out/'G5_R0_INDEPENDENT_VERIFICATION.json',report)
      print(json.dumps({'status':report['status'],'verification_sha256':report['verification_sha256'],'cold_science_sha256':cold.get('science_sha256')},sort_keys=True))
      raise SystemExit(0 if report['status']=='PASS' else 2)
    finally: shutil.rmtree(coldroot,ignore_errors=True)
if __name__=='__main__': main()
