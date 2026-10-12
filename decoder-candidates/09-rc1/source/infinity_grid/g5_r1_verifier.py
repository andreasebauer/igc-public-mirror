from __future__ import annotations
import argparse,json,shutil,tempfile
from pathlib import Path
import networkx as nx
from .canon import canonical_sha256,write_json_atomic
from .uplift_campaign import UpliftCampaignEngine
from .uplift_g5_r1 import run_g5_r1_fiber_graft_audit,compare_cold_replay,r1_spec

def _load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def _independent_shape_check(max_n=12):
    exp=list(r1_spec()['bounded_shape_census']['expected_unlabelled_tree_counts_n1_to_nmax']); rows=[];f=[]
    for n in range(1,max_n+1):
        cnt=1 if n==1 else sum(1 for _ in nx.generators.nonisomorphic_trees(n))
        ok=cnt==exp[n-1]
        if not ok:f.append({'n':n,'observed':cnt,'expected':exp[n-1]})
        rows.append({'n':n,'networkx_nonisomorphic_tree_count':cnt,'expected':exp[n-1],'pass':ok})
    o={'schema_id':'IG_G5_R1_INDEPENDENT_SHAPE_CHECK_V1','status':'PASS' if not f else 'FAIL','method':'NETWORKX_NONISOMORPHIC_TREES_INDEPENDENT_OF_DECODER_TREE_CANON','rows':rows,'failures':f,'verification_only_dependency':'networkx'};o['science_sha256']=canonical_sha256(o);return o

def _independent_graft_check(primary):
    w=primary['relation_valued_graft_witness']; cs=w['children']; failures=[]
    gs=[]
    for c in cs:
        g=nx.Graph();g.add_nodes_from(range(5));g.add_edges_from([tuple(x) for x in c['edges']]);gs.append(g)
        if not nx.is_tree(g):failures.append('CHILD_NOT_TREE')
    if nx.is_isomorphic(gs[0],gs[1]):failures.append('CHILDREN_ISOMORPHIC')
    if not w.get('same_graduated_public_output'):failures.append('PUBLIC_OUTPUT_MISMATCH')
    if int(w.get('distinct_exact_child_count',0))<2:failures.append('NO_RELATION_VALUED_BRANCHING')
    o={'schema_id':'IG_G5_R1_INDEPENDENT_GRAFT_CHECK_V1','status':'PASS' if not failures else 'FAIL','method':'NETWORKX_EXACT_UNDECORATED_TOPOLOGY_CHECK_PLUS_STORED_PUBLIC_OUTCOME_COMPARISON','children_nonisomorphic_networkx':not nx.is_isomorphic(gs[0],gs[1]),'same_graduated_public_output':w.get('same_graduated_public_output'),'failures':failures,'limitations':['WITNESS_ONLY','UNDECORATED_TOPOLOGY_CHECK_IS_SUFFICIENT_FOR_NONISOMORPHISM_HERE_BECAUSE_ALL_DECORATIONS_ARE_IDENTICAL']};o['science_sha256']=canonical_sha256(o);return o

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primary',required=True);ap.add_argument('--r0-primary',required=True);ap.add_argument('--r0-verification',required=True);ap.add_argument('--r0-closeout',required=True);ap.add_argument('--out-dir',required=True);ns=ap.parse_args();out=Path(ns.out_dir);out.mkdir(parents=True,exist_ok=True)
    primary=_load(ns.primary);r0p=_load(ns.r0_primary);r0v=_load(ns.r0_verification);r0c=_load(ns.r0_closeout)
    coldroot=Path(tempfile.mkdtemp(prefix='g5_r1_cold_'))
    try:
      with UpliftCampaignEngine(coldroot,requested_workers=1) as eng:
        cold=run_g5_r1_fiber_graft_audit(engine=eng,r0_primary=r0p,r0_verification=r0v,r0_closeout=r0c);cold['source_sha256']=primary.get('source_sha256');cold['source_version']=primary.get('source_version');cold['registry_sha256']=eng.registry['registry_sha256']
      comp=compare_cold_replay(primary,cold);shape=_independent_shape_check(12);graft=_independent_graft_check(primary)
      report={'schema_id':'IG_G5_R1_INDEPENDENT_VERIFICATION_V1','status':'PASS' if comp['status']=='PASS' and shape['status']=='PASS' and graft['status']=='PASS' else 'FAIL','comparison':comp,'independent_shape_check':shape,'independent_graft_check':graft,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'authority_effect':'NONE','limitations':['G5_R1_ONLY','NON_PROMOTING','NO_PHYSICAL_GEOMETRY']};report['verification_sha256']=canonical_sha256(report)
      write_json_atomic(out/'G5_R1_COLD_RESULT.json',cold);write_json_atomic(out/'G5_R1_COLD_COMPARISON.json',comp);write_json_atomic(out/'G5_R1_INDEPENDENT_SHAPE_CHECK.json',shape);write_json_atomic(out/'G5_R1_INDEPENDENT_GRAFT_CHECK.json',graft);write_json_atomic(out/'G5_R1_INDEPENDENT_VERIFICATION.json',report)
      print(json.dumps({'status':report['status'],'verification_sha256':report['verification_sha256'],'cold_science_sha256':cold.get('science_sha256')},sort_keys=True));raise SystemExit(0 if report['status']=='PASS' else 2)
    finally:shutil.rmtree(coldroot,ignore_errors=True)
if __name__=='__main__':main()
