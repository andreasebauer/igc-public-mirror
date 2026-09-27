from __future__ import annotations
import argparse,json,shutil,tempfile
from pathlib import Path
import networkx as nx
from .canon import canonical_sha256,write_json_atomic
from .uplift_campaign import UpliftCampaignEngine
from .uplift_g5_r2 import run_g5_r2_predictive_hidden_read,compare_cold_replay,r2_spec

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def _plain_tree(n,edges):
    g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from([tuple(map(int,e)) for e in edges]);return g

def independent_lower_bound(primary):
    rows=[];fail=[]
    for d in range(0,9):
        n=d+3
        pe=[(i,i+1) for i in range(n-1)]
        be=[(i,i+1) for i in range(d)]+[(d,d+1),(d,d+2)]
        gp=_plain_tree(n+1,pe+[(0,n)]);gb=_plain_tree(n+1,be+[(0,n)])
        noniso=not nx.is_isomorphic(gp,gb)
        if not noniso: fail.append(f'NETWORKX_CHILD_ISO_D{d}')
        rows.append({'depth':d,'children_nonisomorphic_networkx':noniso})
    o={'schema_id':'IG_G5_R2_INDEPENDENT_LOWER_BOUND_CHECK_V1','status':'PASS' if not fail else 'FAIL','method':'NETWORKX_UNDECORATED_TREE_ISOMORPHISM_ON_IDENTICALLY_DECORATED_PATH_BROOM_WITNESS','rows':rows,'failures':fail,'limitations':['INDEPENDENT_CHECK_D0_TO_D8_ONLY','DECORATIONS_IDENTICAL_SO_UNDECORATED_NONISOMORPHISM_IS_DECISIVE']};o['science_sha256']=canonical_sha256(o);return o

def _expanded_decorated_graph(tree):
    # Independent endpoint-decoration representation: V nodes carry H class;
    # each tree edge is represented by an E center and one incidence node per
    # endpoint carrying the local endpoint type.
    g=nx.Graph()
    for v,h in enumerate(tree['H']):g.add_node(('v',v),kind='V',label=str(h))
    for i,((a,b),(x,y)) in enumerate(zip(tree['edges'],tree['ops'])):
        e=('e',i);ia=('i',i,0);ib=('i',i,1)
        g.add_node(e,kind='E',label='E');g.add_node(ia,kind='I',label=str(x));g.add_node(ib,kind='I',label=str(y))
        g.add_edge(('v',int(a)),ia);g.add_edge(ia,e);g.add_edge(e,ib);g.add_edge(ib,('v',int(b)))
    return g

def independent_operator_sentinel(primary):
    node_match=nx.algorithms.isomorphism.categorical_node_match(['kind','label'],[None,None]);fail=[];rows=[]
    for r in primary['all_operator_endpoint_sentinel']['rows']:
        a,b=map(int,r['operator'])
        t1={'H':['C','D'],'edges':[(0,1)],'ops':[(a,b)]}
        t2={'H':['C','D'],'edges':[(1,0)],'ops':[(b,a)]}
        eq=nx.is_isomorphic(_expanded_decorated_graph(t1),_expanded_decorated_graph(t2),node_match=node_match)
        if not eq:fail.append(f'REVERSAL:{a}>{b}')
        distinct=None
        if a!=b:
            tm={'H':['C','D'],'edges':[(0,1)],'ops':[(b,a)]}
            distinct=not nx.is_isomorphic(_expanded_decorated_graph(t1),_expanded_decorated_graph(tm),node_match=node_match)
            if not distinct:fail.append(f'MISMATCH_COLLAPSE:{a}>{b}')
        rows.append({'operator':[a,b],'storage_reversal_equal_networkx':eq,'asymmetric_assignment_distinguished_networkx':distinct})
    o={'schema_id':'IG_G5_R2_INDEPENDENT_OPERATOR_SENTINEL_V1','status':'PASS' if not fail else 'FAIL','method':'NETWORKX_SUBDIVIDED_ENDPOINT_DECORATION_GRAPH_ISOMORPHISM','rows':rows,'failures':fail};o['science_sha256']=canonical_sha256(o);return o

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primary',required=True);ap.add_argument('--r1-primary',required=True);ap.add_argument('--r1-verification',required=True);ap.add_argument('--r1-closeout',required=True);ap.add_argument('--out-dir',required=True);ns=ap.parse_args();out=Path(ns.out_dir);out.mkdir(parents=True,exist_ok=True)
    primary=load(ns.primary);r1p=load(ns.r1_primary);r1v=load(ns.r1_verification);r1c=load(ns.r1_closeout)
    coldroot=Path(tempfile.mkdtemp(prefix='g5_r2_cold_'))
    try:
        with UpliftCampaignEngine(coldroot,requested_workers=1) as eng:
            cold=run_g5_r2_predictive_hidden_read(engine=eng,r1_primary=r1p,r1_verification=r1v,r1_closeout=r1c);cold['source_sha256']=primary.get('source_sha256');cold['source_version']=primary.get('source_version');cold['registry_sha256']=eng.registry['registry_sha256']
        comp=compare_cold_replay(primary,cold);lb=independent_lower_bound(primary);ops=independent_operator_sentinel(primary)
        ok=comp['status']=='PASS' and lb['status']=='PASS' and ops['status']=='PASS'
        report={'schema_id':'IG_G5_R2_INDEPENDENT_VERIFICATION_V1','status':'PASS' if ok else 'FAIL','comparison':comp,'independent_lower_bound_check':lb,'independent_operator_sentinel':ops,'primary_science_sha256':primary.get('science_sha256'),'cold_science_sha256':cold.get('science_sha256'),'authority_effect':'NONE','limitations':['G5_R2_ONLY','NON_PROMOTING','NO_PHYSICAL_GEOMETRY','BOUNDED_FRESH_PANELS']};report['verification_sha256']=canonical_sha256(report)
        write_json_atomic(out/'G5_R2_COLD_RESULT.json',cold);write_json_atomic(out/'G5_R2_COLD_COMPARISON.json',comp);write_json_atomic(out/'G5_R2_INDEPENDENT_LOWER_BOUND_CHECK.json',lb);write_json_atomic(out/'G5_R2_INDEPENDENT_OPERATOR_SENTINEL.json',ops);write_json_atomic(out/'G5_R2_INDEPENDENT_VERIFICATION.json',report)
        print(json.dumps({'status':report['status'],'verification_sha256':report['verification_sha256'],'cold_science_sha256':cold.get('science_sha256')},sort_keys=True));raise SystemExit(0 if ok else 2)
    finally:shutil.rmtree(coldroot,ignore_errors=True)
if __name__=='__main__':main()
