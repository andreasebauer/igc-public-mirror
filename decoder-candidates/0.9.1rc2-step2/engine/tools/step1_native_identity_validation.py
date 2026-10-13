#!/usr/bin/env python3
"""Read-only native retained-panel and bounded generated-state validation.

Run in its own process with the pinned runtime. Historical archives are unchanged.
No full scientific census, old capture resume, or release promotion is performed.
"""


def _require_integrity(condition, message=None):
    """Acceptance invariant retained under optimized Python."""
    if not condition:
        if message is None:
            raise AssertionError()
        raise AssertionError(message())

import argparse
from collections import defaultdict
from itertools import combinations_with_replacement
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infinity_grid.o7_c1_typed_baseline import ORIGINAL
from infinity_grid.o7_legacy_library_comparison import ARCHIVE, compare_o7_survivor_topology
from infinity_grid.o7_symmetry_library_comparison import compare_o7_typed_owner_symmetries
from infinity_grid.o7_c1_graph_comparison import compare_c1_outer_graphs
from infinity_grid.o7_edge_canonicalization import canonicalize_with_witness, canonicalize_edges, verify_relabeling_witness, state_identity
from infinity_grid.o6_resource_observer import ResourceMerkle


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);r=out/'native_inputs';r.mkdir(exist_ok=True)
    with zipfile.ZipFile(ORIGINAL) as z:z.extractall(r)
    os.environ['OSCOUT_DATA_ROOT']=str(r.resolve())
    spec=importlib.util.spec_from_file_location('_step1_original',r/'02_CODE/o7_live_engine.py');engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine);engine.O6=engine.import_o6()
    parents=engine.load_parent_records();rng=random.Random(20261001);t0=time.monotonic()
    with zipfile.ZipFile(ARCHIVE) as z:
        survivors=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json'))))['records']
        cases=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/C1_EXACT_EXTERNAL_CLOSURE.json'))))
    selected=json.loads((r/'03_RESULTS/O7_SELECTED_COMPONENT_PROFILES.json').read_text())['records']
    def select(digest,lane):
        rows=[r for r in selected if r['state_digest']==digest and r['lane']==lane]
        _require_integrity(rows)
        _require_integrity(len({(tuple(r['parent_exact_colors']),tuple(tuple(e) for e in r['edges'])) for r in rows})==1)
        return rows[0]
    def skins(ctx,edges):return sorted((p.exact_color,ResourceMerkle(engine.materialize_owner(p.h6,edges,i)).digest) for i,p in enumerate(ctx.parents))
    results=[]
    def check(ctx,raw,kind,index):
        raw=tuple(tuple(e) for e in raw);new=canonicalize_with_witness(ctx,raw);old=engine.canonicalize_edges(ctx,raw)
        _require_integrity(verify_relabeling_witness(ctx,raw,new['edges'],new['witness']))
        _require_integrity(canonicalize_edges(ctx,new['edges'])==new['edges'])
        _require_integrity(skins(ctx,raw)==skins(ctx,new['edges']))
        shuffled=list(raw);rng.shuffle(shuffled)
        # Reverse all endpoint orientations as well as edge occurrence order.
        reversed_edges=tuple((e[7],*e[8:13],e[13],e[0],*e[1:6],e[6]) for e in shuffled)
        _require_integrity(canonicalize_edges(ctx,reversed_edges)==new['edges'])
        identity=state_identity(ctx,raw)
        rec={'kind':kind,'index':index,'lane':ctx.name,'parent_exact_colors':[p.exact_color for p in ctx.parents],
             'legacy_digest':engine.state_digest(old),'new_edge_digest':engine.state_digest(new['edges']),
             'global_identity':identity['object_identity'],
             'identity_schema':identity['identity_schema'],
             'identity_canonicalization_version':identity['identity_canonicalization_version'],
             'legacy_context_identity':identity['legacy_context_identity'],
             'legacy_representative_changed':old!=new['edges'],
             'legacy_materialization_preserved':skins(ctx,raw)==skins(ctx,old),
             'new_materialization_preserved':True,'witness_verified':True,'idempotent':True,'order_orientation_invariant':True}
        results.append(rec)
        return new['edges']
    for i,row in enumerate(survivors):check(engine._profile_row_context(row,parents),row['edges'],'O7_SURVIVOR',i)
    print('Retained O7 survivors checked:',len(survivors),flush=True)
    for kind in ['O7xO6','O7xO7']:
        for i,case in enumerate(cases[kind]):
            if kind=='O7xO6':
                left=select(case['input'],case['lane']);a=engine._profile_row_context(left,parents)
                ctx=engine.LaneContext(case['lane']+'_C1',list(a.parents)+[parents[case['parent']]],list(a.source_order)+[case['parent']]);raw=tuple(tuple(e) for e in left['edges'])
            else:
                left=select(case['A'],case['lanes'][0]);right=select(case['B'],case['lanes'][1]);a=engine._profile_row_context(left,parents);b=engine._profile_row_context(right,parents)
                ctx=engine.LaneContext('+'.join(case['lanes'])+'_C1',list(a.parents)+list(b.parents),list(a.source_order)+list(b.source_order))
                shifted=[engine.make_edge(c+a.n,p,t,d+a.n,q,u) for c,p,t,d,q,u in map(engine.eparts,right['edges'])];raw=tuple(tuple(e) for e in left['edges'])+tuple(shifted)
            action=case['action'];raw+=(engine.make_edge(action['c'],tuple(action['lmin']),action['a'],action['d'],tuple(action['rmin']),action['b']),)
            check(ctx,raw,kind,i)
        print('Retained C1 cases checked:',kind,len(cases[kind]),flush=True)
    twin=parents['O6_TOPOLOGY_TWIN_A'];ctx=engine.LaneContext('TWIN4_BOUNDED',[twin,parents['O6_TOPOLOGY_TWIN_B'],twin,parents['O6_TOPOLOGY_TWIN_B']],['A','B','A','B'])
    paths=[(a,0,0,0,i) for a in range(3) for i in range(2)];target=(0,0,0,0,0);groups=defaultdict(set)
    for i,(p,q) in enumerate(combinations_with_replacement(paths,2)):
        raw=(engine.make_edge(0,p,0,2,target,0),engine.make_edge(0,q,0,2,target,0))
        new=check(ctx,raw,'GENERATED_TWIN4_TWO_EDGE',i);groups[engine.state_digest(engine.canonicalize_edges(ctx,raw))].add(new)
    bounded_collisions=sum(len(v)>1 for v in groups.values())
    _require_integrity(bounded_collisions>0)
    # Lawful swap of identical O5 bags in the native six-edge witness.
    ctx2=engine.LaneContext('TWIN_EQUIVALENCE',[twin,twin],[twin.pid]*2);ps=[(a,0,0,0,0) for a in range(3)];pairs=[(a,b) for a in range(3) for b in range(3) if a!=b];swap={0:1,1:0,2:2}
    raw=tuple(engine.make_edge(0,ps[a],0,1,ps[b],0) for a,b in pairs);alt=tuple(engine.make_edge(0,ps[a],0,1,ps[swap[b]],0) for a,b in pairs)
    _require_integrity(canonicalize_edges(ctx2,raw)==canonicalize_edges(ctx2,alt))
    _require_integrity(engine.canonicalize_edges(ctx2,raw)!=engine.canonicalize_edges(ctx2,alt))
    panels={}
    for name,fn in [('O7_topology',compare_o7_survivor_topology),('O7_symmetry',compare_o7_typed_owner_symmetries),('C1_outer',compare_c1_outer_graphs)]:
        p=fn();_require_integrity(p['status']=='PASS');panels[name]={'status':p['status'],'counts':p['counts']}
    identity_groups=defaultdict(set)
    for row in results:
        if row['kind']=='O7_SURVIVOR':identity_groups[(tuple(row['parent_exact_colors']),row['legacy_digest'])].add(row['global_identity'])
    summary={'schema_id':'IG_STEP1_NATIVE_IDENTITY_VALIDATION_V1','status':'PASS','records':results,'panels':panels,
        'retained_survivors':len(survivors),'retained_C1_cases':sum(len(cases[k]) for k in ['O7xO6','O7xO7']),
        'bounded_generated_TWIN4_states':21,'bounded_legacy_collision_groups':bounded_collisions,
        'retained_legacy_materialization_changes':sum(not r['legacy_materialization_preserved'] for r in results if r['kind']!='GENERATED_TWIN4_TWO_EDGE'),
        'retained_old_key_multiple_new_identity_groups':sum(len(v)>1 for v in identity_groups.values()),
        'retained_representative_changes':sum(r['legacy_representative_changed'] for r in results if r['kind']!='GENERATED_TWIN4_TWO_EDGE'),
        'lawful_six_edge_relabeling_repaired':True,'wall_seconds':time.monotonic()-t0,
        'limitations':['NO_FULL_CENSUS_REGENERATION','NO_PROOF_SAVED_ALGEBRA_INVALID','NO_RECOVERY_OF_PREDEDUP_DISCARDED_STATES','RESOURCE_CHECKS_NOT_FULL_TYPED_FUTURE_REPLAY','SIX_EDGE_WITNESS_OUTSIDE_FROZEN_TWIN4_EDGE_CAP']}
    (out/'NATIVE_IDENTITY_VALIDATION.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['records','limitations']},indent=2),flush=True)

if __name__=='__main__':main()
