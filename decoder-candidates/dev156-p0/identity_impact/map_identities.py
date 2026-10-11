"""Read-only, resumable mapping of pinned retained E7 records; no migration."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile

def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write(path, value):
    raw=(json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    tmp=path.with_suffix(path.suffix+'.pending')
    with tmp.open('wb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def independent_graph(ctx, edges):
    import networkx as nx
    g=nx.Graph()
    def add(o,p,v,d):
        key=('tree',o,p)
        color=('OWNER',ctx.parents[o].exact_color) if d==0 else ('BAG',d)
        if d==5:color=('SITE',tuple(v[0]),tuple(v[1]))
        g.add_node(key,color=color)
        if d<5:
            for i,ch in enumerate(v):g.add_edge(key,add(o,p+(i,),ch,d+1))
        return key
    for o,p in enumerate(ctx.parents):add(o,(),p.h6,0)
    for i,e in enumerate(edges):
        center=('edge',i);g.add_node(center,color=('EDGE',))
        for side,offset in enumerate((0,7)):
            endpoint=('end',i,side);g.add_node(endpoint,color=('END',e[offset+6]))
            g.add_edge(center,endpoint);g.add_edge(endpoint,('tree',e[offset],tuple(e[offset+1:offset+6])))
    return g

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.source.resolve()))
    require(sys.flags.optimize==0,'OPTIMIZED_RUNTIME_REFUSED')
    from infinity_grid.isolated_runtime import _verify_tree,BUNDLED_MANIFEST,BUNDLED_MANIFEST_SHA256
    require(sha(BUNDLED_MANIFEST)==BUNDLED_MANIFEST_SHA256,'RUNTIME_MANIFEST')
    runtime=Path(sys.prefix).parent
    _verify_tree(runtime,json.loads(BUNDLED_MANIFEST.read_text()))
    from infinity_grid.o7_edge_canonicalization import state_identity,verify_relabeling_witness,_tree
    from infinity_grid.o7_c1_typed_baseline import ORIGINAL,ORIGINAL_SHA256
    from infinity_grid.o7_legacy_library_comparison import ARCHIVE,ARCHIVE_SHA256
    from infinity_grid.regime_scanner import O7State
    from infinity_grid.o7_owner_materialization import materialize_owner
    require(sha(ORIGINAL)==ORIGINAL_SHA256 and sha(ARCHIVE)==ARCHIVE_SHA256,'INPUT_ARCHIVE_DRIFT')
    manifest=json.loads((a.overlay/'SOURCE_MANIFEST.json').read_text())
    def inventory(root):
        return {q.relative_to(root).as_posix():{'bytes':q.stat().st_size,'sha256':sha(q)}
                for q in sorted(root.rglob('*')) if q.is_file() and not any(x in q.parts for x in ('__pycache__','.pytest_cache')) and q.suffix!='.pyc'}
    require(inventory(a.source)==manifest['candidate_files'],'SOURCE_INVENTORY')
    require(inventory(a.baseline)==manifest['baseline_files'],'BASELINE_INVENTORY')
    old=module('infinity_grid._mapping_pinned_v2',a.baseline/'infinity_grid/o7_edge_canonicalization.py')
    fixture=a.output/'native_inputs';fixture.mkdir(exist_ok=True)
    with zipfile.ZipFile(ORIGINAL) as z:
        pins=json.loads(z.read('MANIFEST_SHA256.json'))
        for n,h in pins.items():require(hashlib.sha256(z.read(n)).hexdigest()==h,'FIXTURE_MEMBER_DRIFT')
        z.extractall(fixture)
    os.environ['OSCOUT_DATA_ROOT']=str(fixture.resolve())
    engine=module('_identity_mapping_original',fixture/'02_CODE/o7_live_engine.py');engine.O6=engine.import_o6()
    parents=engine.load_parent_records()
    selected=json.loads((fixture/'03_RESULTS/O7_SELECTED_COMPONENT_PROFILES.json').read_text())['records']
    with zipfile.ZipFile(ARCHIVE) as z:
        sn=next(n for n in z.namelist() if n.endswith('/graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json'))
        cn=next(n for n in z.namelist() if n.endswith('/C1_EXACT_EXTERNAL_CLOSURE.json'))
        survivors=json.loads(z.read(sn))['records'];cases=json.loads(z.read(cn))
    require(len(survivors)==205 and len(selected)==205,'SURVIVOR_CENSUS')
    require(survivors==selected,'SELECTED_AND_SURVIVOR_SNAPSHOT_DIFFER')
    require(sum(len(cases[k]) for k in ('O7xO6','O7xO7'))==192,'C1_CENSUS')
    def select(h,lane):
        rows=[r for r in selected if r['state_digest']==h and r['lane']==lane]
        require(rows and len({digest([r['parent_ids'],r['edges']]) for r in rows})==1,'C1_PARENT_ALIAS_AMBIGUOUS')
        return rows[0]
    tasks=[]
    for i,row in enumerate(survivors):
        ctx=engine._profile_row_context(row,parents);edges=tuple(tuple(e) for e in row['edges'])
        tasks.append((f'O7_SURVIVOR:{i:03d}',ctx,edges,row['state_digest'],'retained'))
    for kind in ('O7xO6','O7xO7'):
        for i,c in enumerate(cases[kind]):
            if kind=='O7xO6':
                left=select(c['input'],c['lane']);ctx0=engine._profile_row_context(left,parents)
                ctx=engine.LaneContext(kind,list(ctx0.parents)+[parents[c['parent']]],list(ctx0.source_order)+[c['parent']])
                edges=tuple(tuple(e) for e in left['edges'])
            else:
                left=select(c['A'],c['lanes'][0]);right=select(c['B'],c['lanes'][1])
                ca=engine._profile_row_context(left,parents);cb=engine._profile_row_context(right,parents)
                ctx=engine.LaneContext(kind,list(ca.parents)+list(cb.parents),list(ca.source_order)+list(cb.source_order))
                edges=tuple(tuple(e) for e in left['edges'])+tuple(engine.make_edge(o+ca.n,x,t,d+ca.n,y,u) for o,x,t,d,y,u in map(engine.eparts,right['edges']))
            ac=c['action'];edges+=(engine.make_edge(ac['c'],tuple(ac['lmin']),ac['a'],ac['d'],tuple(ac['rmin']),ac['b']),)
            legacy=engine.canonicalize_edges(ctx,edges)
            require(engine.state_digest(legacy)==c['state'],'HISTORICAL_C1_ALIAS_MISMATCH')
            tasks.append((f'{kind}:{i:03d}:raw',ctx,edges,c['state'],'retained_raw_C1'))
            tasks.append((f'{kind}:{i:03d}:stored',ctx,legacy,c['state'],'historical_C1_representative'))
    primary=[t for t in tasks if t[4]!='historical_C1_representative']
    for name,ctx,edges,alias,role in primary:
        moved_parents=[replace(q,h6=tuple(reversed(q.h6))) for q in ctx.parents]
        moved=engine.LaneContext(ctx.name,moved_parents,list(ctx.source_order))
        def path(owner,x):return (len(ctx.parents[owner].h6)-1-x[0],*x[1:])
        moved_edges=tuple(engine.make_edge(o,path(o,x),t,d,path(d,y),u) for o,x,t,d,y,u in map(engine.eparts,edges))
        # Independently encode both sides and transport every nested site by the explicit bijection.
        import networkx as nx
        ga=independent_graph(ctx,edges);gb=independent_graph(moved,moved_edges)
        def label(v):
            if v[0]=='tree' and v[2]:return ('tree',v[1],path(v[1],v[2]))
            return v
        transported=nx.relabel_nodes(ga,{v:label(v) for v in ga},copy=True)
        require(dict(transported.nodes(data='color'))==dict(gb.nodes(data='color')) and {frozenset(e) for e in transported.edges}=={frozenset(e) for e in gb.edges},'INDEPENDENT_RELABELING_BIJECTION')
        tasks.append((name+':relabeled',moved,moved_edges,alias,'lawful_relabeling_control'))
    witness=json.loads((a.audit/'audit_outputs/backend_actual_parent_probe_result.json').read_text())
    byid={q.pid:q for q in parents.values()};q=byid[witness['parent_pid']];other=byid[witness['other_parent_pid']]
    bags=list(q.h6);i,j=witness['R6_child_bag_swap'];bags[i],bags[j]=bags[j],bags[i]
    changed=replace(q,h6=tuple(bags));edge=tuple(tuple(e) for e in witness['edge'])
    for suffix,first in [('left',q),('right',changed)]:
        ctx=engine.LaneContext('NATIVE_COLLISION',[first,other],[q.pid,other.pid])
        tasks.append(('NATIVE_COLLISION:'+suffix,ctx,edge,None,'nonisomorphic_collision_control'))
    require(_tree(q.h6)==_tree(changed.h6),'NATIVE_PARENT_BASE')
    require(_tree(materialize_owner(q.h6,edge,0))!=_tree(materialize_owner(changed.h6,edge,0)),'NATIVE_MATERIALIZATION_NEGATIVE')
    prereg={'schema':'IG_RETAINED_V2_V3_MAPPING_REGISTRATION_V1','status':'READ_ONLY_ENGINEERING_MAPPING_NOT_MIGRATION',
        'source_inventory_sha256':digest(manifest['candidate_files']),'baseline_inventory_sha256':digest(manifest['baseline_files']),
        'original_archive_sha256':ORIGINAL_SHA256,'material_archive_sha256':ARCHIVE_SHA256,
        'script_sha256':sha(__file__),'runtime_manifest_sha256':BUNDLED_MANIFEST_SHA256,
        'retained_primary_records':397,'historical_C1_representatives':192,'lawful_relabeling_controls':397,'native_collision_controls':2,
        'task_count':len(tasks),'task_inputs_sha256':digest([[n,[p.h6 for p in ctx.parents],[p.exact_color for p in ctx.parents],e,h,r] for n,ctx,e,h,r in tasks]),
        'workers':1,'memory_budget_bytes':4294967296,'overall_deadline':None,
        'stop':'First unexplained source/input/identity/witness mismatch','historical_rewrites':False,'controller_takeover':False,
        'limitations':['NO_PREDEDUP_DISCARDED_STATE_RECOVERY','NO_FULL_MASTER_OR_Q2_RECONCILIATION','NO_RC_OR_RELEASE_ACCEPTANCE']}
    reg=a.output/'PREREGISTRATION.json'
    if reg.exists():require(json.loads(reg.read_text())==prereg,'REGISTRATION_DRIFT')
    else:write(reg,prereg)
    print('REGISTERED',len(tasks),'identity records',flush=True)
    folder=a.output/'rows';folder.mkdir(exist_ok=True);rows=[];contexts={};start=time.monotonic();reused=0
    for index,(name,ctx,edges,alias,role) in enumerate(tasks):
        pin=digest([prereg,name,[q.h6 for q in ctx.parents],[q.exact_color for q in ctx.parents],edges,alias,role])
        path=folder/f'{index:04d}.json'
        if path.exists():
            saved=json.loads(path.read_text());require(saved['input_binding']==pin,'ROW_RESUME_DRIFT')
            row=saved['row'];require(saved['row_sha256']==digest(row),'ROW_RESUME_CORRUPTION');reused+=1
        else:
            before=old.state_identity(ctx,edges);after=state_identity(ctx,edges)
            require(before['object_identity']==after['legacy_context_identity'],'PINNED_V2_ALIAS_MISMATCH')
            require(before['edges']==after['edges'],'FIXED_CONTEXT_REPRESENTATIVE_DRIFT')
            require(verify_relabeling_witness(ctx,edges,after['edges'],after['witness']),'HIERARCHICAL_WITNESS')
            st=O7State(engine,ctx,edges)
            oldscanner=digest({'schema_id':'IG_O7_SCANNER_CONSTRUCTION_V2','level':7,'base_identity':before['object_identity'],'reserve_counts':[0]*7})
            row={'record':name,'role':role,'historical_state_alias':alias,'old_context_identity':before['object_identity'],
                'new_global_identity':after['object_identity'],'new_identity_schema':after['identity_schema'],
                'parent_ids':[q.pid for q in ctx.parents],'parent_colors':[q.exact_color for q in ctx.parents],
                'parent_exact_coordinates_sha256':digest([q.h6 for q in ctx.parents]),'raw_edges':edges,
                'new_edges_sha256':digest(after['edges']),'witness_sha256':digest(after['witness']),
                'old_scanner_construction':oldscanner,'new_scanner_construction':st.construction_digest,
                'scanner_scope':'ZERO_RESERVATION_ONLY','witness_verified':True}
            write(path,{'input_binding':pin,'row':row,'row_sha256':digest(row)})
        rows.append(row);contexts[name]=(ctx,edges)
        if index%40==0:print('MAPPED',index+1,'/',len(tasks),flush=True)
    byname={r['record']:r for r in rows};relabel_changes=[]
    for name,*_ in primary:
        x,y=byname[name],byname[name+':relabeled']
        require(x['new_global_identity']==y['new_global_identity'],'V3_LAWFUL_RELABELING_SPLIT')
        if x['old_context_identity']!=y['old_context_identity']:relabel_changes.append(name)
    l,r=byname['NATIVE_COLLISION:left'],byname['NATIVE_COLLISION:right']
    require(l['old_context_identity']==r['old_context_identity']==witness['object_identity'],'NATIVE_V2_COLLISION_REPRODUCTION')
    require(l['new_global_identity']!=r['new_global_identity'],'NATIVE_V3_COLLISION_NOT_SEPARATED')
    retained=[r for r in rows if r['role'] in ('retained','retained_raw_C1')]
    def groups(items,key,value):
        d=defaultdict(lambda:defaultdict(list))
        for x in items:d[x[key]][x[value]].append(x['record'])
        return {k:dict(v) for k,v in sorted(d.items()) if len(v)>1}
    splits=groups(retained,'old_context_identity','new_global_identity')
    merges=groups(retained,'new_global_identity','old_context_identity')
    proofs=[]
    from networkx.algorithms.isomorphism import categorical_node_match
    for direction,gg in [('OLD_KEY_SPLIT',splits),('OLD_KEYS_MERGE',merges)]:
        for key,parts in gg.items():
            representatives=[v[0] for v in parts.values()]
            for left in representatives[1:]:
                right=representatives[0];ca,ea=contexts[left];cb,eb=contexts[right]
                slots_a=[(q.exact_color,_tree(q.h6)) for q in ca.parents];slots_b=[(q.exact_color,_tree(q.h6)) for q in cb.parents]
                equivalent=slots_a==slots_b and nx.is_isomorphic(independent_graph(ca,ea),independent_graph(cb,eb),node_match=categorical_node_match('color',None))
                require(equivalent==(direction=='OLD_KEYS_MERGE'),'INDEPENDENT_PARTITION_DISAGREEMENT')
                proofs.append({'direction':direction,'key':key,'left':left,'right':right,'independent_isomorphic':equivalent})
    historical_changes=[]
    for name,ctx,edges,alias,role in tasks:
        if role!='retained_raw_C1':continue
        stored=name[:-3]+'stored'
        if byname[name]['new_global_identity']!=byname[stored]['new_global_identity']:historical_changes.append({'raw':name,'stored':stored,'historical_alias':alias})
    summary={'schema':'IG_RETAINED_V2_V3_MAPPING_RESULT_V1','status':'PASS_DECLARED_RETAINED_MAPPING_ONLY','registration_sha256':sha(reg),
        'mapped_rows':len(rows),'retained_primary_records':len(retained),'historical_C1_representatives':192,
        'retained_unique_v2':len({r['old_context_identity'] for r in retained}),'retained_unique_v3':len({r['new_global_identity'] for r in retained}),
        'retained_old_key_split_groups':len(splits),'retained_new_key_merge_groups':len(merges),
        'lawful_relabeling_false_splits_in_v2':len(relabel_changes),'lawful_relabeling_checks':397,
        'native_collision_v2_same_v3_distinct':True,'C1_raw_vs_historical_representative_distinct_v3':len(historical_changes),
        'affected_identity_namespaces_all_retained_records':397,'historical_V1_impact_is_separate':True,
        'historical_rows_rewritten':0,'scientific_jobs_dispatched':0,'controller_changed':False,
        'reused_completed_rows':reused,'wall_seconds':time.monotonic()-start,'limitations':prereg['limitations']}
    write(a.output/'PARTITION_IMPACT.json',{'retained_splits':splits,'retained_merges':merges,'independent_proofs':proofs,
        'lawful_relabeling_false_splits':relabel_changes,'historical_C1_object_changes':historical_changes,
        'native_collision':{'left':l,'right':r}})
    (a.output/'IDENTITY_MAPPING.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    require(inventory(a.source)==manifest['candidate_files'],'SOURCE_CHANGED_DURING_MAPPING')
    write(a.output/'SUMMARY.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
