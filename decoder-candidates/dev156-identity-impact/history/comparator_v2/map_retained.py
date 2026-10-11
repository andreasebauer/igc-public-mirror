"""Frozen-input identity migration inventory; never rewrites scientific inputs."""
import argparse, hashlib, importlib.util, json, os, resource, sys, zipfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

def digest_bytes(b): return hashlib.sha256(b).hexdigest()
def digest(o): return digest_bytes(json.dumps(o,sort_keys=True,separators=(',',':')).encode())
def require(ok, text):
    if not ok: raise RuntimeError(text)
def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def groups(rows, before, after):
    a=defaultdict(set); b=defaultdict(set)
    for r in rows: a[r[before]].add(r[after]); b[r[after]].add(r[before])
    return {'old_keys':len(a),'new_keys':len(b),
        'splits':[{'old':k,'new':sorted(v)} for k,v in sorted(a.items()) if len(v)>1],
        'merges':[{'new':k,'old':sorted(v)} for k,v in sorted(b.items()) if len(v)>1]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',type=Path,required=True);a=ap.parse_args()
    reg=json.loads(a.registration.read_text()); base=a.registration.parent
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    for row in reg['inputs']:
        p=Path(row['path']);require(p.stat().st_size==row['bytes'] and digest_bytes(p.read_bytes())==row['sha256'],'INPUT_MISMATCH '+str(p))
    require(digest_bytes(Path(__file__).read_bytes())==reg['script_sha256'],'SCRIPT_MISMATCH')
    source=Path(reg['candidate_source']);sys.path.insert(0,str(source))
    from infinity_grid.o7_edge_canonicalization import state_identity,verify_relabeling_witness,_tree
    from infinity_grid.o7_owner_materialization import materialize_owner
    from infinity_grid.o6_resource_observer import ResourceMerkle
    old=load_module('infinity_grid._frozen_v2',reg['v2_module'])
    native=base/'native_inputs';native.mkdir(exist_ok=True)
    with zipfile.ZipFile(reg['original_archive']) as z:
        for n in z.namelist():
            require(not Path(n).is_absolute() and '..' not in Path(n).parts,'UNSAFE_ARCHIVE')
        z.extractall(native)
    os.environ['OSCOUT_DATA_ROOT']=str(native.resolve())
    engine=load_module('_identity_map_frozen_engine',native/'02_CODE/o7_live_engine.py');engine.O6=engine.import_o6()
    parents=engine.load_parent_records()
    selected=json.loads((native/'03_RESULTS/O7_SELECTED_COMPONENT_PROFILES.json').read_text())['records']
    with zipfile.ZipFile(reg['material_archive']) as z:
        survivors=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json'))))['records']
        cases=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/C1_EXACT_EXTERNAL_CLOSURE.json'))))
    rows=[];objects={};witnesses=[]
    def projection(ctx,edges):
        return sorted((p.exact_color,digest(_tree(p.h6)),ResourceMerkle(materialize_owner(p.h6,edges,i)).digest) for i,p in enumerate(ctx.parents))
    def check(ctx, raw, kind, index, historical_ref=None):
        edges=tuple(tuple(e) for e in raw)
        obj={'parents':[{'exact_color':p.exact_color,'h6':p.h6} for p in ctx.parents],'edges':edges}
        object_sha=digest(obj); objects.setdefault(object_sha,obj)
        v2=old.state_identity(ctx,edges);v3=state_identity(ctx,edges)
        require(v2['object_identity']==v3['legacy_context_identity'],'V2_ALIAS_MISMATCH')
        require(v2['edges']==v3['edges'],'FIXED_CONTEXT_CHANGED')
        require(verify_relabeling_witness(ctx,edges,v3['edges'],v3['witness']),'WITNESS_MISMATCH')
        require(projection(ctx,edges)==projection(ctx,v3['edges']),'RESOURCE_PROJECTION_MISMATCH')
        old_edges=engine.canonicalize_edges(ctx,edges)
        context_sha=digest(v2['identity_payload']['parent_slots'])
        v1_edge=engine.state_digest(old_edges)
        if historical_ref is not None:
            require(v1_edge==historical_ref,'FROZEN_V1_REFERENCE_MISMATCH '+kind+str(index))
        row={'kind':kind,'index':index,'object_sha256':object_sha,'lane':getattr(ctx,'name',None),
            'context_sha256':context_sha,'historical_v1_edge_digest':v1_edge,
            'historical_v1_context_key':digest([context_sha,v1_edge]),
            'fixed_context_v2_edge_digest':engine.state_digest(v2['edges']),
            'v2_global_alias':v2['object_identity'],'v3_global_identity':v3['object_identity'],
            'v1_materialization_preserved':projection(ctx,edges)==projection(ctx,old_edges),
            'v3_resource_projection_verified':True,'hierarchical_witness_verified':True}
        rows.append(row); return row
    for kind,records in [('SELECTED_PROFILE',selected),('IMMUTABLE_SURVIVOR',survivors)]:
        for i,r in enumerate(records):check(engine._profile_row_context(r,parents),r['edges'],kind,i,r['state_digest'])
        print(kind,len(records),flush=True)
    def select(d,lane):
        found=[r for r in selected if r['state_digest']==d and r['lane']==lane]
        require(bool(found),'MISSING_C1_INPUT')
        require(len({digest([r['parent_exact_colors'],r['edges']]) for r in found})==1,'AMBIGUOUS_C1_INPUT')
        return found[0]
    for kind in ['O7xO6','O7xO7']:
        for i,c in enumerate(cases[kind]):
            if kind=='O7xO6':
                r=select(c['input'],c['lane']);x=engine._profile_row_context(r,parents)
                ctx=engine.LaneContext(c['lane']+'_C1',list(x.parents)+[parents[c['parent']]],list(x.source_order)+[c['parent']]);edges=tuple(tuple(e) for e in r['edges'])
            else:
                r=select(c['A'],c['lanes'][0]);s=select(c['B'],c['lanes'][1]);x=engine._profile_row_context(r,parents);y=engine._profile_row_context(s,parents)
                ctx=engine.LaneContext('+'.join(c['lanes'])+'_C1',list(x.parents)+list(y.parents),list(x.source_order)+list(y.source_order))
                edges=tuple(tuple(e) for e in r['edges'])+tuple(engine.make_edge(o+x.n,p,t,d+x.n,q,u) for o,p,t,d,q,u in map(engine.eparts,s['edges']))
            act=c['action'];edges+=(engine.make_edge(act['c'],tuple(act['lmin']),act['a'],act['d'],tuple(act['rmin']),act['b']),)
            check(ctx,edges,kind,i)
        print(kind,len(cases[kind]),flush=True)
    # Exact previously retained false-merge input, plus its lawful relabeling.
    probe=json.loads(Path(reg['synthetic_probe']).read_text())['nested_coordinates_false_merge']
    control=[]
    for i,o in enumerate(probe['objects']):
        ctx=SimpleNamespace(n=len(o['parents']),parents=[SimpleNamespace(**p) for p in o['parents']])
        control.append(check(ctx,o['edges'],'SYNTHETIC_FALSE_MERGE',i))
    require(control[0]['v2_global_alias']==control[1]['v2_global_alias'] and control[0]['v3_global_identity']!=control[1]['v3_global_identity'],'FALSE_MERGE_NOT_SEPARATED')
    o=probe['objects'][1]; moved=[list(e) for e in o['edges']];moved[0][1]=1
    ctx=SimpleNamespace(n=2,parents=[SimpleNamespace(**p) for p in o['parents']]);lawful=check(ctx,moved,'SYNTHETIC_LAWFUL_RELABEL',0)
    require(lawful['v3_global_identity']==control[0]['v3_global_identity'],'RELABEL_MISMATCH')
    native_probe=json.loads(Path(reg['native_probe']).read_text()); bypid={p.pid:p for p in parents.values()}
    p=bypid[native_probe['parent_pid']];q=bypid[native_probe['other_parent_pid']];bags=list(p.h6);i,j=native_probe['R6_child_bag_swap'];bags[i],bags[j]=bags[j],bags[i]
    native_control=[]
    for k,parent in enumerate([p,replace(p,h6=tuple(bags))]):
        ctx=engine.LaneContext('NATIVE_COLLISION_CONTROL',[parent,q],[p.pid,q.pid]);native_control.append(check(ctx,native_probe['edge'],'NATIVE_FALSE_MERGE',k))
    require(native_control[0]['v2_global_alias']==native_probe['object_identity']==native_control[1]['v2_global_alias'],'NATIVE_OLD_WITNESS_MISMATCH')
    require(native_control[0]['v3_global_identity']!=native_control[1]['v3_global_identity'],'NATIVE_FALSE_MERGE_NOT_SEPARATED')
    retained=[r for r in rows if 'SYNTHETIC' not in r['kind'] and 'NATIVE_FALSE' not in r['kind']]
    require(len(retained)==reg['expected_retained_occurrences'],'SCOPE_COUNT_MISMATCH')
    # Independent typed incidence graph, including ordered owner contract.
    import networkx as nx
    def graph(obj):
        g=nx.Graph()
        def tree(owner,path,value,depth):
            key=('tree',owner,path); color=('owner',owner,obj['parents'][owner]['exact_color']) if depth==0 else ('bag',depth)
            if depth==5:color=('site',tuple(value[0]),tuple(value[1]))
            g.add_node(key,color=color)
            if depth<5:
                for i,ch in enumerate(value):g.add_edge(key,tree(owner,path+(i,),ch,depth+1))
            return key
        for i,p in enumerate(obj['parents']):tree(i,(),p['h6'],0)
        for i,e in enumerate(obj['edges']):
            center=('edge',i);g.add_node(center,color=('edge',))
            for side,offset in enumerate((0,7)):
                end=('end',i,side);g.add_node(end,color=('end',e[offset+6]));g.add_edge(center,end);g.add_edge(end,('tree',e[offset],tuple(e[offset+1:offset+6])))
        return g
    def independent(a,b,expected,label):
        iso=nx.is_isomorphic(graph(objects[a['object_sha256']]),graph(objects[b['object_sha256']]),node_match=nx.algorithms.isomorphism.categorical_node_match('color',None))
        require(iso==expected,'INDEPENDENT_ISOMORPHISM_MISMATCH '+label)
        witnesses.append({'label':label,'left_object':a['object_sha256'],'right_object':b['object_sha256'],'independent_isomorphic':iso})
    independent(control[0],control[1],False,'synthetic false merge')
    independent(control[0],lawful,True,'lawful nested relabeling')
    independent(native_control[0],native_control[1],False,'native false merge')
    result={'schema':'IG_IDENTITY_IMPACT_RETAINED_V1','status':'RETAINED_PANEL_MAPPED_NOT_GLOBAL_MIGRATION_ACCEPTANCE',
        'registration_sha256':digest_bytes(a.registration.read_bytes()),'records':rows,
        'retained_occurrences':len(retained),'retained_unique_objects':len({r['object_sha256'] for r in retained}),
        'historical_v1_to_v2_register':groups(retained,'historical_v1_context_key','v2_global_alias'),
        'global_v2_to_v3_register':groups(retained,'v2_global_alias','v3_global_identity'),
        'controls_v2_to_v3_register':groups(rows[len(retained):],'v2_global_alias','v3_global_identity'),
        'independent_witnesses':witnesses,'retained_v1_projection_changes':sum(not r['v1_materialization_preserved'] for r in retained),
        'global_population_complete':False,'migration_applied':False,'new_scientific_generation':0,'full_p0_accepted':False}
    (base/'results/IDENTITY_MAP.json').write_text(json.dumps(result,indent=2)+'\n')
    with (base/'results/OBJECTS.jsonl').open('w') as f:
        for sha,o in sorted(objects.items()):f.write(json.dumps({'sha256':sha,'object':o},separators=(',',':'))+'\n')
    for r in reg['inputs']:require(digest_bytes(Path(r['path']).read_bytes())==r['sha256'],'INPUT_MUTATED')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
if __name__=='__main__':main()
