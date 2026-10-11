"""Read-only finite recipe and historical DAG dependency binding; no native dispatch."""
import argparse, ast, collections, gzip, hashlib, json, resource, sys, time
from pathlib import Path
from types import SimpleNamespace

def require(ok, label):
    if not ok:
        raise RuntimeError(label)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def digest(obj):
    h=hashlib.sha256()
    for s in json.JSONEncoder(sort_keys=True,separators=(',',':'),ensure_ascii=False).iterencode(obj):h.update(s.encode())
    return h.hexdigest()

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n');tmp.replace(path)

def load(path):return json.loads(path.read_text())

def pure_recipe_function(source):
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='enumerate_candidate_recipes')
    ga=((0,1),(1,2),(1,3),(2,4),(3,5));gb=((0,4),(0,5),(1,3),(2,3),(3,4))
    ns={'rs':SimpleNamespace(GA=ga,GB=gb)}
    module=ast.Module(body=[node],type_ignores=[])
    # Only this reviewed pure function executes; no producer imports or native engine.
    exec(compile(module,str(source),'exec',flags=__import__('__future__').annotations.compiler_flag),ns)
    return ns['enumerate_candidate_recipes']

def run(root,output):
    start=time.monotonic();reg=load(output/'PREREGISTRATION.json')
    for name,pin in reg['inputs'].items():
        p=root/name;require(p.is_file() and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'INPUT_PIN:'+name)
    candidate=root/'decoder/source';manifest=load(root/'recovery/overlay/dev156/SOURCE_MANIFEST.json')
    for name,pin in manifest['candidate_files'].items():
        p=candidate/name;require(p.is_file() and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'SOURCE_PIN:'+name)
    actual={p.relative_to(candidate).as_posix() for p in candidate.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    require(actual==set(manifest['candidate_files']),'SOURCE_FILE_SET')
    seal=root/'recovery/g1';sealed=load(seal/'SEAL_MANIFEST.json')['payload']
    for name,pin in sealed['files'].items():
        p=seal/name;require(sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'HISTORICAL_SEAL:'+name)
    spec=load(candidate/'infinity_grid/resources/decoder/O_REGIME_ADAPTIVE_SCANNER_SPEC_v1.json')
    motifs=load(candidate/'infinity_grid/resources/decoder/O_REGIME_MOTIF_LIBRARY_v1.json')['motifs']
    recipe_fn=pure_recipe_function(candidate/'infinity_grid/maturation_parallel.py')
    recipe_sets={str(t):recipe_fn(motifs,center_supports_twin=t,pairs_count=31) for t in (False,True)}
    require(len(recipe_sets['False'])==191 and len(recipe_sets['True'])==193,'RECIPE_CENSUS')
    hist_source=load(seal/'HISTORICAL_DEPTH100_SOURCE_INPUT.json')
    hist_motifs={r['motif_id'] for r in hist_source['materialized_discovery_evidence']['candidate_cohort']['motif_structural_signatures']}
    require(hist_motifs=={r['motif_id'] for r in recipe_sets['True']},'HISTORICAL_RECIPE_MOTIF_SET')
    write(output/'RECIPE_REGISTER.json',{'schema':'IG_RAW_RECIPE_BOUNDARY_V1','recipes':recipe_sets,'motif_count':len(motifs),'panel':spec['panel'],
      'operator_count':31,'operator_order_binding':'Schedule indexes only; native ordered alphabet must bind before materialization',
      'candidate_cap_active_on_complete_recipe_set':193>spec['panel']['candidate_cap'],
      'earlier_O7_action_stream_cap_unresolved':True,'generation_calls':0})
    print('RECIPE_REGISTER_PASS',flush=True)
    with gzip.open(seal/'BOOTSTRAP100_HISTORICAL.json.gz','rt') as f:bootstrap=json.load(f)
    dag=bootstrap['dag'];nodes=dag['nodes'];roots=dag['roots']
    require(len(nodes)==16528 and len(roots)==193 and len(set(roots))==193,'DAG_CENSUS')
    require(digest({k:v for k,v in dag.items() if k!='science_sha256'})==dag['science_sha256']==sealed['dag_science_sha256'],'DAG_SCIENCE_HASH')
    require(len(bootstrap['beam_roots'])==24 and set(bootstrap['beam_roots'])<=set(roots),'TERMINAL_BEAM')
    retained=[json.loads(l) for l in (root/'recovery/identity/results/IDENTITY_MAPPING.jsonl').read_text().splitlines() if json.loads(l)['role']=='retained']
    aliases=collections.defaultdict(list)
    for r in retained:aliases[r['historical_state_alias']].append(r)
    levels=collections.Counter();bindings={};anchors=set();nonzero=0
    for key,n in nodes.items():
        require(n['construction_digest']==key,'NODE_KEY')
        if n['kind']!='O7':continue
        source=n['source_id'];opts=[r for r in aliases[source] if r['record'].startswith('O7_SURVIVOR:')]
        require(bool(opts),'MISSING_O7_AUTHORITY_ANCHOR:'+source)
        matching=[]
        for r in opts:
            old={'level':7,'parents':r['parent_ids'],'edges':r['raw_edges'],'reserve_counts':n['reserve_counts'],'source':source}
            if digest(old)==key:matching.append(r)
        require(bool(matching),'HISTORICAL_O7_COORDINATE_KEY:'+key)
        require(len({r['new_global_identity'] for r in matching})==1,'AMBIGUOUS_ANCHOR_BASE')
        r=matching[0];nz=any(n['reserve_counts']);nonzero+=nz;anchors.add(source);levels[7]+=1
        bindings[key]={'old_reference':key,'kind':'O7','level':7,'anchor_source_id':source,
          'base_v3_identity':r['new_global_identity'],'reserve_counts':n['reserve_counts'],
          'migration_status':'MATERIALIZED_SKIN_AND_COORDINATE_BASE_REQUIRED' if nz else 'ZERO_OVERLAY_BASE_MAP_AVAILABLE',
          'anchor_record':r['record'],'dependencies':[], 'anchor_sources':[source]}
    print('O7_OVERLAY_BINDINGS_PASS',len(bindings),'nonzero',nonzero,flush=True)
    remaining=set(nodes)-set(bindings)
    while remaining:
        ready=[k for k in sorted(remaining) if all(c in bindings for c in nodes[k].get('children',[]))]
        require(bool(ready),'CYCLE_OR_MISSING_CHILD')
        for key in ready:
            n=nodes[key];require(n['kind']=='LIFT','UNKNOWN_NODE_KIND')
            require(n['children'] and all(bindings[c]['level']==n['level']-1 for c in n['children']),'LEVEL_OR_CHILD_BOUNDARY')
            p={'level':n['level'],'children':n['children'],'edges':n['top_edges_full'],'lane':n['lane'],'motif':n['motif_id']}
            require(digest(p)==key,'HISTORICAL_LIFT_KEY:'+key)
            aset=sorted({a for c in n['children'] for a in bindings[c]['anchor_sources']})
            bindings[key]={'old_reference':key,'kind':'LIFT','level':n['level'],'dependencies':n['children'],
              'exact_witness_sha256':digest(n['top_edges_full']),'anchor_sources':aset,
              'migration_status':'RECURSIVE_CHILD_KEYS_AND_RESOURCE_WITNESSES_REQUIRE_MATCHED_REPLAY','motif_id':n['motif_id'],'lane':n['lane']}
            levels[n['level']]+=1;remaining.remove(key)
        print('DAG_BOUND_LEVEL',max(bindings[k]['level'] for k in ready),'nodes',len(bindings),flush=True)
    reachable=set();stack=list(roots)
    while stack:
        k=stack.pop()
        if k in reachable:continue
        reachable.add(k);stack.extend(bindings[k]['dependencies'])
    require(reachable==set(nodes),'UNREACHABLE_SEALED_NODES')
    terminal=[{'historical_root':k,'motif_id':nodes[k]['motif_id'],'lane':nodes[k]['lane'],
        'anchor_sources':bindings[k]['anchor_sources'],'in_historical_beam':k in bootstrap['beam_roots'],
        'migration_status':bindings[k]['migration_status']} for k in roots]
    require({x['motif_id'] for x in terminal}==hist_motifs,'DAG_ROOT_RECIPE_JOIN')
    write(output/'G1_DEPENDENCY_REGISTER.json',{'schema':'IG_HISTORICAL_G1_DEPENDENCY_V1','roots':terminal,'nodes':bindings,
      'scope':'Historical V1 source and witness keys verified; no V3 overlay or descendant equivalence claim'})
    stored=load(root/'recovery/identity/saved_export_mapping/MAPPING.json');missing=collections.defaultdict(list)
    for r in stored['occurrence_routes']:
        if not r['source_root_available']:missing[tuple(r['occurrence_key'][:3])].append(r)
    require(len(missing)==68 and sum(map(len,missing.values()))==89,'MISSING_ROOT_CENSUS')
    write(output/'MISSING_O7_ROOT_REGISTER.json',{'roots':[{'scope':list(k),'occurrences':v,'status':'WHOLE_ROOT_AND_RAW_PREDEDUP_ANCESTRY_NOT_RECOVERED'} for k,v in sorted(missing.items())]})
    observer=load(root/'recovery/audit/audit_inputs/current_master/continuation0285/PREREGISTRATION.json')
    require(sum(x['case_count'] for x in observer['specifications'])==observer['selected_members']==7058,'OBSERVER_CASE_CENSUS')
    require(sum(x['class_count'] for x in observer['specifications'])==observer['selected_complete_classes']==3523,'OBSERVER_CLASS_CENSUS')
    bridge={'schema':'IG_G1_Q2_OBSERVER_SCOPE_BRIDGE_V1','historical_G1_roots':roots,
      'frozen_observer_members':7058,'frozen_observer_classes':3523,'frozen_observer_chunks':len(observer['specifications']),
      'historical_record_stream_sha256':observer['historical_record_stream_sha256'],
      'historical_signature_stream_sha256':observer['historical_signature_stream_sha256'],
      'matched_member_specifications':observer['specifications'],'exact_member_to_root_routes_recovered_this_step':False,
      'prior_Q2_campaign_separate':True,'new_generation_calls':0,'native_observer_dispatch':False,
      'next':'Recover frozen member specifications/Q2 operational witness routes; match overlay and serializer with historical source; no bare-hash substitutions'}
    write(output/'Q2_OBSERVER_BOUNDARY.json',bridge)
    result={'schema':'IG_P0_AFFECTED_SCOPE_BINDING_V1','status':'PASS_DECLARED_STATIC_AND_HISTORICAL_DEPENDENCY_BINDING',
      'source_files_verified':len(actual),'historical_seal_files_verified':len(sealed['files']),
      'motifs':len(motifs),'raw_recipes_without_twins':191,'raw_recipes_with_twins':193,'candidate_cap':220,'beam':24,
      'historical_DAG_nodes':len(nodes),'historical_roots':len(roots),'O7_overlay_nodes':levels[7],
      'O7_nonzero_overlay_nodes':nonzero,'distinct_O7_authority_anchors':len(anchors),
      'levels':dict(sorted(levels.items())),'unavailable_O7_whole_roots':68,'unavailable_root_occurrences':89,
      'migration':'OPEN','full_P0':'OPEN','candidate_generation':0,'native_runtime_qualification':False,
      'historical_witness_keys_verified':True,'V3_nonzero_overlay_keys_calculated':False,
      'pre_dedup_state_population_closed':False,'wall_seconds':time.monotonic()-start}
    write(output/'SUMMARY.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    run(args.root,args.output)
