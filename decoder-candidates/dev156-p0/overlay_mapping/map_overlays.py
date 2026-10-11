"""Frozen 193-state O7 overlay mapping with independent literal-tree reference."""
import collections, hashlib, importlib.util, json, os, resource, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'decoder/source'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def require(x,label):
    if not x:raise RuntimeError(label)
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp')
    with q.open('w') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    q.replace(p)
def tree_sig(x,depth=0):
    if depth==5:
        b=b'SITE|'+json.dumps([list(x[0]),list(x[1])],separators=(',',':')).encode()
    else:b=(('R6','O5','O4','O3','O2')[depth]+'|'+'|'.join(sorted(tree_sig(c,depth+1) for c in x))).encode()
    return hashlib.sha256(b).hexdigest()
def sites(x,path=()):
    if len(path)==5:yield path,x
    else:
        for i,c in enumerate(x):yield from sites(c,path+(i,))
def replace_tree(x,leaves,path=()):
    if len(path)==5:return leaves[path]
    return tuple(replace_tree(c,leaves,path+(i,)) for i,c in enumerate(x))
def reference(ctx,edges,counts):
    used=collections.Counter()
    for e in edges:
        used[(e[0],tuple(e[1:6]),e[6])]+=1;used[(e[7],tuple(e[8:13]),e[13])]+=1
    owners=[];base=[];route=[]
    for oi,p in enumerate(ctx.parents):
        leaves={}
        for path,(ports,free) in sites(p.h6):
            f=[int(free[t])-used[(oi,path,t)] for t in range(7)]
            require(min(f)>=0,'REFERENCE_EDGE_UNDERFLOW')
            leaves[path]=(tuple(ports),f)
        base.append(replace_tree(p.h6,leaves));owners.append(leaves)
    for t,cnt in enumerate(counts):
        rem=cnt
        for oi,leaves in enumerate(owners):
            for path,(ports,f) in sorted(leaves.items()):
                take=min(rem,f[t])
                if take:f[t]-=take;rem-=take;route.append([oi,list(path),t,take])
                if not rem:break
            if not rem:break
        require(rem==0,'REFERENCE_OVERLAY_UNDERFLOW')
    material=[replace_tree(p.h6,leaves) for p,leaves in zip(ctx.parents,owners)]
    caps=[tuple(sum(f[t] for _,f in leaves.values()) for t in range(7)) for leaves in owners]
    skin=hashlib.sha256(('R7|'+'|'.join(sorted(tree_sig(x) for x in material))).encode()).hexdigest()
    return base,material,caps,skin,route

def main():
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    require(shutil.disk_usage(ROOT).free>=10*1024**3,'FREE_SPACE_FLOOR')
    out=ROOT/'overlay_step/results';reg=load(out/'PREREGISTRATION.json')
    for n,pin in reg['inputs'].items():require(sha(ROOT/n)==pin['sha256'],'INPUT_PIN:'+n)
    require(sha(Path(__file__))==reg['script_sha256'],'SCRIPT_PIN')
    manifest=load(ROOT/'recovery/overlay/dev156/SOURCE_MANIFEST.json')['candidate_files']
    source=ROOT/'decoder/source'
    for n,p in manifest.items():require((source/n).stat().st_size==p['bytes'] and sha(source/n)==p['sha256'],'SOURCE_PIN:'+n)
    require({p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}==set(manifest),'SOURCE_FILE_SET')
    from infinity_grid.isolated_runtime import verified_runtime
    runtime=verified_runtime()
    targets=load(ROOT/'scope_step/results/NEXT_OVERLAY_GATE.json')['targets'];require(len(targets)==193,'TARGET_COUNT')
    retained={r['record']:r for r in map(json.loads,(ROOT/'recovery/identity/results/IDENTITY_MAPPING.jsonl').read_text().splitlines()) if r['role']=='retained'}
    complete=[];missing=[]
    for t in targets:
        p=out/'rows'/f"{t['old_reference']}.json"
        if p.exists():
            r=load(p);require(r['target']==t and r['preregistration_sha256']==sha(out/'PREREGISTRATION.json') and digest(r['result'])==r['result_sha256'],'CHECKPOINT_BINDING');complete.append(r)
        else:missing.append(t)
    if missing:
        from infinity_grid.frontier import _load_engine
        from infinity_grid.regime_scanner import O7State
        from infinity_grid.o7_edge_canonicalization import state_identity
        kernel=next((ROOT/'overlay_step/kernel').iterdir());engine=_load_engine(kernel);parents=engine.load_parent_records()
        records=load(ROOT/reg['immutable_records'])['records'];contexts={}
        spec=importlib.util.spec_from_file_location('sealed_historical_scanner',ROOT/'recovery/g1/decoder/regime_scanner.py');hist=importlib.util.module_from_spec(spec);sys.modules[spec.name]=hist;spec.loader.exec_module(hist)
        for name in sorted({t['anchor_record'] for t in missing}):
            row=records[int(name.split(':')[1])];old=retained[name]
            require(row['edges']==old['raw_edges'] and row['parent_ids']==old['parent_ids'],'ANCHOR_COORDINATES')
            ctx=engine._profile_row_context(row,parents);edges=tuple(map(tuple,row['edges']));identity=state_identity(ctx,edges)
            require(identity['object_identity']==old['new_global_identity'],'ANCHOR_GLOBAL_IDENTITY')
            contexts[name]=(ctx,edges,identity)
        print('ANCHORS_VERIFIED',len(contexts),flush=True)
        for i,t in enumerate(missing):
            ctx,edges,identity=contexts[t['anchor_record']];counts=tuple(t['reserve_counts']);require(any(counts),'NONZERO_REQUIRED')
            state=O7State(engine,ctx,edges,counts,t['anchor_source_id']);historical=hist.O7State(engine,ctx,edges,counts,t['anchor_source_id'])
            require(historical.construction_digest==t['old_reference'],'HISTORICAL_CONSTRUCTION')
            require(identity['object_identity']==t['base_v3_identity'],'FROZEN_BASE_IDENTITY')
            base,material,caps,skin,route=reference(ctx,edges,counts)
            # base trees captured before overlays share mutable leaf lists; reconstruct zero-overlay separately.
            base=reference(ctx,edges,(0,)*7)[1]
            require(digest(base)==digest([x[0] for x in state.base_data]),'LITERAL_BASE_RESOURCE')
            require(state.skin==historical.skin==skin,'RESOURCE_SKIN')
            require(state.owner_caps==historical.owner_caps==caps,'OWNER_CAPACITIES')
            payload={'schema_id':'IG_O7_SCANNER_CONSTRUCTION_V3','level':7,'base_identity_schema':identity['identity_schema'],'base_identity':identity['object_identity'],'reserve_counts':list(counts),'materialized_resource_skin':skin,'coordinate_base_key':state.base_key}
            new=state.construction_digest;require(new==digest(payload),'V3_CONSTRUCTION')
            result={'new_construction':new,'v3_payload':payload,'skin':skin,'owner_caps':caps,'total_caps':list(state.total_caps),'reservation_route':route,'materialized_literal_tree_sha256':digest(material),'base_literal_tree_sha256':digest(base),'historical_construction_verified':True,'historical_resource_verified':True,'independent_tree_verified':True}
            r={'target':t,'preregistration_sha256':sha(out/'PREREGISTRATION.json'),'result':result,'result_sha256':digest(result)}
            write(out/'rows'/f"{t['old_reference']}.json",r);complete.append(r)
            print('OVERLAY_PASS',i+1,len(missing),t['old_reference'],flush=True)
    complete.sort(key=lambda r:r['target']['old_reference'])
    require(len(complete)==193 and len({r['target']['old_reference'] for r in complete})==193,'MAPPING_CENSUS')
    write(out/'MAPPING.json',complete)
    write(out/'SUMMARY.json',{'status':'PASS','targets':193,'anchors':len({r['target']['anchor_record'] for r in complete}),'distinct_v3_constructions':len({r['result']['new_construction'] for r in complete}),'executed':len(missing),'reused':193-len(missing),'runtime':str(runtime),'historical_references_preserved':True,'native_resource_comparisons':len(missing),'independent_host_qualification':False,'full_p0_closed':False})
    print('SUMMARY',load(out/'SUMMARY.json'),flush=True)
if __name__=='__main__':main()
