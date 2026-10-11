"""Bind catalog-pinned saved O7 exports to the V2/V3 mapping without rewriting."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from map_identities import require,sha,digest,write,module

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.source.resolve()))
    require(sys.flags.optimize==0,'OPTIMIZED_RUNTIME')
    from infinity_grid.o7_edge_canonicalization import state_identity,verify_relabeling_witness
    old=module('infinity_grid._saved_export_v2',a.baseline/'infinity_grid/o7_edge_canonicalization.py')
    engine=module('_saved_export_original',a.inputs/'native_inputs/02_CODE/o7_live_engine.py')
    os.environ['OSCOUT_DATA_ROOT']=str((a.inputs/'native_inputs').resolve());engine.O6=engine.import_o6()
    parents=engine.load_parent_records();byid={v.pid:v for v in parents.values()}
    catalog=json.loads(a.catalog.read_text());contents=[];roots=[]
    for i in (0,1):
        archive=a.inputs/f'saved_o7_{i}.zip';root=a.inputs/f'saved_o7_{i}'
        pin=next(s for s in catalog['slices'] if s.get('archive',{}).get('sha256')==sha(archive))
        require(archive.stat().st_size==pin['archive']['size_bytes'],'SAVED_EXPORT_ARCHIVE_SIZE')
        require(sha(root/'ROOT.json')==pin['scientific_root_sha256'],'SAVED_EXPORT_ROOT_BINDING')
        r=json.loads((root/'ROOT.json').read_text());data={}
        for item in r['content']:
            q=root/'content'/f"{item['sha256']}.blob"
            require(q.stat().st_size==item['bytes'] and sha(q)==item['sha256'],'SAVED_EXPORT_BLOB')
            try:data[item['name']]=json.loads(q.read_text())
            except (UnicodeDecodeError,json.JSONDecodeError):data[item['name']]=None
        contents.append(data);roots.append(r)
    d0,d1=contents
    require(len(d0['states'])==74 and len(d0['components'])==82 and len(d1['roots'])==24 and len(d1['objects'])==164 and len(d1['occurrences'])==205,'SAVED_EXPORT_CENSUS')
    require(digest(d0['selected'])==digest(json.loads(Path(engine.PROTO_PATH).read_text())),'EXACT_SELECTED_PROTOTYPES')
    require(digest(d0['twins'])==digest(json.loads(Path(engine.TWINS_PATH).read_text())),'EXACT_TWIN_PROTOTYPES')
    # Minimal C1 fixture deliberately omits lane specification. Bind the exact
    # saved specification blob through the catalog root, without synthesizing it.
    specs=d0['spec']['frozen']['homogeneous_heterogeneous_lane_sizes']['lanes']
    lanes={}
    for name,spec in specs.items():
        ids=spec.get('prototype_ids') or [spec['prototype_id']]*spec['owner_count']
        lanes[name]=(engine.LaneContext(name,[byid[k] for k in ids],ids),spec)
    for q in d0['bindings']:
        # O5_bindings are inputs before O6 relations consume resources. The
        # stored ordered O6 digest binds the resulting parent used by E7.
        parent=byid[q['prototype_id']]
        require(digest(parent.h6)==q['ordered_resource_sha256'] and parent.exact_color==q['exact_representative_sha256'],'ADMITTED_ORDERED_O6_RESOURCE_BINDING')
    for pid,o in d1['owners'].items():
        if 'binding' in o:
            q=o['binding'];parent=byid[pid]
            require(digest(parent.h6)==q['ordered_resource_sha256'] and parent.exact_color==q['exact_representative_sha256'],'ADDITIONAL_O6_RESOURCE_BINDING')
        else:
            h=engine.nested_h6(o['literal_root']['ordered_resource_O5_carriers'])
            require(h==byid[pid].h6,'EXTERNAL_TWIN_RESOURCE_BINDING')
    out=a.inputs/'saved_export_mapping';out.mkdir(exist_ok=True)
    registration={'schema':'IG_SAVED_O7_V2_V3_MAP_REGISTRATION_V1','script_sha256':sha(__file__),
        'catalog_sha256':sha(a.catalog),'archives':[sha(a.inputs/f'saved_o7_{i}.zip') for i in (0,1)],
        'whole_roots':98,'admitted_component_routes':82,'exact_component_objects':164,'component_occurrences':205,
        'boundary':'Read-only available admitted saved scope; no discarded-state/fresh-population/whole-G2 claim'}
    reg=out/'PREREGISTRATION.json'
    if reg.exists():require(json.loads(reg.read_text())==registration,'EXPORT_REGISTRATION_DRIFT')
    else:write(reg,registration)
    rows=[]
    def mapped(name,ctx,edges,historical,literal):
        edges=tuple(tuple(e) for e in edges);before=old.state_identity(ctx,edges);after=state_identity(ctx,edges)
        require(before['object_identity']==after['legacy_context_identity'],'EXPORT_V2_ALIAS')
        require(verify_relabeling_witness(ctx,edges,after['edges'],after['witness']),'EXPORT_WITNESS')
        row={'record':name,'historical_reference':historical,'literal_payload_sha256':literal,
            'parent_ids':[q.pid for q in ctx.parents],'parent_exact_colors':[q.exact_color for q in ctx.parents],
            'parent_coordinates_sha256':digest([q.h6 for q in ctx.parents]),'edges':edges,
            'old_context_identity':before['object_identity'],'new_global_identity':after['object_identity'],
            'identity_schema':after['identity_schema'],'witness_verified':True}
        rows.append(row);return row
    for i,r in enumerate(d0['states']):
        require(digest(r['record'])==r['payload_sha256'],'SAVED_STATE_LITERAL_HASH')
        ctx=lanes[r['lane']][0];edges=r['record']['edges'];require(engine.state_digest(edges)==r['digest'],'SAVED_ROOT_E7_DIGEST')
        mapped(f'SAVED_ROOT:{i:03d}',ctx,edges,[r['lane'],r['rank'],r['digest']],r['payload_sha256'])
    for i,r in enumerate(d1['roots']):
        require(digest(r['record'])==r['literal_payload_sha256'],'ADDITIONAL_ROOT_LITERAL_HASH')
        ctx=engine.LaneContext(r['root_key'][0],[byid[k] for k in r['parent_ids']],r['parent_ids'])
        require([q.exact_color for q in ctx.parents]==r['parent_exact_colors'],'ADDITIONAL_ROOT_COLORS')
        require(engine.state_digest(r['record']['edges'])==r['root_key'][2],'ADDITIONAL_ROOT_E7')
        mapped(f'ADDITIONAL_ROOT:{i:03d}',ctx,r['record']['edges'],r['root_key'],r['literal_payload_sha256'])
    objectmap={}
    for i,r in enumerate(d1['objects']):
        payload=r['payload'];identity=payload['identity'];require(digest(payload)==r['payload_sha256'] and digest(identity)==r['object_id'],'EXACT_COMPONENT_IDENTITY_HASH')
        ctx=engine.LaneContext('SAVED_COMPONENT',[byid[k] for k in identity['parent_ids']],identity['parent_ids'])
        require([q.exact_color for q in ctx.parents]==identity['parent_exact_colors'],'SAVED_COMPONENT_COLORS')
        require(engine.state_digest(payload['edges'])==identity['E7_digest'],'SAVED_COMPONENT_E7')
        objectmap[r['object_id']]=mapped(f'EXACT_COMPONENT:{i:03d}',ctx,payload['edges'],r['object_id'],r['payload_sha256'])
    routes=[]
    panel=[json.loads(line) for line in (a.inputs/'results/IDENTITY_MAPPING.jsonl').read_text().splitlines()]
    panel={int(r['record'].split(':')[1]):r for r in panel if r['role']=='retained'}
    for r in d1['occurrences']:
        require(digest(r['record'])==r['literal_payload_sha256'],'OCCURRENCE_LITERAL_HASH')
        mappedrow=objectmap[r['object_id']];prior=panel[r['record_index']]
        require(mappedrow['new_global_identity']==prior['new_global_identity'] and mappedrow['old_context_identity']==prior['old_context_identity'],'OCCURRENCE_PANEL_MAP')
        routes.append({'occurrence_key':r['occurrence_key'],'object_id':r['object_id'],'new_global_identity':mappedrow['new_global_identity'],
            'retained_panel_record':prior['record'],'source_root_available':r['source_root_available']})
    componentkeys={(tuple(r['parent_ids']),digest(r['edges'])):r for r in rows if r['record'].startswith('EXACT_COMPONENT:')}
    componentroutes=[]
    for r in d0['components']:
        key=(tuple(r['prototype_ids']),digest(r['edges']));require(key in componentkeys,'SAVED_COMPONENT_ROUTE_MISSING')
        componentroutes.append({'state_key':r['state_key'],'source_owners':r['source_owners'],
            'exact_component_record':componentkeys[key]['record'],'new_global_identity':componentkeys[key]['new_global_identity']})
    def partition(key,value):
        groups=defaultdict(lambda:defaultdict(list))
        for r in rows:groups[r[key]][r[value]].append(r['record'])
        return {k:dict(v) for k,v in groups.items() if len(v)>1}
    splits=partition('old_context_identity','new_global_identity');merges=partition('new_global_identity','old_context_identity')
    summary={'schema':'IG_AVAILABLE_SAVED_O7_MAP_V1','status':'PASS_ALL_AVAILABLE_SAVED_O7_EXPORT_RECORDS',
        'whole_roots':98,'exact_component_objects':164,'mapped_records':len(rows),'component_occurrence_routes':205,
        'component_source_routes':82,'missing_source_root_references':sum(not r['source_root_available'] for r in routes),
        'unique_v2':len({r['old_context_identity'] for r in rows}),'unique_v3':len({r['new_global_identity'] for r in rows}),
        'v2_split_groups':len(splits),'v3_merge_groups':len(merges),'historical_rewrites':0,'new_scientific_jobs':0,
        'limitations':['NO_PREDEDUP_POPULATION_RECOVERY','NO_COMPLETE_SOURCE_OWNER_MAPPING','NO_G1_G2_MIGRATION']}
    write(out/'MAPPING.json',{'rows':rows,'occurrence_routes':routes,'component_routes':componentroutes,'splits':splits,'merges':merges})
    write(out/'SUMMARY.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
