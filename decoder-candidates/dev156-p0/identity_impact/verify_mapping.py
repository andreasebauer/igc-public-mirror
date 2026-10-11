"""Check delivered identity tables, namespace partitions and route closure."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from map_identities import require,digest,sha,write

def verify(root):
    reg=json.loads((root/'results/PREREGISTRATION.json').read_text())
    rows=[json.loads(line) for line in (root/'results/IDENTITY_MAPPING.jsonl').read_text().splitlines()]
    require(len(rows)==reg['task_count']==988,'MAPPING_CENSUS')
    require(len({r['record'] for r in rows})==len(rows),'DUPLICATE_RECORD')
    byname={r['record']:r for r in rows}
    require(Counter(r['role'] for r in rows)=={'retained':205,'retained_raw_C1':192,'historical_C1_representative':192,'lawful_relabeling_control':397,'nonisomorphic_collision_control':2},'ROLE_CENSUS')
    for i,r in enumerate(rows):
        checkpoint=json.loads((root/'results/rows'/f'{i:04d}.json').read_text())
        require(digest(r)==checkpoint['row_sha256'] and digest(checkpoint['row'])==digest(r),'CHECKPOINT_TABLE_MISMATCH')
        require(r['witness_verified'] is True and r['new_identity_schema']=='IG_E7_GLOBAL_INCIDENCE_IDENTITY_V3','ROW_ACCEPTANCE')
        require(r['old_scanner_construction']!=r['new_scanner_construction'],'SCANNER_NAMESPACE_CHANGED')
    primary=[r for r in rows if r['role'] in ('retained','retained_raw_C1')]
    false_splits=0
    for r in primary:
        alt=byname[r['record']+':relabeled']
        require(r['new_global_identity']==alt['new_global_identity'],'LAWFUL_EQUIVALENCE')
        false_splits+=r['old_context_identity']!=alt['old_context_identity']
    require(false_splits==395,'EXPECTED_LAWFUL_FALSE_SPLIT_CENSUS')
    stored=json.loads((root/'saved_export_mapping/MAPPING.json').read_text());mapped=stored['rows']
    require(len(mapped)==262 and len({r['record'] for r in mapped})==262,'EXPORT_MAP_CENSUS')
    require(len(stored['occurrence_routes'])==205 and len(stored['component_routes'])==82,'ROUTE_CENSUS')
    require(len({tuple(r['occurrence_key']) for r in stored['occurrence_routes']})==205,'DUPLICATE_OCCURRENCE')
    component={r['historical_reference']:r for r in mapped if r['record'].startswith('EXACT_COMPONENT:')}
    for r in stored['occurrence_routes']:
        require(r['object_id'] in component and component[r['object_id']]['new_global_identity']==r['new_global_identity']==byname[r['retained_panel_record']]['new_global_identity'],'OCCURRENCE_JOIN')
    unavailable=[r for r in stored['occurrence_routes'] if not r['source_root_available']]
    missing_roots={tuple(r['occurrence_key'][:3]) for r in unavailable}
    require(len(unavailable)==89 and len(missing_roots)==68,'MISSING_ROOT_SCOPE')
    c1changes=[r['record'] for r in primary if r['role']=='retained_raw_C1' and r['new_global_identity']!=byname[r['record'][:-3]+'stored']['new_global_identity']]
    require(len(c1changes)==21,'SEPARATE_HISTORICAL_V1_IMPACT')
    merged=primary+mapped
    groups=defaultdict(set);back=defaultdict(set)
    for r in merged:
        groups[r['old_context_identity']].add(r['new_global_identity']);back[r['new_global_identity']].add(r['old_context_identity'])
    require(all(len(v)==1 for v in groups.values()) and all(len(v)==1 for v in back.values()),'CROSS_EXPORT_PANEL_PARTITION_CHANGE')
    return {'schema':'IG_IDENTITY_MAP_TABLE_VERIFICATION_V1','status':'PASS',
        'retained_control_mapping_rows':988,'saved_export_mapping_rows':262,'table_rows_with_overlap':1250,
        'mapped_original_and_raw_record_references_with_overlap':len(merged),'combined_unique_v2':len(groups),'combined_unique_v3':len(back),
        'unavailable_source_root_occurrences':89,'unavailable_distinct_source_roots':68,
        'lawful_relabeling_false_splits_v2':false_splits,'historical_C1_differences':21,
        'verification_scope':'Independent table/route/partition checks; source-bound computations and full-bijection controls retained separately; not independent-host RC'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();result=verify(a.root)
    write(a.root/'TABLE_VERIFICATION.json',result);print(json.dumps(result,indent=2))
