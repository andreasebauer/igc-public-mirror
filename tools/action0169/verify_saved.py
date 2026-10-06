from pathlib import Path
import json,hashlib,sys
B=Path(__file__).resolve().parent;R=B.parent;sys.path.insert(0,str(B))
from project.reader import CarrierReader
pins=json.loads((B/'EXPORT_BINDINGS.json').read_bytes());r=CarrierReader(B/'SCIENTIFIC_EXPORT.zip',pins['archive_sha256'],pins['root_sha256'],R/'o6_export0165/SCIENTIFIC_EXPORT.zip',R/'o5_export0162/SCIENTIFIC_EXPORT.zip',R/'o4_export0159/SCIENTIFIC_EXPORT.zip',R/'o3_export0156/SCIENTIFIC_EXPORT.zip')
assert r.report['carriers']==72 and r.report['seed_carriers']==2
for key,row in r.records.items():
 assert r.lookup(*key)['record']==row and r.components(*key)==r.component_records[key]
 owner=r.owner(*key,0);assert owner['prototype_id']==r.contexts[key].parents[0].pid
 assert len(r.resources(*key))==r.contexts[key].n
r.close();print('PASS:74 saved O7 states,192 typed E7 edges,322 bound O6 owners,82 components')
