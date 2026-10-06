"""Registered export-only selection of exact qualified scientific bytes."""
from pathlib import Path
import json,hashlib,zipfile,io
from infinity_grid.canon import canonical_bytes
from infinity_grid.v05_chain import ChainExecutionResult
from .reader import RecursiveReader

def handler(stage,runtime):
 inputs=stage['input_artifacts'];params=stage['execution']['parameters'];root_raw=Path(inputs['qualified_view']).read_bytes()
 if hashlib.sha256(root_raw).hexdigest()!=params['qualified_view_sha256']:raise ValueError('VIEW_HASH')
 view=json.loads(root_raw);locations={};bundle=b''.join(Path(inputs['saved_contents_part'+str(i).zfill(2)]).read_bytes() for i in range(params['saved_contents_part_count']))
 if hashlib.sha256(bundle).hexdigest()!=params['saved_contents_bundle_sha256']:raise ValueError('BUNDLE_HASH')
 expected={view['primitive_packet_sha256'],view['parent_packet_sha256']}|{r['sha256'] for s in view['sources'] for r in s['shards']}
 with zipfile.ZipFile(io.BytesIO(bundle)) as z:
  if len(z.namelist())!=len(expected) or set(z.namelist())!={h+'.bin' for h in expected}:raise ValueError('BUNDLE_CLOSURE')
  for h in expected:locations[h]=z.read(h+'.bin')
 r=RecursiveReader(inputs['qualified_view'],params['qualified_view_sha256'],locations)
 if r.report['all_rows_and_dependency_fields_checked']!=199579:raise ValueError('EXPORT_COVERAGE')
 scientific={'schema_id':'IG_RECURSIVE_SCIENTIFIC_EXPORT_V1','dataset_id':'NODE_NATIVE_ROOTED_DEPTH1_DEPTH2_V1','scope':{'depths':[1,2],'native_events':13,'depth1_constructions':1391,'depth2_constructions':198188,'selection':'EXHAUSTIVE_LAWFUL_ENDPOINTS_WITHIN_FROZEN_13_EVENT_ALPHABET'},'identity':'ROOTED_CONSTRUCTION_AND_FORMATION_IDS_WITH_EXPLICIT_PROJECTION_PREIMAGES','qualified_view_sha256':params['qualified_view_sha256'],'scientific_members':[],'dependencies':{'admitted_j3_catalog_sha256':view['catalog_sha256'],'foundation_root_sha256':r.packet['parent_root_sha256'],'primitive_packet_sha256':view['primitive_packet_sha256'],'parent_packet_sha256':view['parent_packet_sha256']},'depth2_ordered_stream_sha256':r.report['full_depth2_stream_sha256'],'historical_selection_weights_qualified':False,'historical_l13_reproduced':False}
 contents={params['qualified_view_sha256']:root_raw}
 for h,raw in locations.items():contents[h]=raw
 for h,raw in sorted(contents.items()):
  if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('EXPORT_CONTENT_HASH')
  scientific['scientific_members'].append({'sha256':h,'size_bytes':len(raw)})
 publication=runtime.publish_bulk_shard('RECURSIVE_SCIENTIFIC_ROOT',scientific)
 r.close()
 return ChainExecutionResult(result={'outcome':'PASS','export_root':publication,'scientific_members':len(contents),'scientific_payload_bytes':sum(len(b) for b in contents.values()),'depth1_constructions':1391,'depth2_constructions':198188,'complete_saved_rows_verified':199579,'generator_calls':0,'scientific_master_admission':False,'historical_selection_weights_qualified':False})
