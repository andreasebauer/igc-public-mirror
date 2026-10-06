"""Freeze a candidate view over qualified saved bytes; no official export/admission."""
from pathlib import Path
import json,sys,hashlib
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(B/'unified139'))
from ig_master import UnifiedMasterReader
from canonical_kernel import canonical_bytes,canonical_sha256
load=lambda p:json.loads(p.read_text());sha=lambda raw:hashlib.sha256(raw).hexdigest()
packet=load(B/'node_adapter/PRIMITIVE_PACKET.json');catalog=(B/'unified139/CATALOG_0139.json').read_bytes();assert sha(catalog)==packet['accepted_catalog_sha256']
with UnifiedMasterReader(B/'unified139',canonical_directory=B/'canonical14',archive_directory=B/'unified139/archives') as master:
 for event in packet['events']:
  carrier=master.lookup_j3(event['selector']['exact_tid'])['carrier'];assert carrier['j3_id']==event['j3_id'];matches=[e for e in carrier['events'] if e['event_id']==event['event_id']];assert len(matches)==1;admitted=matches[0];assert admitted['record']==event['record'] and admitted['realizations']==event['realizations'] and len(admitted['realizations'])==event['realization_count']
proof={'status':'PASS_ALL_13_PRIMITIVE_EVENTS_AGAINST_ADMITTED_J3','catalog_sha256':sha(catalog),'all_records_and_realizations_match':True,'generator_calls':0}
(R/'ADMITTED_PRIMITIVE_CLOSURE.json').write_text(json.dumps(proof,indent=2)+'\n')
union=load(B/'recursive_native/CAMPAIGN_SAVED_UNION_VERIFICATION.json');assert union['status']=='PASS_COMPLETE_SIX_TRANCHE_SAVED_UNION';sources=[];locations={};external={}
for t in range(7):
 D=B/('node_adapter' if t==0 else 'recursive_native/'+('tranche01_repack' if t==1 else f'tranche{t:02d}'));result=load(D/'NATIVE_RESULT.json');qraw=(D/'RECIPE_CONTRACT.json').read_bytes();recipe=json.loads(qraw);cold=load(D/'COLD_REUSE_RESULT.json');assert cold['status']=='PASS' and cold['pending_bytes']==0
 for k in ['completion_sha256','result_sha256']:assert result[k]==cold[k]
 mapping=load(D/'CHECKPOINT_READBACK_MAPPING.json');source={'depth':1 if t==0 else 2,'tranche':t,'rows':1391 if t==0 else recipe['scope']['expected_lawful_rooted_constructions'],'recipe_sha256':sha(qraw),'packet_field':'primitive_packet_sha256' if t==0 else 'parent_packet_sha256','packet_sha256':sha((B/'node_adapter/PRIMITIVE_PACKET.json').read_bytes()) if t==0 else recipe['parent_packet_sha256'],'completion_sha256':result['completion_sha256'],'result_sha256':result['result_sha256'],'shards':[]}
 if t:
  source.update(parent_start=recipe['scope']['parent_start'],parent_stop_exclusive=recipe['scope']['parent_stop_exclusive'],saved_stream_sha256=load(D/'SAVED_OUTPUT_VERIFICATION.json')['saved_order_stream_sha256'])
 else:assert load(D/'NATIVE_EXPORT_VERIFICATION.json')['all_rows_independently_checked']==1391
 for ref in result['result']['data_shards']:
  m=mapping[ref['sha256']];raw=Path(m['path']).read_bytes();assert sha(raw)==ref['sha256'] and len(raw)==ref['size_bytes'];source['shards'].append({k:ref[k] for k in ['sha256','size_bytes']});locations[ref['sha256']]=m['path'];external[ref['sha256']]={k:m[k] for k in ['drive_file_id','sha256','size_bytes']}
 sources.append(source)
for D,n in [(B/'node_adapter','PRIMITIVE_PACKET.json'),(B/'l13_lineage','PARENT_PACKET.json')]:
 raw=(D/n).read_bytes();h=sha(raw);locations[h]=str((R/n).resolve());(R/n).write_bytes(raw)
root={'schema_id':'IG_RECURSIVE_QUALIFIED_VIEW_V1','status':'QUALIFIED_CANDIDATE_READ_VIEW_NOT_MASTER_ADMISSION','master_admission':False,'catalog_sha256':sha(catalog),'primitive_packet_sha256':sha((R/'PRIMITIVE_PACKET.json').read_bytes()),'parent_packet_sha256':sha((R/'PARENT_PACKET.json').read_bytes()),'depth1_identity_recipe_sha256':sources[0]['recipe_sha256'],'depth2_identity_recipe_sha256':union['tranches'] and load(B/'l13_lineage/DRY_VERIFICATION.json')['recipe_sha256'],'full_depth2_stream_sha256':union['ordered_stream_sha256'],'sources':sources,'historical_selection_weights_qualified':False,'historical_l13_reproduced':False}
(R/'ROOT.json').write_bytes(canonical_bytes(root));(R/'LOCATIONS.json').write_text(json.dumps(locations,indent=2)+'\n');(R/'EXTERNAL_OBJECTS.json').write_text(json.dumps(external,indent=2)+'\n')
profile={'status':'FROZEN_CANDIDATE_EXPORT_PROFILE','root_sha256':canonical_sha256(root),'format':'Preserve exact qualified native rows and their frozen source shard envelopes','counts':{'primitive_events':13,'depth1_constructions':1391,'depth2_constructions':198188,'total_constructions':199579,'depth1_boundaries':108,'depth2_boundaries':802},'identity':'Rooted construction and formation IDs; projected boundaries are explicit many-to-one views','required_fields':'Every field retained; full parent, formation, ordered-port, projection occurrence, component, target-origin and realization product references','scientific_dependency_closure':'Admitted J3 roots/catalog and exact13-event realization packet; all1391 depth1 parents must match parent packet; full198188 depth2 rows','authority':'Read-only candidate over exact native saved data. Register current-runtime scientific export/admission separately before catalog publication.','historical_weights':'NOT_QUALIFIED: return native preimages and realization counts without interpreting them as D/B/S/O/L weights','stopping_rule':'Fail on any saved-byte, scope, reference, constructor, completeness, root or dependency mismatch','next':'Native registered export and preservation, independent exported-byte reader qualification, then scoped admission. Historical selection-weight qualification remains separate prerequisite for historical replay.'}
(R/'EXPORT_PROFILE.json').write_text(json.dumps(profile,indent=2)+'\n');print(json.dumps({'root_sha256':profile['root_sha256'],'external_shards':len(external),'primitive_closure':proof['status']}))
