"""Dry binding of historical selection counts to saved native qualification rows."""
from pathlib import Path
import json,ast,itertools,hashlib
from collections import Counter
B=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
primitives=json.loads((B/'engine/infinity_grid/resources/replay/NODE_IN_ORIGINAL_FROZEN_PRIMITIVES.json').read_text())['primitive']
packet=json.loads((B/'node_adapter/PRIMITIVE_PACKET.json').read_text());events={e['event_id']:e for e in packet['events']}
source=(B/'node_input_audit/sources/run_level.py').read_text();tree=ast.parse(source);ns={};pure=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'dtup','canon','bridge','compose'}],type_ignores=[]);exec(compile(pure,'original_pure_constructor','exec'),ns)
legacy={};boundaries=set()
for a,b in itertools.product(primitives,repeat=2):
 for sa,sb in itertools.product(range(3),repeat=2):
  result=ns['compose'](ast.literal_eval(a['record']),sa,ast.literal_eval(b['record']),sb)
  if result is None:continue
  key=(a['tid'],a['rank'],b['tid'],b['rank'],sa,sb);legacy[key]=result;boundaries.add(result)
saved=json.loads((B/'node_adapter/CHECKPOINT_READBACK_MAPPING.json').read_text());native=json.loads((B/'node_adapter/NATIVE_RESULT.json').read_text());counts=Counter()
for shard in native['result']['data_shards']:
 raw=Path(saved[shard['sha256']]['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==shard['sha256']
 for row in json.loads(raw)['rows']:
  a,b=[events[c['event_id']] for c in row['components']];sa,sb=row['identity']['bridge_slots'];la=a['selector'];lb=b['selector']
  key=(la['exact_tid'],la['historical_rank'],lb['exact_tid'],lb['historical_rank'],a['projected_to_native_ports'].index(sa),b['projected_to_native_ports'].index(sb))
  assert key in legacy
  r=row['projected_boundary'];as_tuple=(r[0],tuple(tuple(p) for p in r[1]),r[2],tuple(r[3]),r[4]);assert as_tuple==legacy[key];counts[key]+=1
assert set(counts)==set(legacy)
variants=Counter((e['selector']['exact_tid'],e['selector']['historical_rank']) for e in packet['events'])
assert all(n==variants[key[:2]]*variants[key[2:4]] for key,n in counts.items())
report={'status':'PASS_PROJECTION_PREIMAGE_BINDING','execution':'Dry mapping of saved qualification occurrences; no population job launched.','historical_primitive_selectors':9,'native_event_variants':13,'historical_lawful_endpoint_occurrences':len(legacy),'native_rooted_occurrences':sum(counts.values()),'historical_and_native_projected_boundary_count':len(boundaries),'every_historical_endpoint_occurrence_has_complete_native_preimage':True,'native_preimage_multiplicity_histogram':dict(sorted(Counter(counts.values()).items())),'selector_scoring_rule':'Historical D/B/S/O/L witness/parent weights were computed in the nine projected-primitive domain. Preserve that scoring domain explicitly when reproducing historical lanes; retain the expanded native occurrence preimages separately. Substituting native occurrence counts for historical witness counts changes B/O ranking weights and may change selected frontiers.','higher_depth_requirement':'At each extension, preserve a composable projected-endpoint to exact-construction endpoint correspondence plus all parent construction alternatives; this depth-one result does not recover the historical bootstrap lineage.'}
(D/'SELECTION_PROJECTION_BINDING.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
