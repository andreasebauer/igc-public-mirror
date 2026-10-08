from pathlib import Path
import json,sys
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204');sys.path.insert(0,str(B))
from project.partitions import assemble
C=Path('/tmp/ig_native0209/forensic_cold');R=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'));M=json.loads((R/'artifacts/g1_partition_depth_100_manifest.json').read_text());nodes=json.loads((C/'runtime/intake/artifacts'/(M['base']['sha256']+'.bin')).read_text())['dag']['nodes']
for ref in M['partitions']:
 part=json.loads((R/'artifacts'/Path(ref['path']).name).read_text())
 for k,v in part['nodes'].items():
  assert k not in nodes or nodes[k]==v;nodes[k]=v
D=assemble(nodes,M['roots'],M['science_sha256']);print('nodes',len(D['nodes']));print('root keys',list(D['nodes'][D['roots'][0]]));print('first root',str(D['nodes'][D['roots'][0]])[:1500]);refsha=json.loads((B/'SPEC.json').read_text())['execution']['parameters']['bindings']['reference'];reference=json.loads((C/'runtime/intake/artifacts'/(refsha+'.bin')).read_text());seen=set(D['roots']);expected={x['carrier_ref'] for x in reference['interfaces']};print('carrier_ref overlap',len(seen&expected),'observed',len(seen),'reference',len(expected))
(B/'ROOT_REF_MISMATCH.json').write_text(json.dumps({'observed_roots':sorted(seen),'reference_roots':sorted(expected),'common_roots':sorted(seen&expected),'observed_only':sorted(seen-expected),'reference_only':sorted(expected-seen),'science_sha256':D['science_sha256']},indent=2))
