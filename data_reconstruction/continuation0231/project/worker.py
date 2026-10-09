"""Restore saved terminal100 and compare exactly under registered historical V1."""
from pathlib import Path
import json,hashlib,gzip

def read_bound(path,digest):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('INPUT_IDENTITY')
 return json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)

def evaluate(payload):
 from .historical import maturation_parallel as mp
 from .public_interface import extract_interface_population,implementation_spec
 from infinity_grid.canon import canonical_sha256
 parent=read_bound(payload['bootstrap_path'],payload['bootstrap_sha256'])
 assert parent['level']==100 and parent['selected_count']==193 and parent['candidate_count']==193
 spec=read_bound(payload['v1_spec_path'],payload['v1_spec_sha256'])
 assert implementation_spec()==spec and spec['spec_sha256']==payload['historical_spec_identity']
 _,states=mp._states_from_dag(parent['dag'])
 assert len(states)==193 and mp.state_dag_wire(states)==parent['dag']
 reference=read_bound(payload['reference_path'],payload['reference_sha256'])
 observed=extract_interface_population(states,source_stage='G1:R100',source_authority_sha256=reference['source_authority_sha256'])
 if observed!=reference:raise ValueError('HISTORICAL_V1_FULL_POPULATION_MISMATCH')
 anchor=read_bound(payload['anchor_path'],payload['anchor_sha256'])
 by_ref={x.construction_digest:x for x in states};beam={k:{'skin':str(by_ref[k].skin),'caps':list(by_ref[k].total_caps)} for k in parent['beam_roots']}
 expected={x['construction_digest']:{'skin':x['resource_skin_sha256'],'caps':x['total_free_by_type']} for x in anchor['materialized_discovery_evidence']['state_probes']}
 assert len(beam)==24 and beam==expected
 # V2 check uses the separately retained, byte-identical current extractor.
 from .current_interface import extract_interface_population as current_population
 v2=current_population(states,source_stage='G1:R100',source_authority_sha256=reference['source_authority_sha256'])
 assert v2==read_bound(payload['diagnostic_v2_path'],payload['diagnostic_v2_sha256'])
 data=dict(level=100,restored_roots=193,all193_DAG_roundtrip=True,dag_science_sha256=parent['dag']['science_sha256'],historical_depth100_beam_anchor='PASS24_EXACT_ROOT_SKIN_CAPS',terminal_comparison='PASS193_EXACT_HISTORICAL_V1_PUBLIC_INTERFACES',historical_implementation_spec_sha256=spec['spec_sha256'],current_v2_population_reproduced=True,historical_interface_population=observed,candidate_generation_calls=0,new_admissions=0,master_slices=151)
 return {'states':[{'identity':data,'state':data}], 'metrics':{'restored_roots':193,'candidate_generation_calls':0}}
