"""Pure exact-state evaluator; multiprocessing and persistence belong to Decoder."""
import hashlib,json,statistics
from pathlib import Path
from infinity_grid.canon import canonical_sha256

def read_bound(path,digest):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('INPUT_IDENTITY')
 return json.loads(raw)

def evaluate(payload):
 from .historical import maturation_parallel as mp,regime_scanner as rs
 from .historical.materialized_discovery import _get_process_o7_runtime
 mode=payload['mode'];level=payload['level']
 if mode=='SEED':
  kernel=_get_process_o7_runtime(rs.load_regime_scanner_spec());states=kernel['base_states']
  if kernel['base_candidate_count']!=205 or len(states)!=24:raise ValueError('BASE_CENSUS')
  dag=mp.state_dag_wire(states);data={'level':7,'dag':dag,'candidate_count':205,'selected_count':24}
 else:
  from .partitions import load_parent
  parent=load_parent(payload['parent_path'],payload['parent_sha256'])
  engine,prev=mp._states_from_dag(parent['dag'])
  if mode=='RESTORE':
   if level!=100 or len(prev)!=193:raise ValueError('RESTORE_CENSUS')
   from .public_interface import extract_interface_population
   ref=read_bound(payload['reference_path'],payload['reference_sha256'])
   observed=extract_interface_population(prev,source_stage='G1:R100',source_authority_sha256=ref['source_authority_sha256'])
   if observed['interfaces']!=ref['interfaces']:raise ValueError('INTERFACE_MISMATCH')
   data={'level':100,'restored_roots':len(prev),'interfaces_checked':193,'dag_science_sha256':parent['dag']['science_sha256']}
  elif mode=='LEVEL':
   if not 8<=level<=100 or parent['level']!=level-1 or len(prev)!=24:raise ValueError('PARENT_SCOPE')
   prev=sorted(prev,key=lambda s:s.construction_digest);med=statistics.median(sum(s.total_caps) for s in prev)
   center=min(prev,key=lambda s:(abs(sum(s.total_caps)-med),s.construction_digest))
   pairs=[tuple(x) for x in read_bound(payload['pairs_path'],payload['pairs_sha256'])['pairs']]
   motifs=rs.load_motif_library();spec=rs.load_regime_scanner_spec()
   recipes=mp.enumerate_candidate_recipes(motifs,center_supports_twin=center.total_caps[0]>=6,pairs_count=len(pairs))
   frozen=read_bound(payload['recipes_path'],payload['recipes_sha256'])
   if recipes!=frozen:raise ValueError('RECIPE_MISMATCH')
   mp.install_candidate_context(engine=engine,prev=prev,level=level,pairs=pairs,motifs=motifs,center=center)
   try:
    rows=[mp._candidate_worker({'recipe':r}) for r in recipes]
    if len(rows)!=193 or any(not r['success'] for r in rows):raise ValueError('CANDIDATE_CENSUS')
    selected=rows if level==100 else mp._farthest_select_descriptors(rows,24,must_include_motif_ids={'TWIN:A','TWIN:B'})
    states=[mp._build_recipe_state(r['recipe']) for r in selected]
    states=sorted(states,key=lambda s:s.construction_digest)
    dag=mp.state_dag_wire(states)
   finally:mp.clear_candidate_context()
   data={'level':level,'dag':dag,'candidate_count':193,'selected_count':len(states)}
  else:raise ValueError('MODE')
 if mode=='LEVEL' and level==50:
  anchor=read_bound(payload['anchor_path'],payload['anchor_sha256'])
  observed={str(x.construction_digest):{'skin':str(x.skin),'caps':list(x.total_caps)} for x in states}
  expected={r['construction_digest']:{'skin':r['resource_skin_sha256'],'caps':r['total_free_by_type']} for r in anchor['materialized_discovery_evidence']['state_probes']}
  if observed!=expected:raise ValueError('HISTORICAL_DEPTH50_ANCHOR_MISMATCH')
  data['historical_depth50_anchor']='PASS24_EXACT_ROOT_SKIN_CAPS'
 if mode=='LEVEL':
  from .partitions import delta
  data=delta(parent,data)
 # Full new-node bytes and root manifest form the identity; ancestor bytes are hash-bound inputs.
 return {'states':[{'identity':data,'state':data}], 'metrics':{'level':level,'mode':mode,'candidate_build_calls':193 if mode=='LEVEL' else 0}}
