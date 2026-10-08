"""Verify saved exact historical phases. No candidate-producing operation."""
import hashlib,json
from pathlib import Path
from infinity_grid.canon import canonical_sha256
from infinity_grid.structural_encoding import structural_canonical_bytes
from .partitions import assemble

def read_bound(path,digest):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('INPUT_IDENTITY')
 return json.loads(raw)

def verify(payload):
 from .historical import maturation_parallel as mp
 x={k:read_bound(payload[k+'_path'],payload[k+'_sha256']) for k in ('bootstrap','phases','anchor','prior_audit','prior_export')}
 if x['prior_audit']['status']!='PASS_COLD_HISTORICAL_DEPTH80_PAUSED_NATIVE_BUDGET':raise ValueError('PRIOR_AUDIT')
 phase_evidence=x['phases']
 if phase_evidence['prior_capture_id']!=payload['prior_capture_id'] or phase_evidence['prior_checkpoint_sha256']!=x['prior_export']['sha256']:raise ValueError('PRIOR_BINDING')
 base=x['bootstrap'];nodes=dict(base['dag']['nodes']);previous=base['dag']['science_sha256'];dags=[];rows=[]
 if base['level']!=74 or len(phase_evidence['phases'])!=6:raise ValueError('SCOPE')
 for n,saved in zip(range(75,81),phase_evidence['phases'],strict=True):
  d=saved['state'];exact=structural_canonical_bytes(d)
  if hashlib.sha256(exact).hexdigest()!=saved['index_digest'] or saved['generation_tasks']!=1:raise ValueError('NATIVE_EXACT_IDENTITY')
  if d['level']!=n or d['candidate_count']!=193 or d['selected_count']!=24 or d['parent_science_sha256']!=previous:raise ValueError('PHASE_SCOPE')
  for k,v in d['nodes'].items():
   if k in nodes and nodes[k]!=v:raise ValueError('NODE_COLLISION')
   nodes[k]=v
  dag=assemble(nodes,d['roots'],d['science_sha256']);dags.append(dag);previous=dag['science_sha256']
  rows.append({'level':n,'roots':24,'science_sha256':previous,'index_digest':saved['index_digest']})
 union_nodes={};roots=[]
 for dag in dags:
  roots.extend(dag['roots'])
  for k,v in dag['nodes'].items():
   if k in union_nodes and union_nodes[k]!=v:raise ValueError('UNION_COLLISION')
   union_nodes[k]=v
 union={'schema_id':'IG_MATURATION_STATE_DAG_V1','roots':roots,'nodes':union_nodes};union['science_sha256']=canonical_sha256(union)
 _,states=mp._states_from_dag(union)
 if len(states)!=144:raise ValueError('ROUNDTRIP_CENSUS')
 for i,dag in enumerate(dags):
  if mp.state_dag_wire(states[i*24:(i+1)*24])!=dag:raise ValueError('DAG_ROUNDTRIP')
 observed={s.construction_digest:{'skin':s.skin,'caps':list(s.total_caps)} for s in states[-24:]}
 expected={s['construction_digest']:{'skin':s['resource_skin_sha256'],'caps':s['total_free_by_type']} for s in x['anchor']['materialized_discovery_evidence']['state_probes']}
 if observed!=expected or len(expected)!=24:raise ValueError('DEPTH80_ANCHOR')
 if previous!=payload['final_science_sha256']:raise ValueError('FINAL_SCIENCE')
 return {'outcome':'PASS_HISTORICAL_DEPTH80_NATIVE_RECOVERY','completed_depth':80,'roots':24,'historical_depth80_anchor':'PASS24_EXACT_ROOT_SKIN_CAPS','saved_native_exact_identity_phases':6,'DAG_roundtrip_roots':144,'candidate_regeneration_calls':0,'prior_capture_id':payload['prior_capture_id'],'prior_capture_status':'PAUSED_WORKSPACE_BUDGET','prior_checkpoint_sha256':x['prior_export']['sha256'],'final_science_sha256':previous,'rows':rows,'master_scientific_slices':151,'new_admissions':0,'terminal_comparison':'NOT_RUN','Q2_payload_generated':False}

def evaluate(payload):
 result=verify(payload)
 return {'states':[{'identity':result,'state':result}], 'metrics':{'candidate_build_calls':0,'saved_phases_verified':6}}
