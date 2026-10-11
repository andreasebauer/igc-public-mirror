"""Native O9 recovery qualification; frozen comparison, no scientific admission."""
from pathlib import Path
import json
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from infinity_grid import maturation_parallel as mp,regime_scanner as rs
_READY=False

def evaluate(payload):
 global _READY
 here=Path(__file__).resolve().parent
 if not _READY:
  context=json.loads((here/'context.json').read_text());mp._candidate_initializer({'context_path':str(here/'context.json'),'context_sha256':canonical_sha256(context)});_READY=True
 recipe=payload['recipe'];s=mp._build_recipe_state(recipe)
 if s is None:raise RuntimeError('NATIVE_O9_BUILD_FAILED')
 def physical(st):
  if st.level==7:return {'source':st.source_id,'counts':list(st.reserve_counts),'skin':st.skin,'caps':list(st.total_caps)}
  return {'skin':st.skin,'caps':list(st.total_caps),'children':[physical(c) for c in st.children],'edges':[rs._edge_json(e) for e in st.top_edges_full]}
 result={'recipe':recipe,'physical':physical(s),'construction':s.construction_digest,'execution_policy':s.execution_policy,'execution_order_key':rs.execution_order_key(s),'pre_features':rs._state_pre_features(s),'motif_id':s.motif_id}
 assert canonical_sha256(result)==payload['expected_sha256'],'NATIVE_RAW_COMPARISON'
 return {'signature':{'construction':s.construction_digest},'outcome_count':1,'metrics':{'matched_raw_sha256':canonical_sha256(result),'motif_id':s.motif_id,'execution_policy':s.execution_policy}}

def handler(stage,runtime):
 rows=json.loads((Path(__file__).resolve().parent/'tasks.json').read_text());tasks=[TaskSpec(task_id=f'o9-{i:04d}',task_kind='P0_NATIVE_O9_RECOVERY',binding_sha256=canonical_sha256({'question':stage['question_sha256'],'payload':row}),payload=row,cost_weight=1.0) for i,row in enumerate(rows)]
 part=runtime.run_structural_partition(phase_id='O9_NATIVE_PARTITION',tasks=tasks,evaluator_ref='project.fixture:evaluate',requested_workers=1,max_tasks=193,stream_shard_task_limit=1)
 return ChainExecutionResult(result={'outcome':'PASS','partition':part.summary,'execution':part.execution_metadata})
