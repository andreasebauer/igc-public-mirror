"""Qualification observation wrapper around production initializer and native builder."""
from infinity_grid import maturation_parallel as mp,regime_scanner as rs

def build(recipe):
 s=mp._build_recipe_state(recipe)
 if s is None:raise RuntimeError('candidate build failed')
 def physical(st):
  if st.level==7:return {'source':st.source_id,'counts':list(st.reserve_counts),'skin':st.skin,'caps':list(st.total_caps)}
  return {'skin':st.skin,'caps':list(st.total_caps),'children':[physical(c) for c in st.children],'edges':[rs._edge_json(e) for e in st.top_edges_full]}
 return {'recipe':recipe,'physical':physical(s),'construction':s.construction_digest,'execution_policy':s.execution_policy,'execution_order_key':rs.execution_order_key(s),'pre_features':rs._state_pre_features(s),'motif_id':s.motif_id}
