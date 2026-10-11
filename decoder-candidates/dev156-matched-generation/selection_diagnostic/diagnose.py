"""Saved-observation selection counterfactuals; no native materialization or dispatch."""
import ast,collections,hashlib,json,pathlib,statistics,types
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'selection_diagnostic'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(s):return json.loads((ROOT/s).read_text())
def write(n,x):(OUT/n).write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def main():
 reg=read('selection_diagnostic/REGISTRATION.json')
 for p,h in reg['pins'].items():assert sha(ROOT/p)==h,p
 data={a:read('matched_run2/'+a+'/BASE.json') for a in ('historical','dev156')}
 traces={a:read('matched_run2/'+a+'/O7_SELECTION.json') for a in data}
 features={}
 for a in data:
  features[a]=dict(zip(traces[a]['pool'],traces[a]['vectors']))
  assert len(features[a])==164
  for row in data[a]:row['vector']=features[a][row['construction']]
 def partition(rows):
  d=collections.defaultdict(list)
  for r in rows:d[r['construction']].append(r['index'])
  return sorted(d.values())
 assert partition(data['historical'])==partition(data['dev156'])
 for h,n in zip(data['historical'],data['dev156']):
  assert h['index']==n['index'] and h['vector']==n['vector'] and h['caps']==n['caps'] and h['skin']==n['skin']
 funcs={}
 for a,p in [('historical','recovery/g1/decoder/regime_scanner.py'),('dev156','decoder156/infinity_grid/regime_scanner.py')]:
  tree=ast.parse((ROOT/p).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_farthest_select')
  scope={'Any':object,'_state_pre_features':lambda s:s.vector}
  exec(compile(ast.Module(body=[node],type_ignores=[]),p,'exec'),scope);funcs[a]=scope['_farthest_select']
 def states(a,old_order=False):
  return [types.SimpleNamespace(construction_digest=data['historical'][r['index']]['construction'] if old_order else r['construction'],actual_v3_identity=data['dev156'][r['index']]['construction'],index=r['index'],vector=r['vector'],caps=r['caps']) for r in data[a]]
 def summarize(chosen):
  ordered=sorted(chosen,key=lambda s:s.construction_digest);med=statistics.median(sum(s.caps) for s in ordered)
  center=min(ordered,key=lambda s:(abs(sum(s.caps)-med),s.construction_digest))
  return {'beam':[s.index for s in chosen],'ordered_parents':[s.index for s in ordered],'median':med,'center':center.index}
 experiments={}
 for a in data:
  native=funcs[a](states(a),24);experiments[a]=summarize(native)
  assert experiments[a]['beam']==read('matched_run2/'+a+'/SUMMARY.json')['o7_beam_occurrences']
 compat=states('dev156',True);experiments['dev156_historical_order_proxy']=summarize(funcs['dev156'](compat,24))
 assert experiments['dev156_historical_order_proxy']==experiments['historical']
 ns=states('dev156');hseed=experiments['historical']['beam'][0]
 experiments['dev156_seed_only']=summarize(funcs['dev156'](ns,24,[ns[hseed]]))
 # Existing ancestry is a selected, subsequently reserved subset, not a raw O8 panel.
 retained=read('selection_diagnostic/RETAINED_O8.json');overlays={r['target']['old_reference']:r for r in read('recovery/overlay/results/MAPPING.json')}
 bindings=read('matched_protocol/O7_COHORT_BINDING.json')
 raw=[json.loads(p.read_text()) for p in sorted((ROOT/'matched_run2/historical/recipes').glob('*.json'))]
 comparisons=[]
 for r in raw:
  candidates=[(k,n) for k,n in retained.items() if n['motif_id']==r['call']['motif_id'] and n['lane']==r['call']['lane']]
  sources=[bindings[o['occurrence']]['historical_source_id'] for o in r['call']['owners']]
  same_sources=[];same_top=[];same_ancestry=[];exact_child_counts=[]
  for k,n in candidates:
   old_sources=[overlays[c]['target']['anchor_source_id'] for c in n['children']]
   if old_sources==sources:same_sources.append(k)
   if n['top_edges_full']==r['result']['edges']:same_top.append(k)
   counts=[overlays[c]['target']['reserve_counts'] for c in n['children']]
   raw_counts=[c['counts'] for c in r['result']['children']]
   if old_sources==sources and n['top_edges_full']==r['result']['edges'] and all(a>=b for aa,bb in zip(counts,raw_counts) for a,b in zip(aa,bb)):
    same_ancestry.append(k)
    if counts==raw_counts:exact_child_counts.append(k)
  comparisons.append({'motif':r['call']['motif_id'],'retained_variants':len(candidates),'same_ordered_source_aliases':same_sources,'same_top_witnesses':same_top,'same_source_and_top_with_monotone_reservations':same_ancestry,'exact_child_reserve_counts':exact_child_counts,'exact_construction_retained':r['result']['state']['construction'] in retained})
 write('EXPERIMENTS.json',experiments);write('ANCESTRY_COMPARISON.json',comparisons)
 result={'status':'PASS_BOUND_SELECTION_COUNTERFACTUAL_DIAGNOSTIC','starting_occurrences':205,'same_dedup_partition_groups':164,'same_feature_rows':205,'native_reference_replays':2,'historical_order_proxy_reproduces_beam_order_median_center':True,'seed_only_reproduces_complete_decisions':experiments['dev156_seed_only']==experiments['historical'],'retained_o8_nodes':len(retained),'retained_distinct_motifs':len({n['motif_id'] for n in retained.values()}),'raw_recipes':len(raw),'exact_raw_construction_matches':sum(r['exact_construction_retained'] for r in comparisons),'motifs_with_matching_ordered_source_aliases':sum(bool(r['same_ordered_source_aliases']) for r in comparisons),'motifs_with_matching_top_witnesses':sum(bool(r['same_top_witnesses']) for r in comparisons),'generation_calls':0,'source_modified':False,'controller_changed':False,'O9_dispatched':False,'full_p0_closed':False,'compatibility_policy':'Separate immutable V3 object identity from a versioned historical execution-order key. Preserve historical dedup preimages/last-occurrence representative, pool order, tie order, parent order, median/center ties, cap order and recursive eligible-child reservation order. Historical aliases alone are insufficient; bind exact occurrences and coordinates. Proxy evidence qualifies O7 decision replay only.','next':'Register a native compatibility arm with separate ordering metadata and exact original occurrence binding; then regenerate O8 raw recipes, compare reservations/resources and record cuts. Retained O100 ancestry alone cannot establish raw completeness.'}
 result['motifs_with_joint_ancestry_match']=sum(bool(r['same_source_and_top_with_monotone_reservations']) for r in comparisons)
 result['motifs_with_exact_child_counts']=sum(bool(r['exact_child_reserve_counts']) for r in comparisons)
 write('RESULT.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
