"""Register frozen O7 lane/rank caps and pre-dedup loss boundaries; no generation."""
import argparse, ast, hashlib, json
from pathlib import Path
from bind_scopes import load,sha,write,require

def main(root,output):
    p=root/'recovery/o7-export';r=load(p/'ROOT.json');spec=None
    for x in r['content']:
        b=(p/'content'/(x['sha256']+'.blob')).read_bytes()
        require(len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['sha256'],'EXPORT_BLOB_PIN')
        if x['name']=='spec':spec=json.loads(b)
    require(spec is not None,'LANE_SPEC_REQUIRED')
    lanes=spec['frozen']['homogeneous_heterogeneous_lane_sizes']['lanes']
    routes=load(root/'recovery/identity/saved_export_mapping/MAPPING.json')['occurrence_routes'];rows=[]
    for lane,d in lanes.items():
        for rank in range(d['max_m7']+1):
            selected=[r for r in routes if r['occurrence_key'][0]==lane and r['occurrence_key'][1]==rank]
            roots={r['occurrence_key'][2] for r in selected}
            unavailable={r['occurrence_key'][2] for r in selected if not r['source_root_available']}
            rows.append({'lane':lane,'rank':rank,'owner_count':d['owner_count'],'beam_cap':d['beam_cap'],
              'reservoir_cap':d['reservoir_cap'],'required_previous_rank':rank-1 if rank else None,
              'known_component_source_roots':len(roots),'known_missing_whole_roots':len(unavailable),
              'raw_action_scope':'parent-beam occurrence x legal owner/type/site action before old orbit/canonicalization and digest cuts',
              'raw_action_stream_recovered':False,'missing_whole_roots_are_not_raw_candidate_census':True})
    require(len(rows)==18 and sum(x['known_missing_whole_roots'] for x in rows)==68,'RANK_SCOPE_CENSUS')
    source=next((root/'recovery/o7_seed').rglob('07_INPUT_SNAPSHOTS/o7_live_engine.py'))
    text=source.read_text();tree=ast.parse(text);evidence=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in ['_candidate_keep_add','_multi_block_worker','_range_worker','generation_successors','run_lane_rank','outer_orbits','farthest']:
            excerpt='\n'.join(text.splitlines()[n.lineno-1:n.end_lineno])+'\n'
            evidence.append({'function':n.name,'start_line':n.lineno,'end_line':n.end_lineno,
              'sha256':hashlib.sha256(excerpt.encode()).hexdigest(),'source_excerpt':excerpt})
    register={'schema':'IG_O7_RAW_ACTION_SELECTION_BOUNDARY_V1','lanes':lanes,'rank_scopes':rows,
      'spec_blob_sha256':'71b407991dc13f51ef535778e15f2fa4437e6cf829b9cda575df23942e78fdd7',
      'material_seed_engine_sha256':sha(source),'reviewed_function_boundaries':evidence,
      'discard_boundary':'Canonicalize and old edge digest dedup precede bounded keep trimming; worker keeps trim at >8x reservoir and export first2x reservoir. Saved beam recovery cannot close lost action occurrences.',
      'raw_population_closed':False,'next_local_sufficiency_gate':'Frozen native lane inputs, exact parent beam, legal raw action indexing and saved checkpoint contract before a V3 missing-only re-enumeration'}
    write(output/'O7_RAW_BOUNDARY.json',register);print('PASS_18_LANE_RANK_BOUNDARIES')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);p=a.parse_args();main(p.root,p.output)
