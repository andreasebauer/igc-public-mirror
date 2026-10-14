"""Bounded writer or separate cold-reader job; invoke only through native capture."""
from copy import deepcopy
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import time
from infinity_grid import regime_scanner as rs
from infinity_grid import materialized_discovery as discovery
from infinity_grid.maturation_parallel import state_dag_wire, _states_from_dag
from infinity_grid.maturation_dag_contract import validate_state_dag
from infinity_grid.canon import canonical_sha256
from g3_dag_export import assemble_g3_dag_export, canonical_hash
from marker_full_export import exact_json
from independent_g3_reference import lower_sources, reference_for_dag


def save(name, obj):
    Path(name).write_text(json.dumps(exact_json(obj), sort_keys=True, indent=2) + '\n')


def sha(name):
    return hashlib.sha256(Path(name).read_bytes()).hexdigest()


def native_snapshot(roots):
    rows = {}; seen = set()
    def visit(state):
        if id(state) in seen:return
        seen.add(id(state))
        record = {'kind': 'O7' if isinstance(state, rs.O7State) else 'LIFT',
                  'level': state.level, 'lane': state.lane, 'execution_policy': state.execution_policy,
                  'construction_digest': state.construction_digest,
                  'execution_order_key': rs.execution_order_key(state),
                  'owner_caps': state.owner_caps, 'total_caps': state.total_caps, 'skin': state.skin}
        if isinstance(state, rs.O7State):
            record.update(source_id=state.source_id, parent_ids=state.ctx.source_order,
                parent_colors=[p.exact_color for p in state.ctx.parents],
                original_parent_resources=[p.h6 for p in state.ctx.parents],
                original_coupled_edges=state.edges, reserve_counts=state.reserve_counts)
        else:
            record.update(ordered_children=[c.construction_digest for c in state.children],
                          top_edges_full=state.top_edges_full, motif_id=state.motif_id)
            for child in state.children:visit(child)
        record = exact_json(record)
        if state.construction_digest in rows:assert rows[state.construction_digest] == record
        rows[state.construction_digest] = record
    for root in roots:visit(root)
    return {'schema_id': 'IG_MASTER_G3_NATIVE_SNAPSHOT_V1', 'nodes': rows}


def compare_snapshot(reference, snapshot):
    assert snapshot['schema_id'] == 'IG_MASTER_G3_NATIVE_SNAPSHOT_V1'
    snapshot = snapshot['nodes']
    assert set(reference['node_materializations']) == set(snapshot)
    for key, expected in reference['node_materializations'].items():
        actual = snapshot[key]
        common = ['kind', 'level', 'owner_caps', 'total_caps', 'skin']
        if expected['kind'] == 'O7':
            common += ['source_id', 'parent_ids', 'parent_colors', 'original_parent_resources',
                       'original_coupled_edges', 'reserve_counts']
        else:common += ['ordered_children', 'top_edges_full']
        for field in common:assert actual[field] == expected[field], (key, field)


def main():
    started = time.perf_counter()
    inputs = json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    parameters = json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    raw = {}
    def load(name):
        data = Path(inputs[name]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == parameters['input_hashes'][name]
        raw[name] = data
        return json.loads(data)
    selection = load('selected_o7_rows');spec = load('scanner_spec')
    seed = Path(inputs['o7_seed']).read_bytes();raw['o7_seed'] = seed
    assert hashlib.sha256(seed).hexdigest() == parameters['input_hashes']['o7_seed'] == selection['source_zip_sha256']
    assert files('infinity_grid').joinpath('resources/decoder/O7_MATERIAL_ROOT_cb6f48641eb9.zip').read_bytes() == seed
    assert files('infinity_grid').joinpath('resources/decoder/O_REGIME_ADAPTIVE_SCANNER_SPEC_v1.json').read_bytes() == raw['scanner_spec']
    selected, parents = lower_sources(seed, selection)
    assert list(selected) == parameters['selected_source_ids']
    required = {name: {'sha256': parameters['input_hashes'][name], 'bytes': len(raw[name])}
                for name in ['o7_seed', 'scanner_spec', 'selected_o7_rows']}
    receipts = [{'path': n, **required[n]} for n in sorted(required)]
    assert canonical_hash(receipts) == parameters['lower_input_root_ref']
    binding = {name: parameters[name] for name in ['recipe_ref','source_ref','runtime_ref','task_ref']}
    binding.update(input_root_ref=parameters['lower_input_root_ref'], identity_scope='DECODER091_NATIVE_V3')
    assert discovery._PROCESS_O7_RUNTIME is None
    assert all(not getattr(rs, n) for n in ['_O7_DATA_CACHE','_O7_SKIN_CACHE','_O7_OWNER_CAP_CACHE'])
    fresh_cache_entry = True
    if parameters['mode'] == 'WRITER':
        begin = time.perf_counter()
        anchor = discovery._get_process_o7_runtime(spec)
        warm_seconds = time.perf_counter() - begin
        originals = []
        for source_id in parameters['selected_source_ids']:
            matches = [s for s in anchor['authority_base_states'] if s.source_id == source_id]
            assert len(matches) == 1
            originals.append(matches[0])
        left, right = originals
        assert (0, 0) in anchor['bridge_pairs']
        begin = time.perf_counter()
        homogeneous = rs._build_lift(anchor['engine'], 8, [left,left], [(0,1)], anchor['bridge_pairs'],
                                    'G3_CODEC_Q01', 'Q01:HOM:EDGE', force_pair=(0,0))
        heterogeneous = rs._build_lift(anchor['engine'], 8, [left,right], [(0,1)], anchor['bridge_pairs'],
                                      'G3_CODEC_Q01', 'Q01:HET:EDGE', force_pair=(0,0))
        nested = rs._build_lift(anchor['engine'], 9, [homogeneous,homogeneous], [(0,1)], anchor['bridge_pairs'],
                               'G3_CODEC_Q01', 'Q01:NESTED:EDGE', force_pair=(0,0))
        assert homogeneous is not None and heterogeneous is not None and nested is not None
        roots = [left,right,homogeneous,heterogeneous,nested,nested]
        dag = state_dag_wire(roots)
        snapshot = native_snapshot(roots)
        construction_seconds = time.perf_counter() - begin
        export = assemble_g3_dag_export(dag=dag,binding=binding,required_inputs=required,
            input_bytes={n:raw[n] for n in required},authority_source_ids=parameters['selected_source_ids'])
        save('NATIVE_DAG.json', dag);save('FULL_G3_DAG_EXPORT.json', export);save('NATIVE_SNAPSHOT.json', snapshot)
        begin = time.perf_counter()
        reference = reference_for_dag(dag, selected, parents)
        save('INDEPENDENT_CORRESPONDENCE.json', reference)
        compare_snapshot(reference, snapshot)
        independent_seconds = time.perf_counter() - begin
        result = {'schema_id': 'IG_MASTER_G3_DAG_WRITER_RESULT_V1', 'status': 'PASS',
            'scientific_outcome': 'G3_DAG_WRITER_CORRESPONDENCE_EARNED_FOR_DECLARED_SCOPE',
            'root_occurrences': len(dag['roots']), 'unique_nodes': len(dag['nodes']),
            'logical_occurrences': reference['logical_occurrence_count'],
            'recursive_endpoint_claims': reference['endpoint_claim_count'],
            'DAG_sha256': sha('NATIVE_DAG.json'), 'snapshot_sha256': sha('NATIVE_SNAPSHOT.json'),
            'full_export_sha256': sha('FULL_G3_DAG_EXPORT.json'),
            'correspondence_sha256': sha('INDEPENDENT_CORRESPONDENCE.json'),
            'anchor_records_loaded': len(anchor['authority_base_states']),
            'warmup_seconds': str(warm_seconds), 'construction_seconds': str(construction_seconds),
            'independent_seconds': str(independent_seconds), 'fresh_cache_entry': fresh_cache_entry,
            'cold_roundtrip_qualified': False}
    else:
        assert parameters['mode'] == 'COLD'
        dag = load('accepted_native_DAG');snapshot = load('accepted_native_snapshot')
        reference = load('accepted_independent_correspondence');writer_result = load('accepted_writer_result')
        assert writer_result['status'] == 'PASS' and writer_result['DAG_sha256'] == parameters['input_hashes']['accepted_native_DAG']
        assert writer_result['snapshot_sha256'] == parameters['input_hashes']['accepted_native_snapshot']
        assert writer_result['correspondence_sha256'] == parameters['input_hashes']['accepted_independent_correspondence']
        assert parameters['writer_completion_sha256'] and parameters['writer_capture_id']
        # These resealed corruptions must fail before any authority materialization.
        refused = []
        for label in ['bool_counter','float_counter','negative_counter','bool_level','float_level',
                      'missing_child','cycle','wrong_policy','bad_witness','extra_field','old_schema','bad_seal']:
            bad=deepcopy(dag);leaf=next(r for r in bad['nodes'].values() if r['kind']=='O7')
            lift=next(r for r in bad['nodes'].values() if r['kind']=='LIFT')
            if label=='bool_counter':leaf['reserve_counts'][0]=True
            elif label=='float_counter':leaf['reserve_counts'][0]=1.0
            elif label=='negative_counter':leaf['reserve_counts'][0]=-1
            elif label=='bool_level':lift['level']=True
            elif label=='float_level':lift['level']=8.0
            elif label=='missing_child':lift['children'][0]='missing'
            elif label=='cycle':lift['children'][0]=lift['construction_digest']
            elif label=='wrong_policy':lift['execution_policy']='HISTORICAL_COORDINATE_V1'
            elif label=='bad_witness':lift['top_edges_full'][0][4]=[0,999999]
            elif label=='extra_field':leaf['unconsumed_payload']='extra'
            elif label=='old_schema':bad['schema_id']='IG_MATURATION_STATE_DAG_V1'
            else:bad['science_sha256']='bad'
            if label!='bad_seal':bad['science_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='science_sha256'})
            try:_states_from_dag(bad)
            except ValueError:refused.append(label)
            else:raise AssertionError('corrupt DAG accepted: '+label)
            assert discovery._PROCESS_O7_RUNTIME is None
        begin=time.perf_counter();_,roots=_states_from_dag(dag);cold_seconds=time.perf_counter()-begin
        regenerated=state_dag_wire(roots);cold_snapshot=native_snapshot(roots)
        save('COLD_NATIVE_DAG.json',regenerated);save('COLD_NATIVE_SNAPSHOT.json',cold_snapshot)
        assert regenerated==dag and cold_snapshot==snapshot
        own_reference=reference_for_dag(dag,selected,parents)
        assert own_reference==reference;compare_snapshot(own_reference,cold_snapshot)
        # Actual frozen-source ambiguity is tested after the fresh cold reconstruction.
        bad=deepcopy(dag);leaf=next(r for r in bad['nodes'].values() if r['kind']=='O7')
        leaf['source_id']=parameters['ambiguous_source_id']
        bad['science_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='science_sha256'})
        try:_states_from_dag(bad)
        except ValueError as exc:
            assert 'SOURCE_ALIAS_AMBIGUOUS_REVIEW_REQUIRED' in str(exc);refused.append('actual_ambiguous_O7_source')
        else:raise AssertionError('ambiguous original source alias silently selected')
        save('REFUSAL_CONTROLS.json',{'status':'PASS','controls':refused})
        result={'schema_id':'IG_MASTER_G3_DAG_COLD_RESULT_V1','status':'PASS',
            'scientific_outcome':'G3_DAG_COLD_CORRESPONDENCE_EARNED_FOR_DECLARED_SCOPE',
            'writer_completion_sha256':parameters['writer_completion_sha256'],
            'writer_capture_id':parameters['writer_capture_id'], 'fresh_cache_entry':fresh_cache_entry,
            'root_occurrences':len(dag['roots']),'unique_nodes':len(dag['nodes']),
            'logical_occurrences':own_reference['logical_occurrence_count'],
            'recursive_endpoint_claims':own_reference['endpoint_claim_count'],
            'exact_DAG_roundtrip':True,'exact_snapshot_roundtrip':True,
            'correspondence_sha256':parameters['input_hashes']['accepted_independent_correspondence'],
            'refusal_controls':len(refused),'cold_seconds':str(cold_seconds),
            'cold_roundtrip_qualified':True}
    assert result['root_occurrences']==6 and result['unique_nodes']==9
    assert result['logical_occurrences']==22 and result['recursive_endpoint_claims']==16
    result.update(scope='Two exact available O7 component records; homogeneous/heterogeneous level8 lifts and a nested level9 lift; duplicate roots and child slots',
        source_identity_scope='DECODER091_NATIVE_V3', lower_input_root_ref=parameters['lower_input_root_ref'],
        wall_seconds=str(time.perf_counter()-started),pipeline_qualified=False,
        actual_G1_G3_consumer_qualified=False, historical_V1_rebased=False,
        official_canary=False,full_generation_allowed=False)
    save('QUALIFICATION_RESULT.json', result)
    print(json.dumps(result))


if __name__ == '__main__':main()
