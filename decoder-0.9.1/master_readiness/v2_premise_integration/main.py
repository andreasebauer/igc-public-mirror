"""New native V2 producer integration; no historical job is executed here."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from infinity_grid import regime_scanner as rs, maturation_parallel as mp
from infinity_grid import materialized_discovery as md, theorem_registry as tr
from infinity_grid import fixed_grammar_transport as ft, exact_carrier_acceleration as acceleration
from infinity_grid import semantic_sentinel as ss
from infinity_grid.carrier_reuse import load_certified_carrier_set
from infinity_grid.canon import canonical_sha256


def write(name,value):Path(name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def digest(value):return canonical_sha256(value)
def json_tree(value):return json.loads(json.dumps(value,sort_keys=True))


def main():
    started=time.monotonic()
    inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    parameters=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for name,expected in parameters['input_hashes'].items():
        assert hashlib.sha256(Path(inputs[name]).read_bytes()).hexdigest()==expected
    protocol=json.loads(Path(inputs['audit_protocol']).read_text())
    parent=json.loads(Path(inputs['parent_semantic_status']).read_text())['current_snapshot']
    old={row['action_id']:row['callable_semantic_sha256'] for row in parent['actions']}
    snapshot=ss.native_semantics_snapshot()
    changed=[row['action_id'] for row in snapshot['actions'] if row['callable_semantic_sha256']!=old[row['action_id']]]
    assert changed==['FIXED_GRAMMAR_FRONTIER_ADVANCE_V1']
    semantic=ss.verify_native_semantics(raise_on_change=False)
    theorems=tr.verify_all_theorems(raise_on_stale=False)
    assert semantic['status']=='REOPEN_REQUIRED' and theorems['status']=='STALE_REOPEN_REQUIRED'
    assert len(theorems['stale_theorems'])==4
    write('SEMANTIC_AND_THEOREM_STATUS.json',{'status':'PASS','registered_callable_changes':changed,'semantic':semantic,'theorems':theorems})
    # These real records must refuse before any O7 authority warmup or output write.
    assert md._PROCESS_O7_RUNTIME is None
    cache_before={name:len(getattr(rs,name)) for name in ['_O7_DATA_CACHE','_O7_SKIN_CACHE','_O7_OWNER_CAP_CACHE']}
    no_output=Path('guard_output_must_not_exist')
    service='O_DEPTH_ERASED_RELATION_ADD_SERVICE_V1'
    guard_calls=[
        ('earned_registry',rs.load_earned_regime_laws,tr.StaleTheoremError,None),
        ('cached_inherited_law',lambda:rs._current_inherited_law({service:{'status':'EARNED'}},service),tr.StaleTheoremError,None),
        ('streaming_baseline',ft.streaming_recipe_baseline,tr.StaleTheoremError,None),
        ('registered_frontier',lambda:ft._advance_frontier({},{},{},{}),tr.StaleTheoremError,None),
        ('streaming_reset',lambda:ft.run_fixed_grammar_transport(no_output,through=100,reset=True),tr.StaleTheoremError,None),
        ('exact_transport',lambda:acceleration.theorem_transport_certificate(level=8,parent_certificate_sha256='a'*64,
            material_instance_sha256='b'*64,authority_audit={'status':'PASS'}),tr.StaleTheoremError,None),
        ('materialized_calibration',md.MaterializedDiscoverySession,md.MaterializedDiscoveryError,'SCANNER_DIAGNOSTIC_CALIBRATION_REOPEN_REQUIRED'),
        ('frozen_streaming_record',ft._streaming_recipe_baseline_record,RuntimeError,'STREAMING_DIAGNOSTIC_CALIBRATION_REOPEN_REQUIRED'),
    ]
    refusals=[]
    for name,call,error,code in guard_calls:
        try:call()
        except error as exc:
            if code is not None:assert code in str(exc)
            refusals.append({'consumer':name,'exception':type(exc).__name__,'message':str(exc)})
        else:raise AssertionError('AUTHORITY_PATH_DID_NOT_REFUSE:'+name)
        assert not no_output.exists() and md._PROCESS_O7_RUNTIME is None and not md._DISCOVERY_SESSION_LOCK.locked()
        assert cache_before=={key:len(getattr(rs,key)) for key in cache_before}
    assert len(refusals)==protocol['expected_refusals']
    write('AUTHORITY_CONSUMPTION_REFUSALS.json',{'status':'PASS','refusal_count':len(refusals),'refusals':refusals,
        'before_authority_warmup':True,'no_output_written':True,'scanner_caches_unchanged':True,'session_lock_untouched':True,
        'baseline_updated':False,'calibration_adopted':False})
    frozen=json.loads(Path(inputs['extended_cases']).read_text())
    accepted=json.loads(Path(inputs['accepted_v2_cases']).read_text())
    references={row['case_id']:row for row in accepted['cases']}
    rows=[];panels=defaultdict(list);analyses={};synthetic_count=0

    def inspect_state(case_id,state,pairs,reference=None,dataset='synthetic'):
        before=mp.state_wire(state);payload={'state':before,'bridge_pairs':pairs}
        analysis=mp._selected_state_worker(payload)
        assert mp.state_wire(state)==before
        assert analysis['schema_id']=='IG_MATURATION_SELECTED_STATE_ANALYSIS_V2'
        assert analysis['state_input_sha256']==digest(before)
        assert analysis['bridge_pairs_sha256']==digest(sorted(tuple(pair) for pair in pairs))
        if reference is not None:
            assert analysis['action']==reference['expected'], 'ACCEPTED_V2_ACTION_MISMATCH:'+case_id
            assert analysis['organizational_sha256']==digest(reference['organizational_signature']), 'ACCEPTED_V2_ORGANIZATION_MISMATCH:'+case_id
        if state.construction_digest in analyses:
            assert analyses[state.construction_digest]==analysis, 'SAME_CONSTRUCTION_ANALYSIS_MISMATCH:'+case_id
        analyses[state.construction_digest]=analysis
        level=state.level if dataset=='actual_lower' else 8
        key=(dataset,level,bool(analysis['graph']['connected']),digest(sorted(tuple(pair) for pair in pairs)))
        panels[key].append((case_id,state,pairs))
        row={'case_id':case_id,'dataset':dataset,'construction_digest':state.construction_digest,
             'state_input_sha256':analysis['state_input_sha256'],'action':analysis['action'],
             'organizational_sha256':analysis['organizational_sha256'],'accepted_reference_equal':reference is not None,
             'state_unchanged':True}
        rows.append(row)
        with Path('STATE_JOURNAL.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')

    def synthetic(row,relabel=False):
        caps=[tuple(c) for c in row['caps']];edges=[tuple(e) for e in row['edges']]
        if relabel:
            p=row['permutation'];new_caps=[None]*len(p)
            for old,new in enumerate(p):new_caps[new]=caps[old]
            caps=new_caps;new_edges=[]
            for u,v,a,b in edges:
                u,v=p[u],p[v]
                new_edges.append((u,v,a,b) if u<v else (v,u,b,a))
            edges=sorted(new_edges)
        state=SimpleNamespace(owner_caps=caps,typed_edges=edges,top_pairs=[(u,v) for u,v,a,b in edges],
            total_caps=tuple(map(sum,zip(*caps))),construction_digest=digest(['synthetic',row['case_id'],caps,edges]),
            skin=digest(['synthetic-resource',list(map(sum,zip(*caps)))]),lane='SYNTHETIC',motif_id=row['case_id'],
            leaf_count=len(caps),relation_count_total=len(edges))
        inspect_state(row['case_id'],state,row['bridge_pairs'],references[row['case_id']])
    for row in frozen['cases']:synthetic(row);synthetic_count+=1
    for row in frozen['relabel_cases']:synthetic(row,True);synthetic_count+=1
    assert synthetic_count==protocol['expected_synthetic_states']
    # Load the actual preserved six lower roots through the exact existing loader.
    # Producer identities stay fixed; these states are not relabelled as R100.
    root=Path('actual_saved_lower_packet');root.mkdir()
    (root/'CERTIFIED_CARRIER_SET_V1.json').write_bytes(Path(inputs['actual_lower_manifest']).read_bytes())
    (root/'EXACT_STATE_DAG_V1.json').write_bytes(Path(inputs['actual_lower_dag']).read_bytes())
    manifest=json.loads((root/'CERTIFIED_CARRIER_SET_V1.json').read_text())
    first=manifest['carriers'][0]
    identity={key:first[key] for key in ['source_sha256','registry_sha256','producer_experiment_id','authority_stage','authority_science_sha256']}
    loaded,loaded_manifest=load_certified_carrier_set(root,**{'expected_'+key:value for key,value in identity.items()})
    assert loaded_manifest==manifest and len(loaded)==protocol['expected_actual_lower_states']
    actual_rows=[]
    pairs=list(md._PROCESS_O7_RUNTIME['bridge_pairs'])
    for row in manifest['carriers']:
        state=loaded[row['carrier_id']];public=row['public_state']
        assert state.level==public['level'] and state.execution_policy==public['execution_policy']=='NATIVE_V3'
        assert state.construction_digest==public['construction_digest'] and state.skin==public['resource_skin_sha256']
        assert json_tree(state.owner_caps)==public['owner_caps'] and list(state.total_caps)==public['total_caps']
        assert public['formal_G1_G3_authority'] is False and state.level in [7,8,9]
        inspect_state('actual:'+row['carrier_id'],state,pairs,dataset='actual_lower')
        actual_rows.append({'carrier_id':row['carrier_id'],'level':state.level,'construction_digest':state.construction_digest,
            'public_descriptor_equal':True,'formal_G1_G3_authority':False})
    actual_unique=len({row['construction_digest'] for row in actual_rows})
    assert actual_unique==protocol['expected_actual_unique_constructions']
    write('ACTUAL_LOWER_PACKET_INTEGRATION.json',{'status':'PASS','carrier_count':len(actual_rows),
        'unique_constructions':actual_unique,'duplicate_root_occurrences':len(actual_rows)-actual_unique,'producer_identity':identity,
        'carriers':actual_rows,'data_regenerated':False,'current_R100_qualified':False})
    panel_results=[];disconnected_panels=[];duplicate_panels=[]
    for panel_key,items in sorted(panels.items()):
        states=[item[1] for item in items];pairs=items[0][2]
        by_construction=defaultdict(list)
        for item in items:by_construction[item[1].construction_digest].append(item)
        if len(by_construction)!=len(items):
            assert panel_key[0]=='actual_lower', 'UNEXPECTED_SYNTHETIC_CONSTRUCTION_ALIAS'
            try:mp.scan_level_from_analysis(states,panel_key[1],pairs,'FINITE_V2_INTEGRATION',analyses)
            except ValueError as exc:assert 'NONEMPTY_UNIQUE_STATES_REQUIRED' in str(exc)
            else:raise AssertionError('DUPLICATE_CONSTRUCTION_PANEL_ACCEPTED')
            for occurrences in by_construction.values():
                assert all(mp.state_wire(item[1])==mp.state_wire(occurrences[0][1]) for item in occurrences)
            duplicate_panels.append({'panel':panel_key,'root_occurrence_count':len(items),
                'unique_construction_count':len(by_construction),'duplicate_root_occurrences':len(items)-len(by_construction),
                'reducer_refuses_duplicate_roots':True,'no_occurrence_weighted_reducer_summary_created':True,
                'construction_to_case_ids':{key:[item[0] for item in values] for key,values in sorted(by_construction.items())}})
            # This is an explicit construction-set panel, not an occurrence-weighted census.
            items=[values[0] for key,values in sorted(by_construction.items())]
            states=[item[1] for item in items]
        if not panel_key[2]:
            for route,call in [('reducer',lambda:mp.scan_level_from_analysis(states,panel_key[1],pairs,'FINITE_V2_INTEGRATION',analyses)),
                               ('serial',lambda:rs._scan_level(states,panel_key[1],pairs,'FINITE_V2_INTEGRATION'))]:
                try:call()
                except ValueError as exc:assert 'CONNECTED_PANEL_REQUIRED' in str(exc)
                else:raise AssertionError('DISCONNECTED_PANEL_ACCEPTED:'+route)
            disconnected_panels.append({'panel':panel_key,'state_count':len(states),'case_ids':[item[0] for item in items],
                'serial_and_reducer_refuse':True,'no_summary_created':True})
            continue
        parallel=mp.scan_level_from_analysis(states,panel_key[1],pairs,'FINITE_V2_INTEGRATION',analyses)
        serial=rs._scan_level(states,panel_key[1],pairs,'FINITE_V2_INTEGRATION')
        if parallel!=serial:
            write('FIRST_FAILURE.json',{'panel':panel_key,'parallel':parallel,'serial':serial})
            raise AssertionError('FULL_SERIAL_REDUCER_PACKET_MISMATCH')
        assert parallel[0]['diagnostic_schema']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2
        assert parallel[0]['normalized_signature']['schema_id']==rs.ORGANIZATIONAL_SCHEMA_V2
        assert all(value=='PERSISTS' for value in rs._classify_genealogy(serial[0],parallel[0]).values())
        panel_results.append({'panel':panel_key,'state_count':len(states),'case_ids':[item[0] for item in items],
            'full_raw_and_normalized_equal':True,'packet_sha256':digest(json_tree(parallel[0])),
            'serial':json_tree(serial[0]),'reducer':json_tree(parallel[0])})
    compared=sum(row['state_count'] for row in panel_results)
    disconnected=sum(row['state_count'] for row in disconnected_panels)
    duplicates=sum(row['duplicate_root_occurrences'] for row in duplicate_panels)
    assert compared+disconnected+duplicates==synthetic_count+len(actual_rows)
    assert compared==protocol['expected_compared_states'] and disconnected==protocol['expected_disconnected_probe_states']
    assert len(panel_results)==protocol['expected_serial_reducer_panels']
    assert duplicates==protocol['expected_duplicate_root_occurrences']
    assert len(duplicate_panels)==protocol['expected_duplicate_panel_refusals']
    write('STATE_RESULTS.json',{'status':'PASS','synthetic_states':synthetic_count,'actual_lower_states':len(actual_rows),'states':rows})
    write('SERIAL_REDUCER_COMPARISONS.json',{'status':'PASS','panel_count':len(panel_results),'state_count':compared,
        'disconnected_refused_panels':len(disconnected_panels),'disconnected_probe_states':disconnected,
        'duplicate_root_occurrences':duplicates,'duplicate_refused_panels':len(duplicate_panels),
        'duplicate_panels':duplicate_panels,'disconnected_panels':disconnected_panels,'panels':panel_results})
    result={'schema_id':'IG_STEP3_V2_INTEGRATION_AUDIT_RESULT_V2','status':'PASS','scientific_outcome':protocol['scientific_outcome'],
        'synthetic_worker_comparisons':synthetic_count,'actual_lower_states':len(actual_rows),'serial_reducer_panels':len(panel_results),
        'states_covered_by_panels':compared,'disconnected_refused_panels':len(disconnected_panels),
        'disconnected_probe_states':disconnected,'authority_refusals':len(refusals),
        'actual_unique_constructions':actual_unique,'duplicate_root_occurrences':duplicates,
        'duplicate_refused_panels':len(duplicate_panels),
        'registered_callable_changes':changed,'semantic_baseline_status':'REOPEN_REQUIRED','theorem_status':'STALE_REOPEN_REQUIRED',
        'baseline_updated':False,'calibration_adopted':False,'actual_current_R100_cohort_qualified':False,
        'full_generation_allowed':False,'historical_jobs_replayed':False,'actual_data_regenerated':False,
        'wall_seconds':time.monotonic()-started,'scope':protocol['scope']}
    write('QUALIFICATION_RESULT.json',result);print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
