"""Prospective native SCRIPT audit for the changed V2 diagnostic source."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from infinity_grid import regime_scanner as rs
from infinity_grid import semantic_sentinel as ss
from infinity_grid import graph_library_backend as backend
from infinity_grid.theorem_registry import verify_all_theorems, StaleTheoremError
from oracle import summary, owner_maps, normalized_edge, digest


def write(name,value):
    Path(name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


class ProbeState:
    def __init__(self,caps,edges):
        self._caps=[tuple(row) for row in caps]
        self._edges=list(edges)
        self._top=[(u,v) for u,v,a,b in edges]
        self.reads=set()
    @property
    def owner_caps(self):
        self.reads.add('owner_caps');return self._caps
    @property
    def typed_edges(self):
        self.reads.add('typed_edges');return self._edges
    @property
    def top_pairs(self):
        self.reads.add('top_pairs');return self._top
    def __getattr__(self,name):raise AssertionError('UNDECLARED_STATE_READ:'+name)
    def value(self):return (tuple(self._caps),tuple(self._edges),tuple(self._top))


def main():
    started=time.monotonic()
    inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for name,expected in params['input_hashes'].items():
        assert hashlib.sha256(Path(inputs[name]).read_bytes()).hexdigest()==expected
    protocol=json.loads(Path(inputs['audit_protocol']).read_text())
    cases=json.loads(Path(inputs['extended_cases']).read_text())
    prior=json.loads(Path(inputs['v1_accepted_results']).read_text())
    prior_by_id={row['case_id']:row['expected'] for row in prior['cases']}
    prior_relabel=json.loads(Path(inputs['v1_relabel_control']).read_text())
    parent=json.loads(Path(inputs['parent_semantic_snapshot']).read_text())
    current=ss.native_semantics_snapshot()
    current_by_id={row['action_id']:row for row in current['actions']}
    assert all(current_by_id[row['action_id']]['callable_semantic_sha256']==row['callable_semantic_sha256']
               for row in parent['actions'])
    status=ss.verify_native_semantics(raise_on_change=False)
    assert status['status']=='REOPEN_REQUIRED'
    refusals=[]
    for name,call in [('native_semantics',ss.verify_native_semantics),('action_contract',ss.current_action_contract)]:
        try:call()
        except ss.NativeSemanticsReopenRequired:refusals.append(name)
        else:raise AssertionError('GUARD_DID_NOT_REFUSE:'+name)
    theorems=verify_all_theorems(raise_on_stale=False)
    assert theorems['status']=='STALE_REOPEN_REQUIRED' and len(theorems['stale_theorems'])==protocol['expected_stale_theorems']
    try:verify_all_theorems()
    except StaleTheoremError:refusals.append('theorem_registry')
    else:raise AssertionError('THEOREM_GUARD_DID_NOT_REFUSE')
    registry=ss.action_registry()
    diagnostic_rows=[row for row in registry['actions'] if row['action_id'].startswith('RESOURCE_SCANNER_')]
    assert len(diagnostic_rows)==3
    assert all(row['scientific_role']!='RELATION_ADD_SEMANTICS' and row['successor_semantics']=='NONE_DIAGNOSTICS_ONLY'
               and row['features']['hidden_topology_read'] for row in diagnostic_rows)
    write('SEMANTIC_BASELINE_STATUS.json',status)
    write('THEOREM_PREMISE_STATUS.json',theorems)
    write('REGISTRY_SCOPE.json',{'status':'PASS','registry_sha256':registry['registry_sha256'],
        'diagnostics':diagnostic_rows,'complete_transition_counter':False,'baseline_adopted':False})
    rows=[];mode_counts=Counter();observed_reads=set();organizations={}
    core_fields=['legal_action_labels','total_action_copies','owner_pair_support','action_orbits','automorphism_size','owner_orbits']

    def check(case_id,caps,edges,pairs,modes):
        caps=tuple(tuple(row) for row in caps);edges=tuple(tuple(edge) for edge in edges);pairs=tuple(tuple(pair) for pair in pairs)
        expected=summary(caps,edges,pairs);maps=list(owner_maps(caps,edges));actuals={};orgs={}
        for mode in modes:
            st=ProbeState(caps,edges);before=st.value();kwargs={} if mode=='default' else {'automorphism_backend':mode}
            actual=rs._action_aggregate_v2(st,list(pairs),**kwargs)
            actual_maps=rs._automorphisms(len(caps),[tuple(int(x>0) for x in row) for row in caps],list(edges),backend=None if mode=='default' else mode)
            if actual!=expected or actual_maps!=maps or st.value()!=before:
                write('FIRST_FAILURE.json',{'case_id':case_id,'mode':mode,'expected':expected,'actual':actual,
                    'expected_maps':maps,'actual_maps':actual_maps,'caps':caps,'edges':edges,'pairs':pairs,'immutable':st.value()==before})
                raise AssertionError('V2_ORACLE_GROUP_OR_IMMUTABILITY_MISMATCH:'+case_id+':'+mode)
            orgs[mode]=rs._state_organizational_signature_v2(st,actual)
            actuals[mode]=actual;mode_counts[mode]+=1;observed_reads.update(st.reads)
        assert len({digest(org) for org in orgs.values()})==1
        assert all(expected[k]==prior_by_id[case_id][k] for k in core_fields)
        organizations[case_id]=orgs['default']
        row={'case_id':case_id,'input_sha256':digest([caps,edges,pairs]),'expected':expected,'actual_by_mode':actuals,
             'exact_owner_maps':maps,'organizational_signature':orgs['default'],'input_unchanged':True,'v1_physical_counts_and_orbits_preserved':True}
        rows.append(row)
        with Path('CASE_JOURNAL.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        return expected

    original={}
    magnitude_controls=[]
    for row in cases['cases']:
        expected=check(row['case_id'],row['caps'],row['edges'],row['bridge_pairs'],row['modes'])
        original[row['case_id']]=expected
        scaled=ProbeState([[3*x for x in c] for c in row['caps']],[tuple(e) for e in row['edges']])
        scaled_action=rs._action_aggregate_v2(scaled,row['bridge_pairs'])
        assert scaled_action['total_action_copies']==9*expected['total_action_copies']
        assert all(scaled_action[k]==v for k,v in expected.items() if k!='total_action_copies')
        assert rs._state_organizational_signature_v2(scaled,scaled_action)==organizations[row['case_id']]
        magnitude_controls.append({'case_id':row['case_id'],'copies_factor':9,'organization_unchanged':True})
    assert len(original)==protocol['expected_extended_cases']
    relabels=[]
    for row in cases['relabel_cases']:
        p=row['permutation'];caps=[None]*len(p)
        for old,new in enumerate(p):caps[new]=row['caps'][old]
        edges=sorted(normalized_edge(p[u],p[v],a,b) for u,v,a,b in row['edges'])
        got=check(row['case_id'],caps,edges,row['bridge_pairs'],['legacy','igraph_bliss','default'])
        assert got==original[row['original_case_id']], 'FULL_DIAGNOSTIC_RELABEL_FAILURE:'+row['case_id']
        assert organizations[row['case_id']]==organizations[row['original_case_id']], 'ORGANIZATION_RELABEL_FAILURE:'+row['case_id']
        relabels.append({'case_id':row['case_id'],'original_case_id':row['original_case_id'],'permutation':p,
                        'all_diagnostic_fields_equal':True,'organizational_signature_equal':True})
    assert len(relabels)==protocol['expected_relabels']
    outside=0
    for row in cases['cases']:
        if 'igraph_bliss' in row['modes']:continue
        try:rs._action_aggregate_v2(ProbeState(row['caps'],[tuple(e) for e in row['edges']]),row['bridge_pairs'],automorphism_backend='igraph_bliss')
        except ValueError as exc:
            assert str(exc)=='Outside qualified igraph automorphism domain';outside+=1
        else:raise AssertionError('OUTSIDE_DOMAIN_ACCEPTED')
    assert outside==protocol['expected_outside_refusals']
    path,star=original['control:path4'],original['control:star4']
    assert all(path[k]==star[k] for k in ['legal_action_labels','total_action_copies','type_pair_support','owner_pair_support'])
    assert path!=star and observed_reads=={'owner_caps','top_pairs','typed_edges'}
    assert len(rows)==protocol['expected_cases'] and sum(mode_counts.values())==protocol['expected_comparisons']
    write('CASE_RESULTS.json',{'schema_id':'IG_DIAGNOSTIC_V2_FINITE_CASE_RESULTS_V1','status':'PASS','case_count':len(rows),'cases':rows})
    write('RELABEL_AND_MAGNITUDE_CONTROLS.json',{'status':'PASS','relabel_cases':len(relabels),'relabels':relabels,
        'complete_diagnostic_relabelling_invariant_on_saved_family':True,'organization_relabelling_invariant_on_saved_family':True,
        'magnitude_controls':magnitude_controls,'legacy_v1_counterexample_count':prior_relabel['counterexample_count'],
        'legacy_v1_results_unchanged':True})
    write('TOPOLOGY_SCOPE_CONTROL.json',{'status':'PASS','path4':path,'star4':star,'observed_state_reads':sorted(observed_reads),
        'diagnostic_topology_sensitive':True,'matched_physical_counts_equal':True,'diagnostic_is_complete_transition_counter':False})
    result={'schema_id':'IG_DIAGNOSTIC_V2_FINITE_AUDIT_RESULT_V1','status':'PASS','scientific_outcome':protocol['scientific_outcome'],
        'cases':len(rows),'summary_comparisons':sum(mode_counts.values()),'exact_owner_group_comparisons':sum(mode_counts.values()),
        'modes':dict(mode_counts),'relabel_cases':len(relabels),'magnitude_controls':len(magnitude_controls),'outside_domain_refusals':outside,
        'guard_refusals':refusals,'all_parent_action_callable_hashes_preserved':True,'registry_scope_repaired':True,
        'complete_diagnostic_relabelling_invariant_on_saved_family':True,'organization_relabelling_invariant_on_saved_family':True,
        'semantic_baseline_status':'REOPEN_REQUIRED','theorem_status':'STALE_REOPEN_REQUIRED','theorems_stale':theorems['stale_theorems'],
        'baseline_updated':False,'guard_weakened':False,'actual_current_R100_cohort_qualified':False,'full_generation_allowed':False,
        'historical_jobs_replayed':False,'case_results_sha256':digest(rows),'backend_usage_observed':backend.usage(),
        'wall_seconds':time.monotonic()-started,'scope':protocol['scope']}
    write('QUALIFICATION_RESULT.json',result);print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
