"""Native finite consumer integration; only new motif qualification is constructed."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import integration as ig
from infinity_grid import complete_resource_read as read
from infinity_grid import materialized_counter_reference as ref, maturation_parallel as mp
from infinity_grid import regime_scanner as rs, semantic_sentinel as ss, theorem_registry as tr
from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid.v05_execution_authority import digest as source_digest, source_tree_digest


def parent_evidence(path,expected_capture,expected_completion):
    with zipfile.ZipFile(path) as z:items={n:z.read(n) for n in z.namelist()}
    cap=json.loads(items['CAPTURE.json']);done=json.loads(items['COMPLETION.json'])
    assert cap['capture_id']==expected_capture==canonical_sha256({k:v for k,v in cap.items() if k!='capture_id'})
    assert done['completion_sha256']==expected_completion==canonical_sha256({k:v for k,v in done.items() if k!='completion_sha256'})
    assert done['status']=='COMPLETED' and done['evidence_status']=='VERIFIED'
    assert done['source_sha256']==cap['workspace']['source_sha256']
    assert done['registration_sha256']==cap['job']['registration_sha256']
    evidence={r['path']:r for r in done['evidence']}
    for name,raw in items.items():
        if name in ('CAPTURE.json','COMPLETION.json','PRESERVED.json','PACKAGE_MODULES.json'):continue
        row=evidence['script/work/'+name]
        assert hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row['size_bytes']
    preserved=json.loads(items['PRESERVED.json'])
    assert preserved['capture_id']==expected_capture and preserved['outbox']['status']=='PRESERVED'
    assert preserved['outbox']['pending_bytes']==0
    if 'PACKAGE_MODULES.json' in items:
        # Compatibility check for reusing old projections after the additive Q09
        # complete-read module. This does not rebind the parent's source identity.
        package=Path(read.__file__).parent
        manifest=json.loads(items['PACKAGE_MODULES.json'])
        assert source_digest({'schema_id':'IG_DECODER_EXECUTABLE_SOURCE_TREE_V1',
            'files':[[k,v] for k,v in sorted(manifest.items(),key=lambda item:Path(item[0]).parts)]})==cap['workspace']['package_sha256']
        for relative,sha in manifest.items():assert hashlib.sha256((package/relative).read_bytes()).hexdigest()==sha
        current={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file() and
                 '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
        assert current-set(manifest)=={'complete_resource_read.py'}
    else:
        assert source_tree_digest(Path(read.__file__).parent)==cap['workspace']['package_sha256']
    return cap,done,items


def main():
    started=time.monotonic()
    inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role,sha in params['input_hashes'].items():assert hashlib.sha256(Path(inputs[role]).read_bytes()).hexdigest()==sha
    contract=json.loads(Path(inputs['consumer_contract']).read_text());assert contract==ig.make_contract()
    read_contract=json.loads(Path(inputs['read_contract']).read_text());assert read_contract==read.make_contract()
    cap8,done8,p8=parent_evidence(inputs['q08_evidence'],params['q08_capture'],params['q08_completion'])
    cap9,done9,p9=parent_evidence(inputs['q09_evidence'],params['q09_capture'],params['q09_completion'])
    old=json.loads(p8['TEST_DISPATCHES.json'])['dispatches']
    saved_counters=json.loads(p9['SAVED_COUNTERS.json'])
    assert contract['finite_admission']['counter_anchor_ids']==sorted({r['construction_digest'] for r in saved_counters['rows']})
    assert contract['finite_admission']['saved_diagnostic_cases']==sorted([[r['domain'],r['level']] for r in old])
    assert contract['finite_admission']['parents']=={
        'q08':{'capture_id':cap8['capture_id'],'completion_sha256':done8['completion_sha256']},
        'q09':{'capture_id':cap9['capture_id'],'completion_sha256':done9['completion_sha256']}}
    baseline=ss.verify_native_semantics(raise_on_change=False);theorems=tr.verify_all_theorems(raise_on_stale=False)
    assert baseline['status']=='REOPEN_REQUIRED' and len(theorems['stale_theorems'])==4
    consumed=[ig.consume_saved(old,row['domain'],row['level'],contract,
        {'capture_id':cap8['capture_id'],'completion_sha256':done8['completion_sha256']}) for row in old]
    assert len(consumed)==10 and sum(len(r['original_dispatch']['executions']) for r in consumed)==100
    write_json_atomic(Path('REUSED_DIAGNOSTIC_CONSUMPTION.json'),{'status':'PASS','packets':10,
      'original_test_executions_reused':100,'test_reexecutions':0,'dispatches':consumed})
    with tempfile.TemporaryDirectory(prefix='ig_consumer_q10_') as directory:
        root=Path(directory)
        with zipfile.ZipFile(inputs['saved_inputs']) as z:
            names=z.namelist();assert len(names)==len(set(names))
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names);z.extractall(root)
        from infinity_grid import qualified_discovery_inputs as qi
        binding=json.loads(Path(inputs['saved_binding']).read_text());qi.preflight(root,binding)
        dag=json.loads((root/'PREFIX_DAG.json').read_text());index=json.loads((root/'PREFIX_INDEX.json').read_text())['roots']
        assert not any(row.get('lane')=='BACKBONE' or str(row.get('motif_id','')).startswith('BACKBONE:') for row in dag['nodes'].values())
        engine,states=mp._states_from_dag(dag);pairs=sorted(tuple(p) for p in engine.O6.load_rules()[1])
        mapper=ref.SavedScannerActionMapper();reader=read.CompleteResourceRead(engine,pairs,read_contract)
        bysource={};seen=set()
        def visit(s):
            if id(s) in seen:return
            seen.add(id(s))
            if type(s) is rs.O7State:bysource.setdefault(s.source_id,[]).append(s)
            else:
                for c in s.children:visit(c)
        for s in states:visit(s)
        expected={(r['source_id'],r['type']):r for r in saved_counters['rows']}
        sources=sorted({r['source_id'] for r in saved_counters['rows']})
        counter_exes=[];calls=set()
        def profile(frame,event,arg):
            if event=='call':calls.add((str(frame.f_globals.get('__name__','')),frame.f_code.co_name))
        assert sys.getprofile() is None
        sys.setprofile(profile)
        packets={};families={};representations={};all_new=[]
        try:
            for source in sources:
                s=min(bysource[source],key=lambda s:(sum(s.reserve_counts),s.construction_digest))
                exe=ig.consume_complete(reader,mapper.map(s),s.construction_digest,contract)
                for row in exe.evidence_payload['rows']:
                    original=expected[source,row['type']]
                    assert s.construction_digest==original['construction_digest']
                    got=canonical_sha256(sorted((repr(k),v) for k,v in row['successors']))
                    assert got==original['counter_sha256'] and row['site_occurrences']==original['site_occurrences']
                counter_exes.append(exe)
            selected={level:[s for row,s in zip(index,states) if row['domain']=='selected' and row['level']==level]
                      for level in range(7,14)}
            for level in range(8,14):
                backbone,packet=ig.construct_backbone(engine,selected[level-1],level,contract,pairs)
                assert len(backbone)==7 and len(packet['matched_backbone']['rows'])==7
                # Real native edge witnesses survive new DAG serialization. No
                # historical selected-panel packet is overwritten or relabelled.
                for st in backbone:
                    assert st.lane=='BACKBONE' and st.level==level and st.execution_policy==rs.NATIVE_EXECUTION_POLICY
                    assert all(edge[2:4]==(0,0) for edge in st.top_edges_full)
                    mapper.map(st)
                all_new.extend(backbone);packets[level]=packet
                fam,representation=ig.matched_findings(packet,contract,families.get(level-1))
                families[level]=fam;representations[level]=representation
            windows=[ig.temporal_window(packets,families,level,contract) for level in range(9,13)]
        finally:sys.setprofile(None)
        assert len(counter_exes)==6 and len(all_new)==42 and len(windows)==4
        forbidden=[list(c) for c in sorted(calls) if c[0].endswith('fixed_grammar_transport') or
          c[1] in ('_scan_level','_read_probe','_resource_future_signature','_build_next_panel','_farthest_select','load_earned_regime_laws')]
        assert not forbidden
        assert ('infinity_grid.regime_scanner','_backbone_signature') in calls
        assert ('infinity_grid.regime_scanner','_action_aggregate_v2') in calls
        write_json_atomic(Path('COMPLETE_READ_CONSUMER.json'),{'status':'PASS','anchor_count':6,
          'complete_reservation_counters':42,'unit':read.UNIT,'executions':[ig.as_json(e) for e in counter_exes]})
        write_json_atomic(Path('NEW_MATCHED_BACKBONES.json'),{'status':'PASS','levels':list(range(8,14)),
          'state_count':42,'new_qualification_construction':True,'historical_recovery':False,
          'matching_basis':contract['matching_basis'],'exact_individual_ancestry_matched':False,
          'packets':{str(k):v for k,v in packets.items()}})
        write_json_atomic(Path('NEW_MATCHED_BACKBONE_DAG.json'),mp.state_dag_wire(all_new))
        write_json_atomic(Path('MATCHED_TEST_FINDINGS.json'),{'status':'PASS','family_findings':24,
          'representation_findings':6,'families':{str(l):{r:ig.as_json(e) for r,e in vals.items()} for l,vals in families.items()},
          'representation':{str(l):ig.as_json(e) for l,e in representations.items()}})
        write_json_atomic(Path('TEMPORAL_WINDOWS.json'),{'status':'PASS','closed_windows':4,
          'candidate_windows':sum(e.finding['scientific_status']=='CANDIDATE' for e in windows),
          'promotion_authorized':False,'exact_individual_ancestry_matched':False,
          'matching_basis':contract['matching_basis'],'windows':[ig.as_json(e) for e in windows],
          'terminal_context':{'stopped_at':13,'depth_13_next_depth_check':'NOT_OBSERVED_BY_PREREGISTERED_STOP',
            'depth_8_context':'REFERENCE_ONLY_NO_PRIOR_MATCHED_DEPTH','further_generation_authorized':False}})
        refusals=[]
        def refuse(name,fn,kind=ig.ConsumerError):
            try:fn()
            except kind as exc:refusals.append({'name':name,'reason':str(exc)})
            else:raise AssertionError('missing refusal:'+name)
        for kind,level,caps in [('complete_read',8,[ig.COUNTER]),('matched_backbone',7,[ig.BACKBONE]),
          ('matched_backbone',14,[ig.BACKBONE]),('complete_read',7,[ig.BACKBONE]),
          ('matched_backbone',8,[ig.COUNTER]),('saved_diagnostics',13,[ig.COUNTER]),
          ('historical_backbone',8,[ig.BACKBONE]),('complete_read',7,['RESOURCE_COUNTERS'])]:
            refuse(str((kind,level,caps)),lambda kind=kind,level=level,caps=caps:ig.compile_case(contract,kind,level,caps))
        for caps in ([],[ig.COUNTER,ig.COUNTER]):
            refuse('request:'+str(caps),lambda caps=caps:ig.compile_case(contract,'complete_read',7,caps))
        for key,value in [('production_authority','PRODUCTION'),('generation_allowed',True),
          ('matching_basis','EXACT_ANCESTRY'),('complete_read_contract_sha256','0'*64)]:
            bad=deepcopy(contract);bad[key]=value;bad=ig.sealed({k:v for k,v in bad.items() if k!='science_sha256'})
            refuse(key,lambda bad=bad:ig.compile_case(bad,'complete_read',7,[ig.COUNTER]))
        refuse('unsaved_full_pool_7',lambda:ig.consume_saved(old,'full_pool',7,contract,{}))
        refuse('wrong_saved_parent',lambda:ig.consume_saved(old,'selected',7,contract,{}))
        refuse('unbound_counter_anchor',lambda:ig.consume_complete(reader,mapper.map(s),'UNBOUND',contract))
        refuse('boolean_level',lambda:ig.compile_case(contract,'complete_read',True,[ig.COUNTER]))
        refuse('no_next_depth',lambda:ig.temporal_window(packets,families,13,contract))
        missing=dict(families);missing.pop(10)
        refuse('missing_window_context',lambda:ig.temporal_window(packets,missing,9,contract))
        malformed=deepcopy(packets);malformed[9]['level']=10
        refuse('changed_packet',lambda:ig.temporal_window(malformed,families,9,contract))
        malformed=deepcopy(families);malformed[9].pop(next(iter(malformed[9])))
        refuse('missing_dependency',lambda:ig.temporal_window(packets,malformed,9,contract))
        malformed=deepcopy(families);key=next(iter(malformed[9]));malformed[9][key].finding['regime_ref']='LEGACY'
        refuse('wrong_Regime',lambda:ig.temporal_window(packets,malformed,9,contract))
        malformed=deepcopy(families);key=next(iter(malformed[9]));malformed[9][key].evidence_payload['projection']=[]
        refuse('changed_Finding_evidence',lambda:ig.temporal_window(packets,malformed,9,contract))
        assert len(refusals)==24
        write_json_atomic(Path('REFUSALS.json'),{'status':'PASS','count':len(refusals),'checks':refusals})
        qi.unchanged(root,binding);reader.close()
    for ident in theorems['stale_theorems']:
        try:tr.require_current_theorems((ident,))
        except tr.StaleTheoremError:pass
        else:raise AssertionError('stale theorem admitted')
    assert ss.verify_native_semantics(raise_on_change=False)==baseline
    assert tr.verify_all_theorems(raise_on_stale=False)==theorems
    write_json_atomic(Path('CONSUMER_TRACE.json'),{'status':'PASS','forbidden_calls':forbidden,
      'measured_calls':[list(c) for c in sorted(calls)],'profile_restored':sys.getprofile() is None,
      'new_qualification_backbones':42,'historical_selected_scanner_reruns':0,'prior_Test_reexecutions':0})
    write_json_atomic(Path('QUALIFICATION_RESULT.json'),{'status':'PASS',
      'scientific_outcome':'FINITE_COMPLETE_READ_AND_NEW_MATCHED_CONTEXT_CONSUMER_QUALIFIED',
      'reused_diagnostic_Test_executions':100,'complete_read_anchors':6,'complete_reservation_counters':42,
      'new_backbone_states':42,'matched_family_Findings':24,'representation_Findings':6,
      'closed_temporal_windows':4,'refusal_checks':24,'stale_theorems':4,'historical_backbones_recovered':False,
      'exact_individual_ancestry_matched':False,'production_authority':'NONE','generation_allowed':False,
      'package2_status':'BOUNDED_CONSUMER_COMPLETE_PRODUCTION_CUTOVER_NOT_QUALIFIED',
      'wall_seconds':time.monotonic()-started})


if __name__=='__main__':main()
