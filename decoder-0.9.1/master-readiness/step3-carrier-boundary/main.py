"""Invoke only through native capture; bounded carrier handoff is not G1/G3 graduation."""
from copy import deepcopy
import gc
import gzip
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import time
from infinity_grid import carrier_reuse as cr
from infinity_grid import maturation_parallel as mp
from infinity_grid import materialized_discovery as discovery
from infinity_grid import regime_scanner as rs
from infinity_grid.canon import canonical_sha256
from independent_carrier_reference import expected_rows

def save(path, value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')

def public(state):
    return {'schema_id':'IG_MASTER_BOUNDED_CARRIER_PUBLIC_INTERFACE_V1',
        'construction_digest':state.construction_digest,'execution_policy':state.execution_policy,
        'level':state.level,'resource_skin_sha256':state.skin,
        'owner_caps':[list(c) for c in state.owner_caps],'total_caps':list(state.total_caps),
        'formal_G1_G3_authority':False}

def snapshot(roots):
    nodes={};seen=set()
    def visit(state):
        if id(state) in seen:return
        seen.add(id(state))
        row={'kind':'O7' if isinstance(state,rs.O7State) else 'LIFT','level':state.level,
            'lane':state.lane,'execution_policy':state.execution_policy,
            'construction_digest':state.construction_digest,'execution_order_key':rs.execution_order_key(state),
            'owner_caps':state.owner_caps,'total_caps':state.total_caps,'skin':state.skin}
        if isinstance(state,rs.O7State):
            row.update(source_id=state.source_id,parent_ids=state.ctx.source_order,
                parent_colors=[p.exact_color for p in state.ctx.parents],
                original_parent_resources=[p.h6 for p in state.ctx.parents],
                original_coupled_edges=state.edges,reserve_counts=state.reserve_counts)
        else:
            row.update(ordered_children=[c.construction_digest for c in state.children],
                top_edges_full=state.top_edges_full,motif_id=state.motif_id)
            for child in state.children:visit(child)
        exact=json.loads(json.dumps(row));alias=state.construction_digest
        if alias in nodes:assert nodes[alias]==exact
        nodes[alias]=exact
    for root in roots:visit(root)
    return {'schema_id':'IG_MASTER_G3_NATIVE_SNAPSHOT_V1','nodes':nodes}

def main():
    started=time.perf_counter()
    inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    def raw(name):
        b=Path(inputs[name]).read_bytes();assert hashlib.sha256(b).hexdigest()==params['input_hashes'][name];return b
    def load(name):return json.loads(raw(name))
    dag=load('accepted_g3_DAG');native_snapshot=load('accepted_g3_snapshot')
    reference=load('accepted_g3_correspondence');result=load('accepted_g3_result')
    completion=load('accepted_g3_completion');capture=load('accepted_g3_capture')
    assert completion['status']=='COMPLETED' and completion['evidence_status']=='VERIFIED'
    assert completion['completion_sha256']==params['g3_completion_sha256']
    assert capture['capture_id']==params['g3_capture_id']
    assert result['status']=='PASS' and result['DAG_sha256']==params['input_hashes']['accepted_g3_DAG']
    assert result['snapshot_sha256']==params['input_hashes']['accepted_g3_snapshot']
    assert result['correspondence_sha256']==params['input_hashes']['accepted_g3_correspondence']
    assert dag['schema_id']=='IG_MATURATION_STATE_DAG_V2' and dag['execution_policy']=='NATIVE_V3'
    assert dag['science_sha256']==canonical_sha256({k:v for k,v in dag.items() if k!='science_sha256'})
    assert files('infinity_grid').joinpath('resources/decoder/O7_MATERIAL_ROOT_cb6f48641eb9.zip').read_bytes()==raw('o7_seed')
    binding=params['producer_binding']
    assert binding['source_sha256']==completion['source_sha256']
    assert binding['registry_sha256']==completion['registration_sha256']
    assert binding['producer_experiment_id']==capture['job']['job_id']
    assert binding['authority_stage']==capture['job']['question']['stage_id']
    assert binding['authority_science_sha256']==dag['science_sha256']
    assert discovery._PROCESS_O7_RUNTIME is None
    assert all(not getattr(rs,n) for n in ['_O7_DATA_CACHE','_O7_SKIN_CACHE','_O7_OWNER_CAP_CACHE'])
    fresh=True;expected=expected_rows(dag,reference);assert expected['carrier_count']==6
    refusals=[]
    if params['mode']=='WRITER':
        _,roots=mp._states_from_dag(dag)
        assert snapshot(roots)==native_snapshot
        carriers={r['carrier_id']:(state,public(state)) for r,state in zip(expected['ordered_rows'],roots)}
        assert [{'carrier_id':cid,'public_state':p} for cid,(_,p) in carriers.items()]==expected['ordered_rows']
        manifest=cr.save_certified_carrier_set('carrier_set',carriers=carriers,**binding)
        actual=json.loads(Path('carrier_set/EXACT_STATE_DAG_V1.json').read_text())
        assert actual==dag and manifest['carrier_count']==6
        for row,want in zip(manifest['carriers'],expected['ordered_rows']):
            assert row['carrier_id']==want['carrier_id'] and row['public_state']==want['public_state']
        save('PUBLIC_CORRESPONDENCE.json',expected)
        mode_result={'manifest_sha256':hashlib.sha256(Path('carrier_set/CERTIFIED_CARRIER_SET_V1.json').read_bytes()).hexdigest(),
            'bundle_sha256':hashlib.sha256(Path('carrier_set/EXACT_STATE_DAG_V1.json').read_bytes()).hexdigest(),
            'correspondence_sha256':hashlib.sha256(Path('PUBLIC_CORRESPONDENCE.json').read_bytes()).hexdigest(),
            'cold_consumer_qualified':False}
    else:
        assert params['mode']=='COLD'
        manifest=load('accepted_carrier_manifest');bundle=load('accepted_carrier_bundle')
        writer=load('accepted_carrier_writer_result');prior_projection=load('accepted_carrier_public_correspondence')
        assert writer['status']=='PASS' and writer['manifest_sha256']==params['input_hashes']['accepted_carrier_manifest']
        assert writer['bundle_sha256']==params['input_hashes']['accepted_carrier_bundle']
        assert writer['correspondence_sha256']==params['input_hashes']['accepted_carrier_public_correspondence']
        assert prior_projection==expected and bundle==dag
        args={'expected_'+k:v for k,v in binding.items()}
        names=list(binding)+['bool_count','float_count','duplicate_id','public_hash','root_digest',
            'bad_seal','payload_path','extra_field','missing_root','extra_root','swapped_roots']
        assert len(names)==16
        for name in names:
            bad=deepcopy(manifest);bad_dag=deepcopy(bundle);bad_args=dict(args)
            if name in binding:bad_args['expected_'+name]='f'*64 if name.endswith('sha256') else 'DIFFERENT'
            elif name=='bool_count':bad['carrier_count']=True
            elif name=='float_count':bad['carrier_count']=6.0
            elif name=='duplicate_id':bad['carriers'][1]['carrier_id']=bad['carriers'][0]['carrier_id']
            elif name=='public_hash':bad['carriers'][0]['public_state']['total_caps'][0]+=1
            elif name=='root_digest':bad['carriers'][0]['exact_carrier_root_digest']='f'*64
            elif name=='bad_seal':bad['artifact_sha256']='f'*64
            elif name=='payload_path':bad['carriers'][0]['exact_carrier_payload']='../outside.json'
            elif name=='extra_field':bad['extra']='unconsumed'
            elif name=='missing_root':bad_dag['roots'].pop()
            elif name=='extra_root':bad_dag['roots'].append(bad_dag['roots'][0])
            else:bad_dag['roots'][0],bad_dag['roots'][1]=bad_dag['roots'][1],bad_dag['roots'][0]
            if name!='bad_seal':bad['artifact_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='artifact_sha256'})
            bad_dag['science_sha256']=canonical_sha256({k:v for k,v in bad_dag.items() if k!='science_sha256'})
            folder=Path('controls')/name;save(folder/'CERTIFIED_CARRIER_SET_V1.json',bad);save(folder/'EXACT_STATE_DAG_V1.json',bad_dag)
            try:cr.load_certified_carrier_set(folder,**bad_args)
            except cr.CertifiedCarrierReuseError as exc:refusals.append({'control':name,'reason':str(exc),'authority_loaded':False})
            else:raise AssertionError('invalid carrier boundary accepted: '+name)
            assert discovery._PROCESS_O7_RUNTIME is None
        # The genuine historical packet keeps its original V1 schema and identities.
        historical=json.loads(gzip.decompress(raw('historical_bootstrap_gzip')))
        old=historical['dag'];assert historical['level']==100 and historical['candidate_count']==193
        assert old['schema_id']=='IG_MATURATION_STATE_DAG_V1' and len(old['roots'])==193 and len(old['nodes'])==16528
        historical_meta={'schema_id':'IG_MASTER_HISTORICAL_G1_ADMISSION_CONTROL_V1',
            'level':100,'root_count':193,'node_count':16528,'original_schema':old['schema_id'],
            'policy_labels_added':False,'historical_rerun':False}
        p=Path('historical_control/EXACT_STATE_DAG_V1.json');p.parent.mkdir(exist_ok=True)
        with p.open('w') as output:json.dump(old,output,sort_keys=True,separators=(',',':'))
        del old,historical;gc.collect()
        try:cr._load_exact_state_bundle(p.parent,schema='IG_MATURATION_STATE_DAG_V1',payload_name=p.name)
        except cr.CertifiedCarrierReuseError as exc:
            assert 'DAG reconstruction failed' in str(exc)
            historical_meta.update(status='EXPECTED_REFUSAL',reason=str(exc),authority_loaded=False)
            refusals.append({'control':'actual_historical_G1_V1','reason':str(exc),'authority_loaded':False})
        else:raise AssertionError('historical V1 silently admitted as current source')
        assert discovery._PROCESS_O7_RUNTIME is None
        save('HISTORICAL_ADMISSION_CONTROL.json',historical_meta)
        Path('carrier_set').mkdir();Path('carrier_set/CERTIFIED_CARRIER_SET_V1.json').write_bytes(raw('accepted_carrier_manifest'))
        Path('carrier_set/EXACT_STATE_DAG_V1.json').write_bytes(raw('accepted_carrier_bundle'))
        loaded,observed=cr.load_certified_carrier_set('carrier_set',**args)
        assert observed==manifest and list(loaded)==[x['carrier_id'] for x in expected['ordered_rows']]
        roots=[loaded[x['carrier_id']] for x in expected['ordered_rows']]
        assert mp.state_dag_wire(roots)==dag and snapshot(roots)==native_snapshot
        actual=[{'carrier_id':cid,'public_state':public(state)} for cid,state in loaded.items()]
        assert actual==expected['ordered_rows']
        save('COLD_PUBLIC_CORRESPONDENCE.json',expected);save('COLD_NATIVE_SNAPSHOT.json',snapshot(roots))
        save('REFUSAL_CONTROLS.json',{'schema_id':'IG_MASTER_CURRENT_CARRIER_REFUSALS_V1','status':'PASS','control_count':len(refusals),'controls':refusals})
        mode_result={'cold_consumer_qualified':True,'refusal_controls':len(refusals),
            'exact_DAG_roundtrip':True,'exact_full_snapshot_roundtrip':True,'exact_public_correspondence':True,
            'writer_capture_id':params['writer_capture_id'],'writer_completion_sha256':params['writer_completion_sha256']}
    outcome='CURRENT_CARRIER_'+params['mode']+'_CORRESPONDENCE_EARNED_FOR_DECLARED_SCOPE'
    result={'schema_id':'IG_MASTER_CURRENT_CARRIER_'+params['mode']+'_RESULT_V1','status':'PASS',
        'scientific_outcome':outcome,'carrier_count':6,'unique_DAG_nodes':9,'logical_occurrences':22,
        'recursive_endpoint_claims':16,'fresh_cache_entry':fresh,'producer_binding':binding,
        'scope':'Existing carrier save/load API on the earned two-component level7/8/9 DAG; six ordered carrier IDs including duplicate roots',
        'formal_certification_granted':False,'actual_G1_G3_consumer_qualified':False,
        'pipeline_qualified':False,'historical_V1_rebased':False,'full_generation_allowed':False,
        'wall_seconds':str(time.perf_counter()-started),**mode_result}
    save('QUALIFICATION_RESULT.json',result);print(json.dumps(result))

if __name__=='__main__':main()
