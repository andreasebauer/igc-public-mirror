"""Engineering fixtures only. No R100 scientific authority is supplied here."""
import copy
import hashlib
import json
from types import SimpleNamespace
import pytest
from infinity_grid import carrier_reuse as cr
from infinity_grid import r100_reuse as rr
from infinity_grid import uplift_campaign as uc
from infinity_grid.canon import canonical_sha256

BINDING=dict(source_sha256='1'*64,registry_sha256='2'*64,producer_experiment_id='ENGINEERING_ONLY',
             authority_stage='ENGINEERING_ONLY',authority_science_sha256='3'*64)
def alias(value):return hashlib.sha256(str(value).encode()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def seal_manifest(m):m['artifact_sha256']=canonical_sha256({k:v for k,v in m.items() if k!='artifact_sha256'})
def seal_dag(d):d['science_sha256']=canonical_sha256({k:v for k,v in d.items() if k!='science_sha256'})
def fixture(root):
    root.mkdir(exist_ok=True);nodes={};child=alias('leaf')
    nodes[child]=dict(kind='O7',construction_digest=child,execution_policy='NATIVE_V3',execution_order_key=child,
                      lane='ENGINEERING_ONLY',source_id='ENGINEERING_ONLY',reserve_counts=[0]*7)
    for level in range(8,100):
        key=alias(('chain',level));nodes[key]=dict(kind='LIFT',construction_digest=key,execution_policy='NATIVE_V3',
          execution_order_key=key,lane='ENGINEERING_ONLY',level=level,children=[child],top_edges_full=[],motif_id='ENGINEERING_ONLY');child=key
    rows=[];states={}
    for i,key in enumerate(sorted(alias(('root',i)) for i in range(193))):
        nodes[key]=dict(kind='LIFT',construction_digest=key,execution_policy='NATIVE_V3',execution_order_key=key,
                       lane='ENGINEERING_ONLY',level=100,children=[child],top_edges_full=[],motif_id='ENGINEERING_ONLY')
        state=SimpleNamespace(level=100,execution_policy='NATIVE_V3',construction_digest=key,skin=alias(('skin',i)),total_caps=[2**200+i]*7)
        states[key]=state;p=rr.public_descriptor(state)
        rows.append(dict(carrier_id=key,construction_digest=key,exact_carrier_root_digest=key,exact_carrier_payload='EXACT_STATE_DAG_V1.json',
                         public_state=p,public_state_sha256=canonical_sha256(p),certification_status='CERTIFIED_PASS',**BINDING))
    dag=dict(schema_id='IG_MATURATION_STATE_DAG_V2',execution_policy='NATIVE_V3',roots=list(states),nodes=nodes);seal_dag(dag)
    manifest=dict(schema_id='CERTIFIED_CARRIER_SET_V1',exact_payload_schema='IG_MATURATION_STATE_DAG_V1',status='CERTIFIED_PASS',carrier_count=193,carriers=rows);seal_manifest(manifest)
    write(root/'CERTIFIED_CARRIER_SET_V1.json',manifest);write(root/'EXACT_STATE_DAG_V1.json',dag)
    return states,manifest,dag
class Session:
    instances=[]
    def __init__(self,**kwargs):self.closed=False;self.advance_calls=0;self.engine=object();self.bridge_pairs=[(0,0)];self.instances.append(self)
    def close(self):self.closed=True
    def advance_to(self,level):self.advance_calls+=1;raise AssertionError('unexpected regeneration')
def campaign(monkeypatch,root):
    monkeypatch.setattr(uc,'load_uplift_experiment_registry',lambda:{'registry_sha256':'2'*64,'experiments':[]})
    monkeypatch.setattr(uc,'MaterializedDiscoverySession',Session);Session.instances=[]
    return uc.UpliftCampaignEngine(root/'run',requested_workers=1,r100_carrier_root=root,r100_producer_binding=BINDING)
def test_complete_metadata_preflight_loads_no_authority(tmp_path,monkeypatch):
    states,m,d=fixture(tmp_path)
    monkeypatch.setattr(cr,'_load_exact_state_bundle',lambda *a,**k:pytest.fail('authority loaded'))
    ticket=rr.preflight(tmp_path,BINDING);assert len(ticket.ordered_roots)==193
    assert ticket.manifest_sha256==hashlib.sha256((tmp_path/'CERTIFIED_CARRIER_SET_V1.json').read_bytes()).hexdigest()
def test_saved_campaign_reuses_one_load_without_regeneration(tmp_path,monkeypatch):
    states,m,d=fixture(tmp_path);calls=[]
    monkeypatch.setattr(cr,'load_certified_carrier_set',lambda *a,**k:(calls.append(k) or (states,m)))
    engine=campaign(monkeypatch,tmp_path)
    with engine:
        a=engine.ensure_g1_r100_population();b=engine.ensure_g1_r100_population()
        assert [x.construction_digest for x in a]==list(states) and all(x is y for x,y in zip(a,b))
        a.clear();engine.carrier_map.clear();assert len(engine.ensure_g1_r100_population())==193 and len(engine.carrier_map)==193
        assert engine.materialization_count==0 and engine.r100_reuse_count==1 and len(calls)==1
        assert Session.instances[0].advance_calls==0
    assert Session.instances[0].closed
    with pytest.raises(uc.UpliftCampaignError,match='NO_REGENERATION'):engine.ensure_g1_r100_population()
def test_invalid_metadata_refuses_before_opening_session(tmp_path,monkeypatch):
    _,m,_=fixture(tmp_path);m['carriers']=m['carriers'][:6];m['carrier_count']=6;seal_manifest(m);write(tmp_path/'CERTIFIED_CARRIER_SET_V1.json',m)
    engine=campaign(monkeypatch,tmp_path)
    with pytest.raises(uc.UpliftCampaignError,match='193_COHORT'):engine.__enter__()
    assert not Session.instances and engine.materialization_count==0 and engine.r100_reuse_count==0
    with pytest.raises(uc.UpliftCampaignError,match='NO_REGENERATION'):engine.ensure_g1_r100_population()
def test_provenance_refuses_before_session_or_decode(tmp_path,monkeypatch):
    fixture(tmp_path);engine=campaign(monkeypatch,tmp_path);engine._r100_producer_binding['source_sha256']='f'*64
    with pytest.raises(uc.UpliftCampaignError,match='identity mismatch'):engine.__enter__()
    assert not Session.instances
def test_public_scope_policy_and_exact_vector_required(tmp_path):
    _,m,_=fixture(tmp_path)
    for key,value in [('source_stage','OTHER'),('level',True),('level',99),('execution_policy','HISTORICAL_COORDINATE_V1'),('total_caps',[6.0]*7),('total_caps',[False]*7)]:
        bad=copy.deepcopy(m);bad['carriers'][0]['public_state'][key]=value
        bad['carriers'][0]['public_state_sha256']=canonical_sha256(bad['carriers'][0]['public_state']);seal_manifest(bad);write(tmp_path/'CERTIFIED_CARRIER_SET_V1.json',bad)
        with pytest.raises(rr.R100ReuseError):rr.preflight(tmp_path,BINDING)
def test_native_policy_and_root_closure_required(tmp_path):
    _,m,d=fixture(tmp_path)
    bad=copy.deepcopy(d);bad['roots'][0],bad['roots'][1]=bad['roots'][1],bad['roots'][0];seal_dag(bad);write(tmp_path/'EXACT_STATE_DAG_V1.json',bad)
    with pytest.raises(rr.R100ReuseError,match='ORDERED_ROOT'):rr.preflight(tmp_path,BINDING)
    bad=copy.deepcopy(d);bad['execution_policy']='HISTORICAL_COORDINATE_V1'
    for row in bad['nodes'].values():row['execution_policy']='HISTORICAL_COORDINATE_V1'
    seal_dag(bad);write(tmp_path/'EXACT_STATE_DAG_V1.json',bad)
    with pytest.raises(rr.R100ReuseError,match='CURRENT_DAG_POLICY'):rr.preflight(tmp_path,BINDING)
def test_root_ids_and_count_cannot_hide_duplicates(tmp_path):
    _,m,_=fixture(tmp_path);bad=copy.deepcopy(m)
    bad['carriers'][0]['carrier_id']='ORDINAL';seal_manifest(bad);write(tmp_path/'CERTIFIED_CARRIER_SET_V1.json',bad)
    with pytest.raises(rr.R100ReuseError,match='CONSTRUCTION_IDS'):rr.preflight(tmp_path,BINDING)
    bad=copy.deepcopy(m);bad['carrier_count']=193.0;seal_manifest(bad);write(tmp_path/'CERTIFIED_CARRIER_SET_V1.json',bad)
    with pytest.raises(rr.R100ReuseError,match='EXACT_POSITIVE_COUNT'):rr.preflight(tmp_path,BINDING)
def test_changed_input_refuses_before_loader(tmp_path,monkeypatch):
    fixture(tmp_path);ticket=rr.preflight(tmp_path,BINDING)
    with (tmp_path/'EXACT_STATE_DAG_V1.json').open('a') as f:f.write(' ')
    monkeypatch.setattr(cr,'load_certified_carrier_set',lambda *a,**k:pytest.fail('decoder loaded'))
    with pytest.raises(rr.R100ReuseError,match='INPUT_CHANGED'):rr.load(ticket)
def test_loaded_public_mismatch_closes_session_without_cache(tmp_path,monkeypatch):
    states,m,_=fixture(tmp_path);next(iter(states.values())).total_caps[0]+=1
    monkeypatch.setattr(cr,'load_certified_carrier_set',lambda *a,**k:(states,m));engine=campaign(monkeypatch,tmp_path)
    with pytest.raises(uc.UpliftCampaignError,match='PUBLIC_CORRESPONDENCE'):engine.__enter__()
    assert engine.session is None and engine._r100_carriers is None and engine._carrier_map is None
    assert Session.instances[0].closed and engine.materialization_count==0 and engine.r100_reuse_count==0
def test_configuration_pair_and_legacy_path_remain_explicit(tmp_path,monkeypatch):
    with pytest.raises(uc.UpliftCampaignError,match='REQUIRED_TOGETHER'):uc.UpliftCampaignEngine(tmp_path,r100_carrier_root=tmp_path)
    monkeypatch.setattr(uc,'load_uplift_experiment_registry',lambda:{'experiments':[]});engine=uc.UpliftCampaignEngine(tmp_path,requested_workers=1)
    engine.session=SimpleNamespace(advance_to=lambda n:None,level_states={99:[]},engine=None,bridge_pairs=[],motifs=[],spec={})
    states=[SimpleNamespace(construction_digest=alias(i)) for i in range(193)];count=[]
    monkeypatch.setattr(uc,'parallel_candidate_descriptors',lambda **kwargs:([],None,None,None))
    monkeypatch.setattr(uc,'rebuild_selected_states',lambda *a,**k:(count.append(1) or states))
    assert len(engine.ensure_g1_r100_population())==193 and engine.materialization_count==1 and engine.r100_reuse_count==0
    assert len(engine.ensure_g1_r100_population())==193 and len(count)==1
def test_campaign_stage_threads_saved_cohort_to_real_dispatch(tmp_path,monkeypatch):
    states,m,_=fixture(tmp_path);monkeypatch.setattr(cr,'load_certified_carrier_set',lambda *a,**k:(states,m))
    monkeypatch.setattr(uc,'load_uplift_experiment_registry',lambda:{'experiments':[]})
    monkeypatch.setattr(uc,'MaterializedDiscoverySession',Session);Session.instances=[]
    called=[]
    def stage(*,engine,**inputs):
        called.append(engine.r100_reuse_count)
        assert len(engine.ensure_g1_r100_population())==193 and engine.materialization_count==0 and engine.r100_reuse_count==1
        return {'status':'PASS','engineering_only':True}
    monkeypatch.setattr(uc,'run_g3_s0_native',stage)
    result=uc.run_native_campaign_stage(run_root=tmp_path/'run',experiment_id='G3:S0',inputs={},requested_workers=1,
                                      r100_carrier_root=tmp_path,r100_producer_binding=BINDING)
    assert result['status']=='COMPLETE' and called==[1] and Session.instances[0].advance_calls==0 and Session.instances[0].closed
def test_stage_refusal_creates_no_running_science_or_fallback(tmp_path,monkeypatch):
    _,m,_=fixture(tmp_path);m['carriers']=m['carriers'][:6];m['carrier_count']=6;seal_manifest(m);write(tmp_path/'CERTIFIED_CARRIER_SET_V1.json',m)
    monkeypatch.setattr(uc,'load_uplift_experiment_registry',lambda:{'experiments':[]})
    monkeypatch.setattr(uc,'MaterializedDiscoverySession',Session);Session.instances=[]
    monkeypatch.setattr(uc,'run_g3_s0_native',lambda **k:pytest.fail('science dispatched'))
    with pytest.raises(uc.UpliftCampaignError,match='193_COHORT'):
        uc.run_native_campaign_stage(run_root=tmp_path/'run',experiment_id='G3:S0',inputs={},requested_workers=1,
                                    r100_carrier_root=tmp_path,r100_producer_binding=BINDING)
    assert not Session.instances and not (tmp_path/'run/CAMPAIGN_STATE.json').exists()
def test_literal_sentinel_preserves_all_registered_callable_hashes():
    from infinity_grid import semantic_sentinel as sentinel
    snapshot=sentinel.native_semantics_snapshot()
    # Exact snapshot of the qualified parent5e3f, obtained by read-only source inspection.
    assert snapshot['science_sha256']=='1bcf3b5d3b22933fec35e5684f76114ed85f04603f6aacca124d141506504357'
    assert len(snapshot['actions'])==4
    result=sentinel.verify_native_semantics(raise_on_change=False)
    assert result['status']=='REOPEN_REQUIRED'
    assert result['comparison']['changes']==[{'action_id':'RESOURCE_SCANNER_RELATION_ADD_AGGREGATE_V1',
        'change':'ACTION_CONTRACT_OR_IMPLEMENTATION_CHANGED','fields':['callable_semantic_sha256']}]
    with pytest.raises(sentinel.NativeSemanticsReopenRequired):sentinel.verify_native_semantics()
def test_literal_sentinel_refuses_unknown_module_and_campaign_preflight_passes():
    from pathlib import Path
    from infinity_grid import semantic_sentinel as sentinel, workflow_guard
    with pytest.raises(sentinel.NativeSemanticsError,match='unregistered callable module'):
        sentinel._resolve_callable('os:path')
    source=Path(uc.__file__).resolve().parent.parent
    result=workflow_guard.preflight(source,[source/'infinity_grid/uplift_campaign.py'])
    assert result['status']=='PASS' and any(x['module_path'].endswith('/semantic_sentinel.py') for x in result['modules'])
