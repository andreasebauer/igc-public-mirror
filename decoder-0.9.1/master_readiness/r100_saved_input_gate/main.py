"""Registered read-only admission control; no R100 producer or G3 science executes."""
import hashlib
import json
import os
from pathlib import Path
from infinity_grid import materialized_discovery as discovery
from infinity_grid import regime_scanner as rs
from infinity_grid import semantic_sentinel as sentinel
from infinity_grid import uplift_campaign as campaign

def save(name,value):Path(name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def main():
    inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    parameters=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    def raw(name):
        value=Path(inputs[name]).read_bytes()
        assert hashlib.sha256(value).hexdigest()==parameters['input_hashes'][name]
        return value
    manifest=json.loads(raw('carrier_manifest'))
    writer=json.loads(raw('carrier_writer_result'))
    completion=json.loads(raw('carrier_writer_completion'))
    capture=json.loads(raw('carrier_writer_capture'))
    assert completion['status']=='COMPLETED' and completion['evidence_status']=='VERIFIED'
    assert completion['completion_sha256']==parameters['writer_completion_sha256']
    assert capture['capture_id']==parameters['writer_capture_id']
    assert writer['status']=='PASS' and writer['manifest_sha256']==parameters['input_hashes']['carrier_manifest']
    assert writer['bundle_sha256']==parameters['input_hashes']['carrier_bundle']
    assert manifest['carrier_count']==6 and writer['carrier_count']==6
    assert discovery._PROCESS_O7_RUNTIME is None
    assert all(not getattr(rs,n) for n in ['_O7_DATA_CACHE','_O7_SKIN_CACHE','_O7_OWNER_CAP_CACHE'])
    folder=Path('saved_carriers');folder.mkdir()
    (folder/'CERTIFIED_CARRIER_SET_V1.json').write_bytes(raw('carrier_manifest'))
    (folder/'EXACT_STATE_DAG_V1.json').write_bytes(raw('carrier_bundle'))
    try:
        campaign.run_native_campaign_stage(run_root='campaign',experiment_id='G3:S0',inputs={},requested_workers=1,
            r100_carrier_root=folder,r100_producer_binding=writer['producer_binding'])
    except campaign.UpliftCampaignError as exc:
        assert str(exc)=='R100_COMPLETE_193_COHORT_REQUIRED'
        refusal={'schema_id':'IG_MASTER_R100_ACTUAL_CAMPAIGN_INPUT_REFUSAL_V1',
            'status':'EXPECTED_REFUSAL','reason':str(exc),'input_carrier_count':6,'required_carrier_count':193,
            'authority_loaded':False,'scientific_stage_started':False,'regeneration_started':False}
    else:raise AssertionError('six lower roots admitted as complete current R100')
    assert discovery._PROCESS_O7_RUNTIME is None
    assert all(not getattr(rs,n) for n in ['_O7_DATA_CACHE','_O7_SKIN_CACHE','_O7_OWNER_CAP_CACHE'])
    assert not Path('campaign/CAMPAIGN_STATE.json').exists()
    parent=json.loads(raw('parent_semantic_snapshot'))
    current=sentinel.native_semantics_snapshot()
    assert current==parent and len(current['actions'])==4
    verification=sentinel.verify_native_semantics(raise_on_change=False)
    prior=json.loads(raw('parent_semantic_baseline_status'))
    assert verification==prior and verification['status']=='REOPEN_REQUIRED'
    assert verification['comparison']['changes']==[{'action_id':'RESOURCE_SCANNER_RELATION_ADD_AGGREGATE_V1',
        'change':'ACTION_CONTRACT_OR_IMPLEMENTATION_CHANGED','fields':['callable_semantic_sha256']}]
    save('CAMPAIGN_INPUT_REFUSAL.json',refusal)
    save('SEMANTIC_PARENT_CORRESPONDENCE.json',{'schema_id':'IG_MASTER_R100_LITERAL_SENTINEL_CORRESPONDENCE_V1',
        'status':'PASS','registered_action_count':4,'parent_snapshot_sha256':parent['science_sha256'],
        'current_snapshot_sha256':current['science_sha256'],'all_parent_hashes_preserved':True,
        'existing_baseline_status':'REOPEN_REQUIRED','baseline_updated':False,'guard_weakened':False})
    save('SEMANTIC_BASELINE_STATUS.json',verification)
    result={'schema_id':'IG_MASTER_R100_SAVED_INPUT_GATE_RESULT_V1','status':'PASS',
        'scientific_outcome':'R100_ACTUAL_CAMPAIGN_SAVED_INPUT_REFUSAL_AND_PARENT_HASH_CORRESPONDENCE_EARNED_FOR_DECLARED_SCOPE',
        'input_carrier_count':6,'required_carrier_count':193,'expected_refusal_count':1,
        'authority_loaded':False,'regeneration_started':False,'scientific_stage_started':False,
        'all_four_parent_action_hashes_preserved':True,'existing_semantic_baseline_status':'REOPEN_REQUIRED',
        'actual_current_R100_cohort_qualified':False,'actual_G1_G3_consumer_qualified':False,
        'pipeline_qualified':False,'full_generation_allowed':False,'historical_V1_rebased':False,
        'writer_capture_id':parameters['writer_capture_id'],'writer_completion_sha256':parameters['writer_completion_sha256']}
    save('QUALIFICATION_RESULT.json',result);print(json.dumps(result))

if __name__=='__main__':main()
