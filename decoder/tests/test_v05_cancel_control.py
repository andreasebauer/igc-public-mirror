from __future__ import annotations
import json, inspect
import pytest

def _science_request(rid='science-x'):
    return {'schema_id':'IG_DECODER_PASSIVE_REQUEST_V1','request_id':rid,'registered_job_id':'DECODER.G6.SCIENCE','requested_operation_id':'G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE','parent_source_sha256':'a'*64,'input_artifacts':[]}

def test_cancel_submission_is_data_only(tmp_path):
    from infinity_grid.v05_cancel_control import submit_science_cancel_request,CANCEL_SCHEMA
    req={'schema_id':CANCEL_SCHEMA,'request_id':'cancel-x','operation':'CANCEL_SCIENCE_REQUEST','target_request_id':'science-x','parent_source_sha256':'a'*64}
    got=submit_science_cancel_request(tmp_path,req)
    assert got['state']=='PENDING_PASSIVE_CONTROL'

def test_cancel_archives_pending_science_without_running_it(tmp_path):
    from infinity_grid.v05_passive_intake import submit_passive_request
    from infinity_grid.v05_cancel_control import submit_science_cancel_request,process_next_cancel_request,CANCEL_SCHEMA
    intake=tmp_path/'intake';submit_passive_request(intake,_science_request())
    submit_science_cancel_request(intake,{'schema_id':CANCEL_SCHEMA,'request_id':'cancel-x','operation':'CANCEL_SCIENCE_REQUEST','target_request_id':'science-x','parent_source_sha256':'a'*64})
    got=process_next_cancel_request(tmp_path,expected_source_sha256='a'*64)
    assert got['action']=='ARCHIVED_PENDING_SCIENCE' and got['terminate_controller_child'] is False
    assert (intake/'cancelled_operator/science-x.json').is_file()

def test_cancel_active_science_requests_controller_tree_stop(tmp_path):
    from infinity_grid.v05_passive_intake import submit_passive_request
    from infinity_grid.v05_cancel_control import submit_science_cancel_request,process_next_cancel_request,CANCEL_SCHEMA
    intake=tmp_path/'intake';submit_passive_request(intake,_science_request())
    status=tmp_path/'status';status.mkdir();(status/'ACTIVE_REQUEST.json').write_text(json.dumps({'schema_id':'IG_DECODER_ACTIVE_REQUEST_V1','state':'ACTIVE','request_id':'science-x','registered_job_id':'DECODER.G6.SCIENCE'}))
    submit_science_cancel_request(intake,{'schema_id':CANCEL_SCHEMA,'request_id':'cancel-x','operation':'CANCEL_SCIENCE_REQUEST','target_request_id':'science-x','parent_source_sha256':'a'*64})
    got=process_next_cancel_request(tmp_path,expected_source_sha256='a'*64)
    assert got['terminate_controller_child'] is True

def test_cancel_rejects_non_science_target(tmp_path):
    from infinity_grid.v05_passive_intake import submit_passive_request
    from infinity_grid.v05_cancel_control import submit_science_cancel_request,process_next_cancel_request,CANCEL_SCHEMA,CancelControlError
    req=_science_request();req['registered_job_id']='DECODER.ENGINEERING.SOURCE_CHANGE';req['requested_operation_id']='APPLY_SOURCE_CHANGE';submit_passive_request(tmp_path/'intake',req)
    submit_science_cancel_request(tmp_path/'intake',{'schema_id':CANCEL_SCHEMA,'request_id':'cancel-x','operation':'CANCEL_SCIENCE_REQUEST','target_request_id':'science-x','parent_source_sha256':'a'*64})
    with pytest.raises(CancelControlError,match='CANCEL_TARGET_NOT_SCIENCE'):process_next_cancel_request(tmp_path,expected_source_sha256='a'*64)

def test_supervisor_poll_loop_owns_cancel_and_controller_publishes_active_request():
    import infinity_grid.v05_c6_recovery as r, infinity_grid.v05_controller_event_loop as loop
    assert 'process_next_cancel_request' in inspect.getsource(r.supervisor_main)
    assert 'ACTIVE_REQUEST.json' in inspect.getsource(loop.controller_child_main)
