"""Data-only planning/refusal tests, not a substitute for the native wave gate."""
from pathlib import Path
import pytest
from infinity_grid import result_contracts as rc, preservation as pr, submission as sub
from infinity_grid.canon import canonical_sha256,write_json_atomic
from infinity_grid.v05_validation_runtime import _planned_wave,ValidationRuntimeError
from tests.test_decoder06_results_preservation import report


def test_wave_limit_is_opt_in_and_registered():
    assert rc.policy({})['validation_wave_selectors']==0
    assert rc.policy({'preservation':{'validation_wave_selectors':1}})['validation_wave_selectors']==1
    for value in (-1,True,'1',1.0):
        with pytest.raises(sub.SubmissionError):rc.policy({'preservation':{'validation_wave_selectors':value}})
    with pytest.raises(sub.SubmissionError):rc.policy({'preservation':{'interval_seconds':0}})


def test_wave_plan_only_selects_missing_and_keeps_original_order(tmp_path):
    selectors=['tests/fixture.py','tests/second.py','tests/third.py']
    collection={'binding':'bound','selectors':[selectors[0]],
                'nodes':['tests/fixture.py::test_pass','tests/fixture.py::test_partial']}
    write_json_atomic(tmp_path/'collections'/'fixture.json',collection)
    report(tmp_path,'tests/fixture.py::test_pass','passed')
    report(tmp_path,'tests/fixture.py::test_partial','passed',finished=False)
    assert _planned_wave(tmp_path,'bound',selectors,1)==['tests/fixture.py::test_partial']
    assert _planned_wave(tmp_path,'bound',selectors,2)==['tests/fixture.py::test_partial','tests/second.py']
    assert _planned_wave(tmp_path,'bound',selectors,0)==['tests/fixture.py::test_partial','tests/second.py','tests/third.py']


def test_prior_failure_or_skip_blocks_next_wave(tmp_path):
    selectors=['tests/fixture.py','tests/second.py']
    for outcome in ('failed','skipped'):
        report(tmp_path,'tests/fixture.py::test_bad',outcome)
        with pytest.raises(ValidationRuntimeError,match='PRIOR_NONPASS'):
            _planned_wave(tmp_path,'bound',selectors,1)


def test_prior_collection_error_blocks_next_wave(tmp_path):
    write_json_atomic(tmp_path/'collection_errors'/'error.json',{'binding':'bound','error':'observed'})
    with pytest.raises(ValidationRuntimeError,match='PRIOR_COLLECTION_ERROR'):
        _planned_wave(tmp_path,'bound',['tests/fixture.py'],1)


def test_wave_drain_requires_logical_roles_even_when_bytes_saved(tmp_path,monkeypatch):
    monkeypatch.setattr(rc,'contract_for',lambda root:{'preservation':{'validation_wave_selectors':1}})
    admission={'workspace':tmp_path,'job':{'execution':{'kind':'VALIDATION'}}}
    with pytest.raises(sub.SubmissionError,match='VALIDATION_SAVE_DRAIN_REQUIRED'):
        pr.require_validation_drain(admission,{'pending_bytes':0,'pending_objects':[{'obligation_id':'pending'}]})
    pr.require_validation_drain(admission,{'pending_objects':[]})


def test_wave_schedule_does_not_remove_single_selector_backlog_limits(tmp_path,monkeypatch):
    limits={'preservation':{'validation_wave_selectors':1}}
    monkeypatch.setattr(rc,'contract_for',lambda root:limits)
    maximum=rc.policy(limits)['max_pending_commits']
    monkeypatch.setattr(pr,'status',lambda root:{'pending_bytes':0,
        'pending_checkpoints':[{}]*maximum,'oldest_pending_age_seconds':0})
    with pytest.raises(sub.SubmissionError,match='SAVE_BACKLOG_PAUSE'):
        pr.backlog(tmp_path,reserve=True)
