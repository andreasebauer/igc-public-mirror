from pathlib import Path
import copy
import pytest

from infinity_grid import candidate_workflow as flow

PARENT = '1'*64


def candidate(version, source, files, contracts, outcome='PASS'):
    return flow.make_candidate(version=version, parent_source_sha256=PARENT,
        candidate_source_sha256=source*64, changed_files=files, contracts=contracts,
        test_results=[{'name': 'focused', 'outcome': outcome}], reason='bounded repair')


def test_independent_candidates_merge_without_owner_token(tmp_path):
    a=candidate('0.7.0.dev2','2',['infinity_grid/a.py'],['save.receipts'])
    b=candidate('0.7.0.dev3','3',['infinity_grid/b.py'],['stage.project'])
    for row in (a,b):flow.save_candidate(tmp_path,row)
    plan=flow.merge_plan(tmp_path,[b['record_sha256'],a['record_sha256']])
    assert plan['status']=='MERGEABLE_NONCONFLICTING'
    assert plan['candidate_ids']==sorted([a['record_sha256'],b['record_sha256']])


def test_same_contract_refuses_with_witness(tmp_path):
    rows=[candidate('0.7.0.dev2','2',['a.py'],['save.receipts']),candidate('0.7.0.dev3','3',['b.py'],['save.receipts'])]
    for row in rows:flow.save_candidate(tmp_path,row)
    with pytest.raises(flow.CandidateError,match='CANDIDATE_CONTRACT_CONFLICT:save.receipts'):
        flow.merge_plan(tmp_path,[row['record_sha256'] for row in rows])


def test_same_file_refuses_even_when_contract_labels_differ(tmp_path):
    rows=[candidate('0.7.0.dev2','2',['same.py'],['a']),candidate('0.7.0.dev3','3',['same.py'],['b'])]
    for row in rows:flow.save_candidate(tmp_path,row)
    with pytest.raises(flow.CandidateError,match='CANDIDATE_FILE_CONFLICT:same.py'):
        flow.merge_plan(tmp_path,[row['record_sha256'] for row in rows])


def test_task_local_selection_never_writes_shared_pointer(tmp_path):
    row=candidate('0.7.0.dev2','2',['a.py'],['a']);flow.save_candidate(tmp_path,row)
    result=flow.select_task_local(tmp_path,'PROJECT.1',row['record_sha256'])
    assert result['selection']['scope']=='TASK_LOCAL_ONLY_NOT_SHARED_POINTER'
    assert not (tmp_path/'CURRENT_SHARED.json').exists()
    assert not (tmp_path/'CURRENT_LOCAL.json').exists()


def test_failed_candidate_cannot_be_selected(tmp_path):
    row=candidate('0.7.0.dev2','2',['a.py'],['a'],'FAIL');flow.save_candidate(tmp_path,row)
    with pytest.raises(flow.CandidateError,match='CANDIDATE_TESTS_NOT_PASSING'):
        flow.select_task_local(tmp_path,'PROJECT.1',row['record_sha256'])


def test_candidate_record_is_immutable(tmp_path):
    row=candidate('0.7.0.dev2','2',['a.py'],['a']);flow.save_candidate(tmp_path,row)
    changed=copy.deepcopy(row);changed['reason']='different'
    with pytest.raises(flow.CandidateError,match='CANDIDATE_RECORD_INVALID'):
        flow.save_candidate(tmp_path,changed)


@pytest.mark.parametrize("version", ["0.8.0.dev2+lib", "0.8.0+lib", "0.8lib", "0.7.0.dev4"])
def test_candidate_version_roundtrip(version, tmp_path):
    row = candidate(version, "2", ["a.py"], ["version.contract"])
    saved = flow.save_candidate(tmp_path, row)
    assert flow._load(tmp_path, saved["candidate_id"])["version"] == version


@pytest.mark.parametrize("version", ["0.8.0.dev2+", "0.8.0.dev2+lib+other", "0.8.0.dev2+other", "0.8.0.dev2+lib\n", "../0.8lib", "", None])
def test_invalid_local_candidate_versions_refused(version):
    with pytest.raises(flow.CandidateError, match="CANDIDATE_VERSION_INVALID"):
        candidate(version, "2", ["a.py"], ["version.contract"])


def test_local_suffix_does_not_relax_project_names(tmp_path):
    row = candidate("0.8.0.dev2+lib", "2", ["a.py"], ["version.contract"])
    flow.save_candidate(tmp_path, row)
    with pytest.raises(flow.CandidateError, match="PROJECT_ID_INVALID"):
        flow.select_task_local(tmp_path, "PROJECT+lib", row["record_sha256"])
