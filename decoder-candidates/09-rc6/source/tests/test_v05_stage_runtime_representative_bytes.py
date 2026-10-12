from pathlib import Path
import pytest
from infinity_grid import submission as sub
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid._representative_qualification_cases import test_old_partition_schema_migrates_additively

@pytest.fixture(scope='module')
def qualification(tmp_path_factory):
    source = Path(loop.__file__).resolve().parents[1]
    record = sub.capture_record(source.parent)
    row = next(x for x in record['job']['input_artifacts'] if x['logical_name'] == 'saved_representative_qualification')
    archive = source.parent/'runtime/intake/artifacts'/(row['sha256']+'.bin')
    root = tmp_path_factory.mktemp('native-representative')/'workspace'
    loop.restore_workspace(archive, root, row['sha256'])
    result = loop.run_workspace_job(root, sub.capture_record(root)['job']['job_id'])
    assert result['status'] == 'COMPLETED'
    return result['result']['cases']

def test_257_distinct_then_duplicate_needs_no_controller_reopen(qualification):
    assert qualification["test_257_distinct_then_duplicate_needs_no_controller_reopen"] == "PASS"

def test_cold_multicore_over_256_classes_keeps_exact_grouping_and_durable_bytes(qualification):
    assert qualification["test_cold_multicore_over_256_classes_keeps_exact_grouping_and_durable_bytes"] == "PASS"

def test_partial_resume_reuses_durable_rep_bytes_without_reopen(qualification):
    assert qualification["test_partial_resume_reuses_durable_rep_bytes_without_reopen"] == "PASS"

def test_legacy_null_rep_binds_view_once_backfills_then_reuses(qualification):
    assert qualification["test_legacy_null_rep_binds_view_once_backfills_then_reuses"] == "PASS"

