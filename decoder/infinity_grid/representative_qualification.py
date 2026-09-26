"""Native engineering-stage qualification for representative-byte regression.

No alternate ingress or runtime permit is created. The registered dispatcher
provides the runtime; every case uses its genuine output binding and permit.
This handler requires pytest only in a declared engineering test environment.
"""
from types import SimpleNamespace
from .v05_origin_guard import require_controller_execution_origin


def handler(stage, runtime):
    require_controller_execution_origin('representative-byte-qualification')
    if stage['stage_id'] != 'ENG:REPRESENTATIVE_QUALIFICATION':
        raise RuntimeError('REPRESENTATIVE_QUALIFICATION_STAGE_REQUIRED')
    import pytest
    from . import _representative_qualification_cases as cases
    names = ('test_257_distinct_then_duplicate_needs_no_controller_reopen', 'test_cold_multicore_over_256_classes_keeps_exact_grouping_and_durable_bytes', 'test_partial_resume_reuses_durable_rep_bytes_without_reopen', 'test_legacy_null_rep_binds_view_once_backfills_then_reuses')
    result = {}
    for name in names:
        with pytest.MonkeyPatch.context() as patches:
            getattr(cases, name)(patches, runtime._chain_dir, runtime)
        result[name] = 'PASS'
    return SimpleNamespace(result={'outcome': 'PASS', 'cases': result})
