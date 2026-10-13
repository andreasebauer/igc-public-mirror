from __future__ import annotations


def test_controller_resume_fallback_binds_declared_s7_kernel_view():
    from infinity_grid.v05_stage_runtime import _bind_controller_fallback_kernel_view
    from infinity_grid.v05_kernel_services import current_kernel_view, clear_kernel_view
    ref = 'infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator'
    clear_kernel_view()
    _bind_controller_fallback_kernel_view(ref, 'a' * 64)
    snap = current_kernel_view().snapshot()
    assert snap['evaluator_ref'] == ref
    assert set(snap['allowed_kernel_services']) == {'AUTHORITY_BASIS','EXACT_RELATION','EXACT_RELATION_PROFILE','EXACT_RELATION_PROFILE_BATCH','EXACT_RELATION_PROFILE_FAMILY'}
    clear_kernel_view()
