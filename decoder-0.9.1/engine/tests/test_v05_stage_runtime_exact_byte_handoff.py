from __future__ import annotations

import hashlib

import infinity_grid.v05_stage_runtime as rt
from infinity_grid.structural_encoding import structural_canonical_bytes


def test_partition_worker_hands_exact_signature_bytes_to_controller():
    old_eval=rt._PARTITION_EVALUATOR
    old_override=rt._PARTITION_ALLOW_TEST_DIGEST_OVERRIDE
    try:
        rt._PARTITION_EVALUATOR=lambda payload: {'signature':('PASS',payload['x']),'metrics':{'x':payload['x']}}
        rt._PARTITION_ALLOW_TEST_DIGEST_OVERRIDE=False
        out=rt._partition_worker({'x':17})
    finally:
        rt._PARTITION_EVALUATOR=old_eval
        rt._PARTITION_ALLOW_TEST_DIGEST_OVERRIDE=old_override
    expected=structural_canonical_bytes(('PASS',17))
    assert out['signature_canonical_bytes']==expected
    assert out['signature_sha256']==hashlib.sha256(expected).hexdigest()
    assert out['engineering_digest_override'] is False


def test_partition_controller_fresh_current_signature_is_not_reopened():
    src=open(rt.__file__,encoding='utf-8').read()
    assert 'cur_sig = result.get("signature_canonical_bytes")' in src
    assert 'cur_sig = self._evaluate_signature_bytes(evaluator_ref, task.payload)' not in src
    assert 'rep_sig = self._evaluate_signature_bytes(evaluator_ref, task_by_id[rep_tid].payload)' in src


def test_runtime_telemetry_uses_phase_cumulative_rate_not_completion_burst(monkeypatch, tmp_path):
    import infinity_grid.runtime_telemetry as tm
    now={'t':1000.0}
    monkeypatch.setattr(tm.time,'monotonic',lambda:now['t'])
    rt=tm.RuntimeTelemetry(tmp_path/'telemetry','TEST',wall_interval_seconds=999)
    rt.phase('SCIENCE_CENSUS',tasks_total=64,kernels_total=64)
    now['t']=1700.0
    rt._tasks_completed=8; rt._kernels_completed=8
    # A bursty pair of recent samples would imply 400/s under the old estimator.
    rt._completion_samples=[(1699.98,4,4),(1699.99,8,8)]
    tr,kr=rt._rates()
    assert abs(tr-(8/700))<1e-12
    assert abs(kr-(8/700))<1e-12
    eta,confidence=rt._eta(kr)
    assert confidence=='MEDIUM'
    assert eta>4000
