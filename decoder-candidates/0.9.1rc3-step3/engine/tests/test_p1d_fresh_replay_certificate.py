from __future__ import annotations

import json
from pathlib import Path

from infinity_grid import __version__


ROOT = Path(__file__).resolve().parents[1]
CERT = ROOT / "infinity_grid/resources/replay/P1D_AUTHORIZED_PREFIX_REPLAY_CERTIFICATE_V1.json"


def test_p1d_certificate_stops_after_g8() -> None:
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    assert cert["status"] == "QUALIFIED_PASS_WAITING_AT_GLOBAL"
    assert cert["authorized_replay_through_layer"] == "G8"
    assert cert["next_wait_layer"] == "GLOBAL"
    assert cert["fail_closed"] is True
    assert cert["decoder_version"] == "0.8.0.dev42+lib"
    # The current identity is tested centrally; this record keeps its historical pin.


def test_p1d_certificate_preserves_evidence_modes_and_counts() -> None:
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    fresh = cert["fresh_recomputation"]
    assert fresh["formal_foundation"]["byte_identical_files"] == 10
    assert fresh["finite_l2"]["completed_checks"] == 9
    assert fresh["finite_l2"]["bridge_truth_table_cases"] == 64 ** 4
    assert fresh["o1_o7"]["completed_targets"] == 10
    assert fresh["o1_o7"]["completed_obligations"] == 36
    assert fresh["g6_named_executors"]["tests_passed"] == 22
    assert cert["g1_g8_assertion_source_integrity"]["assertion_mappings"] == 32
    assert any("not relabelled" in item for item in cert["nonclaims"])
