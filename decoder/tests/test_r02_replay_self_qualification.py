from __future__ import annotations

import json
from pathlib import Path

from infinity_grid import __version__


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "infinity_grid/resources/replay/R02_REPLAY_SELF_QUALIFICATION_MANIFEST_V1.json"


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_r02_manifest_remains_a_dev47_record_and_does_not_claim_full_completion():
    manifest = load_manifest()
    # The current identity is tested centrally; this record keeps its historical pin.
    assert manifest["decoder_version"] == "0.8.0.dev47+lib"
    assert manifest["status"] == "PORTABLE_R02_CORE_PASS_WITH_22_DECLARED_NONPORTABLE_GAPS"
    assert manifest["r02_complete"] is False
    assert manifest["production_regressions_detected"] == 0


def test_r02_portable_and_compatibility_counts_are_exact():
    runs = load_manifest()["qualification_runs"]
    assert runs["portable_core"]["passed"] == 340
    assert runs["portable_core"]["skipped"] == 1
    assert runs["portable_core"]["deselected"] == 3
    assert runs["dev33_archive_compatibility"]["passed"] == 9
    assert runs["dev33_archive_compatibility"]["deselected"] == 4
    assert runs["p1_compiler_regression"]["passed"] == 57


def test_all_22_gaps_are_named_once_and_category_counts_agree():
    manifest = load_manifest()
    gaps = manifest["nonportable_gaps"]
    nodes = [node for group in gaps for node in group["nodes"]]
    assert sum(group["count"] for group in gaps) == manifest["gap_count"] == 22
    assert all(group["count"] == len(group["nodes"]) for group in gaps)
    assert len(nodes) == len(set(nodes)) == 22


def test_historical_archives_and_non_authoritative_capture_are_pinned():
    manifest = load_manifest()
    assert len(manifest["historical_archives"]) == 3
    assert all(len(row["sha256"]) == 64 for row in manifest["historical_archives"])
    assert manifest["fixture_note"]["status"] == "SAVE_REQUIRED"
    assert manifest["fixture_note"]["authority"] == "NON_AUTHORITATIVE_FIXTURE_ACCESS_ONLY"


def test_manifest_defines_pass_as_reproduction_not_confirmation():
    manifest = load_manifest()
    assert "reproduced" in manifest["interpretation"]
    assert "does not independently confirm" in manifest["interpretation"]
    assert "Full R02 remains incomplete" in manifest["conclusion"]
