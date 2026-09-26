from pathlib import Path
import hashlib
import json

import pytest

from infinity_grid import certified_base as cb


def add(store, content=b'{"n":1}', *, category="core_objects", producer=None,
        schema="IG_FIXTURE_V1", meaning="fixture.object", domain="engineering.fixture",
        coverage=None, invariants=None, canonicalization="CANONICAL_JSON_V1"):
    return cb.add_block(
        store, category=category, content=content,
        canonicalization=canonicalization, schema_id=schema,
        meaning=meaning, domain=domain,
        coverage=coverage or ["fixture.n"],
        invariants=invariants or ["fixture.n.integer"], producer=producer,
    )


def certify(store, block_id):
    return cb.certify_block(store, block_id, authority="DECODER.V07.STEP2",
                            basis=["exact.bytes", "declared.semantics", "declared.invariants"])


def test_same_content_across_producer_versions_is_one_admissible_block(tmp_path):
    first = add(tmp_path, producer={"decoder_version": "0.6.2.dev2", "test_version": "G7.C9C.V1"})
    second = add(tmp_path, producer={"decoder_version": "0.7.0.dev2", "test_version": "G7.C9C.V2"})
    assert first["block_id"] == second["block_id"]
    assert first["physical_object_created"] is True
    assert second["physical_object_created"] is False
    assert first["producer_record_id"] != second["producer_record_id"]
    certify(tmp_path, first["block_id"])
    result = cb.admit_block(tmp_path, first["block_id"], schema_id="IG_FIXTURE_V1",
                            meaning="fixture.object", domain="engineering.fixture",
                            required_coverage=["fixture.n"],
                            required_invariants=["fixture.n.integer"])
    assert result["status"] == "DATA_BLOCK_ADMITTED"


def test_canonical_json_deduplicates_key_order(tmp_path):
    one = add(tmp_path, b'{"b":2,"a":1}')
    two = add(tmp_path, b'{ "a": 1, "b": 2 }')
    assert one["content_sha256"] == two["content_sha256"]
    assert one["block_id"] == two["block_id"]
    assert len(list((tmp_path / "objects").glob("*.bin"))) == 1


def test_changed_bytes_are_a_new_block(tmp_path):
    one = add(tmp_path, b'{"n":1}')
    two = add(tmp_path, b'{"n":2}')
    assert one["content_sha256"] != two["content_sha256"]
    assert one["block_id"] != two["block_id"]


def test_corrupted_physical_bytes_refuse(tmp_path):
    row = add(tmp_path)
    block = cb.load_block(tmp_path, row["block_id"])
    (tmp_path / "objects" / (block["content_sha256"] + ".bin")).write_bytes(b"corrupt")
    with pytest.raises(cb.CertifiedBaseError, match="DATA_OBJECT_CORRUPT"):
        cb.load_block(tmp_path, row["block_id"])


@pytest.mark.parametrize("field,value", [
    ("schema_id", "IG_OTHER_V1"),
    ("meaning", "other.meaning"),
    ("domain", "other.domain"),
])
def test_incompatible_semantics_refuse(tmp_path, field, value):
    row = add(tmp_path); certify(tmp_path, row["block_id"])
    args = dict(schema_id="IG_FIXTURE_V1", meaning="fixture.object",
                domain="engineering.fixture", required_coverage=["fixture.n"],
                required_invariants=["fixture.n.integer"])
    args[field] = value
    with pytest.raises(cb.CertifiedBaseError, match="DATA_SEMANTICS_INCOMPATIBLE"):
        cb.admit_block(tmp_path, row["block_id"], **args)


def test_missing_coverage_and_invariant_refuse(tmp_path):
    row = add(tmp_path); certify(tmp_path, row["block_id"])
    with pytest.raises(cb.CertifiedBaseError, match="DATA_COVERAGE_MISSING"):
        cb.admit_block(tmp_path, row["block_id"], schema_id="IG_FIXTURE_V1",
                       meaning="fixture.object", domain="engineering.fixture",
                       required_coverage=["fixture.absent"], required_invariants=[])
    with pytest.raises(cb.CertifiedBaseError, match="DATA_INVARIANT_MISSING"):
        cb.admit_block(tmp_path, row["block_id"], schema_id="IG_FIXTURE_V1",
                       meaning="fixture.object", domain="engineering.fixture",
                       required_coverage=[], required_invariants=["fixture.absent"])


def test_revocation_is_exact_and_blocks_dependents(tmp_path):
    bad = add(tmp_path, b'{"n":1}'); good = add(tmp_path, b'{"n":2}')
    certify(tmp_path, bad["block_id"]); certify(tmp_path, good["block_id"])
    cb.revoke(tmp_path, bad["block_id"], reason="fixture.defect", evidence=["TICKET-1"])
    with pytest.raises(cb.CertifiedBaseError, match="DATA_BLOCK_REVOKED"):
        cb.admit_block(tmp_path, bad["block_id"], schema_id="IG_FIXTURE_V1",
                       meaning="fixture.object", domain="engineering.fixture",
                       required_coverage=[], required_invariants=[])
    assert cb.admit_block(tmp_path, good["block_id"], schema_id="IG_FIXTURE_V1",
                          meaning="fixture.object", domain="engineering.fixture",
                          required_coverage=[], required_invariants=[])["status"] == "DATA_BLOCK_ADMITTED"
    with pytest.raises(cb.CertifiedBaseError, match="CERTIFIED_BASE_CONTAINS_REVOKED"):
        cb.create_base(tmp_path, name="BAD.BASE", block_ids=[bad["block_id"]])


def test_clean_materialization_and_tamper_refusal(tmp_path):
    ids = []
    for category, n in (("core_objects", 1), ("core_relations", 2),
                        ("schemas_and_dictionaries", 3)):
        row = add(tmp_path, json.dumps({"n": n}).encode(), category=category)
        certify(tmp_path, row["block_id"]); ids.append(row["block_id"])
    base = cb.create_base(tmp_path, name="ENGINEERING.BASE.V1", block_ids=ids, activate=True)
    result = cb.materialize_base(tmp_path, base["base_id"], tmp_path / "clean")
    assert result["materialization_sha256"] == base["materialization_sha256"]
    active = json.loads((tmp_path / "ACTIVE_BASE.json").read_text())
    assert active["base_id"] == base["base_id"]
    catalog = cb.load_base(tmp_path, base["base_id"])
    victim = tmp_path / "clean" / catalog["materialization_files"][0]["path"]
    victim.write_bytes(b"tampered")
    with pytest.raises(cb.CertifiedBaseError, match="CERTIFIED_BASE_FILE_CORRUPT"):
        cb.verify_materialization(tmp_path / "clean", catalog)


def test_base_plus_delta_reconstructs_exact_target_and_refcounts(tmp_path):
    rows = [add(tmp_path, json.dumps({"n": n}).encode()) for n in range(3)]
    for row in rows: certify(tmp_path, row["block_id"])
    base1 = cb.create_base(tmp_path, name="BASE.V1", block_ids=[r["block_id"] for r in rows[:2]])
    base2 = cb.create_base(tmp_path, name="BASE.V2", block_ids=[r["block_id"] for r in rows[1:]])
    delta = cb.create_delta(tmp_path, base1["base_id"], base2["base_id"])
    result = cb.materialize_delta(tmp_path, delta["delta_id"], tmp_path / "from_delta")
    assert result["materialization_sha256"] == base2["materialization_sha256"]
    counts = cb.reference_counts(tmp_path)
    assert counts[rows[1]["content_sha256"]] == 2
    plan = cb.compaction_plan(tmp_path, [base2["base_id"]])
    assert plan["action"] == "DRY_RUN_NO_DELETION"
    assert rows[0]["content_sha256"] in plan["unreferenced_content_sha256"]
    assert (tmp_path / "objects" / (rows[0]["content_sha256"] + ".bin")).is_file()


def test_representative_existing_inputs_are_measured_not_modified(tmp_path):
    source = Path(__file__).parent
    fixtures = [
        (source / "fixtures/G6_S8_PREREGISTRATION_V3.json", "schemas_and_dictionaries",
         "g6.s8.preregistration", "g6.s8", "IG_G6_S8_PREREGISTRATION_V3"),
        (source / "fixtures/g6_s7d2_a30_certified_carriers.json", "public_states",
         "g6.s7.certified.carriers", "g6.s7", "IG_G6_S7_CARRIERS_V1"),
        (source.parent / "infinity_grid/resources/g6/authority/G6__S2R.json", "reference_corpora",
         "g6.s2.authority", "g6.s2", "IG_G6_S2_AUTHORITY_V1"),
        (source.parent / "infinity_grid/resources/engineering/EXACT_TREE_KERNEL_E1_FIXTURE.json", "positive_controls",
         "engineering.tree.kernel.fixture", "engineering", "IG_TREE_KERNEL_FIXTURE_V1"),
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path, *_ in fixtures}
    ids = []
    for path, category, meaning, domain, schema in fixtures:
        row = cb.add_block(tmp_path, category=category, content=path.read_bytes(),
                           canonicalization="CANONICAL_JSON_V1", schema_id=schema,
                           meaning=meaning, domain=domain, coverage=[meaning],
                           invariants=["source.fixture.exact"],
                           producer={"source_path": path.relative_to(source.parent).as_posix(),
                                     "decoder_version": "0.7.0.dev2"})
        certify(tmp_path, row["block_id"]); ids.append(row["block_id"])
        assert cb.read_block(tmp_path, row["block_id"], consumer="STEP2.REPRESENTATIVE.READ")
    base = cb.create_base(tmp_path, name="DECODER.ENGINEERING.BASE.V1", block_ids=ids, activate=True)
    assert cb.read_audit_report(tmp_path) == {
        "schema_id": "IG_DECODER_DATA_READ_AUDIT_REPORT_V1", "event_count": 4,
        "block_ids": sorted(ids), "consumers": ["STEP2.REPRESENTATIVE.READ"]}
    assert cb.load_base(tmp_path, base["base_id"])["name"] == "DECODER.ENGINEERING.BASE.V1"
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path, *_ in fixtures}


def test_raw_large_fixture_can_be_certified_without_json_rewrite(tmp_path):
    fixture = Path(__file__).parent / "fixtures/repair7r_completed_state.zip"
    before = hashlib.sha256(fixture.read_bytes()).hexdigest()
    row = add(tmp_path, fixture.read_bytes(), category="reference_corpora",
              canonicalization="RAW_BYTES_V1", schema="IG_PORTABLE_WORKSPACE_ZIP_V1",
              meaning="decoder.completed.workspace", domain="decoder.recovery",
              coverage=["repair7r.completed"], invariants=["archive.sha256.exact"])
    certify(tmp_path, row["block_id"])
    assert cb.read_block(tmp_path, row["block_id"], consumer="REPAIR7R.REUSE") == fixture.read_bytes()
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == before
