"""Native change qualification for the full-owner exporter import closure."""
from pathlib import Path
import json
from infinity_grid import workflow_guard
from infinity_grid.adapters.g4_accepted import G4AcceptedAdapter

def test_exact_tree_exporter_import_closure_is_admitted(tmp_path):
    source = Path(__file__).resolve().parents[1]
    # The production exporter imports this real kernel; no scientific function
    # is invoked by preflight. The original source refuses this closure.
    receipt = workflow_guard.preflight(source, [source/'infinity_grid/exact_tree_relation_kernel.py'])
    assert receipt['status'] == 'PASS'
    assert any(row['module_path'].endswith('adapters/g4_accepted.py') for row in receipt['modules'])

def test_static_adapter_preserves_exact_literal_authority():
    source = Path(__file__).resolve().parents[1]/'infinity_grid/resources/uplift'
    literal = json.loads((source/'authority_fixtures/G4_S3_REBASE_V03050/RESULT.json').read_text())
    grammar = json.loads((source/'G4_S6_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1.json').read_text())
    rows = sorted(literal['candidate_interface']['rows'], key=lambda row:row['term_key'])
    expected = tuple((r['term_key'],r['candidate_key_sha256'],tuple(r['caps7'])) for r in rows)
    adapter = G4AcceptedAdapter()
    assert adapter.relation_class_table() == expected
    assert tuple(adapter.operator_basis()) == tuple(sorted(map(tuple,grammar['frozen_grammar']['operator_basis'])))
    assert all(type(n) is int for _,_,caps in expected for n in caps)
    assert max(n.bit_length() for _,_,caps in expected for n in caps) > 64
