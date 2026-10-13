"""Preselected C1 lane comparison using the independent O6 observer."""
import json
import zipfile

from infinity_grid.o7_c1_outer_action_profile import independent_o6_hybrid_profile
from infinity_grid.o7_c1_typed_baseline import reconstruct_original_case
from infinity_grid.o7_legacy_library_comparison import ARCHIVE


def _check(tmp_path, kind, lane, index):
    with zipfile.ZipFile(ARCHIVE) as archive:
        member = next(name for name in archive.namelist()
                      if name.endswith('/C1_EXACT_EXTERNAL_CLOSURE.json'))
        selected = [row for row in json.loads(archive.read(member))[kind]
                    if row.get('lane', '+'.join(row.get('lanes', []))) == lane]
    choice = min(selected, key=lambda row: (row['accounting']['K6'],
                                            row['accounting']['m7'], row['case_index']))
    assert choice['case_index'] == index
    result = reconstruct_original_case(tmp_path, kind=kind, index=index,
                                       profile_calculator=independent_o6_hybrid_profile)
    assert result['status'] == 'PASS', result
    assert all(result['checks'].values())


def test_hom6_single(tmp_path):
    _check(tmp_path, 'O7xO6', 'HOM6', 19)


def test_twin4_single(tmp_path):
    _check(tmp_path, 'O7xO6', 'TWIN4', 2)


def test_hom6_twin4_pair(tmp_path):
    _check(tmp_path, 'O7xO7', 'HOM6+TWIN4', 52)


def test_twin4_twin4_pair(tmp_path):
    _check(tmp_path, 'O7xO7', 'TWIN4+TWIN4', 23)
