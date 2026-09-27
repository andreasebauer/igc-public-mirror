"""Preselected C1 lane extension for the bounded hybrid outer-action check."""
import json
import zipfile

from infinity_grid.o7_c1_outer_action_profile import hybrid_profile
from infinity_grid.o7_c1_typed_baseline import reconstruct_original_case
from infinity_grid.o7_legacy_library_comparison import ARCHIVE


def _case(kind, lane):
    with zipfile.ZipFile(ARCHIVE) as z:
        member = next(n for n in z.namelist() if n.endswith('/C1_EXACT_EXTERNAL_CLOSURE.json'))
        rows = json.loads(z.read(member))[kind]
    selected = [r for r in rows if r.get('lane', '+'.join(r.get('lanes', []))) == lane]
    return min(selected, key=lambda r: (r['accounting']['K6'], r['accounting']['m7'], r['case_index']))


def _check(tmp_path, kind, lane, index):
    assert _case(kind, lane)['case_index'] == index
    result = reconstruct_original_case(tmp_path, kind=kind, index=index,
                                       profile_calculator=hybrid_profile)
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
