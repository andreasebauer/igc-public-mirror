"""Fixed maximum cases in the five C1 pair lanes not covered by dev75."""
import json
import zipfile

from infinity_grid.o7_c1_outer_action_profile import independent_materialization_profile
from infinity_grid.o7_c1_typed_baseline import reconstruct_original_case
from infinity_grid.o7_owner_materialization import materialize_owner
from infinity_grid.o7_edge_canonicalization import canonicalize_edges
from infinity_grid.o7_parent_orbits import rebuild_parent_orbits
from infinity_grid.o7_legacy_library_comparison import ARCHIVE


def _check(tmp_path, index, lane):
    with zipfile.ZipFile(ARCHIVE) as archive:
        member = next(n for n in archive.namelist()
                      if n.endswith('/C1_EXACT_EXTERNAL_CLOSURE.json'))
        rows = json.loads(archive.read(member))['O7xO7']
    choices = sorted((r for r in rows if '+'.join(r['lanes']) == lane),
                     key=lambda r: (r['accounting']['K6'], r['accounting']['m7'],
                                    r['case_index']))
    assert choices[-1]['case_index'] == index

    def checked(ctx, edges, templates, pairs):
        from sys import modules
        reference = modules['_ig_c1_original_pinned_engine']
        for owner, parent in enumerate(ctx.parents):
            assert materialize_owner(parent.h6, edges, owner) == reference.materialize_owner(parent.h6, edges, owner)
        return independent_materialization_profile(ctx, edges, templates, pairs)

    def checked_canonicalizer(ctx, raw):
        from sys import modules
        reference = modules['_ig_c1_original_pinned_engine']
        result = canonicalize_edges(ctx, raw)
        assert result == reference.canonicalize_edges(ctx, raw)
        return result

    def checked_parents(reference):
        rebuilt = rebuild_parent_orbits(reference)
        assert rebuilt.keys() == reference.keys()
        for key in reference:
            assert rebuilt[key].base_p2k == reference[key].base_p2k
            assert rebuilt[key].base_k2p == reference[key].base_k2p
        return rebuilt

    result = reconstruct_original_case(tmp_path, kind='O7xO7', index=index,
                                       profile_calculator=checked,
                                       canonicalizer=checked_canonicalizer,
                                       parent_transform=checked_parents)
    assert result['status'] == 'PASS', result
    assert all(result['checks'].values())


def test_het4_het4(tmp_path):
    _check(tmp_path, 18, 'HET4+HET4')


def test_het4_twin4(tmp_path):
    _check(tmp_path, 50, 'HET4+TWIN4')


def test_hom6_hom6(tmp_path):
    _check(tmp_path, 45, 'HOM6+HOM6')


def test_hom6_twin4(tmp_path):
    _check(tmp_path, 40, 'HOM6+TWIN4')


def test_twin4_twin4(tmp_path):
    _check(tmp_path, 53, 'TWIN4+TWIN4')
