"""Frozen remaining 151 C1 case comparisons, selected by case key."""
import json
import zipfile

from infinity_grid.o7_c1_outer_action_profile import independent_materialization_profile
from infinity_grid.o7_c1_typed_baseline import reconstruct_original_case
from infinity_grid.o7_owner_materialization import materialize_owner
from infinity_grid.o7_edge_canonicalization import canonicalize_edges
from infinity_grid.o7_parent_orbits import rebuild_parent_orbits
from infinity_grid.o7_legacy_library_comparison import ARCHIVE


def _check(tmp_path, kind, index, lane=None):
    if lane is not None:
        with zipfile.ZipFile(ARCHIVE) as archive:
            member = next(n for n in archive.namelist()
                          if n.endswith('/C1_EXACT_EXTERNAL_CLOSURE.json'))
            rows = json.loads(archive.read(member))[kind]
        choices = sorted((r for r in rows if r.get('lane', '+'.join(r.get('lanes', []))) == lane),
                         key=lambda r: (r['accounting']['K6'], r['accounting']['m7'],
                                        r['case_index']))
        assert any(choice['case_index'] == index for choice in choices)
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

    result = reconstruct_original_case(tmp_path, kind=kind, index=index,
                                       profile_calculator=checked,
                                       canonicalizer=checked_canonicalizer,
                                       parent_transform=checked_parents)
    assert result['status'] == 'PASS', result
    assert all(result['checks'].values())






def test_o7xo6_44(tmp_path):
    _check(tmp_path, 'O7xO6', 44, 'TWIN4')


def test_o7xo6_46(tmp_path):
    _check(tmp_path, 'O7xO6', 46, 'HOM6')


def test_o7xo6_47(tmp_path):
    _check(tmp_path, 'O7xO6', 47, 'TWIN4')


def test_o7xo6_49(tmp_path):
    _check(tmp_path, 'O7xO6', 49, 'HOM6')


def test_o7xo6_59(tmp_path):
    _check(tmp_path, 'O7xO6', 59, 'TWIN4')


def test_o7xo6_62(tmp_path):
    _check(tmp_path, 'O7xO6', 62, 'TWIN4')


def test_o7xo6_64(tmp_path):
    _check(tmp_path, 'O7xO6', 64, 'HOM6')


def test_o7xo6_67(tmp_path):
    _check(tmp_path, 'O7xO6', 67, 'HOM6')


def test_o7xo6_68(tmp_path):
    _check(tmp_path, 'O7xO6', 68, 'TWIN4')


def test_o7xo6_71(tmp_path):
    _check(tmp_path, 'O7xO6', 71, 'TWIN4')


def test_o7xo6_73(tmp_path):
    _check(tmp_path, 'O7xO6', 73, 'HOM6')


def test_o7xo6_74(tmp_path):
    _check(tmp_path, 'O7xO6', 74, 'TWIN4')


def test_o7xo6_77(tmp_path):
    _check(tmp_path, 'O7xO6', 77, 'TWIN4')


def test_o7xo6_82(tmp_path):
    _check(tmp_path, 'O7xO6', 82, 'HOM6')


def test_o7xo6_83(tmp_path):
    _check(tmp_path, 'O7xO6', 83, 'TWIN4')


def test_o7xo6_85(tmp_path):
    _check(tmp_path, 'O7xO6', 85, 'HOM6')


def test_o7xo6_86(tmp_path):
    _check(tmp_path, 'O7xO6', 86, 'TWIN4')


def test_o7xo6_89(tmp_path):
    _check(tmp_path, 'O7xO6', 89, 'TWIN4')


def test_o7xo6_94(tmp_path):
    _check(tmp_path, 'O7xO6', 94, 'HOM6')


def test_o7xo6_101(tmp_path):
    _check(tmp_path, 'O7xO6', 101, 'TWIN4')


def test_o7xo6_104(tmp_path):
    _check(tmp_path, 'O7xO6', 104, 'TWIN4')


def test_o7xo6_107(tmp_path):
    _check(tmp_path, 'O7xO6', 107, 'TWIN4')


def test_o7xo6_108(tmp_path):
    _check(tmp_path, 'O7xO6', 108, 'HET4')


def test_o7xo6_109(tmp_path):
    _check(tmp_path, 'O7xO6', 109, 'HOM6')


def test_o7xo6_111(tmp_path):
    _check(tmp_path, 'O7xO6', 111, 'HET4')


def test_o7xo6_112(tmp_path):
    _check(tmp_path, 'O7xO6', 112, 'HOM6')


def test_o7xo6_113(tmp_path):
    _check(tmp_path, 'O7xO6', 113, 'TWIN4')


def test_o7xo6_6(tmp_path):
    _check(tmp_path, 'O7xO6', 6, 'HET4')


def test_o7xo6_8(tmp_path):
    _check(tmp_path, 'O7xO6', 8, 'TWIN4')


def test_o7xo6_9(tmp_path):
    _check(tmp_path, 'O7xO6', 9, 'HET4')


def test_o7xo6_10(tmp_path):
    _check(tmp_path, 'O7xO6', 10, 'HOM6')


def test_o7xo6_12(tmp_path):
    _check(tmp_path, 'O7xO6', 12, 'HET4')


def test_o7xo6_13(tmp_path):
    _check(tmp_path, 'O7xO6', 13, 'HOM6')


def test_o7xo6_14(tmp_path):
    _check(tmp_path, 'O7xO6', 14, 'TWIN4')


def test_o7xo6_15(tmp_path):
    _check(tmp_path, 'O7xO6', 15, 'HET4')


def test_o7xo6_17(tmp_path):
    _check(tmp_path, 'O7xO6', 17, 'TWIN4')


def test_o7xo6_18(tmp_path):
    _check(tmp_path, 'O7xO6', 18, 'HET4')


def test_o7xo6_20(tmp_path):
    _check(tmp_path, 'O7xO6', 20, 'TWIN4')


def test_o7xo6_24(tmp_path):
    _check(tmp_path, 'O7xO6', 24, 'HET4')


def test_o7xo6_40(tmp_path):
    _check(tmp_path, 'O7xO6', 40, 'HOM6')


def test_o7xo6_45(tmp_path):
    _check(tmp_path, 'O7xO6', 45, 'HET4')


def test_o7xo6_54(tmp_path):
    _check(tmp_path, 'O7xO6', 54, 'HET4')


def test_o7xo6_57(tmp_path):
    _check(tmp_path, 'O7xO6', 57, 'HET4')


def test_o7xo6_60(tmp_path):
    _check(tmp_path, 'O7xO6', 60, 'HET4')


def test_o7xo6_66(tmp_path):
    _check(tmp_path, 'O7xO6', 66, 'HET4')


def test_o7xo6_69(tmp_path):
    _check(tmp_path, 'O7xO6', 69, 'HET4')


def test_o7xo6_72(tmp_path):
    _check(tmp_path, 'O7xO6', 72, 'HET4')


def test_o7xo6_75(tmp_path):
    _check(tmp_path, 'O7xO6', 75, 'HET4')


def test_o7xo6_76(tmp_path):
    _check(tmp_path, 'O7xO6', 76, 'HOM6')


def test_o7xo6_88(tmp_path):
    _check(tmp_path, 'O7xO6', 88, 'HOM6')


def test_o7xo6_90(tmp_path):
    _check(tmp_path, 'O7xO6', 90, 'HET4')


def test_o7xo6_92(tmp_path):
    _check(tmp_path, 'O7xO6', 92, 'TWIN4')


def test_o7xo6_93(tmp_path):
    _check(tmp_path, 'O7xO6', 93, 'HET4')


def test_o7xo6_95(tmp_path):
    _check(tmp_path, 'O7xO6', 95, 'TWIN4')


def test_o7xo6_99(tmp_path):
    _check(tmp_path, 'O7xO6', 99, 'HET4')


def test_o7xo6_103(tmp_path):
    _check(tmp_path, 'O7xO6', 103, 'HOM6')


def test_o7xo6_105(tmp_path):
    _check(tmp_path, 'O7xO6', 105, 'HET4')


def test_o7xo6_106(tmp_path):
    _check(tmp_path, 'O7xO6', 106, 'HOM6')


def test_o7xo6_114(tmp_path):
    _check(tmp_path, 'O7xO6', 114, 'HET4')


def test_o7xo6_120(tmp_path):
    _check(tmp_path, 'O7xO6', 120, 'HET4')


def test_o7xo6_123(tmp_path):
    _check(tmp_path, 'O7xO6', 123, 'HET4')


def test_o7xo6_126(tmp_path):
    _check(tmp_path, 'O7xO6', 126, 'HET4')


def test_o7xo6_29(tmp_path):
    _check(tmp_path, 'O7xO6', 29, 'TWIN4')


def test_o7xo6_32(tmp_path):
    _check(tmp_path, 'O7xO6', 32, 'TWIN4')


def test_o7xo6_50(tmp_path):
    _check(tmp_path, 'O7xO6', 50, 'TWIN4')


def test_o7xo6_53(tmp_path):
    _check(tmp_path, 'O7xO6', 53, 'TWIN4')


def test_o7xo6_116(tmp_path):
    _check(tmp_path, 'O7xO6', 116, 'TWIN4')


def test_o7xo6_119(tmp_path):
    _check(tmp_path, 'O7xO6', 119, 'TWIN4')


def test_o7xo6_117(tmp_path):
    _check(tmp_path, 'O7xO6', 117, 'HET4')


def test_o7xo6_97(tmp_path):
    _check(tmp_path, 'O7xO6', 97, 'HOM6')


def test_o7xo6_3(tmp_path):
    _check(tmp_path, 'O7xO6', 3, 'HET4')


def test_o7xo6_4(tmp_path):
    _check(tmp_path, 'O7xO6', 4, 'HOM6')


def test_o7xo6_21(tmp_path):
    _check(tmp_path, 'O7xO6', 21, 'HET4')


def test_o7xo6_28(tmp_path):
    _check(tmp_path, 'O7xO6', 28, 'HOM6')


def test_o7xo6_35(tmp_path):
    _check(tmp_path, 'O7xO6', 35, 'TWIN4')


def test_o7xo6_51(tmp_path):
    _check(tmp_path, 'O7xO6', 51, 'HET4')


def test_o7xo6_55(tmp_path):
    _check(tmp_path, 'O7xO6', 55, 'HOM6')


def test_o7xo6_61(tmp_path):
    _check(tmp_path, 'O7xO6', 61, 'HOM6')


def test_o7xo6_63(tmp_path):
    _check(tmp_path, 'O7xO6', 63, 'HET4')


def test_o7xo6_65(tmp_path):
    _check(tmp_path, 'O7xO6', 65, 'TWIN4')


def test_o7xo6_80(tmp_path):
    _check(tmp_path, 'O7xO6', 80, 'TWIN4')


def test_o7xo6_98(tmp_path):
    _check(tmp_path, 'O7xO6', 98, 'TWIN4')


def test_o7xo6_115(tmp_path):
    _check(tmp_path, 'O7xO6', 115, 'HOM6')


def test_o7xo6_121(tmp_path):
    _check(tmp_path, 'O7xO6', 121, 'HOM6')


def test_o7xo6_122(tmp_path):
    _check(tmp_path, 'O7xO6', 122, 'TWIN4')


def test_o7xo6_56(tmp_path):
    _check(tmp_path, 'O7xO6', 56, 'TWIN4')


def test_o7xo7_44(tmp_path):
    _check(tmp_path, 'O7xO7', 44, 'HET4+TWIN4')


def test_o7xo6_48(tmp_path):
    _check(tmp_path, 'O7xO6', 48, 'HET4')


def test_o7xo6_78(tmp_path):
    _check(tmp_path, 'O7xO6', 78, 'HET4')


def test_o7xo7_8(tmp_path):
    _check(tmp_path, 'O7xO7', 8, 'HET4+TWIN4')


def test_o7xo7_12(tmp_path):
    _check(tmp_path, 'O7xO7', 12, 'HET4+HET4')


def test_o7xo7_13(tmp_path):
    _check(tmp_path, 'O7xO7', 13, 'HET4+HOM6')


def test_o7xo7_42(tmp_path):
    _check(tmp_path, 'O7xO7', 42, 'HET4+HET4')


def test_o7xo6_39(tmp_path):
    _check(tmp_path, 'O7xO6', 39, 'HET4')


def test_o7xo6_87(tmp_path):
    _check(tmp_path, 'O7xO6', 87, 'HET4')


def test_o7xo7_5(tmp_path):
    _check(tmp_path, 'O7xO7', 5, 'TWIN4+TWIN4')


def test_o7xo7_20(tmp_path):
    _check(tmp_path, 'O7xO7', 20, 'HET4+TWIN4')


def test_o7xo7_47(tmp_path):
    _check(tmp_path, 'O7xO7', 47, 'TWIN4+TWIN4')


def test_o7xo7_56(tmp_path):
    _check(tmp_path, 'O7xO7', 56, 'HET4+TWIN4')


def test_o7xo6_27(tmp_path):
    _check(tmp_path, 'O7xO6', 27, 'HET4')


def test_o7xo6_33(tmp_path):
    _check(tmp_path, 'O7xO6', 33, 'HET4')


def test_o7xo6_36(tmp_path):
    _check(tmp_path, 'O7xO6', 36, 'HET4')


def test_o7xo6_81(tmp_path):
    _check(tmp_path, 'O7xO6', 81, 'HET4')


def test_o7xo6_84(tmp_path):
    _check(tmp_path, 'O7xO6', 84, 'HET4')


def test_o7xo7_37(tmp_path):
    _check(tmp_path, 'O7xO7', 37, 'HET4+HOM6')


def test_o7xo6_16(tmp_path):
    _check(tmp_path, 'O7xO6', 16, 'HOM6')


def test_o7xo6_91(tmp_path):
    _check(tmp_path, 'O7xO6', 91, 'HOM6')


def test_o7xo6_124(tmp_path):
    _check(tmp_path, 'O7xO6', 124, 'HOM6')


def test_o7xo7_4(tmp_path):
    _check(tmp_path, 'O7xO7', 4, 'HOM6+TWIN4')


def test_o7xo7_14(tmp_path):
    _check(tmp_path, 'O7xO7', 14, 'HET4+TWIN4')


def test_o7xo7_15(tmp_path):
    _check(tmp_path, 'O7xO7', 15, 'HOM6+HOM6')


def test_o7xo7_6(tmp_path):
    _check(tmp_path, 'O7xO7', 6, 'HET4+HET4')


def test_o7xo7_38(tmp_path):
    _check(tmp_path, 'O7xO7', 38, 'HET4+TWIN4')


def test_o7xo7_61(tmp_path):
    _check(tmp_path, 'O7xO7', 61, 'HET4+HOM6')


def test_o7xo7_62(tmp_path):
    _check(tmp_path, 'O7xO7', 62, 'HET4+TWIN4')


def test_o7xo7_63(tmp_path):
    _check(tmp_path, 'O7xO7', 63, 'HOM6+HOM6')


def test_o7xo6_1(tmp_path):
    _check(tmp_path, 'O7xO6', 1, 'HOM6')


def test_o7xo7_2(tmp_path):
    _check(tmp_path, 'O7xO7', 2, 'HET4+TWIN4')


def test_o7xo7_11(tmp_path):
    _check(tmp_path, 'O7xO7', 11, 'TWIN4+TWIN4')


def test_o7xo7_49(tmp_path):
    _check(tmp_path, 'O7xO7', 49, 'HET4+HOM6')


def test_o7xo7_24(tmp_path):
    _check(tmp_path, 'O7xO7', 24, 'HET4+HET4')


def test_o7xo7_36(tmp_path):
    _check(tmp_path, 'O7xO7', 36, 'HET4+HET4')


def test_o7xo7_43(tmp_path):
    _check(tmp_path, 'O7xO7', 43, 'HET4+HOM6')


def test_o7xo6_7(tmp_path):
    _check(tmp_path, 'O7xO6', 7, 'HOM6')


def test_o7xo6_31(tmp_path):
    _check(tmp_path, 'O7xO6', 31, 'HOM6')


def test_o7xo6_52(tmp_path):
    _check(tmp_path, 'O7xO6', 52, 'HOM6')


def test_o7xo6_79(tmp_path):
    _check(tmp_path, 'O7xO6', 79, 'HOM6')


def test_o7xo6_100(tmp_path):
    _check(tmp_path, 'O7xO6', 100, 'HOM6')


def test_o7xo6_22(tmp_path):
    _check(tmp_path, 'O7xO6', 22, 'HOM6')


def test_o7xo6_25(tmp_path):
    _check(tmp_path, 'O7xO6', 25, 'HOM6')


def test_o7xo6_58(tmp_path):
    _check(tmp_path, 'O7xO6', 58, 'HOM6')


def test_o7xo6_70(tmp_path):
    _check(tmp_path, 'O7xO6', 70, 'HOM6')


def test_o7xo7_7(tmp_path):
    _check(tmp_path, 'O7xO7', 7, 'HET4+HOM6')


def test_o7xo7_17(tmp_path):
    _check(tmp_path, 'O7xO7', 17, 'TWIN4+TWIN4')


def test_o7xo7_19(tmp_path):
    _check(tmp_path, 'O7xO7', 19, 'HET4+HOM6')


def test_o7xo7_28(tmp_path):
    _check(tmp_path, 'O7xO7', 28, 'HOM6+TWIN4')


def test_o7xo7_35(tmp_path):
    _check(tmp_path, 'O7xO7', 35, 'TWIN4+TWIN4')


def test_o7xo7_55(tmp_path):
    _check(tmp_path, 'O7xO7', 55, 'HET4+HOM6')


def test_o7xo7_22(tmp_path):
    _check(tmp_path, 'O7xO7', 22, 'HOM6+TWIN4')


def test_o7xo7_26(tmp_path):
    _check(tmp_path, 'O7xO7', 26, 'HET4+TWIN4')


def test_o7xo7_34(tmp_path):
    _check(tmp_path, 'O7xO7', 34, 'HOM6+TWIN4')


def test_o7xo7_25(tmp_path):
    _check(tmp_path, 'O7xO7', 25, 'HET4+HOM6')


def test_o7xo7_10(tmp_path):
    _check(tmp_path, 'O7xO7', 10, 'HOM6+TWIN4')


def test_o7xo7_16(tmp_path):
    _check(tmp_path, 'O7xO7', 16, 'HOM6+TWIN4')


def test_o7xo7_46(tmp_path):
    _check(tmp_path, 'O7xO7', 46, 'HOM6+TWIN4')


def test_o7xo7_3(tmp_path):
    _check(tmp_path, 'O7xO7', 3, 'HOM6+HOM6')


def test_o7xo7_33(tmp_path):
    _check(tmp_path, 'O7xO7', 33, 'HOM6+HOM6')


def test_o7xo7_51(tmp_path):
    _check(tmp_path, 'O7xO7', 51, 'HOM6+HOM6')


def test_o7xo7_57(tmp_path):
    _check(tmp_path, 'O7xO7', 57, 'HOM6+HOM6')


def test_o7xo7_21(tmp_path):
    _check(tmp_path, 'O7xO7', 21, 'HOM6+HOM6')


def test_o7xo7_27(tmp_path):
    _check(tmp_path, 'O7xO7', 27, 'HOM6+HOM6')
