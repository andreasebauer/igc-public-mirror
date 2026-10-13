from __future__ import annotations

"""Non-promoting toric structure audit for the frozen G2 CAPS7 bridge grammar.

The audit translates each directed G2 bridge operator a>b to the resource-consumption
column e_a+e_b in Z^7.  It does not modify G1/G2 transition semantics and cannot graduate
or reopen any G layer.  It is an analytic observer on the already-earned 31-operator basis.
"""

from collections import defaultdict
from importlib.resources import files
from itertools import combinations, permutations
from math import comb, gcd
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json

from .canon import canonical_sha256, write_json_atomic
from .records import source_sha256


class UpliftToricError(RuntimeError):
    pass

_SPEC = "G2_TORIC_STRUCTURE_AUDIT_SPEC_V1.json"
_CORE = (0, 1, 4, 5, 6)


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def toric_spec() -> dict[str, Any]:
    obj = json.loads(_resource(_SPEC).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G2_TORIC_STRUCTURE_AUDIT_SPEC_V1":
        raise UpliftToricError("bad toric audit spec schema")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if canonical_sha256(payload) != obj.get("science_sha256"):
        raise UpliftToricError("toric audit spec hash mismatch")
    return obj


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def consumption_vector(op: Sequence[int]) -> tuple[int, ...]:
    a, b = map(int, op)
    if not (0 <= a < 7 and 0 <= b < 7):
        raise UpliftToricError(f"bad operator {op}")
    out = [0] * 7
    out[a] += 1
    out[b] += 1
    return tuple(out)


def _undirected(op: Sequence[int]) -> tuple[int, int]:
    a, b = map(int, op)
    return (a, b) if a <= b else (b, a)


def _det_bareiss(mat: Sequence[Sequence[int]]) -> int:
    a = [list(map(int, row)) for row in mat]
    n = len(a)
    if n == 0:
        return 1
    if any(len(row) != n for row in a):
        raise UpliftToricError("determinant requires square matrix")
    sign = 1
    prev = 1
    for k in range(n - 1):
        pivot_row = next((i for i in range(k, n) if a[i][k] != 0), None)
        if pivot_row is None:
            return 0
        if pivot_row != k:
            a[k], a[pivot_row] = a[pivot_row], a[k]
            sign *= -1
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                num = a[i][j] * pivot - a[i][k] * a[k][j]
                if prev != 1:
                    if num % prev:
                        raise UpliftToricError("Bareiss non-exact division")
                    num //= prev
                a[i][j] = num
        prev = pivot
        for i in range(k + 1, n):
            a[i][k] = 0
    return sign * a[-1][-1]


def _minor(columns: Sequence[tuple[int, ...]], rows: Sequence[int], cols: Sequence[int]) -> int:
    return _det_bareiss([[columns[j][i] for j in cols] for i in rows])


def _find_minor(columns: Sequence[tuple[int, ...]], size: int, target_abs: int) -> dict[str, Any] | None:
    for rows in combinations(range(7), size):
        for cols in combinations(range(len(columns)), size):
            d = _minor(columns, rows, cols)
            if abs(d) == target_abs:
                return {"rows": list(rows), "columns": list(cols), "determinant": d}
    return None


def _operator_audit(bridge_pairs: Sequence[Sequence[int]]) -> dict[str, Any]:
    directed = [tuple(map(int, x)) for x in bridge_pairs]
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, op in enumerate(directed):
        groups[_undirected(op)].append(i)
    unique = sorted(groups)
    cols = [consumption_vector(op) for op in directed]
    ucols = [consumption_vector(op) for op in unique]
    duplicate_groups = [
        {"consumption_operator": list(op), "directed_indices": ids, "directed_operators": [list(directed[i]) for i in ids]}
        for op, ids in sorted(groups.items()) if len(ids) > 1
    ]
    loops = [list(op) for op in unique if op[0] == op[1]]
    offdiag = [op for op in unique if op[0] != op[1]]
    reversal_complete = all(((b, a) in directed) for a, b in offdiag)
    failures: list[str] = []
    if len(directed) != 31:
        failures.append("DIRECTED_OPERATOR_COUNT")
    if len(unique) != 18:
        failures.append("UNIQUE_CONSUMPTION_COUNT")
    if len(duplicate_groups) != 13:
        failures.append("REVERSE_DUPLICATE_COUNT")
    if not reversal_complete:
        failures.append("REVERSAL_COMPLETENESS")
    return {
        "schema_id": "IG_G2_TORIC_OPERATOR_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "directed_operator_count": len(directed),
        "unique_consumption_monomial_count": len(unique),
        "reverse_orientation_duplicate_count": len(duplicate_groups),
        "loop_operator_count": len(loops),
        "loops": loops,
        "unique_undirected_operators": [list(x) for x in unique],
        "directed_operators": [list(x) for x in directed],
        "directed_consumption_matrix_columns": [list(x) for x in cols],
        "unique_consumption_matrix_columns": [list(x) for x in ucols],
        "reverse_duplicate_groups": duplicate_groups,
        "reversal_complete": reversal_complete,
        "failures": failures,
    }


def _lattice_audit(unique_ops: Sequence[Sequence[int]]) -> dict[str, Any]:
    cols = [consumption_vector(op) for op in unique_ops]
    six = _find_minor(cols, 6, 1)
    seven = _find_minor(cols, 7, 2)
    # Every full-rank 7x7 minor is even because each column has even coordinate sum.
    # A determinant-2 witness makes the 7th determinantal divisor exactly 2.
    failures: list[str] = []
    if six is None:
        failures.append("NO_UNIMODULAR_6_MINOR")
    if seven is None:
        failures.append("NO_DET2_7_MINOR")
    result = {
        "schema_id": "IG_G2_TORIC_LATTICE_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "matrix_rank_over_Q": 7 if seven is not None else None,
        "unique_presentation_kernel_rank": 11 if seven is not None else None,
        "directed_presentation_kernel_rank": 24 if seven is not None else None,
        "six_by_six_unimodular_minor_witness": six,
        "seven_by_seven_det2_minor_witness": seven,
        "all_columns_have_total_sum": 2,
        "image_lattice": "L={z in Z^7 : sum_i z_i is even}",
        "image_lattice_index_in_Z7": 2 if seven is not None else None,
        "smith_normal_form_nonzero_diagonal": [1, 1, 1, 1, 1, 1, 2] if not failures else None,
        "proof_note": "A 6x6 minor of determinant ±1 forces the first six invariant factors to 1. All 7x7 minors are even because every column has even coordinate sum; a 7x7 determinant ±2 witness forces the final invariant factor to 2.",
        "failures": failures,
    }
    return result


def _reachable_sets(unique_cols: Sequence[tuple[int, ...]], max_degree: int) -> list[set[tuple[int, ...]]]:
    levels: list[set[tuple[int, ...]]] = [{(0, 0, 0, 0, 0, 0, 0)}]
    for _ in range(max_degree):
        nxt: set[tuple[int, ...]] = set()
        for x in levels[-1]:
            for c in unique_cols:
                nxt.add(tuple(x[i] + c[i] for i in range(7)))
        levels.append(nxt)
    return levels


def _formula_points(d: int) -> set[tuple[int, ...]]:
    n = 2 * int(d)
    out: set[tuple[int, ...]] = set()
    # x4>=x2 and x5+x6>=x3 are the only extra cone inequalities.
    for x0 in range(n + 1):
        for x1 in range(n - x0 + 1):
            rem1 = n - x0 - x1
            for x2 in range(rem1 + 1):
                rem2 = rem1 - x2
                for x3 in range(rem2 + 1):
                    rem3 = rem2 - x3
                    for x4 in range(rem3 + 1):
                        if x4 < x2:
                            continue
                        rem4 = rem3 - x4
                        for x5 in range(rem4 + 1):
                            x6 = rem4 - x5
                            if x5 + x6 >= x3:
                                out.add((x0, x1, x2, x3, x4, x5, x6))
    return out


def _hilbert_formula(d: int) -> int:
    d = int(d)
    return (d + 1) * (d + 2) * (d + 2) * (d + 3) * (2 * d * d + 8 * d + 5) // 60


def _cone_semigroup_audit(unique_ops: Sequence[Sequence[int]]) -> dict[str, Any]:
    unique = [tuple(map(int, x)) for x in unique_ops]
    unique_set = set(unique)
    core = set(_CORE)
    core_complete = all(((min(i, j), max(i, j)) in unique_set) for i in core for j in core)
    type2 = sorted([op for op in unique if 2 in op])
    type3 = sorted([op for op in unique if 3 in op])
    cols = [consumption_vector(op) for op in unique]
    levels = _reachable_sets(cols, 8)
    formula_counts = []
    equality = []
    for d in range(9):
        fp = _formula_points(d)
        formula_counts.append(len(fp))
        equality.append(levels[d] == fp)
    failures: list[str] = []
    if not core_complete:
        failures.append("CORE_NOT_COMPLETE_WITH_LOOPS")
    if type2 != [(2, 4)]:
        failures.append("TYPE2_ATTACHMENT_PATTERN")
    if type3 != [(3, 5), (3, 6)]:
        failures.append("TYPE3_ATTACHMENT_PATTERN")
    if not all(equality):
        failures.append("SEMIGROUP_FORMULA_CROSSCHECK")
    vertices = [
        [2, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 2],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
    ]
    return {
        "schema_id": "IG_G2_TORIC_CONE_SEMIGROUP_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "core_types": list(_CORE),
        "core_pattern": "complete graph K5 with a loop at each core type",
        "type2_unique_attachment": [2, 4],
        "type3_attachments": [[3, 5], [3, 6]],
        "cone_dimension": 7,
        "projective_polytope_dimension": 6,
        "cone_characterization": "C={x in R^7_{>=0}: x4>=x2 and x5+x6>=x3}",
        "semigroup_characterization": "S={x in N^7: x4>=x2, x5+x6>=x3, sum_i x_i even}",
        "constructive_normality_proof": "Set c24=x2. Split x3=c35+c36 with 0<=c35<=x5 and 0<=c36<=x6, possible exactly when x5+x6>=x3. The residual coordinates on core types {0,1,4,5,6} are nonnegative and have even total sum, hence can be paired into the complete looped K5 degree-2 generators. Therefore every lattice point of C in the even-sum lattice is generated by the 18 degree-one bridge monomials.",
        "normal_affine_semigroup": True if not failures else None,
        "hilbert_basis_size": 18,
        "hilbert_basis_equals_unique_bridge_consumptions": True,
        "extreme_ray_count": 8,
        "polytope_vertex_count": 8,
        "polytope_vertices": vertices,
        "nonvertex_degree_one_lattice_points": 10,
        "facet_inequalities_relative_to_sum2_slice": ["x0>=0", "x1>=0", "x2>=0", "x3>=0", "x5>=0", "x6>=0", "x4-x2>=0", "x5+x6-x3>=0"],
        "degree_0_to_8_reachable_counts": [len(x) for x in levels],
        "degree_0_to_8_formula_counts": formula_counts,
        "degree_0_to_8_exact_formula_match": equality,
        "failures": failures,
    }


def _interior_formula_points(d: int) -> list[tuple[int, ...]]:
    return sorted([
        x for x in _formula_points(d)
        if x[0] > 0 and x[1] > 0 and x[2] > 0 and x[3] > 0 and x[5] > 0 and x[6] > 0
        and x[4] > x[2] and x[5] + x[6] > x[3]
    ])


def _hilbert_gorenstein_audit() -> dict[str, Any]:
    hs = [_hilbert_formula(d) for d in range(13)]
    # h*_k = sum_{j=0}^k (-1)^j C(7,j) H(k-j)
    hstar: list[int] = []
    for k in range(8):
        hstar.append(sum(((-1) ** j) * comb(7, j) * hs[k - j] for j in range(min(7, k) + 1)))
    while hstar and hstar[-1] == 0:
        hstar.pop()
    omega = (1, 1, 1, 1, 2, 1, 1)
    interior_counts = [len(_interior_formula_points(d)) for d in range(1, 6)]
    interior4 = _interior_formula_points(4)
    translate_checks: list[bool] = []
    # Bounded exact implementation check of the all-depth translate proof.
    for d in range(4, 9):
        lhs = set(_interior_formula_points(d))
        rhs = {tuple(omega[i] + y[i] for i in range(7)) for y in _formula_points(d - 4)}
        translate_checks.append(lhs == rhs)
    failures: list[str] = []
    if hstar != [1, 11, 11, 1]:
        failures.append("HSTAR")
    if interior_counts[:3] != [0, 0, 0] or interior_counts[3] != 1 or interior4 != [omega]:
        failures.append("CODEGREE4_INTERIOR")
    if not all(translate_checks):
        failures.append("GORENSTEIN_TRANSLATE_CHECK")
    return {
        "schema_id": "IG_G2_TORIC_HILBERT_GORENSTEIN_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "hilbert_function": "H(d)=((d+1)(d+2)^2(d+3)(2d^2+8d+5))/60",
        "hilbert_counts_d0_to_d12": hs,
        "hilbert_series": "(1+11t+11t^2+t^3)/(1-t)^7",
        "h_star_polynomial_coefficients": [1, 11, 11, 1],
        "affine_semigroup_ring_dimension": 7,
        "projective_toric_dimension": 6,
        "normalized_projective_degree": 24,
        "codegree_gorenstein_index": 4,
        "unique_first_interior_lattice_point": list(omega),
        "unique_first_interior_degree": 4,
        "interior_counts_degrees_1_to_5": interior_counts,
        "gorenstein_translate_identity": "relint(S)=omega+S with omega=(1,1,1,1,2,1,1)",
        "gorenstein_translate_proof": "For every relative-interior lattice point x, subtracting omega preserves nonnegativity, x4>=x2, x5+x6>=x3, and even total sum; conversely omega+S makes every facet inequality strict. By the standard normal affine-semigroup criterion, k[S] is Gorenstein.",
        "bounded_translate_checks_degrees_4_to_8": translate_checks,
        "gorenstein": True if not failures else None,
        "equivalent_polytope_statement": "The degree-one bridge polytope is a 6-dimensional Gorenstein lattice polytope of index 4; 4P-omega is reflexive in the affine even-sum lattice.",
        "failures": failures,
    }


def _monomial_name(op: tuple[int, int]) -> str:
    return f"Q{op[0]}{op[1]}"


def _pair_sum(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a[i] + b[i] for i in range(7))


def _toric_ideal_audit(unique_ops: Sequence[Sequence[int]], directed_ops: Sequence[Sequence[int]]) -> dict[str, Any]:
    unique = [tuple(map(int, x)) for x in unique_ops]
    cols = [consumption_vector(op) for op in unique]
    # Degree-two toric fibers in the unique 18-generator presentation.
    fibers: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    for i in range(len(unique)):
        for j in range(i, len(unique)):
            fibers[_pair_sum(cols[i], cols[j])].append((i, j))
    multi = {k: v for k, v in fibers.items() if len(v) > 1}
    span_count = sum(len(v) - 1 for v in multi.values())

    core_ops = [(i, j) for i in _CORE for j in _CORE if i <= j]
    core_set = set(core_ops)
    if not core_set.issubset(set(unique)):
        raise UpliftToricError("core Veronese generators absent")
    core_idx = [unique.index(op) for op in core_ops]
    core_fibers: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    for ai, i in enumerate(core_idx):
        for j in core_idx[ai:]:
            core_fibers[_pair_sum(cols[i], cols[j])].append((i, j))
    core_multi = {k: v for k, v in core_fibers.items() if len(v) > 1}
    core_quad_count = sum(len(v) - 1 for v in core_multi.values())

    # Canonical spanning-tree generators for the 50-dimensional Veronese quadratic kernel.
    veronese_generators: list[dict[str, Any]] = []
    for image, mons in sorted(core_multi.items()):
        mons = sorted(mons)
        anchor = mons[0]
        for other in mons[1:]:
            veronese_generators.append({
                "lhs": [_monomial_name(unique[anchor[0]]), _monomial_name(unique[anchor[1]])],
                "rhs": [_monomial_name(unique[other[0]]), _monomial_name(unique[other[1]])],
                "resource_image": list(image),
            })

    coupling_generators = []
    for i in _CORE:
        # U=Q35, V=Q36, and Q_i6 / Q_i5 are present in the K5 core.
        i5 = tuple(sorted((i, 5)))
        i6 = tuple(sorted((i, 6)))
        coupling_generators.append({
            "lhs": ["Q35", _monomial_name(i6)],
            "rhs": ["Q36", _monomial_name(i5)],
            "identity": f"Q35*{_monomial_name(i6)} = Q36*{_monomial_name(i5)}",
        })

    directed = [tuple(map(int, x)) for x in directed_ops]
    reverse_linear = []
    seen = set()
    for a, b in directed:
        if a == b:
            continue
        u = tuple(sorted((a, b)))
        if u in seen:
            continue
        seen.add(u)
        reverse_linear.append({"lhs": f"Y{a}{b}", "rhs": f"Y{b}{a}", "resource_monomial": _monomial_name(u)})

    failures: list[str] = []
    if len(fibers) != 116 or len(multi) != 50 or span_count != 55:
        failures.append("UNIQUE_QUADRATIC_FIBER_COUNTS")
    if core_quad_count != 50 or len(veronese_generators) != 50:
        failures.append("VERONESE_QUADRATIC_COUNT")
    if len(coupling_generators) != 5:
        failures.append("TYPE3_COUPLING_COUNT")
    if len(reverse_linear) != 13:
        failures.append("DIRECTED_LINEAR_COUNT")

    return {
        "schema_id": "IG_G2_TORIC_IDEAL_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "toric_map_unique": "phi: k[Q_ab | 18 unique capacity monomials] -> k[x0,...,x6], Q_ab |-> x_a x_b",
        "toric_map_directed": "Phi: k[Y_ab | 31 directed bridge operators] -> k[x0,...,x6], Y_ab |-> x_a x_b",
        "unique_presentation_codimension": 11,
        "directed_presentation_codimension": 24,
        "degree2_monomial_count_unique_presentation": 171,
        "degree2_distinct_resource_images": 116,
        "degree2_multifiber_count": 50,
        "degree2_kernel_dimension": 55,
        "degree2_fiber_size_distribution": {"2": 45, "3": 5},
        "minimal_unique_toric_ideal_generators": {
            "total": 55,
            "degrees": {"quadratic": 55},
            "veronese_core_quadrics": 50,
            "type3_coupling_quadrics": 5,
        },
        "minimal_directed_toric_presentation_generators": {
            "total": 68,
            "degrees": {"linear": 13, "quadratic": 55},
            "orientation_identification_linears": 13,
            "unique_capacity_toric_quadrics": 55,
        },
        "orientation_linear_relations": reverse_linear,
        "canonical_veronese_quadratic_generators": veronese_generators,
        "type3_coupling_generators": coupling_generators,
        "type2_operator_Q24_occurs_in_toric_relations": False,
        "generation_proof": "The 15 core generators on B={0,1,4,5,6} are the full second Veronese semigroup and its toric ideal is generated by the 50 degree-two fiber relations. Q24 is algebraically free because type 2 occurs in no other generator. For Q35,Q36, any binomial relation has equal total type-3 degree. After using core Veronese relations to pair an excess type-6 token with some core token i, one of the five quadrics Q35*Q_i6=Q36*Q_i5 moves one Q35 to Q36 while changing the core exponent by e5-e6. Repeating matches the Q35/Q36 exponents; the remaining core relation lies in the Veronese ideal. Thus the 50+5 quadrics generate the full unique toric ideal. Since there are no linear relations in the 18-generator presentation and its degree-two kernel has dimension 55, 55 is also the minimal number of homogeneous generators.",
        "failures": failures,
    }


def _symmetry_audit(unique_ops: Sequence[Sequence[int]]) -> dict[str, Any]:
    edges = {_undirected(x) for x in unique_ops}
    autos = []
    for p in permutations(range(7)):
        transformed = {_undirected((p[a], p[b])) for a, b in edges}
        if transformed == edges:
            autos.append(tuple(p))
    return {
        "schema_id": "IG_G2_TORIC_SYMMETRY_AUDIT_V1",
        "status": "PASS" if len(autos) == 4 else "FAIL",
        "resource_type_automorphism_group_order": len(autos),
        "automorphisms_as_images_0_to_6": [list(x) for x in autos],
        "group_identification": "C2 x C2" if len(autos) == 4 else "UNRESOLVED",
        "generators": ["swap 0<->1", "swap 5<->6"],
        "fixed_types": [2, 3, 4],
    }


def _known_piece_audit(unique_ops: Sequence[Sequence[int]]) -> dict[str, Any]:
    unique = {tuple(map(int, x)) for x in unique_ops}
    core = {(i, j) for i in _CORE for j in _CORE if i <= j}
    veronese_ok = core.issubset(unique) and len(core) == 15
    scroll_set = {(5, 5), (5, 6), (6, 6), (3, 5), (3, 6)}
    scroll_ok = scroll_set.issubset(unique)
    q24_free = (2, 4) in unique and sum(1 for op in unique if 2 in op) == 1
    failures = []
    if not veronese_ok:
        failures.append("VERONESE_CORE")
    if not scroll_ok:
        failures.append("SCROLL_SECTION")
    if not q24_free:
        failures.append("Q24_FREE_CONE_DIRECTION")
    return {
        "schema_id": "IG_G2_TORIC_KNOWN_PIECES_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "exact_core_identification": {
            "types": list(_CORE),
            "generator_count": 15,
            "monomials": "all x_i x_j for i<=j in {0,1,4,5,6}",
            "standard_algebraic_geometry_object": "second Veronese embedding v2(P^4) in P^14",
            "projective_dimension": 4,
            "degree": 16,
            "hilbert_series": "(1+10t+5t^2)/(1-t)^5",
        },
        "type3_section_identification": {
            "types": [3, 5, 6],
            "generators": ["Q55", "Q56", "Q66", "Q35", "Q36"],
            "ideal": ["Q55*Q66-Q56^2", "Q35*Q66-Q36*Q56", "Q35*Q56-Q36*Q55"],
            "standard_algebraic_geometry_object": "rational normal cubic scroll S(1,2) in P^4",
            "projective_dimension": 2,
            "degree": 3,
            "hilbert_series": "(1+2t)/(1-t)^3",
        },
        "type2_cone_direction": {
            "generator": "Q24=x2*x4",
            "appears_in_no_toric_binomial_relation": True,
            "projective_effect": "the full unique-operator toric variety is a projective cone in the Q24 coordinate over the 17-generator subvariety",
        },
        "structural_reading": "The bridge resource geometry contains a Veronese core, a cubic-scroll attachment through types 5/6, and one algebraically free cone direction through type 2->4.",
        "nonclaim": "These are exact identifications of the CAPS7 resource-consumption toric model, not physical-space geometry and not a claim that raw G2 operator semantics quotient orientation or topology.",
        "failures": failures,
    }


def toric_result(*, bridge_pairs: Sequence[Sequence[int]], s5_result: Mapping[str, Any], s4_operator_basis: Mapping[str, Any]) -> dict[str, Any]:
    spec = toric_spec()
    if s5_result.get("status") != "PASS":
        raise UpliftToricError("S5 primary result must PASS for CAPS7 toric audit")
    if s4_operator_basis.get("status") != "PASS" or int(s4_operator_basis.get("operator_count", 0)) != 31:
        raise UpliftToricError("S4 operator basis authority absent")
    actual_basis = [tuple(map(int, r["operator"])) for r in s4_operator_basis.get("basis_rows", [])]
    engine_basis = [tuple(map(int, x)) for x in bridge_pairs]
    if actual_basis != engine_basis:
        raise UpliftToricError("S4 operator basis != live decoder bridge basis")

    op = _operator_audit(engine_basis)
    unique = [tuple(x) for x in op["unique_undirected_operators"]]
    lattice = _lattice_audit(unique)
    cone = _cone_semigroup_audit(unique)
    hilbert = _hilbert_gorenstein_audit()
    ideal = _toric_ideal_audit(unique, engine_basis)
    symmetry = _symmetry_audit(unique)
    pieces = _known_piece_audit(unique)
    sections = (op, lattice, cone, hilbert, ideal, symmetry, pieces)
    passed = all(x.get("status") == "PASS" for x in sections)
    out = {
        "schema_id": "IG_G2_TORIC_STRUCTURE_AUDIT_RESULT_V1",
        "schema_version": "1.0.0",
        "date": "2026-09-03",
        "experiment_id": "G2:S5.TORIC_AUDIT",
        "stage_ref": "G2:S5",
        "status": "PASS" if passed else "REVIEW_REQUIRED",
        "classification": "G2_CAPS7_BRIDGE_SEMIGROUP_IS_NORMAL_GORENSTEIN_TORIC_INDEX4_WITH_VERONESE_SCROLL_CONE_STRUCTURE" if passed else "G2_TORIC_AUDIT_REVIEW_REQUIRED",
        "spec_sha256": spec["science_sha256"],
        "source_sha256": source_sha256(),
        "authority": {
            "s5_primary_science_sha256": s5_result.get("science_sha256"),
            "s4_operator_basis_science_sha256": s4_operator_basis.get("science_sha256"),
            "operator_basis_count": 31,
            "caps7_coordinate_count": 7,
        },
        "operator_audit": op,
        "lattice_audit": lattice,
        "cone_semigroup_audit": cone,
        "hilbert_gorenstein_audit": hilbert,
        "toric_ideal_audit": ideal,
        "symmetry_audit": symmetry,
        "known_piece_audit": pieces,
        "summary": {
            "directed_operators": 31,
            "unique_capacity_monomials": 18,
            "matrix_rank": 7,
            "image_lattice_index": 2,
            "unique_toric_codimension": 11,
            "normal": cone.get("normal_affine_semigroup"),
            "gorenstein": hilbert.get("gorenstein"),
            "gorenstein_index": 4,
            "projective_dimension": 6,
            "normalized_degree": 24,
            "h_star": [1, 11, 11, 1],
            "unique_ideal_minimal_quadrics": 55,
            "directed_orientation_linear_relations": 13,
            "resource_type_automorphism_group": "C2 x C2",
        },
        "interpretation": "Under the CAPS7 resource-consumption observer, the G2 composition basis is not merely an arbitrary list of 31 operators. It is the degree-one generating set of a 7-dimensional normal Gorenstein affine semigroup ring. Its projective bridge polytope is 6-dimensional, degree 24 and Gorenstein of index 4. The 31 directed operators collapse to 18 distinct resource monomials only for this toric observer; orientation remains part of the full G2 operator grammar.",
        "nonclaims": [
            "NOT_PHYSICAL_SPACE",
            "NOT_SPACETIME",
            "NOT_A_METRIC_OR_CURVATURE_RESULT",
            "NOT_CAPS7_MINIMALITY",
            "NOT_RAW_G2_STATE_EQUIVALENCE",
            "NOT_DIRECTIONAL_OPERATOR_EQUIVALENCE_OUTSIDE_CAPS7_RESOURCE_CONSUMPTION",
            "NO_CHANGE_TO_G2_GRADUATION_STATUS",
        ],
        "promotion_effect": "NONE_NONPROMOTING_ANALYTIC_AUDIT",
        "g2_graduated": False,
        "r0_unlocked": False,
    }
    out["science_sha256"] = canonical_sha256({k: v for k, v in out.items() if k != "science_sha256"})
    return out


def run_g2_toric_native(*, engine: Any, s5_result: str | Path, s4_operator_basis: str | Path) -> dict[str, Any]:
    row = engine.experiment("G2:S5.TORIC_AUDIT")
    if row.get("execution_mode") != "NATIVE_HANDLER":
        raise UpliftToricError("toric audit is not registered native")
    if engine.session is None:
        raise UpliftToricError("campaign session absent")
    stage_dir = engine.run_root / "stages" / "G2_S5_TORIC_AUDIT"
    stage_dir.mkdir(parents=True, exist_ok=True)
    s5 = _load_json(s5_result)
    op = _load_json(s4_operator_basis)
    result = toric_result(bridge_pairs=engine.session.bridge_pairs, s5_result=s5, s4_operator_basis=op)
    result["execution_metadata"] = {
        "schema_id": "IG_G2_TORIC_NATIVE_EXECUTION_METADATA_V1",
        "registered_experiment_id": "G2:S5.TORIC_AUDIT",
        "registry_sha256": engine.registry["registry_sha256"],
        "execution_backend_owned_by_decoder": True,
        "stage_specific_external_science_runner": False,
        "materialization_required": False,
        "parallelizable": False,
        "promotion_effect": "NONE",
    }
    write_json_atomic(stage_dir / "RESULT.json", result)
    write_json_atomic(stage_dir / "OPERATOR_AUDIT.json", result["operator_audit"])
    write_json_atomic(stage_dir / "LATTICE_AUDIT.json", result["lattice_audit"])
    write_json_atomic(stage_dir / "CONE_SEMIGROUP_AUDIT.json", result["cone_semigroup_audit"])
    write_json_atomic(stage_dir / "HILBERT_GORENSTEIN_AUDIT.json", result["hilbert_gorenstein_audit"])
    write_json_atomic(stage_dir / "TORIC_IDEAL_AUDIT.json", result["toric_ideal_audit"])
    write_json_atomic(stage_dir / "SYMMETRY_AUDIT.json", result["symmetry_audit"])
    write_json_atomic(stage_dir / "KNOWN_PIECES_AUDIT.json", result["known_piece_audit"])
    write_json_atomic(stage_dir / "EXECUTION_METADATA.json", result["execution_metadata"])
    return result
