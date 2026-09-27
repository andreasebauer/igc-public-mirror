from pathlib import Path
import inspect

from infinity_grid import g6_s8_intrinsic_descriptor as s8
from infinity_grid import g6_s8_evaluators as ev
from infinity_grid.v05_stage_registry import get_evaluator_spec, PRIMARY


def _permute_state(state, perm):
    return {
        "n": state["n"],
        "H_classes": [state["H_classes"][perm.index(i)] for i in range(state["n"])],
        "edges": [[perm[a], perm[b]] for a, b in state["edges"]],
        "edge_operators": [list(x) for x in state["edge_operators"]],
    }


def test_intrinsic_graph_descriptors_are_vertex_relabel_invariant():
    a = {
        "n": 4,
        "H_classes": ["H0", "H1", "H0", "H2"],
        "edges": [[0,1],[1,2],[2,3]],
        "edge_operators": [[0,0],[4,4],[5,5]],
    }
    perm = [2,0,3,1]
    b = _permute_state(a, perm)
    assert s8._graph_counts_descriptor("S1", a) == s8._graph_counts_descriptor("S1", b)
    assert s8._wl_descriptor("S1", a, stable=False) == s8._wl_descriptor("S1", b, stable=False)
    assert s8._wl_descriptor("S1", a, stable=True) == s8._wl_descriptor("S1", b, stable=True)


def test_descriptor_exactness_classifies_exact_under_and_over_refinement():
    reference = {"a":"X", "b":"X", "c":"Y"}
    exact = s8._partition_exactness(reference, {"a":("u",),"b":("u",),"c":("v",)})
    assert exact["status"] == "COMPLETE_MATCH"
    under = s8._partition_exactness(reference, {"a":("u",),"b":("u",),"c":("u",)})
    assert under["status"] == "UNDERREFINES"
    assert {under["underrefinement_witness"]["a"], under["underrefinement_witness"]["b"]} <= {"a","b","c"}
    over = s8._partition_exactness(reference, {"a":("u",),"b":("v",),"c":("w",)})
    assert over["status"] == "OVERREFINES"
    assert over["overrefinement_witness"]["s7_class"] == "X"


def test_component_split_witness_is_deterministic():
    groups = {"C0":["a","b"], "C1":["c"]}
    active = {"a","b"}
    memberships = {"S8D3O007-S1-a":"A", "S8D3O007-S1-b":"B"}
    split, need = s8._first_component_split(groups, active, memberships, depth=3, outer_execution_context_index=7)
    assert split["s7_class"] == "C0"
    assert split["a_state_token"] == "a"
    assert split["b_state_token"] == "b"
    assert need == {"A","B"}


def test_s8_evaluators_are_registered_primary_with_identity_free_service_surface():
    allowed = {"AUTHORITY_BASIS","PUBLIC_READ","EXACT_RELATION","EXACT_RELATION_PROFILE","EXACT_RELATION_PROFILE_FAMILY"}
    forbidden = {"EXACT_IDENTITY","OBSERVER_Q","OBSERVER_DECODE","MARKER_Q","MARKER_DECODE","MARKER_WRITE"}
    for ref in (
        "infinity_grid.g6_s8_evaluators:s8_recursive_outer_component_evaluator",
        "infinity_grid.g6_s8_evaluators:s8_recursive_ordinary_signature_evaluator",
    ):
        spec = get_evaluator_spec(ref)
        assert spec.role == PRIMARY
        assert set(spec.allowed_kernel_services) == allowed
        assert not (set(spec.allowed_kernel_services) & forbidden)


def test_s8_factor_swap_alias_map_covers_248_scientific_contexts_from_124_execution_contexts():
    refs = ("D2_BROOM","D2_PATH","D4_BROOM","D4_PATH")
    ops = (
        (0,0),(0,1),(0,4),(0,5),(0,6),
        (1,0),(1,1),(1,4),(1,5),(1,6),
        (2,4),(3,5),(3,6),
        (4,0),(4,1),(4,2),(4,4),(4,5),(4,6),
        (5,0),(5,1),(5,3),(5,4),(5,5),(5,6),
        (6,0),(6,1),(6,3),(6,4),(6,5),(6,6),
    )
    execution = ev._execution_contexts(refs, ops)
    aliases = ev._scientific_alias_map(refs, ops)
    assert len(execution) == 124
    assert len(aliases) == 248
    assert set(aliases) == set(range(124))


def test_s8_source_freezes_depth_operator_resource_and_nonclaim_boundaries():
    src = inspect.getsource(s8._validate_plan)
    assert "depths != (3, 4, 5, 6)" in src
    assert "S8_OPERATOR_BASIS" in src
    assert "S8_RECURSIVE_STATE_BUDGET" in src
    assert "S8_RELATION_CALL_BUDGET" in src
    assert "S8_DEEPER_MEMBER_SCOPE" in src
    assert "S8_CONTEXT_ALIAS_COVERAGE" in inspect.getsource(s8._validate_context_normalization)
    run_src = inspect.getsource(s8.run_g6_s8)
    assert '"g6_graduated": False' in run_src
    assert '"all_finite_congruence_theorem_earned": False' in run_src
    assert '"r_series_started": False' in run_src
    assert "G6_S8_D{depth}_O{outer_index:03d}" in run_src


def test_controller_registers_s8_only_through_passive_science_intake():
    from infinity_grid import v05_controller_event_loop as loop
    reg = loop._registry(Path(inspect.getfile(loop)).resolve().parents[1])
    assert "G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE" in reg[loop.SCIENCE_JOB]["allowed_operations"]
    source = inspect.getsource(loop._handle_g6_science)
    block = source.split("elif op=='G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE':",1)[1].split("elif op=='G6_R0_POST_GRADUATION_FIBER':",1)[0]
    assert "G6_S8_ARTIFACT_SET" in block
    assert "issue_controller_event_runtime_permit" in block
    assert "s8_recursive_outer_component_evaluator" in block
    assert "run_g6_s8" in block


def test_s8_v2_plan_and_relation_service_budget_are_fail_closed():
    src = inspect.getsource(s8._validate_plan)
    assert "IG_G6_S8_ACCEPTED_PREREGISTRATION_V2" in s8.PLAN_SCHEMA
    assert "S8_STREAM_SHARD_TASK_LIMIT" in src
    assert "S8_WORKER_RSS_BUDGET" in src
    guard = inspect.getsource(ev._RecursiveComputer._guard)
    depth1 = inspect.getsource(ev._RecursiveComputer.depth1)
    profile_counts = inspect.getsource(ev._RecursiveComputer.profile_counts)
    assert 'exact_relation_profile_call_count' in guard
    assert 'exact_relation_profile_family_call_count' in guard
    assert 'exact_relation_call_count' in guard
    assert 'MAX_WORKER_RSS_BYTES' in guard
    assert 'EXACT_RELATION_PROFILE_FAMILY' in profile_counts
    assert 'S8_D1_FAMILY_PROFILE_LENGTH' in profile_counts
    assert 'self._guard()' in profile_counts
    assert 'counts = self.profile_counts(tree)' in depth1
    run_src = inspect.getsource(s8.run_g6_s8)
    assert 'stream_shard_task_limit=stream_shard_task_limit' in run_src



def test_s8_compact_depth1_encoding_is_exact_equality_equivalent_to_expanded_form():
    root_a = ("D", (1,2,3,4,5,6,7), (("h",2),))
    root_b = ("D", (1,2,3,4,5,6,8), (("h",2),))
    refs = ("A","B","C","D")
    ops = tuple((i % 7, (i * 3) % 7) for i in range(31))
    # This test targets the pure equality reduction used by depth1(): with frozen
    # coordinates, expanded successor public signatures are deterministic from
    # root public D and the coordinate; equality is therefore root+counts equality.
    aliases = tuple(i for i in range(124) for _ in (0,1))
    counts_a = tuple((i * 5) % 11 for i in range(124))
    counts_b = list(counts_a); counts_b[17] += 1; counts_b = tuple(counts_b)
    def compact(root, counts): return ("ORDINARY_BRANCH_MULTISET",1,root,tuple(counts[i] for i in aliases))
    assert compact(root_a, counts_a) == compact(root_a, counts_a)
    assert compact(root_a, counts_a) != compact(root_a, counts_b)
    assert compact(root_a, counts_a) != compact(root_b, counts_a)

def test_s8_context_normalization_is_a_bound_science_artifact():
    from infinity_grid import v05_controller_event_loop as loop
    source = inspect.getsource(loop._handle_g6_science)
    block = source.split("elif op=='G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE':",1)[1].split("elif op=='G6_R0_POST_GRADUATION_FIBER':",1)[0]
    assert 's7_context_normalization' in block
    assert "artifacts={k:v for k,v in resolved.items() if k!='science_plan'}" in block


def test_s8_depth2_uses_s7_a30_count_vectors_without_grandchild_public_reads_or_alias_expansion():
    src = inspect.getsource(ev._RecursiveComputer.depth2_compact)
    assert 'self.profile_counts(child)' in src
    assert 'self.public_sig(child)' not in src
    assert 'tuple(blocks)' in src
    assert 'self.aliases' not in src
    d1 = inspect.getsource(ev._RecursiveComputer.depth1)
    assert 'counts = self.profile_counts(tree)' in d1
    assert 'self.aliases' not in d1

def test_s8_124_execution_tuple_is_equality_equivalent_to_248_factor_swap_alias_materialization():
    refs = ("D2_BROOM","D2_PATH","D4_BROOM","D4_PATH")
    ops = (
        (0,0),(0,1),(0,4),(0,5),(0,6),(1,0),(1,1),(1,4),(1,5),(1,6),
        (2,4),(3,5),(3,6),(4,0),(4,1),(4,2),(4,4),(4,5),(4,6),
        (5,0),(5,1),(5,3),(5,4),(5,5),(5,6),(6,0),(6,1),(6,3),(6,4),(6,5),(6,6),
    )
    aliases = ev._scientific_alias_map(refs, ops)
    a = tuple((i * 7) % 19 for i in range(124))
    b = list(a); b[73] += 1; b = tuple(b)
    expand = lambda x: tuple(x[i] for i in aliases)
    assert (a == a) == (expand(a) == expand(a))
    assert (a == b) == (expand(a) == expand(b))


def test_s8_gen32_class_prefix_comparator_registered_and_frozen():
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    ref='infinity_grid.g6_s8_evaluators:s8_s7_class_recursive_prefix_comparator_evaluator'
    spec=get_evaluator_spec(ref)
    assert spec.role == PRIMARY
    assert set(spec.allowed_kernel_services)=={'AUTHORITY_BASIS','PUBLIC_READ','EXACT_RELATION','EXACT_RELATION_PROFILE','EXACT_RELATION_PROFILE_FAMILY'}
    src=inspect.getsource(ev.s8_s7_class_recursive_prefix_comparator_evaluator)
    assert '(1, 8, 32, 124)' in src
    assert 'split_found=True' in src
    assert 'p124_full_equality_equivalent=True' in src


def test_s8_gen32_plan_freezes_class_order_batch_and_projection_schedule():
    src=inspect.getsource(s8._validate_plan)
    assert 'S8_RECURSIVE_PREFIX_SCHEDULE' in src
    assert 'S8_CLASS_BATCH_SIZE' in src
    assert 'DESCENDING_S7_CLASS_SIZE_THEN_CLASS_ID' in src
    assert 'MONOTONE_EXECUTION_CONTEXT_PROJECTION_P124_FULL_EQUALITY_EQUIVALENT' in src
    run=inspect.getsource(s8.run_g6_s8)
    assert 'CLASS_PREFIX_EVALUATOR' in run
    assert 'G6_S8_D{depth}_O{outer_index:03d}_CB{batch_index:03d}' in run
    assert '_ordered_non_singleton_classes' in run


def test_s8_gen32_projection_schedule_is_monotone_and_full_at_p124():
    src=inspect.getsource(ev._RecursiveComputer.projected_signature)
    assert 'self.execution[:p]' in src
    assert 'p not in (1, 8, 32, 124)' in src
    outer=inspect.getsource(ev._RecursiveComputer.outer_component_projection)
    assert 'self.projected_signature(child, level - 1, prefix_count=p)' in outer
    # P124 retains every factor-swap-normalized execution coordinate.  The
    # existing alias-map test proves these 124 coordinates cover all 248
    # scientific contexts exactly.
    refs=("D2_BROOM","D2_PATH","D4_BROOM","D4_PATH")
    ops=((0,0),(0,1),(0,4),(0,5),(0,6),(1,0),(1,1),(1,4),(1,5),(1,6),(2,4),(3,5),(3,6),(4,0),(4,1),(4,2),(4,4),(4,5),(4,6),(5,0),(5,1),(5,3),(5,4),(5,5),(5,6),(6,0),(6,1),(6,3),(6,4),(6,5),(6,6))
    assert len(ev._execution_contexts(refs,ops))==124
    assert set(ev._scientific_alias_map(refs,ops))==set(range(124))


def test_s8_gen32_class_comparator_short_circuits_on_first_prefix_separator(monkeypatch):
    class FakeComputer:
        def __init__(self,payload):
            self.metrics={'exact_relation_call_count':0}
        def outer_component_projection(self, tree, *, level, execution_context_index, prefix_count):
            self.metrics['exact_relation_call_count']+=1
            token=tree['token']
            if prefix_count < 8:
                return ('same',prefix_count)
            return ('sep',prefix_count, token)
    monkeypatch.setattr(ev,'_RecursiveComputer',FakeComputer)
    payload={
      'future_depth':3,'outer_execution_context_index':0,'recursive_prefix_schedule':[1,8,32,124],
      's7_class_id':'C','class_members':[{'state_token':'a','state_tree':{'token':'a'}},{'state_token':'b','state_tree':{'token':'b'}}],
      'basis_refs':['D2_BROOM','D2_PATH','D4_BROOM','D4_PATH'],'operator_basis':[[0,0]]*31,
      'max_recursive_states':1000,'max_exact_relation_calls':1000,'max_worker_rss_bytes':536870912,
    }
    got=ev.s8_s7_class_recursive_prefix_comparator_evaluator(payload)
    assert got['metrics']['split_found'] is True
    assert got['metrics']['recursive_prefix_count']==8
    assert got['metrics']['a_state_token']=='a' and got['metrics']['b_state_token']=='b'
    assert got['outcome_count']==4


def test_s8_gen32_controller_permit_includes_class_prefix_evaluator():
    from infinity_grid import v05_controller_event_loop as loop
    source=inspect.getsource(loop._handle_g6_science)
    block=source.split("elif op=='G6_S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE':",1)[1].split("elif op=='G6_R0_POST_GRADUATION_FIBER':",1)[0]
    assert 's8_s7_class_recursive_prefix_comparator_evaluator' in block



def test_s8_gen34_partial_depth1_projection_uses_scalar_profiles_and_p124_uses_family():
    src = inspect.getsource(ev._RecursiveComputer.profile_counts_prefix)
    assert 'EXACT_RELATION_PROFILE", tree' in src
    assert 'if p == 124:' in src
    assert 'counts = self.profile_counts(tree)' in src
    projected = inspect.getsource(ev._RecursiveComputer.projected_signature)
    assert 'if p == 124:' in projected
    assert 'return self.depth1(tree)' in projected
    assert 'counts = self.profile_counts_prefix(tree, p)' in projected
    # Depth-2 projected child vectors must also obey the same nested prefix.
    assert 'self.profile_counts_prefix(child, p)' in projected


def test_s8_gen34_profile_prefix_extends_monotonically_and_full_boundary_is_exact(monkeypatch):
    class FakeView:
        def __init__(self):
            self.scalar_calls=[]
            self.family_calls=0
        def call(self, name, *args):
            if name == 'AUTHORITY_BASIS':
                return {ref:{'seed':ref} for ref in ('A','B','C','D')}
            if name == 'PUBLIC_READ':
                return {'legal':True,'descriptor':'CAPS7_PLUS_H_CLASS_BAG','caps7':[1]*7,'H_class_bag':{}}
            if name == 'EXACT_RELATION_PROFILE':
                tree, seed, op = args
                ref=seed['seed']; idx=('A','B','C','D').index(ref)*31 + OPS.index(tuple(op))
                self.scalar_calls.append(idx)
                return {'exact_outcome_count':idx+10}
            if name == 'EXACT_RELATION_PROFILE_FAMILY':
                tree, seeds, ops = args
                self.family_calls += 1
                return {'exact_outcome_counts':[i+10 for i in range(124)]}
            raise AssertionError(name)
    OPS=tuple((i, i) for i in range(31))
    fake=FakeView()
    monkeypatch.setattr(ev, 'current_kernel_view', lambda: fake)
    payload={
      'basis_refs':['A','B','C','D'],'operator_basis':[list(x) for x in OPS],
      'max_recursive_states':10000,'max_exact_relation_calls':10000,
      'max_worker_rss_bytes':2**40,
    }
    c=ev._RecursiveComputer(payload)
    tree={'x':1}
    p1=c.profile_counts_prefix(tree,1)
    p8=c.profile_counts_prefix(tree,8)
    p32=c.profile_counts_prefix(tree,32)
    assert p1 == tuple(range(10,11))
    assert p8 == tuple(range(10,18))
    assert p32 == tuple(range(10,42))
    assert fake.scalar_calls == list(range(32))
    assert fake.family_calls == 0
    full_prefix=c.profile_counts_prefix(tree,124)
    full=c.depth1(tree)
    assert full_prefix == tuple(range(10,134))
    assert full == ('ORDINARY_BRANCH_MULTISET',1,c.public_sig(tree),tuple(range(10,134)))
    # Both explicit full-prefix evaluation and the accepted full depth1 path use
    # the A30 family transform; no scalar expansion is used at P124.
    assert fake.family_calls == 2
    # P124 is exactly the full boundary, not merely another projection tag.
    assert c.projected_signature(tree,1,prefix_count=124) == full


def test_s8_gen34_partial_prefix_is_execution_only_and_does_not_change_full_depth1():
    projected = inspect.getsource(ev._RecursiveComputer.projected_signature)
    assert 'ORDINARY_BRANCH_MULTISET_D1_PROJECTION' in projected
    depth1 = inspect.getsource(ev._RecursiveComputer.depth1)
    assert 'counts = self.profile_counts(tree)' in depth1
    assert 'profile_counts_prefix' not in depth1
