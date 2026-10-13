def test_g6_s6r_modules_import_and_registry():
    import infinity_grid.g6_s6r_recursive_closure as s6
    import infinity_grid.g6_s6r_evaluators as ev
    from infinity_grid.v05_stage_registry import CONTROLLER_ONLY_WORKER_EVALUATORS
    refs=set(CONTROLLER_ONLY_WORKER_EVALUATORS)
    assert s6.STAGE_ID=='G6:S6R'
    assert callable(ev.recursive_closure_holdout_evaluator)
    assert 'infinity_grid.g6_s6r_evaluators:recursive_closure_holdout_evaluator' in refs
    assert 'infinity_grid.g6_s6r_evaluators:recursive_closure_independent_evaluator' in refs


def test_s6_holdout_evaluator_reuses_raw_child_observer_states():
    import infinity_grid.g6_s6r_evaluators as ev
    from infinity_grid.g6_stage_executors import _basis
    from infinity_grid.g6_s5r_compositional_read_write import _tree_record
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import bind_kernel_view
    from infinity_grid.v05_kernel_service_providers import build_kernel_service_providers
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel
    b=_basis(); calls={'n':0}; ref='infinity_grid.g6_s6r_evaluators:recursive_closure_holdout_evaluator'
    configure_relation_kernel(scope_identity='O3B:S6:TEST'); spec=get_evaluator_spec(ref); providers=build_kernel_service_providers(spec)
    real=providers['OBSERVER_Q']
    def counted(*args,**kwargs): calls['n']+=1; return real(*args,**kwargs)
    providers['OBSERVER_Q']=counted; bind_kernel_view(spec,providers)
    payload={'left_tree':_tree_record(b['D2_PATH']),'right_tree':_tree_record(b['D2_BROOM']),'operator':[0,0],'observer_probe_ref':'D2_PATH','observer_operator':[0,0]}
    out=ev.recursive_closure_holdout_evaluator(payload); raw_children=int(out['metrics']['raw_exact_child_count'])
    assert calls['n']==2+raw_children
    assert int(out['metrics']['raw_child_observer_states_reused_for_decode'])==raw_children
    assert int(out['metrics']['q_only_parent_decodes_reused'])==2

