from pathlib import Path
from infinity_grid.v05_stage_registry import controller_only_registry_audit, get_evaluator_spec
from infinity_grid.v05_kernel_services import REFERENCE_ORACLE, bind_kernel_view
from infinity_grid.v05_kernel_service_providers import build_kernel_service_providers
from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel

def test_o3b_all_registered_evaluators_are_service_only_and_unexempted():
    audit=controller_only_registry_audit(); assert audit['status']=='PASS'; assert audit['legacy_frozen_evaluator_count']==0
    assert len(audit['evaluators'])==25
    assert any(row['registry_ref']=='infinity_grid.g6_s8_evaluators:s8_s7_class_recursive_prefix_comparator_evaluator' for row in audit['evaluators'])
    for row in audit['evaluators']:
        sem=row['semantic_gate']; assert sem['status']=='PASS'; assert sem['violation_count']==0; assert sem['migration_required'] is False

def test_o3b_runtime_provider_binding_is_exactly_declared_services():
    ref='infinity_grid.g6_s5r_crw_evaluators:observer_state_write_law_evaluator'; spec=get_evaluator_spec(ref)
    configure_relation_kernel(scope_identity='O3B:PROVIDER:TEST'); providers=build_kernel_service_providers(spec); view=bind_kernel_view(spec,providers)
    assert set(view.allowed_services)==set(providers)
    assert 'OBSERVER_DECODE' not in providers  # write service owns its internal decode

def test_o3b_reference_oracle_is_explicit_and_not_primary():
    ref='infinity_grid.g6_s6r_evaluators:recursive_closure_independent_evaluator'; spec=get_evaluator_spec(ref)
    assert spec.role==REFERENCE_ORACLE
    assert 'LEGACY_OBSERVER_DECODE_ORACLE' in spec.allowed_kernel_services
    assert spec.reference_oracle_for==('LEGACY_OBSERVER_DECODE_ORACLE',)

def test_o3b_s6_stage_has_no_direct_relation_kernel_or_adapter_imports():
    import infinity_grid.g6_s6r_recursive_closure as s6
    text=Path(s6.__file__).read_text(encoding='utf-8')
    assert 'exact_tree_relation_kernel' not in text
    assert 'G4AcceptedAdapter' not in text
    assert 'g6_stage_executors import _basis' not in text
    assert 'build_stage_kernel_view' in text
