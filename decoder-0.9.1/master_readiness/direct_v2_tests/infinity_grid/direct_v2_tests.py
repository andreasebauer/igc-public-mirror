"""Source-bound reconnaissance Tests over direct V2 diagnostic packets.

Reuses legacy projection functions without granting their wider capability or
Regime claims. This is a separate protocol, never an alternate legacy gate.
"""
from copy import deepcopy
import inspect

from . import direct_v2_observation as direct
from . import o7_science_compat as legacy
from .canon import canonical_sha256
from .scientific_architecture import (
    ObservationView, ScientificProtocolRegistry, TestExecution, build_finding,
    make_ref, thaw_json, validate_scientific_artifact,
)

PROTOCOL = 'IG_DIRECT_V2_DIAGNOSTIC_TESTS@1.0.0'
ENTITY = 'IG_DIRECT_V2_FINITE_PACKET@1.0.0'
REGIME = 'IG_DIRECT_V2_DIAGNOSTIC_SCOPE@1.0.0'
PROVIDER = 'IG_DIRECT_V2_TEST_PROVIDER@1.0.0'
# Each row gives the ONLY legacy-shaped payload accepted by that unchanged
# projection. Broad legacy capabilities are neither exported nor certified.
READS = {
    legacy.TEST_GRAMMAR: {'TRANSITION_SYSTEM': ['grammar_sha256']},
    legacy.TEST_DIVERSITY: {'RELATION_GRAPH': ['diversity']},
    legacy.TEST_BRANCHING: {'ACTION_MULTIPLICITY': ['branching']},
    legacy.TEST_SYMMETRY: {'CANONICAL_AUTOMORPHISMS': ['symmetry']},
    legacy.TEST_OVERLAP: {'FACTORIZATION': ['overlap_gluing']},
    legacy.TEST_LINEAGE: {'ANCESTRY': ['lineage']},
    legacy.TEST_QUOTIENT: {'OBSERVER_QUOTIENT': ['quotient_observer']},
    legacy.TEST_TOPOLOGY: {'RELATION_GRAPH': ['topology_services']},
    legacy.TEST_OBSTRUCTION: {'ACTION_ENABLEDNESS': ['obstruction_relief']},
    legacy.TEST_SATURATION: {
        'EXACT_ENTITY_IDENTITY': ['states', 'normalized_signature_sha256'],
        'RESOURCE_COUNTERS': ['raw_growth'],
        'OBSERVER_QUOTIENT': ['normalized_signature']},
}


class DiagnosticTestError(RuntimeError):
    pass


def require(value, reason):
    if not value:
        raise DiagnosticTestError(reason)


def diagnostic_ref(legacy_ref):
    return legacy_ref.replace('IG_TEST_O7_', 'IG_TEST_DIRECT_V2_').replace('_COMPAT@', '_DIAGNOSTIC@')


def test_specs():
    registry = ScientificProtocolRegistry()
    result = {}
    for old in READS:
        spec = deepcopy(registry.test(old))
        spec.update(test_id=diagnostic_ref(old).split('@')[0], version='1.0.0',
            status='DIRECT_DIAGNOSTIC_PROJECTION_ONLY',
            scientific_question='What is the source-bound finite V2 diagnostic projection for ' + old + '?',
            required_capabilities=['V2_DIAGNOSTIC_PACKET'], shared_projection_refs=[old],
            scope_class='RECONNAISSANCE', claim_ceiling='RECONNAISSANCE',
            dependency_test_refs=[], sampling_policy='Exact admitted saved packet population; no resampling.',
            observer_definition='Unchanged legacy projection function applied only to explicit V2 diagnostic fields.',
            default_cadence='One preregistered finite diagnostic dispatch.',
            reopen_conditions=['Any source, packet, scope, unit, or field read changes.'],
            evidence_schema_ref='IG_DIRECT_V2_PROJECTED_DIAGNOSTIC_V1')
        spec['admissibility'] = {'entity_class_refs':[ENTITY], 'regime_refs':[REGIME],
            'constraints':['Diagnostic extraction only; no inherited Regime admission.',
                'No complete Counter, successor equivalence, matched trajectory, plateau or Lift authority.']}
        spec['computation'] = {'kind':'NATIVE_MODULE',
            'entrypoint_ref':'infinity_grid.direct_v2_tests.execute',
            'deterministic_science_projection':'Source-pinned legacy projection, explicit limited read adapter.'}
        spec['outcome_rules'] = {'null':'Projection recorded; no positive scientific inference evaluated.',
            'positive':'Not provided by this diagnostic Test.',
            'counterexample':'Changed source, evidence or undeclared read causes refusal.'}
        validate_scientific_artifact(spec)
        result[make_ref(spec['test_id'],spec['version'])] = spec
    return result


def make_contract(direct_contract):
    direct.validate_contract(direct_contract)
    c = {'schema_id':'IG_DIRECT_V2_TEST_CONTRACT_V1', 'protocol':PROTOCOL,
        'direct_contract_sha256':direct_contract['science_sha256'],
        'test_specs':test_specs(), 'read_bindings':deepcopy(READS),
        'projection_source_sha256':{old:canonical_sha256(inspect.getsource(legacy._ADAPTERS[old])) for old in READS},
        'legacy_test_specs_sha256':canonical_sha256(ScientificProtocolRegistry().tests()),
        'scientific_status_policy':'NULL_NO_POSITIVE_INFERENCE_EVALUATED',
        'quantity_policy':direct_contract['quantity_policy'],
        'production_authority':'NONE', 'generation_allowed':False,
        'legacy_regime_gate_bypass':False, 'full_pipeline_qualified':False}
    c['science_sha256'] = canonical_sha256(c)
    return c


def compile_tests(contract, direct_contract, requested):
    require(contract == make_contract(direct_contract), 'TEST_SOURCE_OR_SCOPE_CHANGED')
    require(type(requested) in (tuple,list) and requested and
        all(type(r) is str for r in requested) and len(set(requested)) == len(requested), 'TEST_REQUEST_REQUIRED')
    unknown = sorted(set(requested)-set(contract['test_specs']))
    require(not unknown, 'TEST_NOT_PROVIDED:' + ','.join(unknown))
    return {'protocol':PROTOCOL, 'test_refs':list(requested),
        'contract_sha256':contract['science_sha256'], 'required_theorems':[],
        'production_authority':'NONE', 'generation_allowed':False}


class _ProjectionReads:
    """Private, limited legacy SHAPE adapter; not a capability provider."""
    def __init__(self, old, packet):
        self.old = old
        self.packet = packet
        self.reads = []

    def read(self, key):
        require(key in READS[self.old], 'UNDECLARED_PROJECTION_READ:' + key)
        self.reads.append(key)
        return {field:deepcopy(self.packet[field]) for field in READS[self.old][key]}


def execute(session, domain, level, contract, direct_contract, requested):
    compiled = compile_tests(contract, direct_contract, requested)
    require(type(session) is direct.SavedDirectSession and session.contract == direct_contract,
        'SOURCE_BOUND_SAVED_SESSION_REQUIRED')
    observed = session.dispatch(domain, level, ['V2_DIAGNOSTIC_PACKET'])
    packet = observed['observations']['V2_DIAGNOSTIC_PACKET']
    source = observed['science_sha256']
    executions = []
    reverse = {diagnostic_ref(old):old for old in READS}
    for ref in requested:
        spec = contract['test_specs'][ref]
        view = ObservationView(entity_instance_ref=f'direct:{domain}:O{level}:{observed["packet_sha256"]}',
            entity_class_ref=ENTITY, regime_ref=REGIME, regime_depth=level,
            test_ref=ref, required_capabilities=spec['required_capabilities'],
            capability_payloads=observed['observations'], source_science_sha256=source, provider_ref=PROVIDER)
        p = thaw_json(view.read('V2_DIAGNOSTIC_PACKET'))
        old = reverse[ref]
        facade = _ProjectionReads(old, p)
        projection = legacy._ADAPTERS[old](facade)
        require(set(facade.reads) == set(READS[old]), 'PROJECTION_READ_SET_CHANGED')
        evidence = {'schema_id':'IG_DIRECT_V2_PROJECTED_DIAGNOSTIC_V1',
            'projection':projection, 'legacy_projection_ref':old,
            'contract_sha256':contract['science_sha256'], 'packet_sha256':observed['packet_sha256'],
            'provenance':observed['provenance'], 'field_reads':deepcopy(READS[old]),
            'quantity_policy':contract['quantity_policy'], 'interpretation':'FINITE_DIAGNOSTIC_ONLY',
            'complete_successor_counter':'NOT_PROVIDED', 'matched_backbone':'NOT_PROVIDED'}
        finding = build_finding(test_spec=spec, entity_instance_ref=view.entity_instance_ref,
            regime_ref=REGIME, regime_depth=level, source_science_sha256=source,
            evidence_payload=evidence, scientific_status='NULL',
            summary='Finite diagnostic projection recorded; no positive inference, inherited Regime admission, or future-equivalence claim evaluated.')
        exe = TestExecution(ref, evidence, canonical_sha256(evidence), finding, view.artifact())
        executions.append({'test_ref':exe.test_ref, 'evidence_payload':exe.evidence_payload,
            'evidence_sha256':exe.evidence_sha256, 'finding':dict(exe.finding),
            'observation_view':dict(exe.observation_view), 'actual_shape_reads':facade.reads})
    result = {'schema_id':'IG_DIRECT_V2_TEST_DISPATCH_V1', 'compiled':compiled,
        'domain':domain, 'level':level, 'packet_sha256':observed['packet_sha256'],
        'executions':executions, 'production_authority':'NONE', 'generation_allowed':False}
    result['science_sha256'] = canonical_sha256(result)
    return result


def compatibility_report():
    rows = []
    for spec in ScientificProtocolRegistry().tests():
        old = make_ref(spec['test_id'],spec['version'])
        row = {'legacy_test_ref':old, 'legacy_required_capabilities':spec['required_capabilities'],
            'legacy_execution_authorized':False}
        if old in READS:
            row.update(status='SEPARATE_DIAGNOSTIC_PROJECTION_AVAILABLE',
                diagnostic_test_ref=diagnostic_ref(old), field_reads=deepcopy(READS[old]))
        elif old == legacy.TEST_REPRESENTATION:
            row.update(status='REFUSED_MATCHED_BACKBONE_ABSENT', required_evidence=['source-bound matched longitudinal backbone'])
        else:
            row.update(status='REFUSED_MATCHED_TEMPORAL_CONTEXT_ABSENT',
                required_evidence=['sealed dependency Findings in admitted Regime',
                    'matched population/trajectory', 'next-depth persistence and stopping context'],
                dependency_test_refs=spec['dependency_test_refs'])
        rows.append(row)
    return {'schema_id':'IG_DIRECT_V2_TEST_COMPATIBILITY_V1', 'tests':rows,
        'legacy_regime_gate_bypass':False, 'complete_counter_provided':False,
        'legacy_test_count':12, 'diagnostic_projections':10, 'explicit_evidence_refusals':2,
        'full_pipeline_qualified':False, 'package2_status':'OPEN_COUNTER_READ_AND_MATCHED_LONGITUDINAL_QUALIFICATION'}
