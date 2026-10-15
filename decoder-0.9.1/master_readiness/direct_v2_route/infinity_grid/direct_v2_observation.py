"""Explicit direct V2 diagnostic route; no inherited theorem consumption.

New protocol/capabilities deliberately separate direct diagnostic observations
from complete successor Counters, action witnesses and matched-backbone claims.
The existing RunPlan/transport providers and their guards are unchanged.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .canon import canonical_sha256
from . import regime_scanner as rs
from . import semantic_sentinel as ss
from . import qualified_discovery_inputs as qi

ROUTE = 'IG_DIRECT_MATERIALIZED_V2_DIAGNOSTICS@1.0.0'
SCOPE = 'DIRECT_FINITE_OBSERVATIONS_NO_THEOREM_OR_GENERATION_AUTHORITY'
KERNEL_PINS = {'O7': '49e167058059b3ee35df82d54fa0c723cbf5b1bd40eec61839f406b6a5c1b776',
               'O6': '75eed545afd017ba01d0f1e7d84078db65e7a1d81032f5a6ab0d9744f44ef72a'}
CAPABILITIES = ('V2_DIAGNOSTIC_PACKET', 'V2_BRANCHING_DIAGNOSTIC',
                'V2_SYMMETRY_DIAGNOSTIC', 'V2_LINEAGE_DIAGNOSTIC',
                'V2_TOPOLOGY_SERVICE_DIAGNOSTIC')
READ_SET = ('hierarchical_owner_occurrences', 'owner_capacity_vectors',
            'top_pairs', 'typed_edges', 'source_construction_identity',
            'native_Bridge', 'level', 'finite_population_membership',
            'resource_skins', 'leaf_counts', 'total_relation_counts',
            'reverse_topology_factor_fingerprints')


class DirectObservationError(RuntimeError):
    pass


def require(value, code):
    if not value:
        raise DirectObservationError(code)


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def source_snapshot():
    """Bind all package Python/resources, not merely a wrapper's function hash."""
    root = Path(__file__).resolve().parent
    files = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc', '.pyo'):
            continue
        files.append((p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()))
    return canonical_sha256(files)


def make_contract():
    """Prospective native capture must freeze this contract before dispatch.

    A seal binds content; it does not grant source trust or production authority.
    Native capture/save/run supplies the execution/source admission separately.
    """
    c = {'schema_id': 'IG_DIRECT_V2_OBSERVATION_CONTRACT_V1', 'route': ROUTE,
         'scope': SCOPE, 'production_authority': 'NONE', 'generation_allowed': False,
         'theorem_transport': 'NOT_CONSUMED', 'complete_successor_counter': 'NOT_PROVIDED',
         'matched_backbone': 'NOT_PROVIDED', 'inherited_no_read_theorem': 'NOT_CONSUMED',
         'read_set': list(READ_SET), 'capabilities': list(CAPABILITIES),
         'kernel_source_sha256': dict(KERNEL_PINS),
         'runtime_manifest_sha256': '6a205e5d39a3acd7261b23caaf06997d41cc65ebd1329b1766c143f5dacf78b5',
         'package_payload_sha256': source_snapshot(),
         'native_semantics_snapshot_sha256': ss.native_semantics_snapshot()['science_sha256'],
         'diagnostic_schema': rs.ACTION_DIAGNOSTIC_SCHEMA_V2,
         'organizational_schema': rs.ORGANIZATIONAL_SCHEMA_V2,
         'admitted_levels': [7, 100],
         'quantity_policy': 'V2_CAPACITY_PRODUCTS_ARE_DIAGNOSTICS_NOT_SITE_OCCURRENCE_COUNTERS'}
    c['science_sha256'] = canonical_sha256(c)
    return c


def validate_contract(contract):
    require(type(contract) is dict, 'DIRECT_CONTRACT_REQUIRED')
    # Compare complete structure as well as the seal: extra claims are refused.
    require(contract == make_contract(), 'DIRECT_CONTRACT_SOURCE_OR_SCOPE_CHANGED')


def compile_requests(contract, requested):
    validate_contract(contract)
    require(type(requested) in (list, tuple) and requested and
            all(type(x) is str for x in requested) and len(set(requested)) == len(requested),
            'DIRECT_CAPABILITY_REQUEST_REQUIRED')
    missing = sorted(set(requested) - set(CAPABILITIES))
    require(not missing, 'DIRECT_REQUIRED_EVIDENCE_NOT_PROVIDED:' + ','.join(missing))
    return {'schema_id': 'IG_DIRECT_V2_COMPILED_REQUEST_V1', 'route': ROUTE,
            'scope': SCOPE, 'requested': list(requested), 'required_theorems': [],
            'transport_bindings': [], 'contract_sha256': contract['science_sha256'],
            'generation_allowed': False, 'production_authority': 'NONE'}


def _packet(packet):
    p = normalized(packet)
    require(p.get('diagnostic_schema') == rs.ACTION_DIAGNOSTIC_SCHEMA_V2 and
            p.get('normalized_signature', {}).get('schema_id') == rs.ORGANIZATIONAL_SCHEMA_V2,
            'DIRECT_V2_PACKET_REQUIRED')
    require(type(p.get('level')) is int and 7 <= p['level'] <= 100 and
            type(p.get('states')) is int and p['states'] > 0, 'DIRECT_PACKET_DOMAIN')
    for lane in ('branching', 'symmetry', 'lineage', 'topology_services'):
        require(type(p.get(lane)) is dict, 'DIRECT_DIAGNOSTIC_LANE_MISSING:' + lane)
    # No silent attachment of inherited scientific conclusions to this new route.
    require(not p.get('inherited_law_events') and not p.get('theorem_transport_delta') and
            not p.get('theorem_transport_auxiliary_audit'), 'DIRECT_INHERITED_CLAIM_REFUSED')
    return p


def _dispatch(packet, contract, requested, provenance):
    compiled = compile_requests(contract, requested)
    p = _packet(packet)
    values = {'V2_DIAGNOSTIC_PACKET': p, 'V2_BRANCHING_DIAGNOSTIC': p['branching'],
              'V2_SYMMETRY_DIAGNOSTIC': p['symmetry'], 'V2_LINEAGE_DIAGNOSTIC': p['lineage'],
              'V2_TOPOLOGY_SERVICE_DIAGNOSTIC': p['topology_services']}
    result = {'schema_id': 'IG_DIRECT_V2_OBSERVATION_RESULT_V1', 'route': ROUTE,
              'scope': SCOPE, 'status': 'OBSERVED_FINITE_DIAGNOSTICS_ONLY',
              'compiled_request': compiled, 'provenance': deepcopy(provenance),
              'packet_sha256': canonical_sha256(p),
              'observations': {key: deepcopy(values[key]) for key in requested},
              'theorem_transport': 'NOT_CONSUMED', 'complete_successor_counter': 'NOT_PROVIDED',
              'matched_backbone': 'NOT_PROVIDED', 'production_authority': 'NONE'}
    result['science_sha256'] = canonical_sha256(result)
    return result


def observe_materialized(states, level, bridge_pairs, contract, requested=CAPABILITIES):
    """Execute native V2 scanner diagnostics on an actual finite state population.

    Does not create carriers, call a construction sampler/read probe, consume
    inherited service laws, or request theorem transport. Native scanner logic
    retains its topology/support reads and capacity-product diagnostic unit.
    """
    compile_requests(contract, requested)
    require(type(level) is int and 7 <= level <= 100 and type(states) in (list, tuple) and states,
            'DIRECT_MATERIALIZED_DOMAIN')
    require(all(type(s) in (rs.O7State, rs.LiftState) and s.level == level and
                s.execution_policy == rs.NATIVE_EXECUTION_POLICY for s in states),
            'DIRECT_NATIVE_SCANNER_STATES_REQUIRED')
    require(len({s.construction_digest for s in states}) == len(states), 'DIRECT_DUPLICATE_ROOT')
    require(type(bridge_pairs) in (list, tuple) and bridge_pairs and
            all(type(p) in (list, tuple) and len(p) == 2 and
                all(type(t) is int and 0 <= t < 7 for t in p) for p in bridge_pairs),
            'DIRECT_BRIDGE_DOMAIN')
    pairs = [tuple(p) for p in bridge_pairs]
    require(len(set(pairs)) == len(pairs) and
            {(b, a) for a, b in pairs} == set(pairs) and
            all(type(t) is int and 0 <= t < 7 for p in pairs for t in p) and
            all(len(p) == 2 for p in pairs), 'DIRECT_BRIDGE_DOMAIN')
    engine = states[0].engine
    require(all(s.engine is engine for s in states), 'DIRECT_MIXED_NATIVE_ENGINES')
    require(hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest() == KERNEL_PINS['O7'] and
            hashlib.sha256(Path(engine.O6.__file__).read_bytes()).hexdigest() == KERNEL_PINS['O6'],
            'DIRECT_NATIVE_KERNEL_SOURCE_CHANGED')
    _ports, native_pairs = states[0].engine.O6.load_rules()
    require(set(pairs) == {tuple(p) for p in native_pairs}, 'DIRECT_NATIVE_BRIDGE_PIN')
    packet, _signature = rs._scan_level(list(states), level, pairs, rs.GRAMMAR_EXPECTED)
    return _dispatch(packet, contract, requested, {'kind': 'NATIVE_MATERIALIZED_OBSERVATION',
        'level': level, 'ordered_roots': [s.construction_digest for s in states],
        'parent_projection_scope': 'SAVED_OR_NATIVE_STAGE_STATE_INPUTS',
        'generated_by_this_route': False})


class SavedDirectSession:
    """Consume exact source-bound packets, without reconstructing qualified DAGs."""
    def __init__(self, root, binding, contract):
        validate_contract(contract)
        admission = qi.preflight(root, binding)
        self.root = admission.root
        self.binding = json.loads(admission.binding_json)
        self.contract = deepcopy(contract)
        self.closed = False
        self.packets = {}
        for role, domain, key in [('materialized_audit', 'selected', 'levels'),
                                  ('full_audit', 'full_pool', 'depths')]:
            data = qi.read(self.root/self.binding['artifacts'][role]['file_name'])
            for row in data[key]:
                self.packets[domain, row['level']] = _packet(row['scan'])
        require(set(self.packets) == {('selected', i) for i in range(7, 14)} |
                {('full_pool', i) for i in (11, 12, 13)}, 'DIRECT_FINITE_PACKET_DOMAIN')
        qi.unchanged(self.root, self.binding)

    def dispatch(self, domain, level, requested=CAPABILITIES):
        require(not self.closed, 'DIRECT_SESSION_CLOSED')
        validate_contract(self.contract)
        qi.unchanged(self.root, self.binding)
        require(type(domain) is str and type(level) is int and (domain, level) in self.packets,
                'DIRECT_UNQUALIFIED_LEVEL_OR_DOMAIN')
        return _dispatch(self.packets[domain, level], self.contract, requested,
            {'kind': 'QUALIFIED_SAVED_PACKET', 'domain': domain, 'level': level,
             'producing_capture_id': self.binding['prefix_capture_id'],
             'producing_completion_sha256': self.binding['prefix_completion_sha256'],
             'generated_by_this_route': False})

    def close(self):
        self.packets.clear()
        self.closed = True
