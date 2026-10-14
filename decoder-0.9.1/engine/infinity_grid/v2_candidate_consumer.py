"""Versioned finite V2 candidate consumption. No production or theorem authority."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .canon import canonical_sha256
from .maturation_dag_contract import validate_state_dag

SCOPE = 'FINITE_SAVED_V2_QUALIFICATION_ONLY'
LANES = ('branching', 'symmetry', 'lineage', 'topology_services')

class CandidateAdmissionError(RuntimeError):
    pass

def _require(value, code):
    if not value:
        raise CandidateAdmissionError(code)

def normalized(value):
    """The producer's explicit JSON-tree normalization policy, including int keys."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))

def normalized_hash(value):
    return canonical_sha256(normalized(value))

def _read(path):
    def pairs(items):
        out = {}
        for key, value in items:
            _require(key not in out, 'DUPLICATE_JSON_KEY')
            out[key] = value
        return out
    return json.loads(path.read_bytes(), object_pairs_hook=pairs)

def _seal(obj):
    _require(obj.get('science_sha256') == normalized_hash({k:v for k,v in obj.items() if k != 'science_sha256'}), 'CANDIDATE_CONTENT_SEAL')

def _unchanged(root, binding, package):
    for name, pin in binding['artifacts'].items():
        _require(name in {'materialized', 'streaming', 'dag', 'index', 'verification', 'completion', 'capture'}, 'ARTIFACT_ROLE')
        rel = Path(pin['file_name'])
        _require(not rel.is_absolute() and len(rel.parts) == 1, 'ARTIFACT_PATH')
        raw = (root/rel).read_bytes()
        _require(type(pin['size_bytes']) is int and len(raw) == pin['size_bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256'], 'INPUT_BYTES_CHANGED:'+name)
    for name, pin in binding['producer_closure'].items():
        rel = Path(name)
        _require(not rel.is_absolute() and '..' not in rel.parts, 'CLOSURE_PATH')
        raw = (package/rel).read_bytes()
        _require(len(raw) == pin['size_bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256'], 'PRODUCER_CLOSURE_CHANGED:'+name)

@dataclass(frozen=True)
class Admission:
    root: Path
    binding_json: str
    package: Path

def preflight(root, binding):
    """Refuse invalid data before O7 reconstruction; caller freezes trusted bindings."""
    try:
        root = Path(root).resolve(strict=True)
        package = Path(__file__).resolve().parent
        _require(type(binding) is dict and set(binding) == {'schema_id','scope','producer_source_sha256','producer_capture_id','producer_completion_sha256','runtime_manifest_sha256','artifacts','producer_closure'}, 'BINDING_FIELDS')
        _require(binding['schema_id'] == 'IG_FINITE_V2_CONSUMER_BINDING_V1' and binding['scope'] == SCOPE, 'QUALIFICATION_SCOPE_REQUIRED')
        _require(set(binding['artifacts']) == {'materialized','streaming','dag','index','verification','completion','capture'}, 'COMPLETE_ARTIFACT_BINDING_REQUIRED')
        _require(binding['producer_closure'], 'PRODUCER_CLOSURE_REQUIRED')
        _unchanged(root, binding, package)
        data = {name:_read(root/pin['file_name']) for name,pin in binding['artifacts'].items()}
        verification = data['verification']
        _require(verification['status'] == 'VERIFIED' and verification['failures'] == [] and verification['scientific_outcome'] == 'FINITE_V2_CALIBRATION_CANDIDATES_QUALIFIED_NOT_ADOPTED', 'PRODUCER_RESULT_REQUIRED')
        completion = data['completion']
        capture = data['capture']
        _require(completion.get('completion_sha256') == binding['producer_completion_sha256'] and completion.get('status') == 'COMPLETED' and completion.get('evidence_status') == 'VERIFIED', 'PRODUCER_COMPLETION_PIN')
        _require(capture.get('capture_id') == binding['producer_capture_id'] and completion.get('source_sha256') == capture['workspace']['source_sha256'], 'PRODUCER_CAPTURE_BINDING')
        evidence = {row['path']:row for row in completion['evidence']}
        for name in ('materialized','streaming','index'):
            pin = binding['artifacts'][name]
            row = evidence.get('script/work/'+pin['file_name'],{})
            _require(row.get('sha256') == pin['sha256'] and row.get('size_bytes') == pin['size_bytes'], 'PRODUCER_ARTIFACT_CORRESPONDENCE:'+name)
        for name in ('materialized','streaming'):
            obj = data[name]
            _require(obj['status'] == 'CANDIDATE_NO_CONSUMPTION_AUTHORITY' and obj['diagnostic_schema'] == 'IG_RESOURCE_ACTION_DIAGNOSTIC_V2' and obj['organizational_schema'] == 'IG_RESOURCE_ORGANIZATIONAL_SIGNATURE_V2', 'CANDIDATE_VERSION_SCOPE')
            _require(obj['source_binding']['engine_source_sha256'] == binding['producer_source_sha256'], 'PRODUCER_SOURCE_BINDING')
            _seal(obj)
            for pin in obj['source_binding']['resource_bindings']:
                rel = Path(pin['path'])
                _require(rel.parts[0] == 'infinity_grid' and '..' not in rel.parts, 'PRODUCER_RESOURCE_PATH')
                raw = package.joinpath(*rel.parts[1:]).read_bytes()
                _require(len(raw) == pin['size_bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256'], 'PRODUCER_RESOURCE_BINDING')
        materialized, streaming, dag, index = (data[n] for n in ('materialized','streaming','dag','index'))
        _require(materialized['execution_policy'] == 'NATIVE_V3' and materialized['calibration_installed'] is False and materialized['theorems_adopted'] is False and streaming['authority_adopted'] is False, 'NO_ADOPTION_SCOPE')
        _require(set(materialized['levels']) == {str(x) for x in range(7,14)}, 'MATERIALIZED_LEVEL_DOMAIN')
        _require(type(streaming['recipe_count']) is int and streaming['recipe_count'] == 193 and len(streaming['recipe_catalogue']) == 193, 'RECIPE_DOMAIN')
        _require(normalized_hash(streaming['recipe_catalogue']) == streaming['recipe_catalogue_sha256'], 'RECIPE_CATALOGUE_SEAL')
        _require(set(streaming['lane_hashes']) == set(LANES) and set(streaming['normalized_lane_payloads']) == set(LANES), 'LANE_DOMAIN')
        for lane in LANES:
            _require(normalized_hash(streaming['normalized_lane_payloads'][lane]) == streaming['lane_hashes'][lane], 'LANE_PAYLOAD_HASH:'+lane)
            _require(streaming['full_recipe_scan']['normalized_signature'][lane] == streaming['normalized_lane_payloads'][lane], 'LANE_PACKET_CORRESPONDENCE:'+lane)
        validate_state_dag(dag)
        _require(dag['execution_policy'] == 'NATIVE_V3' and len(dag['nodes']) == 4144, 'CURRENT_DAG_SCOPE')
        rows = index['roots']
        _require(type(index['root_occurrences']) is int and index['root_occurrences'] == len(rows) == 747 and type(index['unique_constructions']) is int and index['unique_constructions'] == len(set(dag['roots'])) == 675 and index['no_carrier_certification_adopted'] is True, 'ROOT_OCCURRENCE_DOMAIN')
        _require([r['construction_digest'] for r in rows] == dag['roots'], 'ORDERED_ROOT_CORRESPONDENCE')
        counts = {}
        for i, row in enumerate(rows):
            _require(type(row['root_index']) is int and row['root_index'] == i and type(row['level']) is int, 'ROOT_INDEX_TYPE_ORDER')
            key = (row['domain'],row['level'])
            _require(key in {('selected',l) for l in range(7,14)} | {('full_pool',l) for l in (11,12,13)}, 'ROOT_DOMAIN_LEVEL')
            node = dag['nodes'][row['construction_digest']]
            _require(row['level'] == (7 if node['kind'] == 'O7' else node['level']), 'ROOT_LEVEL_CORRESPONDENCE')
            counts[key] = counts.get(key,0) + 1
        _require(counts == {**{('selected',l):24 for l in range(7,14)}, **{('full_pool',l):193 for l in (11,12,13)}}, 'ROOT_GROUP_COUNTS')
        return Admission(root,json.dumps(binding,sort_keys=True,allow_nan=False),package)
    except CandidateAdmissionError:
        raise
    except Exception as exc:
        raise CandidateAdmissionError('CANDIDATE_REFUSED:'+str(exc)) from exc

class FiniteCandidateConsumer:
    """Consume only sealed finite panels; return copies of cached JSON observations."""
    def __init__(self, admission):
        _require(type(admission) is Admission, 'ADMISSION_REQUIRED')
        self._admission = admission
        self._binding = json.loads(admission.binding_json)
        _unchanged(admission.root,self._binding,admission.package)
        from .isolated_runtime import BUNDLED_MANIFEST_SHA256, verified_runtime
        _require(self._binding['runtime_manifest_sha256'] == BUNDLED_MANIFEST_SHA256, 'RUNTIME_BINDING')
        verified_runtime()
        from .maturation_parallel import _states_from_dag, state_wire
        self._data = {n:_read(admission.root/p['file_name']) for n,p in self._binding['artifacts'].items()}
        self._engine, self._states = _states_from_dag(self._data['dag'])
        _require(len(self._states) == 747, 'LOADED_OCCURRENCE_COUNT')
        for state,row in zip(self._states,self._data['index']['roots']):
            _require(state.construction_digest == row['construction_digest'] and normalized_hash(state_wire(state)) == row['public_state_sha256'], 'LOADED_PUBLIC_CORRESPONDENCE')
        _unchanged(admission.root,self._binding,admission.package)
        self._cache = {}

    def scan(self, domain, level):
        _require(type(level) is int and (domain,level) in {('selected',l) for l in range(7,14)} | {('full_pool',l) for l in (11,12,13)}, 'UNSUPPORTED_FINITE_DOMAIN')
        _unchanged(self._admission.root,self._binding,self._admission.package)
        key = (domain,level)
        if key not in self._cache:
            from . import regime_scanner as rs
            from .fixed_grammar_transport import load_streaming_o7_seed
            selected = [s for s,r in zip(self._states,self._data['index']['roots']) if (r['domain'],r['level']) == key]
            pairs = [tuple(x) for x in load_streaming_o7_seed()['bridge_pairs']]
            grammar = self._data['materialized']['levels'][str(level)]['grammar_sha256']
            packet, _ = rs._scan_level(selected,level,pairs,grammar)
            packet = normalized(packet)
            if domain == 'selected':
                expected = self._data['materialized']['levels'][str(level)]
                for lane in LANES:
                    _require(normalized_hash(packet['normalized_signature'][lane]) == expected[lane+'_sha256'], 'SELECTED_LANE_MISMATCH:'+lane)
                _require(normalized_hash(packet['normalized_signature']) == expected['normalized_signature_sha256'], 'SELECTED_SIGNATURE_MISMATCH')
                _require(normalized_hash(packet['diversity']) == expected['diversity_sha256'], 'SELECTED_DIVERSITY_MISMATCH')
            else:
                for lane in LANES:
                    _require(packet['normalized_signature'][lane] == self._data['streaming']['normalized_lane_payloads'][lane], 'FULL_POOL_LANE_MISMATCH:'+lane)
            self._cache[key] = packet
        _unchanged(self._admission.root,self._binding,self._admission.package)
        return deepcopy(self._cache[key])
