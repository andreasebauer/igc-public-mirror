"""Explicit finite joint-source qualification; historical producing pins stay intact.

The unchanged legacy consumer uses one closure map for its execution byte guard.
This adapter supplies a separately declared execution closure to that mechanism,
after proving the precise registry delta and the complete historical closure.
It never persists the internal guard view as historical producer provenance.
Production loaders and theorem admission do not use this adapter.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from . import v2_candidate_consumer as legacy

SCOPE = 'EXACT_JOINT_FINITE_V2_QUALIFICATION_ONLY'
LEGACY_CONSUMER_SHA256 = 'bac2c057b366a6275636c793e8565b5d2951850a295bc07e33b0e4ce20758ce0'
MARKER_EVALUATOR_SHA256 = '0bbab19a5003c84eac56763f8178c0a64700b74d0acc021e9c36e78ddb25108d'
MARKER_REGISTRY_SHA256 = 'e6eeeeb7a27b8d1b3cd843d5654f0256b10f213d14b342a7f37935aa3049a58a'
REGISTRY_ADDITION = """    EvaluatorSpec('infinity_grid.g6_saved_marker_service_evaluators:saved_marker_q_evaluator',PRIMARY,
                  ('EXACT_IDENTITY','EXACT_RELATION','MARKER_Q','MARKER_DECODE','MARKER_WRITE')),
    EvaluatorSpec('infinity_grid.g6_saved_marker_service_evaluators:saved_marker_cold_decode_evaluator',PRIMARY,
                  ('EXACT_IDENTITY','EXACT_RELATION','MARKER_Q','MARKER_DECODE','MARKER_WRITE')),
    EvaluatorSpec('infinity_grid.g6_saved_marker_service_evaluators:saved_marker_write_evaluator',PRIMARY,
                  ('EXACT_IDENTITY','EXACT_RELATION','MARKER_Q','MARKER_DECODE','MARKER_WRITE')),
""".encode()

def _require(value, code):
    if not value:
        raise legacy.CandidateAdmissionError(code)

def _files(package):
    return {str(p.relative_to(package)) for p in package.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}

def _pin(package, name, pin):
    rel = Path(name)
    _require(not rel.is_absolute() and '..' not in rel.parts, 'JOINT_CLOSURE_PATH')
    _require(type(pin) is dict and set(pin)=={'sha256','size_bytes'}, 'JOINT_PIN_FIELDS')
    raw = (package/rel).read_bytes()
    _require(type(pin['size_bytes']) is int and len(raw)==pin['size_bytes'] and
             hashlib.sha256(raw).hexdigest()==pin['sha256'], 'JOINT_EXECUTION_BYTES_CHANGED:'+name)
    return raw

@dataclass(frozen=True)
class JointAdmission:
    legacy_admission: legacy.Admission
    original_binding_json: str
    compatibility_json: str

def preflight_joint(root, original_binding, declaration, producer_registry_bytes):
    """Admission for a prospective finite compatibility job, never production."""
    package = Path(__file__).resolve().parent
    _require(type(declaration) is dict and set(declaration)=={
        'schema_id','scope','qualification_authority','original_binding_sha256',
        'execution_closure','marker_capture_id','marker_completion_sha256'}, 'JOINT_DECLARATION_FIELDS')
    _require(declaration['schema_id']=='IG_EXACT_JOINT_V2_EXECUTION_DECLARATION_V1' and
             declaration['scope']==SCOPE and declaration['qualification_authority']=='NONE', 'JOINT_QUALIFICATION_SCOPE')
    _require(legacy.normalized_hash(original_binding)==declaration['original_binding_sha256'], 'JOINT_ORIGINAL_BINDING_CHANGED')
    historical = original_binding['producer_closure']
    execution = declaration['execution_closure']
    additions = {'v2_candidate_consumer.py','g6_saved_marker_service_evaluators.py','v2_joint_candidate_consumer.py'}
    _require(type(execution) is dict and set(execution)==set(historical)|additions and
             _files(package)==set(execution), 'JOINT_EXECUTION_DOMAIN')
    _require(declaration['marker_capture_id']=='11720a08e64562409faf83689218a9e051ab587e6959058eb3e735a8e7b0867e' and
             declaration['marker_completion_sha256']=='1d894cc05c1238f322c3ff2acdbfd00afe05859365df9954cd1f49bf46d03ed5', 'JOINT_MARKER_PROVENANCE')
    for name, pin in historical.items():
        raw = _pin(package,name,execution[name])
        if name=='v05_stage_registry.py':
            _require(type(producer_registry_bytes) is bytes and
                     len(producer_registry_bytes)==pin['size_bytes'] and
                     hashlib.sha256(producer_registry_bytes).hexdigest()==pin['sha256'], 'JOINT_ORIGINAL_REGISTRY_PIN')
            _require(hashlib.sha256(raw).hexdigest()==MARKER_REGISTRY_SHA256 and
                     raw.count(REGISTRY_ADDITION)==1 and
                     raw.replace(REGISTRY_ADDITION,b'',1)==producer_registry_bytes, 'JOINT_REGISTRY_DELTA')
        else:
            _require(execution[name]==pin, 'JOINT_UNREVIEWED_PRODUCER_CHANGE:'+name)
    for name in additions:
        _pin(package,name,execution[name])
    _require(execution['v2_candidate_consumer.py']['sha256']==LEGACY_CONSUMER_SHA256 and
             execution['g6_saved_marker_service_evaluators.py']['sha256']==MARKER_EVALUATOR_SHA256, 'JOINT_ACCEPTED_MODULE_PINS')
    # The internal copy is an execution guard view. All historical metadata and
    # artifact bindings are preserved, and the original binding is retained below.
    guard_view = deepcopy(original_binding)
    guard_view['producer_closure'] = deepcopy(execution)
    admitted = legacy.preflight(root,guard_view)
    return JointAdmission(admitted,json.dumps(original_binding,sort_keys=True,allow_nan=False),
                          json.dumps(declaration,sort_keys=True,allow_nan=False))

class JointFiniteCandidateConsumer:
    def __init__(self, admission):
        _require(type(admission) is JointAdmission, 'JOINT_ADMISSION_REQUIRED')
        self._admission = admission
        self._consumer = legacy.FiniteCandidateConsumer(admission.legacy_admission)

    def scan(self, domain, level):
        # The unchanged consumer rechecks all declared execution/input bytes on
        # every call, including cache hits, and returns an isolated packet copy.
        return self._consumer.scan(domain,level)

    def provenance(self):
        return {'scope':SCOPE,'historical_producer_binding':json.loads(self._admission.original_binding_json),
                'execution_declaration':json.loads(self._admission.compatibility_json),
                'production_authority':'NONE'}
