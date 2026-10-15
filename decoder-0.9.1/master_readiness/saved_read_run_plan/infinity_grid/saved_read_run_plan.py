"""Source-bound saved-read RunPlan; never grants producer or theorem authority.

Saved diagnostic Tests, complete reservation reads, matched contexts and R100
descriptors retain separate identities and populations. Parent results are reused
under original producing pins. Nothing here runs a scanner or develops a depth.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from .canon import canonical_sha256
from . import direct_v2_observation as direct, qualified_discovery_inputs as qi

ROUTE = 'IG_SOURCE_BOUND_SAVED_READ_RUN_PLAN@1.0.0'
SCHEMA = 'IG_SOURCE_BOUND_SAVED_READ_RUN_PLAN_V1'
PARENT_CAPTURE = '8e0724d1c2f452857f4d7bd37dad5c29e7758f1889996dcc8a273073b089d820'
PARENT_COMPLETION = 'f01c4be664dec8bd0307fc178a5c3d66eabaca92a064a480198ade1e98e85274'
UNIT = 'ONE_POSITIVE_CAPACITY_SITE_TYPE_OCCURRENCE_NOT_CAPACITY_SLOT'
FILES = ('COMPLETE_READ_CONSUMER.json', 'REUSED_DIAGNOSTIC_CONSUMPTION.json',
         'MATCHED_TEST_FINDINGS.json', 'TEMPORAL_WINDOWS.json', 'NEW_MATCHED_BACKBONES.json')


class SavedReadPlanError(RuntimeError):
    pass


def require(value, reason):
    if not value:
        raise SavedReadPlanError(reason)


def seal(value):
    value = deepcopy(value)
    value['science_sha256'] = canonical_sha256(value)
    return value


def package_manifest():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and p.suffix not in ('.pyc', '.pyo')}


def parent_payloads(evidence_path):
    with zipfile.ZipFile(evidence_path) as z:
        require(len(z.namelist()) == len(set(z.namelist())) and
                set(z.namelist()) == set(FILES) | {'CAPTURE.json', 'COMPLETION.json',
                'PRESERVED.json', 'PACKAGE_MODULES.json'}, 'PARENT_ARCHIVE_CLOSURE')
        raw = {n: z.read(n) for n in z.namelist()}
    data = {n: json.loads(v) for n, v in raw.items()}
    cap, done = data['CAPTURE.json'], data['COMPLETION.json']
    for obj, key, expected in [(cap, 'capture_id', PARENT_CAPTURE),
                               (done, 'completion_sha256', PARENT_COMPLETION)]:
        require(obj[key] == expected == canonical_sha256({k: v for k, v in obj.items()
                if k != key}), 'PARENT_NATIVE_SEAL')
    require(done['status'] == 'COMPLETED' and done['evidence_status'] == 'VERIFIED' and
            done['scientific_outcome'] == 'FINITE_COMPLETE_READ_AND_NEW_MATCHED_CONTEXT_CONSUMER_QUALIFIED'
            and done['source_sha256'] == cap['workspace']['source_sha256'] and
            done['registration_sha256'] == cap['job']['registration_sha256'] and
            done['result_sha256'] == canonical_sha256(done['result']), 'PARENT_NATIVE_BINDING')
    evidence = {r['path']: r for r in done['evidence']}
    for n in FILES:
        row = evidence['script/work/' + n]
        require(hashlib.sha256(raw[n]).hexdigest() == row['sha256'] and
                len(raw[n]) == row['size_bytes'], 'PARENT_EVIDENCE_BYTES')
    saved = data['PRESERVED.json']
    require(saved['capture_id'] == PARENT_CAPTURE and saved['outbox']['status'] == 'PRESERVED'
            and saved['outbox']['pending_bytes'] == 0, 'PARENT_PRESERVATION')
    # Full original package pin uses native Path-component ordering.
    from .v05_execution_authority import digest
    old = data['PACKAGE_MODULES.json']
    require(digest({'schema_id': 'IG_DECODER_EXECUTABLE_SOURCE_TREE_V1',
        'files': [[k, v] for k, v in sorted(old.items(), key=lambda p: Path(p[0]).parts)]})
        == cap['workspace']['package_sha256'], 'PARENT_PACKAGE_PIN')
    current = package_manifest()
    require(set(current) - set(old) == {'saved_read_run_plan.py'} and not set(old) - set(current),
            'PACKAGE_CHANGE_DOMAIN')
    require({n for n in old if old[n] != current[n]} ==
            {'materialized_discovery.py', 'run_plan_v2.py'}, 'SCIENTIFIC_SOURCE_REUSE_CHANGED')
    return data


def make_calibration(binding, evidence_path, master_contract):
    data = parent_payloads(evidence_path)
    complete = data['COMPLETE_READ_CONSUMER.json']
    require(complete['unit'] == UNIT and complete['anchor_count'] == 6 and
            complete['complete_reservation_counters'] == 42, 'COMPLETE_READ_UNIT_OR_DOMAIN')
    anchors = sorted(e['evidence_payload']['source_construction_digest'] for e in complete['executions'])
    require(len(set(anchors)) == 6, 'COUNTER_ANCHOR_DOMAIN')
    require(master_contract.get('schema_id') == 'IG_MASTER_READINESS_CONTRACT_V1' and
            master_contract.get('full_master_generation_allowed') is False,
            'FROZEN_MASTER_CONTRACT_REQUIRED')
    return seal({'schema_id': 'IG_SOURCE_BOUND_SAVED_READ_CALIBRATION_V1', 'route': ROUTE,
        'current_package_payload_sha256': direct.source_snapshot(),
        'runtime_manifest_sha256': direct.make_contract()['runtime_manifest_sha256'],
        'input_binding_sha256': canonical_sha256(binding),
        'master_contract_sha256': canonical_sha256(master_contract),
        'parent_archive_sha256': qi.raw_sha(evidence_path),
        'parent_capture_id': PARENT_CAPTURE, 'parent_completion_sha256': PARENT_COMPLETION,
        'counter_anchor_ids': anchors, 'counter_unit': UNIT,
        'saved_packet_cases': [['selected', n] for n in range(7, 14)] +
                              [['full_pool', n] for n in (11, 12, 13)],
        'matched_levels': list(range(8, 14)), 'closed_temporal_depths': list(range(9, 13)),
        'r100_domain': '193_ORIGINAL_CURRENT_Q07_SAVED_DESCRIPTORS_ONLY',
        'required_theorems': [], 'inherited_regime_laws': 'NOT_CONSUMED',
        'generation_allowed': False, 'production_authority': 'NONE',
        'master_contract_coverage': 'SAVED_READER_ONLY_NOT_FULL_PRODUCER_DAG',
        'stop_policy': 'EXPLICIT_REQUEST_LIST_NO_ADVANCE_NO_UNOBSERVED_LOOKAHEAD'})


def compile_plan(plan, calibration):
    require(type(calibration) is dict and calibration.get('schema_id') ==
            'IG_SOURCE_BOUND_SAVED_READ_CALIBRATION_V1' and calibration.get('route') == ROUTE,
            'SAVED_READ_CALIBRATION_REQUIRED')
    require(calibration.get('science_sha256') == canonical_sha256({k: v for k, v in calibration.items()
            if k != 'science_sha256'}) and calibration.get('current_package_payload_sha256') ==
            direct.source_snapshot(), 'SAVED_READ_CALIBRATION_SOURCE_CHANGED')
    require(type(plan) is dict and set(plan) == {'schema_id', 'route', 'calibration_sha256',
            'requests', 'stop_policy', 'production_authority', 'generation_allowed'}, 'PLAN_FIELDS')
    require(plan['schema_id'] == SCHEMA and plan['route'] == ROUTE and
            plan['calibration_sha256'] == calibration['science_sha256'] and
            plan['stop_policy'] == calibration['stop_policy'] and
            plan['production_authority'] == 'NONE' and plan['generation_allowed'] is False,
            'PLAN_SCOPE_OR_CALIBRATION')
    requests = plan['requests']
    require(type(requests) is list and requests and
            len({canonical_sha256(r) for r in requests}) == len(requests), 'REQUEST_LIST_REQUIRED')
    for row in requests:
        require(type(row) is dict and set(row) == {'kind', 'domain', 'level'}, 'REQUEST_FIELDS')
        kind, domain, level = row['kind'], row['domain'], row['level']
        require(type(kind) is str and type(domain) is str and type(level) is int, 'REQUEST_TYPES')
        admitted = ((kind == 'saved_packet' and [domain, level] in calibration['saved_packet_cases']) or
            (kind == 'complete_read' and level == 7 and domain in calibration['counter_anchor_ids']) or
            (kind == 'matched_findings' and domain == 'FROZEN_MOTIFS' and level in calibration['matched_levels']) or
            (kind == 'temporal_window' and domain == 'FROZEN_MOTIFS' and level in calibration['closed_temporal_depths']) or
            (kind == 'r100_descriptors' and domain == 'CURRENT_Q07' and level == 100))
        require(admitted, 'UNADMITTED_CAPABILITY_POPULATION_OR_DEPTH')
    return seal({'schema_id': 'IG_COMPILED_SAVED_READ_RUN_PLAN_V1', 'route': ROUTE,
        'plan_sha256': canonical_sha256(plan), 'calibration_sha256': calibration['science_sha256'],
        'requests': requests, 'required_theorems': [], 'theorem_transport': 'NOT_CONSUMED',
        'binding': 'infinity_grid.materialized_discovery.MaterializedDiscoverySession.from_candidate_read_plan',
        'stop_policy': plan['stop_policy'], 'generation_allowed': False, 'production_authority': 'NONE'})


def preflight(root, binding, calibration, evidence_path, master_contract, plan):
    # All scope/source/provenance and byte checks precede state decoding/session ownership.
    require(calibration == make_calibration(binding, evidence_path, master_contract),
            'CALIBRATION_INPUT_SOURCE_OR_SCOPE_CHANGED')
    compiled = compile_plan(plan, calibration)
    qi.preflight(root, binding)
    return compiled, parent_payloads(evidence_path)


def read_request(session, request):
    require(type(request) is dict and set(request) == {'kind', 'domain', 'level'} and
            type(request['kind']) is str and type(request['domain']) is str and
            type(request['level']) is int, 'REQUEST_TYPES_OR_FIELDS')
    context = getattr(session, '_candidate_read_context', None)
    require(context is not None, 'CANDIDATE_READ_SESSION_REQUIRED')
    reader = session._qualified_saved_inputs
    reader.check()
    require(request in context['compiled']['requests'], 'REQUEST_NOT_COMPILED')
    require(context['calibration'] == make_calibration(reader.binding, context['evidence_path'],
            context['master_contract']), 'CANDIDATE_READ_SOURCE_OR_EVIDENCE_CHANGED')
    kind, domain, level = request['kind'], request['domain'], request['level']
    data = context['data']
    if kind == 'saved_packet':
        packet = session.qualified_saved_packet(domain, level)
        rows = data['REUSED_DIAGNOSTIC_CONSUMPTION.json']['dispatches']
        payload = next(r for r in rows if r['domain'] == domain and r['level'] == level)
        require(payload['original_dispatch']['packet_sha256'] == canonical_sha256(packet),
                'SAVED_PACKET_TEST_BINDING')
    elif kind == 'r100_descriptors':
        payload = {'descriptors': session.qualified_r100_descriptors(),
                   'producer_binding': qi.read(reader.root / reader.binding['artifacts']['r100_binding']['file_name'])}
        require(len(payload['descriptors']) == 193, 'R100_193_DOMAIN')
    elif kind == 'complete_read':
        payload = next(e for e in data['COMPLETE_READ_CONSUMER.json']['executions']
                       if e['evidence_payload']['source_construction_digest'] == domain)
        require(payload['evidence_payload']['unit'] == UNIT and len(payload['evidence_payload']['rows']) == 7,
                'COMPLETE_READ_QUANTITY')
    elif kind == 'matched_findings':
        obj = data['MATCHED_TEST_FINDINGS.json']
        payload = {'families': obj['families'][str(level)],
                   'representation': obj['representation'][str(level)]}
    else:
        payload = next(e for e in data['TEMPORAL_WINDOWS.json']['windows']
                       if e['finding']['regime_depth'] == level)
    return seal({'schema_id': 'IG_SAVED_READ_RUN_PLAN_OBSERVATION_V1', 'request': request,
        'payload': deepcopy(payload), 'compiled_plan_sha256': context['compiled']['science_sha256'],
        'parent_capture_id': PARENT_CAPTURE, 'parent_completion_sha256': PARENT_COMPLETION,
        'scientific_reexecution': False, 'generation_allowed': False, 'production_authority': 'NONE'})
