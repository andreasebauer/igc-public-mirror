from __future__ import annotations

"""Pinned Scout historical claims with their observed failures and scope."""

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256
from .replay_dag_runner import RESULT_SCHEMA
from .replay_reference_data import RECORD_SCHEMA, seal_reference_record

HANDLER_REF = 'infinity_grid.replay_scout_historical_executor:execute_scout_historical'
NODE_IDS = tuple(f'IG/SCOUT_HISTORICAL/{lane}/ASSERTION_MAPPED_{name}' for lane, name in
                 (('S', 'STRUCTURE'), ('R', 'RELATIONS'), ('C', 'INTERFACES'), ('F', 'FALSIFICATIONS')))
NODE_EXECUTOR_BINDINGS = [{'node_id': n, 'handler_ref': HANDLER_REF} for n in NODE_IDS]
V15 = 'p1c_historical_sources/SCOUT2B_V1_5_K128_AUDIT_REPORT.txt'
V17 = 'p1c_historical_sources/SCOUT2B_V1_7_K128_AUDIT_REPORT.txt'
V30 = 'p1c_historical_sources/SCOUT3_V1_0_R64_AUDIT_REPORT.txt'
CLAIMS = {
    'S': {'expected': 'ONE_STEP_TYPES_PASS', 'anchors': {V15: (
        'V1.5 ONE-STEP TRANSITION-REFINED TYPE RESULT: PASS',
        'C1 IS NOT A FIXED POINT', 'exact selected carriers: 941', 'V1.5 C1 classes: 275')}},
    'R': {'expected': 'PRESERVE_PASS_AND_NONASSOCIATIVITY', 'anchors': {
        V15: ('C1 IS NOT A FIXED POINT',),
        V17: ('Q_COMPOSABLE_MACRO_ENTITY_EARNED_V1_7', 'NOT associative')}},
    'C': {'expected': 'Q_COMPOSABLE_MACRO_ENTITY_EARNED_V1_7', 'anchors': {
        V17: ('Q_COMPOSABLE_MACRO_ENTITY_EARNED_V1_7', 'provided the finite composition syntax tree is fixed')}},
    'F': {'expected': 'OBSERVER_INTERPRETATION_FAIL_PRESERVED', 'anchors': {
        V15: ('MATURE HIGHER-ORDER RELATIONAL NODE: NOT EARNED', 'C1 IS NOT A FIXED POINT'),
        V30: ('SCOUT3_V1_0_TRAJECTORY_PASS_OBSERVER_INTERPRETATION_FAIL',
              'TRAJECTORY / EXECUTION: PASS')}},
}


def _record(record_id: str, record_type: str, payload: Mapping[str, Any], sources: list[dict[str, str]],
            *, dependencies: tuple[str, ...] = (), epistemic: str = 'REPLAYED') -> dict[str, Any]:
    return seal_reference_record({
        'schema_id': RECORD_SCHEMA, 'record_id': record_id, 'record_type': record_type,
        'layer': 'SCOUT_HISTORICAL', 'payload': deepcopy(dict(payload)),
        'provenance': {'classification': 'FINITE_COMPUTATIONAL_OBSERVATION', 'status': 'PINNED',
                       'source_hashes': sources, 'explanation': 'Pinned Scout report bytes and explicit text anchors; no new Scout run'},
        'scope': {'layer': 'SCOUT_HISTORICAL', 'execution_class': 'HISTORICAL_OR_SOURCE_INTEGRITY'},
        'nonclaims': ['NO_FRESH_SCOUT_SCIENCE', 'HISTORICAL_REPORT_IS_NOT_INDEPENDENT_PROOF'],
        'dependencies': list(dependencies), 'epistemic_status': epistemic,
        'science_execution': 'NONE', 'authority_effect': 'NONE',
    })


def execute_scout_historical(*, node: Mapping[str, Any], catalogue: Mapping[str, Any],
                             repository_root: Path, attempt: int, inject_stop: bool = False) -> dict[str, Any]:
    node_id = node['canonical_id']
    if node_id not in NODE_IDS or node.get('counts_toward_empty_root_science_replay') is not False:
        raise ValueError('SCOUT historical binding changed')
    lane = node['series']
    required_class = 'SOURCE_INTEGRITY_ONLY' if lane == 'C' else 'HISTORICAL_RESULT_ONLY'
    if node.get('effective_execution_class') != required_class or len(node.get('assertion_mapping_ids', [])) != 1:
        raise ValueError('SCOUT execution class changed')
    mapping_id = node['assertion_mapping_ids'][0]
    mappings = [x for x in catalogue['historical_assertion_mappings'] if x['assertion_id'] == mapping_id]
    if len(mappings) != 1:
        raise ValueError('SCOUT assertion unavailable')
    mapping = mappings[0]
    claim = CLAIMS[lane]
    if mapping['canonical_obligation_ids'] != [node_id] or mapping['execution_class'] != required_class or mapping['expected_outcome'] != claim['expected']:
        raise ValueError('SCOUT mapped claim changed')
    auths = [x for x in catalogue['historical_audit_authorizations'] if x['authorization_id'] in node.get('audit_authorization_ids', [])]
    if len(auths) != 1 or auths[0]['decision'] != 'CONTINUE' or node_id not in auths[0]['canonical_obligation_ids'] or auths[0]['independent_replay_recorded'] is not False:
        raise ValueError('SCOUT historical authorization changed')
    root = Path(repository_root)
    sources = list(mapping['source_hashes'])
    for row in sources + list(auths[0]['historical_source_hashes']) + list(auths[0]['audit_evidence']) + list(auths[0]['audit_provenance']):
        path = root / row['ref']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'SCOUT pinned source mismatch: {row["ref"]}')
    if set(claim['anchors']) != {x['ref'] for x in sources}:
        raise ValueError('SCOUT source set changed')
    for path, anchors in claim['anchors'].items():
        text = (root / path).read_text(encoding='utf-8')
        if any(anchor not in text for anchor in anchors):
            raise ValueError(f'SCOUT pinned assertion anchor absent: {path}')
    identity = canonical_sha256({'assertion_id': mapping_id, 'expected': claim['expected'],
                                 'source_hashes': sources, 'anchors': claim['anchors']})
    contract = {'mode': 'CANONICAL_JSON', 'cardinality_semantics': 'ORDERED', 'compression': 'EXACT',
                'observer': node['observer'], 'certificate_sha256': None}
    historical = _record(f'IGRD/SCOUT_HISTORICAL/EVIDENCE/{lane}/HISTORICAL', 'SRCF_EVIDENCE', {
        'series': lane, 'obligation_id': node_id, 'result_identity': identity, 'outcome': 'REPRODUCED',
        'equality_contract': contract, 'evidence_mode': 'HISTORICAL_RESULT_ONLY',
    }, sources, epistemic='HISTORICAL')
    replay_identity = canonical_sha256({'fault': node_id}) if inject_stop else identity
    replay = _record(f'IGRD/SCOUT_HISTORICAL/EVIDENCE/{lane}/REPLAY' + ('/INJECTED' if inject_stop else ''),
        'SRCF_EVIDENCE', {'series': lane, 'obligation_id': node_id, 'result_identity': replay_identity,
        'outcome': 'DISCREPANCY' if inject_stop else 'REPRODUCED', 'equality_contract': contract,
        'evidence_mode': required_class}, sources)
    compare = _record(f'IGRD/SCOUT_HISTORICAL/COMPARISON/{lane}' + ('/INJECTED' if inject_stop else ''),
        'COMPARISON', {'obligation_id': node_id, 'historical_record_ids': [historical['record_id']],
        'replay_record_ids': [replay['record_id']], 'equality_contract': contract,
        'outcome': 'MISMATCH' if inject_stop else 'REPRODUCED',
        'qualification_ids': list(node.get('known_qualification_ids', []))},
        sources, dependencies=(historical['record_id'], replay['record_id']))
    records = [historical, replay, compare]
    if lane == 'F':
        records.append(_record('IGRD/SCOUT_HISTORICAL/NEGATIVE/F', 'NEGATIVE_RESULT', {
            'obligation_id': node_id, 'tested_scope': {'mapped_assertion_id': mapping_id, 'bounded': True},
            'negative_statement': 'V1.5 C1 not a fixed point; Scout3 V1.0 observer interpretation failed',
            'witnesses': [dict(source=ref, anchors=list(anchors)) for ref, anchors in claim['anchors'].items()],
            'does_not_establish': ['UNBOUNDED_NO_GO', 'FRESH_RECOMPUTATION', 'HIGHER_NODE_GRADUATION'],
        }, sources))
    return {'schema_id': RESULT_SCHEMA, 'node_id': node_id, 'attempt': attempt,
            'comparison_outcome': 'RESULT_MISMATCH' if inject_stop else 'EXACT_HISTORICAL_REPLAY_AUTHORIZED',
            'result_sha256': compare['record_sha256'], 'evidence_sha256': replay['record_sha256'],
            'execution_class': required_class, 'counts_toward_empty_root_science_replay': False,
            'assertion_count': 1, 'reference_records': records}


def seal_frontier(*, runner_state: Mapping[str, Any], manifest: Mapping[str, Any],
                  next_node_id: str, waiting_for_audit: bool) -> dict[str, Any]:
    return _record('IGRD/SCOUT_HISTORICAL/FRONTIER/' + ('AUDIT_STOP' if waiting_for_audit else 'AUTOMATIC_PASS'),
        'RESUME_FRONTIER', {'root_run_id': runner_state['root_run_id'],
        'manifest_dag_sha256': manifest['dag_sha256'], 'runner_state_sha256': runner_state['state_sha256'],
        'completed_node_ids': list(runner_state['completed_node_ids']),
        'checkpoint_sha256_by_node': dict(runner_state['accepted_checkpoint_sha256_by_node']),
        'next_node_id': next_node_id,
        'frontier_status': 'WAITING_FOR_EXTERNAL_AUDIT' if waiting_for_audit else 'READY'},
        [{'ref': 'compiled_manifest.dag_sha256', 'sha256': manifest['dag_sha256']}])
