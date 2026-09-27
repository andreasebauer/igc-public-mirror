from __future__ import annotations

"""P7 NODE_IN mapped evidence and bounded fresh mature-composition replay."""

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256
from .replay_dag_runner import RESULT_SCHEMA
from .replay_node_in_audit import audit, INPUT_HASHES
from .replay_reference_data import RECORD_SCHEMA, seal_reference_record

HANDLER_REF = 'infinity_grid.replay_node_in_executor:execute_node_in'
NODE_IDS = tuple(f'IG/NODE_IN/{lane}/ASSERTION_MAPPED_{name}' for lane, name in
                 (('S', 'STRUCTURE'), ('R', 'RELATIONS'), ('C', 'INTERFACES'), ('F', 'FALSIFICATIONS')))
NODE_EXECUTOR_BINDINGS = [{'node_id': node_id, 'handler_ref': HANDLER_REF} for node_id in NODE_IDS]


def _record(record_id: str, record_type: str, payload: Mapping[str, Any], sources: list[dict[str, str]],
            *, dependencies: tuple[str, ...] = (), science: str = 'NONE', epistemic: str = 'REPLAYED') -> dict[str, Any]:
    return seal_reference_record({
        'schema_id': RECORD_SCHEMA, 'record_id': record_id, 'record_type': record_type,
        'layer': 'NODE_IN', 'payload': deepcopy(dict(payload)),
        'provenance': {'classification': 'FINITE_COMPUTATIONAL_OBSERVATION', 'status': 'PINNED',
                       'source_hashes': sources, 'explanation': 'P7 mapped evidence or bounded fresh library calculation'},
        'scope': {'layer': 'NODE_IN', 'execution_class': 'FRESH_RECOMPUTE' if science == 'EXECUTED' else 'HISTORICAL_RESULT_ONLY'},
        'nonclaims': ['BOUNDED_REGRESSION_DOES_NOT_ESTABLISH_INHERITED_THEOREMS'],
        'dependencies': list(dependencies), 'epistemic_status': epistemic,
        'science_execution': science, 'authority_effect': 'NONE',
    })


def execute_node_in(*, node: Mapping[str, Any], catalogue: Mapping[str, Any], repository_root: Path,
                    attempt: int, inject_stop: bool = False) -> dict[str, Any]:
    node_id = node['canonical_id']
    if node_id not in NODE_IDS or len(node.get('assertion_mapping_ids', [])) != 1:
        raise ValueError('NODE_IN exact binding required')
    lane = node['series']
    mapping_id = node['assertion_mapping_ids'][0]
    mapping = next((m for m in catalogue['historical_assertion_mappings'] if m['assertion_id'] == mapping_id), None)
    if mapping is None or mapping['canonical_obligation_ids'] != [node_id] or mapping['execution_class'] != node['effective_execution_class']:
        raise ValueError('NODE_IN mapped assertion mismatch')
    authorizations = [a for a in catalogue['historical_audit_authorizations']
                      if a['authorization_id'] in node.get('audit_authorization_ids', [])]
    if len(authorizations) != 1 or authorizations[0]['decision'] != 'CONTINUE' or node_id not in authorizations[0]['canonical_obligation_ids']:
        raise ValueError('NODE_IN audit authorization mismatch')
    sources = list(mapping['source_hashes'])
    root = Path(repository_root)
    for row in sources + list(authorizations[0]['audit_provenance']) + list(authorizations[0]['audit_evidence']):
        path = root / row['ref']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'NODE_IN historical source mismatch: {row["ref"]}')
    is_fresh = lane == 'R'
    if is_fresh != (node['effective_execution_class'] == 'FRESH_RECOMPUTE') or is_fresh != node['counts_toward_empty_root_science_replay']:
        raise ValueError('NODE_IN execution class or counting changed')
    fresh = None
    if is_fresh:
        fresh = audit(root / 'infinity_grid/resources/replay')
        if fresh['fresh']['regression_exact_cases'] != 43008 or fresh['fresh']['counter_vs_exact_mismatches'] or fresh['fresh']['closure_failures']:
            raise ValueError('NODE_IN fresh science disagrees with frozen acceptance bounds')
        sources += [{'ref': 'infinity_grid/resources/replay/' + name, 'sha256': digest}
                    for name, digest in INPUT_HASHES.items()]
        source_file = root / 'infinity_grid/replay_node_in_audit.py'
        sources.append({'ref': 'infinity_grid/replay_node_in_audit.py',
                        'sha256': hashlib.sha256(source_file.read_bytes()).hexdigest()})
    contract = {'mode': 'CANONICAL_JSON', 'cardinality_semantics': 'ORDERED', 'compression': 'EXACT',
                'observer': node['observer'], 'certificate_sha256': None}
    historical_identity = canonical_sha256({'assertion_id': mapping_id, 'expected_outcome': mapping['expected_outcome']})
    replay_identity = historical_identity
    if is_fresh:
        replay_identity = canonical_sha256({'assertion_id': mapping_id,
            'expected_outcome': 'ZERO_MISMATCH' if fresh['matches_historical'] else 'DISCREPANCY'})
    if inject_stop:
        replay_identity = canonical_sha256({'fault': node_id})
    historical = _record(f'IGRD/NODE_IN/EVIDENCE/{lane}/HISTORICAL', 'SRCF_EVIDENCE', {
        'series': lane, 'obligation_id': node_id, 'result_identity': historical_identity,
        'outcome': 'REPRODUCED', 'equality_contract': contract, 'evidence_mode': 'HISTORICAL_RESULT_ONLY',
    }, sources, epistemic='HISTORICAL')
    replay = _record(f'IGRD/NODE_IN/EVIDENCE/{lane}/REPLAY' + ('/INJECTED' if inject_stop else ''),
                     'SRCF_EVIDENCE', {
        'series': lane, 'obligation_id': node_id, 'result_identity': replay_identity,
        'outcome': 'DISCREPANCY' if replay_identity != historical_identity else 'REPRODUCED',
        'equality_contract': contract, 'evidence_mode': 'FRESH_RECOMPUTE' if is_fresh else 'HISTORICAL_RESULT_ONLY',
    }, sources, science='EXECUTED' if is_fresh else 'NONE')
    comparison = _record(f'IGRD/NODE_IN/COMPARISON/{lane}' + ('/INJECTED' if inject_stop else ''),
                         'COMPARISON', {
        'obligation_id': node_id, 'historical_record_ids': [historical['record_id']],
        'replay_record_ids': [replay['record_id']], 'equality_contract': contract,
        'outcome': 'REPRODUCED' if replay_identity == historical_identity else 'MISMATCH',
        'qualification_ids': list(node.get('known_qualification_ids', [])),
    }, sources, dependencies=(historical['record_id'], replay['record_id']))
    return {'schema_id': RESULT_SCHEMA, 'node_id': node_id, 'attempt': attempt,
            'comparison_outcome': 'RESULT_MISMATCH' if replay_identity != historical_identity else
                                  'REPRODUCED_WITH_DECLARED_EVIDENCE_MODE' if is_fresh else 'EXACT_HISTORICAL_REPLAY_AUTHORIZED',
            'result_sha256': comparison['record_sha256'], 'evidence_sha256': replay['record_sha256'],
            'execution_class': node['effective_execution_class'],
            'counts_toward_empty_root_science_replay': is_fresh,
            'fresh_metrics': fresh['fresh'] if is_fresh else None,
            'reference_records': [historical, replay, comparison]}


def seal_frontier(*, runner_state: Mapping[str, Any], manifest: Mapping[str, Any],
                  next_node_id: str, waiting_for_audit: bool) -> dict[str, Any]:
    return _record('IGRD/NODE_IN/FRONTIER/' + ('AUDIT_STOP' if waiting_for_audit else 'AUTOMATIC_PASS'),
                   'RESUME_FRONTIER', {
        'root_run_id': runner_state['root_run_id'], 'manifest_dag_sha256': manifest['dag_sha256'],
        'runner_state_sha256': runner_state['state_sha256'],
        'completed_node_ids': list(runner_state['completed_node_ids']),
        'checkpoint_sha256_by_node': dict(runner_state['accepted_checkpoint_sha256_by_node']),
        'next_node_id': next_node_id,
        'frontier_status': 'WAITING_FOR_EXTERNAL_AUDIT' if waiting_for_audit else 'READY',
    }, [{'ref': 'compiled_manifest.dag_sha256', 'sha256': manifest['dag_sha256']}])
