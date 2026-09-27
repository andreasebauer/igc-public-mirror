from __future__ import annotations

"""Exact pinned-result extraction for the historical O1 through O3 catalogue."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256
from .replay_dag_runner import RESULT_SCHEMA
from .replay_reference_data import RECORD_SCHEMA, seal_reference_record

HANDLER_REF = 'infinity_grid.replay_o1_o3_executor:execute_o1_o3'
NODE_IDS = tuple(f'IG/O1_O3/{lane}/ASSERTION_MAPPED_{name}' for lane, name in
                 (('S', 'STRUCTURE'), ('R', 'RELATIONS'), ('C', 'INTERFACES'), ('F', 'FALSIFICATIONS')))
NODE_EXECUTOR_BINDINGS = [{'node_id': n, 'handler_ref': HANDLER_REF} for n in NODE_IDS]
AUDIT = 'p1c_historical_sources/O3_GRADUATION_AUDIT_RESULT.json'
STATEMENT = 'p1c_historical_sources/O3_DISTRIBUTED_CARRIER_GRADUATION_STATEMENT.txt'
SPEC = 'p1c_historical_sources/IG_O123_RECONCILED_FORMAL_SPEC_v1.2_FF2.2.json'
COMPAT = 'p1c_historical_sources/O123_V1_2_FF22_COMPATIBILITY_RESULT.json'


def _record(record_id: str, record_type: str, payload: Mapping[str, Any], sources: list[dict[str, str]],
            *, dependencies: tuple[str, ...] = (), epistemic: str = 'REPLAYED') -> dict[str, Any]:
    return seal_reference_record({
        'schema_id': RECORD_SCHEMA, 'record_id': record_id, 'record_type': record_type,
        'layer': 'O1_O3', 'payload': deepcopy(dict(payload)),
        'provenance': {'classification': 'FINITE_COMPUTATIONAL_OBSERVATION', 'status': 'PINNED',
                       'source_hashes': sources, 'explanation': 'Exact pinned O1–O3 assertion extraction; no new O-chain science'},
        'scope': {'layer': 'O1_O3', 'execution_class': 'HISTORICAL_OR_SOURCE_INTEGRITY'},
        'nonclaims': ['NO_FRESH_O1_O3_RECOMPUTATION', 'PINNED_RESULTS_DO_NOT_PROVE_SCIENTIFIC_CORRECTNESS'],
        'dependencies': list(dependencies), 'epistemic_status': epistemic,
        'science_execution': 'NONE', 'authority_effect': 'NONE',
    })


def _observed(mapping: Mapping[str, Any], root: Path) -> Any:
    locator = mapping['locator']
    source = locator['source_ref']
    selector = locator['selector']
    if mapping['assertion_id'].startswith('IGAM/O1_O3/REGISTRY/'):
        x = json.loads((root / source).read_text())
        idx = int(selector.removeprefix('/tests/'))
        if selector != f'/tests/{idx}' or idx not in (0, 1, 2) or x['tests'][idx]['level'] != ('O1','O2','O3')[idx]:
            raise ValueError('O1–O3 registry locator drift')
        return x['tests'][idx]['execution']
    if mapping['assertion_id'].startswith('IGAM/O1_O3/V026_ORACLE/'):
        x = json.loads((root / source).read_text())
        idx = int(selector.removeprefix('/assertions/'))
        if selector != f'/assertions/{idx}' or idx not in (8, 9, 10):
            raise ValueError('O1–O3 oracle locator drift')
        a = x['assertions'][idx]
        if a['file'] != 'METADATA/O1_O7_PRIMARY_ORACLE.json' or a['pointer'] != f'/levels/{idx-8}/science_sha256':
            raise ValueError('O1–O3 oracle assertion binding drift')
        return json.dumps(a['equals'], ensure_ascii=False, separators=(',', ':'))
    audit = json.loads((root / AUDIT).read_text())
    statement = (root / STATEMENT).read_text(encoding='utf-8')
    if mapping['assertion_id'].startswith('IGAM/O1_O3/R/'):
        if source != AUDIT or any(audit['criteria'][f'G{i}_{name}']['status'] != 'PASS' for i,name in (
            (4,'future_substitution_external_closure'),(5,'exact_scope_of_hiding'),
            (6,'generated_family_component_invariants'),(7,'nontrivial_reduction_necessary_structure'))):
            raise ValueError('O3 graduation G4–G7 evidence changed')
        if 'O3_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_V1_3' not in statement:
            raise ValueError('O3 statement absent')
        return 'PASS'
    if mapping['assertion_id'].startswith('IGAM/O1_O3/C/'):
        if source != SPEC:
            raise ValueError('O123 source locator drift')
        spec = json.loads((root / SPEC).read_text())
        compat = json.loads((root / COMPAT).read_text())
        if (spec['status'] != 'FORMAL_O1_O2_O3_HIERARCHY_RECONCILED_V1_2_FF22_COMPATIBLE'
                or compat['status'] != 'PASS' or compat['checks_passed'] != len(compat['checks'])
                or spec['O3']['status'] != audit['status']):
            raise ValueError('O123 historical compatibility changed')
        return compat['status']
    if mapping['assertion_id'].startswith('IGAM/O1_O3/F/'):
        if source != AUDIT or audit['atomic_status'] != 'ATOMIC_O3_NODE_NOT_EARNED' or 'ATOMIC_O3_NODE_NOT_EARNED' not in statement:
            raise ValueError('O3 atomic nonclaim changed')
        return audit['atomic_status']
    raise ValueError('unregistered O1–O3 assertion')


def execute_o1_o3(*, node: Mapping[str, Any], catalogue: Mapping[str, Any],
                  repository_root: Path, attempt: int, inject_stop: bool = False) -> dict[str, Any]:
    node_id = node['canonical_id']
    if node_id not in NODE_IDS or node.get('counts_toward_empty_root_science_replay') is not False:
        raise ValueError('O1–O3 exact binding changed')
    lane = node['series']
    required_class = 'SOURCE_INTEGRITY_ONLY' if lane == 'R' else 'HISTORICAL_RESULT_ONLY'
    if node['effective_execution_class'] != required_class:
        raise ValueError('O1–O3 execution class drift')
    ids = node.get('assertion_mapping_ids')
    if not isinstance(ids, list) or len(ids) != (6 if lane == 'S' else 1):
        raise ValueError('O1–O3 assertion cardinality changed')
    mapping_by_id = {m['assertion_id']: m for m in catalogue['historical_assertion_mappings']}
    mappings = [mapping_by_id.get(i) for i in ids]
    if any(m is None or m['canonical_obligation_ids'] != [node_id] or
           m['execution_class'] != ('FRESH_RECOMPUTE' if lane == 'S' and '/REGISTRY/' in m['assertion_id']
                                    else required_class) for m in mappings):
        raise ValueError('O1–O3 assertion binding mismatch')
    auths = [a for a in catalogue['historical_audit_authorizations'] if a['authorization_id'] in node.get('audit_authorization_ids', [])]
    if len(auths) != 1 or auths[0]['decision'] != 'CONTINUE' or node_id not in auths[0]['canonical_obligation_ids'] or auths[0]['independent_replay_recorded'] is not False:
        raise ValueError('O1–O3 historical authorization changed')
    root = Path(repository_root)
    sources = []
    for m in mappings:
        for row in m['source_hashes']:
            if row not in sources:sources.append(dict(row))
    for row in sources + list(auths[0]['historical_source_hashes']) + list(auths[0]['audit_evidence']) + list(auths[0]['audit_provenance']):
        p = root / row['ref']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'O1–O3 pinned source changed: {row["ref"]}')
    contract = {'mode': 'CANONICAL_JSON', 'cardinality_semantics': 'ORDERED', 'compression': 'EXACT',
                'observer': node['observer'], 'certificate_sha256': None}
    records = []
    historical_ids=[]; replay_ids=[]
    mismatched=False
    for k,m in enumerate(mappings):
        actual = _observed(m,root)
        mismatch = actual != m['expected_outcome'] or (inject_stop and k == 0)
        mismatched |= mismatch
        ident=canonical_sha256({'assertion_id': m['assertion_id'], 'expected_outcome': m['expected_outcome'], 'locator': m['locator']})
        new_ident=canonical_sha256({'assertion_id': m['assertion_id'], 'observed': actual, 'locator': m['locator']})
        if inject_stop and k == 0:new_ident=canonical_sha256({'fault': node_id})
        short=m['assertion_id'].rsplit('/',1)[-1]
        historical = _record(f'IGRD/O1_O3/EVIDENCE/{lane}/{short}/HISTORICAL', 'SRCF_EVIDENCE', {
            'series': lane, 'obligation_id': node_id, 'result_identity': ident,
            'outcome': 'REPRODUCED', 'equality_contract': contract, 'evidence_mode': 'HISTORICAL_RESULT_ONLY',
        }, sources, epistemic='HISTORICAL')
        replay = _record(f'IGRD/O1_O3/EVIDENCE/{lane}/{short}/REPLAY' + ('/INJECTED' if inject_stop and k == 0 else ''),
            'SRCF_EVIDENCE', {'series': lane, 'obligation_id': node_id, 'result_identity': new_ident,
            'outcome': 'DISCREPANCY' if mismatch else 'REPRODUCED',
            'equality_contract': contract, 'evidence_mode': required_class}, sources)
        records.extend([historical,replay]);historical_ids.append(historical['record_id']);replay_ids.append(replay['record_id'])
    compare = _record(f'IGRD/O1_O3/COMPARISON/{lane}' + ('/INJECTED' if inject_stop else ''),
        'COMPARISON', {'obligation_id': node_id, 'historical_record_ids': historical_ids,
        'replay_record_ids': replay_ids, 'equality_contract': contract,
        'outcome': 'MISMATCH' if mismatched else 'REPRODUCED',
        'qualification_ids': list(node.get('known_qualification_ids', []))},
        sources, dependencies=tuple(historical_ids+replay_ids))
    records.append(compare)
    if lane == 'F':
        records.append(_record('IGRD/O1_O3/NEGATIVE/F', 'NEGATIVE_RESULT', {
            'obligation_id': node_id, 'tested_scope': {'mapped_assertions': ids, 'bounded': True},
            'negative_statement': 'The pinned O3 graduation refuses an atomic O3 node',
            'witnesses': [{'source': AUDIT, 'field': 'atomic_status', 'value': 'ATOMIC_O3_NODE_NOT_EARNED'}],
            'does_not_establish': ['FRESH_O3_RECOMPUTATION', 'UNBOUNDED_ATOMIC_NO_GO'],
        }, sources))
    return {'schema_id': RESULT_SCHEMA, 'node_id': node_id, 'attempt': attempt,
            'comparison_outcome': 'RESULT_MISMATCH' if mismatched else 'EXACT_HISTORICAL_REPLAY_AUTHORIZED',
            'result_sha256': compare['record_sha256'],
            'evidence_sha256': canonical_sha256([r['record_sha256'] for r in records if r['record_type']=='SRCF_EVIDENCE']),
            'execution_class': required_class, 'assertion_count': len(ids),
            'counts_toward_empty_root_science_replay': False, 'reference_records': records}


def seal_frontier(*, runner_state: Mapping[str, Any], manifest: Mapping[str, Any],
                  next_node_id: str, waiting_for_audit: bool) -> dict[str, Any]:
    return _record('IGRD/O1_O3/FRONTIER/' + ('AUDIT_STOP' if waiting_for_audit else 'AUTOMATIC_PASS'),
        'RESUME_FRONTIER', {'root_run_id': runner_state['root_run_id'],
        'manifest_dag_sha256': manifest['dag_sha256'], 'runner_state_sha256': runner_state['state_sha256'],
        'completed_node_ids': list(runner_state['completed_node_ids']),
        'checkpoint_sha256_by_node': dict(runner_state['accepted_checkpoint_sha256_by_node']),
        'next_node_id': next_node_id,
        'frontier_status': 'WAITING_FOR_EXTERNAL_AUDIT' if waiting_for_audit else 'READY'},
        [{'ref': 'compiled_manifest.dag_sha256', 'sha256': manifest['dag_sha256']}])
