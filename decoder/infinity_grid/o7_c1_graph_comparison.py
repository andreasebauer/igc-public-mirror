"""Reconstruct historical C1 outer graphs and compare qualified igraph results.

This does not reconstruct typed endpoint resources or action profiles.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from .graph_metrics_library import supported
from .o7_legacy_library_comparison import ARCHIVE, ARCHIVE_SHA256
from .regime_scanner import _graph_basic


def _member(z: zipfile.ZipFile, suffix: str) -> bytes:
    matches = [n for n in z.namelist() if n.endswith('/' + suffix)]
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous historical member: ' + suffix)
    return z.read(matches[0])


def compare_c1_outer_graphs(archive: Path = ARCHIVE) -> dict:
    archive = Path(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('O7 material root checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        source = _member(z, 'graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json')
        evidence = _member(z, 'graduation_compact/07_INPUT_SNAPSHOTS/C1_EXACT_EXTERNAL_CLOSURE.json')
    rows = json.loads(source)['records']
    historical = json.loads(evidence)
    if len(rows) != 205 or len(historical['O7xO6']) != 128 or len(historical['O7xO7']) != 64:
        raise ValueError('Historical C1 population drift')
    by_state = {}
    for row in rows:
        digest = row['state_digest']
        if digest in by_state and (row['edges'] != by_state[digest]['edges'] or
                                   row['accounting']['K6'] != by_state[digest]['accounting']['K6']):
            raise ValueError('Ambiguous O7 state digest: ' + digest)
        by_state[digest] = row
    results = []
    for kind in ('O7xO6', 'O7xO7'):
        for case in historical[kind]:
            if kind == 'O7xO6':
                left = by_state[case['input']]
                n_left = left['accounting']['K6']
                n = n_left + 1
                edges = [(e[0], e[7]) for e in left['edges']]
            else:
                left, right = by_state[case['A']], by_state[case['B']]
                n_left = left['accounting']['K6']
                n = n_left + right['accounting']['K6']
                edges = [(e[0], e[7]) for e in left['edges']]
                edges += [(e[0] + n_left, e[7] + n_left) for e in right['edges']]
            action = case['action']
            edges.append((action['c'], action['d']))
            if not supported(n, edges):
                results.append({'kind': kind, 'case_index': case['case_index'],
                                'status': 'OUTSIDE_QUALIFIED_DOMAIN', 'owners': n,
                                'edges': len(edges)})
                continue
            graph = _graph_basic(n, edges, backend='igraph')
            old = case['accounting']
            checks = {'owners': n == old['K6'], 'edges': len(edges) == old['m7'],
                      'degree_multiset': graph['degree'] == sorted(old['degrees'], reverse=True),
                      'beta7': graph['beta'] == old['beta7'], 'connected': graph['connected']}
            results.append({'kind': kind, 'case_index': case['case_index'],
                            'status': 'MATCH' if all(checks.values()) else 'MISMATCH',
                            'checks': checks, 'owners': n, 'edges': len(edges)})
    counts = dict(sorted(Counter(r['status'] for r in results).items()))
    return {'schema': 'IG_O7_C1_OUTER_GRAPH_IGRAPH_COMPARISON_V1',
            'status': 'PASS' if counts.get('MISMATCH', 0) == 0 and counts.get('MATCH', 0) > 0 else 'FAIL',
            'archive_sha256': ARCHIVE_SHA256,
            'survivor_snapshot_sha256': hashlib.sha256(source).hexdigest(),
            'c1_evidence_sha256': hashlib.sha256(evidence).hexdigest(),
            'counts': counts, 'cases': results,
            'scope': 'Qualified subset: outer owner topology only, six or fewer owners and edges',
            'nonclaims': ['NO_C1_TYPED_ACTION_RECOMPUTATION', 'NO_C1_PROFILE_RECOMPUTATION',
                          'NO_O7_GRADUATION', 'NO_REPLAY_FRONTIER_ADVANCE']}
