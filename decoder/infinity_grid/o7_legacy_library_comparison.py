"""Bounded igraph comparison against immutable O7 survivor accounting.

This is a qualification probe. It does not promote the O4–O7 replay frontier.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from .graph_metrics_library import supported, graph_basic
from .regime_scanner import _graph_basic_legacy

ARCHIVE_SHA256 = 'cb6f48641eb99d374d19d5dc8d5bfada13e12cf1ecb20ed0e66958629ec08f1b'
ARCHIVE = Path(__file__).parent / 'resources/decoder/O7_MATERIAL_ROOT_cb6f48641eb9.zip'
SNAPSHOT = 'graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json'


def compare_o7_survivor_topology(archive: Path = ARCHIVE) -> dict:
    archive = Path(archive)
    actual_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual_hash != ARCHIVE_SHA256:
        raise ValueError('O7 immutable material root checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        matches = [name for name in z.namelist() if name.endswith('/' + SNAPSHOT)]
        if len(matches) != 1:
            raise ValueError('O7 survivor snapshot path is missing or ambiguous')
        raw = z.read(matches[0])
    rows = json.loads(raw)['records']
    # The selected rows have 163 distinct state digests across 205 lane entries.
    # Multiplicity is part of the historical population, not a deduplication error.
    if len(rows) != 205 or len({r['state_digest'] for r in rows}) != 163:
        raise ValueError('O7 survivor population drift')
    results = []
    for row in rows:
        n = row['accounting']['K6']
        # The frozen O7 edge has 7 fields for each endpoint: owner IDs at 0, 7.
        edges = [(e[0], e[7]) for e in row['edges']]
        if not supported(n, edges):
            results.append({'state_digest': row['state_digest'], 'status': 'OUTSIDE_QUALIFIED_IGRAPH_DOMAIN',
                            'owner_count': n, 'edge_count': len(edges)})
            continue
        observed = graph_basic(n, edges, _graph_basic_legacy, backend='igraph')
        old = row['accounting']
        checks = {'owner_count': n == old['K6'], 'edge_count': len(edges) == old['m7'],
                  'degree_multiset': observed['degree'] == sorted(old['degrees'], reverse=True),
                  'cycle_rank': observed['beta'] == old['beta7'], 'connected': observed['connected'] is True}
        results.append({'state_digest': row['state_digest'], 'status': 'MATCH' if all(checks.values()) else 'MISMATCH',
                        'checks': checks, 'observed': {'degree_multiset': observed['degree'],
                        'cycle_rank': observed['beta'], 'connected': observed['connected']},
                        'historical': {'degree_multiset': sorted(old['degrees'], reverse=True), 'cycle_rank': old['beta7']}})
    counts = dict(sorted(Counter(r['status'] for r in results).items()))
    return {'schema_id': 'IG_O7_IMMUTABLE_TOPOLOGY_LIB_COMPARISON_V1',
            'status': 'PASS' if counts == {'MATCH': 205}
                      else 'FAIL',
            'archive_sha256': actual_hash, 'snapshot_sha256': hashlib.sha256(raw).hexdigest(),
            'backend': 'igraph==1.0.0', 'scientific_scope': 'O7 outer owner support graph only',
            'compared_fields': ['K6', 'm7', 'degree_multiset', 'beta7', 'connected'],
            'nonclaims': ['NO_O7_GRADUATION_RECOMPUTATION', 'NO_TYPED_RESOURCE_ACTION_REPLAY',
                          'NO_O4_O7_FRONTIER_PROMOTION'],
            'counts': counts, 'records': results}
