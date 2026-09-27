"""Compare new igraph metrics to original post-O7 per-row CSV evidence."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from .o7_legacy_library_comparison import ARCHIVE, ARCHIVE_SHA256, SNAPSHOT
from .regime_scanner import _graph_basic

ORIGINAL = Path(__file__).parent / 'resources/replay/O7_POST_STRUCTURAL_ORIGINAL_v02.zip'
ORIGINAL_SHA256 = 'da76005c93d6333a63071d3e3ad34c660a7896074530510d18c3e9a217a6df2e'
CSV = '02_EVIDENCE/O7_OWNER_GRAPH_STRUCTURAL_METRICS.csv'
AUDIT = '01_RESULTS/POST_O7_STRUCTURAL_AUDIT_RECONCILED_RESULT.json'


def _member(z, suffix):
    names = [name for name in z.namelist() if name.endswith('/' + suffix)]
    if len(names) != 1:
        raise ValueError('Original post-O7 member missing or ambiguous: ' + suffix)
    return z.read(names[0])


def compare_structural_audit(original: Path = ORIGINAL, archive: Path = ARCHIVE) -> dict:
    original, archive = Path(original), Path(archive)
    if hashlib.sha256(original.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('Original post-O7 audit archive checksum mismatch')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('O7 source archive checksum mismatch')
    with zipfile.ZipFile(original) as z:
        csv_raw = _member(z, CSV)
        old = list(csv.DictReader(io.StringIO(csv_raw.decode('utf-8'))))
        audit = json.loads(_member(z, AUDIT))
    with zipfile.ZipFile(archive) as z:
        raw = _member(z, SNAPSHOT)
        survivors = json.loads(raw)['records']
    if len(old) != len(survivors) or len(old) != 205:
        raise ValueError('Historical O7 row population drift')
    results, metrics = [], []
    for index, (row, prior) in enumerate(zip(survivors, old)):
        n = row['accounting']['K6']
        edges = [(e[0], e[7]) for e in row['edges']]
        m = _graph_basic(n, edges, backend='igraph')
        metrics.append(m)
        if not m['connected']:
            raise ValueError('O7 historical survivor graph disconnected')
        # The original script padded each local shell to the graph's global
        # diameter. The scanner's generic helper ends at each owner's eccentricity.
        padded_shells = tuple(sorted(tuple(s) + (0,) * (m['diameter'] + 1 - len(s))
                                     for s in m['shells']))
        kappas = tuple(sorted((1 - degree / 2 for degree in m['degree']), reverse=True))
        values = {
            'lane': row['lane'], 'component_index': str(row['component_index']),
            'state_digest': row['state_digest'], 'R7_skin_sha256': row['R7_skin_sha256'],
            'profile_digest': row['profile']['digest'], 'K6': str(n), 'm7': str(len(edges)),
            'beta7': str(m['beta']), 'diameter': str(m['diameter']), 'radius': str(m['radius']),
            'avg_distance': f"{m['distance_sum'] / (n * (n-1) / 2):.8f}",
            'degree_sequence': repr(tuple(m['degree'])),
            'shell_profile_multiset': repr(padded_shells),
            'triangle_count': str(m['triangles']), 'articulation_count': str(m['articulations']),
            'parallel_multiplicities': repr(tuple(m['parallel_multiplicities'])),
            'kappa_multiset': repr(kappas), 'kappa_sum': str(sum(kappas)),
        }
        differences = {key: {'historical': prior[key], 'library': value}
                       for key, value in values.items() if prior[key] != value}
        results.append({'row_index': index, 'state_digest': row['state_digest'],
                        'status': 'MATCH' if not differences else 'MISMATCH', 'differences': differences})
    findings = audit['pregeometry_audit']['findings']
    s2, s3, s5 = (findings[key] for key in ('S2_owner_graph_separation',
                                            'S3_neighborhood_growth', 'S5_cycle_and_local_Euler_defect'))
    dist = lambda field: dict(sorted(Counter(str(m[field]) for m in metrics).items()))
    local = {tuple(s) + (0,) * (m['diameter'] + 1 - len(s)) for m in metrics for s in m['shells']}
    global_shell = {tuple(sorted(tuple(s) + (0,) * (m['diameter'] + 1 - len(s))
                                 for s in m['shells'])) for m in metrics}
    aggregates = {'diameter_distribution': dist('diameter') == s2['diameter_distribution'],
                  'radius_distribution': dist('radius') == s2['radius_distribution'],
                  'beta7_distribution': dist('beta') == s5['beta7_distribution'],
                  'global_shell_signatures': len(global_shell) == s3['distinct_global_shell_signatures'],
                  'local_shell_profiles': len(local) == s3['distinct_local_shell_profiles']}
    counts = dict(sorted(Counter(x['status'] for x in results).items()))
    return {'schema_id': 'IG_O7_HISTORICAL_STRUCTURAL_LIB_COMPARISON_V1',
            'status': 'PASS' if counts == {'MATCH': 205} and all(aggregates.values()) else 'FAIL',
            'historical_audit_archive_sha256': ORIGINAL_SHA256,
            'historical_csv_sha256': hashlib.sha256(csv_raw).hexdigest(),
            'o7_survivor_archive_sha256': ARCHIVE_SHA256,
            'o7_survivor_snapshot_sha256': hashlib.sha256(raw).hexdigest(),
            'backend': 'igraph 1.0.0', 'row_counts': counts,
            'aggregate_checks': aggregates,
            'observed_local_shell_profiles': len(local),
            'observed_global_shell_signatures': len(global_shell),
            'scope': 'Original post-O7 structural CSV and aggregate graph metrics',
            'nonclaims': ['NO_TYPED_RESOURCE_ACTION_RECOMPUTATION', 'NO_O4_O7_FRONTIER_PROMOTION',
                          'NO_SPACETIME_OR_PHYSICAL_GEOMETRY'], 'rows': results}
