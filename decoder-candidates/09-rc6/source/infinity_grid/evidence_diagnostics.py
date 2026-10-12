"""Observations attached to existing fail-closed errors; never repair evidence.

Live inventories are sequential observations, not atomic snapshots. Observer
identity is not writer attribution. No runtime/scientific qualification follows.
"""
from copy import deepcopy
import os
import time
from .canon import canonical_sha256

SCHEMA = 'IG_COMPLETION_EVIDENCE_DIAGNOSTIC_V1'


def _index(rows):
    if not isinstance(rows, list):
        return None
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('path'), str) or row['path'] in result:
            return None
        result[row['path']] = row
    return result


def attach(exc, *, phase, done, completion_path, observed_roots,
           failed_checks, observation, admitted_binding=None):
    expected = done.get('evidence')
    indexed = _index(expected)
    differences = {}
    for root, rows in sorted(observed_roots.items()):
        actual = _index(rows)
        if indexed is None or actual is None:
            differences[root] = {'classification': 'UNINDEXABLE_INVENTORY'}
        else:
            differences[root] = {
                'missing': [indexed[p] for p in sorted(indexed.keys() - actual.keys())],
                'unexpected': [actual[p] for p in sorted(actual.keys() - indexed.keys())],
                'changed': [{'path': p, 'expected': indexed[p], 'observed': actual[p]}
                            for p in sorted(indexed.keys() & actual.keys()) if indexed[p] != actual[p]],
                'exact_sequence_match': expected == rows,
            }
    obj = {'schema_id': SCHEMA, 'phase': phase, 'completion_path': completion_path,
           'producer_binding': {k: done.get(k) for k in (
               'source_sha256', 'registration_sha256', 'request_sha256', 'completion_sha256',
               'evidence_protocol', 'evidence_root')},
           'admitted_binding': admitted_binding,
           'failed_checks': list(failed_checks), 'expected_inventory': expected,
           'observed_roots': observed_roots, 'differences': differences,
           'observation_kind': observation,
           'observer': {'pid': os.getpid(), 'unix_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()},
           'writer_attribution': 'UNKNOWN', 'root_cause_established': False}
    # Freeze caller-owned lists so subsequent mutation cannot change the receipt.
    obj = deepcopy(obj)
    obj['diagnostic_sha256'] = canonical_sha256(obj)
    exc.evidence_diagnostic = obj
    return exc
