"""Qualified BLISS/SymPy owner-group comparison on pinned O7 typed edges."""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from .graph_library_backend import automorphisms, supported
from .regime_scanner import _automorphisms_legacy
from .o7_legacy_library_comparison import ARCHIVE, ARCHIVE_SHA256, SNAPSHOT


def compare_o7_typed_owner_symmetries(archive: Path = ARCHIVE) -> dict:
    archive = Path(archive)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError('O7 immutable material root checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        names = [name for name in z.namelist() if name.endswith('/' + SNAPSHOT)]
        if len(names) != 1:
            raise ValueError('O7 survivor snapshot path is missing or ambiguous')
        raw = z.read(names[0])
    rows = json.loads(raw)['records']
    if len(rows) != 205 or len({r['state_digest'] for r in rows}) != 163:
        raise ValueError('O7 survivor population drift')
    comparisons = []
    for ordinal, row in enumerate(rows):
        n = row['accounting']['K6']
        palette = {value: i for i, value in enumerate(sorted(set(row['parent_exact_colors'])))}
        colors = [(palette[value], 0, 0, 0, 0, 0, 0) for value in row['parent_exact_colors']]
        # O7's 14-field edge stores endpoint owners at 0/7 and port types at 6/13.
        edges = [(e[0], e[7], e[6], e[13]) for e in row['edges']]
        if not supported(n, colors, edges):
            raise ValueError(f'O7 typed graph outside qualified BLISS domain: row {ordinal}')
        old = _automorphisms_legacy(n, colors, edges)
        new = automorphisms(n, colors, edges, _automorphisms_legacy, backend='igraph_bliss')
        comparisons.append({'row_index': ordinal, 'state_digest': row['state_digest'],
                            'owner_count': n, 'edge_count': len(edges), 'group_order': len(new),
                            'status': 'MATCH' if old == new else 'MISMATCH',
                            'old_group_sha256': hashlib.sha256(repr(old).encode()).hexdigest(),
                            'library_group_sha256': hashlib.sha256(repr(new).encode()).hexdigest()})
    counts = dict(sorted(Counter(x['status'] for x in comparisons).items()))
    return {'schema_id': 'IG_O7_TYPED_OWNER_SYMMETRY_LIB_COMPARISON_V1',
            'status': 'PASS' if counts == {'MATCH': 205} else 'FAIL',
            'source_archive_sha256': archive_hash, 'snapshot_sha256': hashlib.sha256(raw).hexdigest(),
            'comparison': 'igraph 1.0.0 BLISS generators, SymPy 1.14.0 group closure versus retained exact owner permutations',
            'scope': 'typed outer-owner graph with parent exact-color equality',
            'nonclaims': ['NO_HISTORICAL_SYMMETRY_RESULT_FIELD_COMPARED', 'NO_O7_ACTION_OR_GRADUATION_RECOMPUTATION',
                          'NO_O4_O7_FRONTIER_PROMOTION'], 'counts': counts, 'records': comparisons}
