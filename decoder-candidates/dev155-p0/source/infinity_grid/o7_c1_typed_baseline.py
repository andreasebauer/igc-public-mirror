"""Bounded, source-pinned original C1 typed baseline for migration comparison."""
from __future__ import annotations


def _require_integrity(condition, message=None):
    """Acceptance invariant retained under optimized Python."""
    if not condition:
        if message is None:
            raise AssertionError()
        raise AssertionError(message())


import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

from .o7_legacy_library_comparison import ARCHIVE, ARCHIVE_SHA256

ORIGINAL = Path(__file__).parent / 'resources/replay/O7_C1_TYPED_ORIGINAL_MINIMAL.zip'
ORIGINAL_SHA256 = '08df93f3b23f71bf33d14a033b7d1117c4cdb3c4ca10cb39ac85ef1b3a485dfd'


def reconstruct_original_case(output: Path, *, kind: str = 'O7xO6', index: int = 0,
                              profile_calculator=None, canonicalizer=None,
                              parent_transform=None) -> dict:
    """Run the global-state historical helper in a disposable forked process."""
    import multiprocessing
    ctx = multiprocessing.get_context('fork')
    receiver, sender = ctx.Pipe(duplex=False)
    def execute():
        receiver.close()
        try:
            result = _reconstruct_original_case(output, kind=kind, index=index,
                profile_calculator=profile_calculator, canonicalizer=canonicalizer,
                parent_transform=parent_transform)
            sender.send(('OK', result))
        except Exception as exc:
            sender.send(('ERROR', (type(exc).__name__, str(exc))))
        finally:
            sender.close()
    child = ctx.Process(target=execute)
    child.start()
    sender.close()
    try:
        if not receiver.poll(900):
            raise RuntimeError('Original C1 comparison exceeded 900 second bound')
        try:
            status, payload = receiver.recv()
        except EOFError as exc:
            raise RuntimeError('Original C1 comparison worker exited without a result') from exc
        if status != 'OK':
            kind, message = payload
            raise (ValueError if kind == 'ValueError' else RuntimeError)(message)
        return payload
    finally:
        receiver.close()
        child.join(timeout=1)
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)


def _reconstruct_original_case(output: Path, *, kind: str = 'O7xO6', index: int = 0,
                               profile_calculator=None, canonicalizer=None,
                               parent_transform=None) -> dict:
    """Replay one original producer case; no new typed implementation is claimed."""
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('Original typed source checksum mismatch')
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('O7 historical evidence checksum mismatch')
    if kind not in ('O7xO6', 'O7xO7') or type(index) is not int or index < 0:
        raise ValueError('Invalid case selection')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ORIGINAL) as z:
        manifest = json.loads(z.read('MANIFEST_SHA256.json'))
        for path, digest in manifest.items():
            if hashlib.sha256(z.read(path)).hexdigest() != digest:
                raise ValueError('Original typed member checksum mismatch')
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(path))
    with zipfile.ZipFile(ARCHIVE) as z:
        cases = [x for x in z.namelist() if x.endswith('/graduation_compact/07_INPUT_SNAPSHOTS/C1_EXACT_EXTERNAL_CLOSURE.json')]
        if len(cases) != 1:
            raise ValueError('Missing or ambiguous C1 evidence')
        historical = json.loads(z.read(cases[0]))
    case = historical[kind][index]
    prior = os.environ.get('OSCOUT_DATA_ROOT')
    os.environ['OSCOUT_DATA_ROOT'] = str(output.resolve())
    try:
        name = '_ig_c1_original_pinned_engine'
        spec = importlib.util.spec_from_file_location(name, output / '02_CODE/o7_live_engine.py')
        engine = importlib.util.module_from_spec(spec)
        sys.modules[name] = engine
        _require_integrity(spec.loader is not None)
        spec.loader.exec_module(engine)
        engine.O6 = engine.import_o6()
        rows = json.loads((output / '03_RESULTS/O7_SELECTED_COMPONENT_PROFILES.json').read_text())['records']
        def selected(digest, lane):
            matches = [r for r in rows if r['state_digest'] == digest and r['lane'] == lane]
            if not matches:
                raise ValueError('Historical state alias unavailable in declared lane')
            keys = {(tuple(r['parent_exact_colors']), tuple(tuple(e) for e in r['edges'])) for r in matches}
            if len(keys) != 1:
                raise ValueError('Historical state alias is ambiguous within declared context')
            return matches[0]
        parents = engine.load_parent_records()
        if parent_transform is not None:
            parents = parent_transform(parents)
        action = case['action']
        if kind == 'O7xO6':
            left = selected(case['input'], case['lane'])
            ctx0 = engine._profile_row_context(left, parents)
            ctx = engine.LaneContext('C1_76', list(ctx0.parents) + [parents[case['parent']]],
                                     list(ctx0.source_order) + [case['parent']])
            raw = tuple(tuple(x) for x in left['edges'])
        else:
            left, right = selected(case['A'], case['lanes'][0]), selected(case['B'], case['lanes'][1])
            a = engine._profile_row_context(left, parents)
            b = engine._profile_row_context(right, parents)
            ctx = engine.LaneContext('C1_77', list(a.parents) + list(b.parents),
                                     list(a.source_order) + list(b.source_order))
            shifted = []
            for row in right['edges']:
                c, path, t, d, path2, t2 = engine.eparts(row)
                shifted.append(engine.make_edge(c+a.n, path, t, d+a.n, path2, t2))
            raw = tuple(tuple(x) for x in left['edges']) + tuple(shifted)
        raw += (engine.make_edge(action['c'], tuple(action['lmin']), action['a'],
                                 action['d'], tuple(action['rmin']), action['b']),)
        use_v2 = canonicalizer is not None
        if canonicalizer is None:
            canonicalizer = engine.canonicalize_edges
        legacy = engine.canonicalize_edges(ctx, raw)
        canonical = canonicalizer(ctx, raw)
        faithful = True
        if use_v2:
            from .o7_edge_canonicalization import canonicalize_with_witness, verify_relabeling_witness
            new = canonicalize_with_witness(ctx, raw)
            faithful = canonical == new['edges'] and verify_relabeling_witness(ctx, raw, canonical, new['witness'])
        if profile_calculator is None:
            profile_calculator = engine.direct_profile
        observed = engine.profile_summary(profile_calculator(ctx, canonical,
                                      engine.O6.load_templates(), engine.O6.load_rules()[1]))
        historical_checks = {'state': engine.state_digest(legacy) == case['state'],
                             'typed_profile': observed == case['profile'],
                             'predicted_profile': observed == case['predicted_profile']}
        checks = dict(historical_checks)
        status = 'PASS' if all(checks.values()) else 'FAIL'
        if use_v2:
            from .o6_resource_observer import ResourceMerkle
            def skins(edges):
                return sorted((p.exact_color, ResourceMerkle(engine.materialize_owner(p.h6, edges, i)).digest)
                              for i,p in enumerate(ctx.parents))
            if skins(raw) != skins(legacy):
                # The historical comparator changed the resource object. Keep
                # its result visible, and independently check both projections.
                raw_profile = engine.profile_summary(profile_calculator(ctx, raw,
                    engine.O6.load_templates(), engine.O6.load_rules()[1]))
                legacy_profile = engine.profile_summary(profile_calculator(ctx, legacy,
                    engine.O6.load_templates(), engine.O6.load_rules()[1]))
                checks = {'state': historical_checks['state'] and faithful,
                          'raw_profile_preserved': observed == raw_profile,
                          'historical_baseline_reproduced': legacy_profile == case['profile'] == case['predicted_profile']}
                status = 'PASS_WITH_EXPLAINED_HISTORICAL_DIFFERENCE' if all(checks.values()) else 'FAIL'
            else:
                checks['faithful_relabeling'] = faithful
                status = 'PASS' if all(checks.values()) else 'FAIL'
        return {'schema': 'IG_O7_C1_ORIGINAL_TYPED_BASELINE_V1',
                'kind': kind, 'index': index, 'checks': checks,
                'status': status, 'historical_checks': historical_checks,
                'observed_profile': observed,
                'canonicalization_version': 'IG_E7_NESTED_BLISS_V2' if use_v2 else 'LEGACY_HISTORICAL',
                'historical_state_alias': case['state'], 'observed_edge_digest': engine.state_digest(canonical),
                'legacy_edge_digest_equal': engine.state_digest(canonical) == case['state'],
                'original_source_sha256': ORIGINAL_SHA256,
                'historical_archive_sha256': ARCHIVE_SHA256,
                'scope': ('Corrected V2 identity and typed profile, with explicit legacy comparison' if use_v2
                          else 'Original producer reproducibility' if profile_calculator is engine.direct_profile
                          else 'Bounded new action-profile calculation; original input materialization, '
                               'canonicalization, and outer Merkle remain dependencies')}
    finally:
        if prior is None:
            os.environ.pop('OSCOUT_DATA_ROOT', None)
        else:
            os.environ['OSCOUT_DATA_ROOT'] = prior
