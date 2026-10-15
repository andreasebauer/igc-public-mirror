"""R100 mapping, resource-tree proof conformance, and V2 read-scope witness."""
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import resource_tree as rt
from infinity_grid import exact_carrier_unblinding as ec
from infinity_grid import materialized_counter_reference as mr
from infinity_grid import qualified_discovery_inputs as qi
from infinity_grid import r100_reuse
from infinity_grid import regime_scanner as rs
from infinity_grid import semantic_sentinel as ss, theorem_registry as tr
from infinity_grid.canon import canonical_sha256, write_json_atomic


def digest_counter(counter):
    return canonical_sha256(sorted((repr(k), v) for k, v in counter.items()))


def main():
    started = time.monotonic()
    paths = json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params = json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role, sha in params['input_hashes'].items():
        assert hashlib.sha256(Path(paths[role]).read_bytes()).hexdigest() == sha
    binding = json.loads(Path(paths['saved_binding']).read_text())
    root = Path('saved_inputs')
    root.mkdir()
    with zipfile.ZipFile(paths['saved_inputs']) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        z.extractall(root)
    # Full origin/contract/byte admission before the R100 reconstruction call.
    qi.preflight(root, binding)
    producer = json.loads((root/'R100_BINDING.json').read_text())
    admission = r100_reuse.preflight(root, producer)
    original = r100_reuse.load(admission)
    mapper = mr.SavedScannerActionMapper()
    mapped = [mapper.map(s) for s in original]
    assert len(mapped) == 193 and mapper.checked == 14465
    manifest = json.loads((root/'CERTIFIED_CARRIER_SET_V1.json').read_text())
    assert [r100_reuse.public_descriptor(s) for s in original] == [r['public_state'] for r in manifest['carriers']]
    # Mapping never edits original states, root order, witnesses, producer metadata.
    after_producer = json.loads((root/'R100_BINDING.json').read_text())
    assert after_producer == producer
    rows = []
    for raw, exact in zip(original, mapped):
        assert raw.skin == exact.skin and raw.total_caps == exact.total_caps
        assert raw.owner_caps == exact.owner_caps and raw.typed_edges == exact.typed_edges
        assert raw.top_edges_full == exact.top_edges_full
        rows.append({'source_construction_digest': raw.construction_digest,
                     'resource_skin': exact.skin, 'total_caps': list(exact.total_caps)})
    write_json_atomic(Path('R100_MAPPING.json'), {'status': 'PASS', 'root_count': len(mapped),
        'mapped_nodes': mapper.checked, 'overlay_site_rows': mapper.overlay_sites,
        'ordered_public_descriptors_match': True, 'projection_and_capacities_equal': True,
        'scanner_witnesses_preserved_as_provenance': True, 'original_producer_unchanged': True,
        'rows': rows, 'scope': 'Saved 193-root R100 cohort only; no exact-action witness rebind'})
    # Reuse one mapped native O7 leaf as source material for new cross-fibre
    # sentinels. No native discovery, selectors or previous Q03 test are rerun.
    base = next(s for s in mapper._memo.values() if isinstance(s, ec.ExactO7State))
    engine = base.engine
    h = base.owner_h6s[0]
    byp = engine.site_orbits_current(h, None)[3]
    live = sorted((tuple(p), t) for p, (_, f, _, _) in byp.items() for t in range(7) if f[t] > 0)
    assert live
    available_types = {t for _, t in live}
    # Keep the first two sites per type, retaining native capacity values.
    keep = {t: {p for p, tt in live if tt == t} for t in available_types}
    keep = {t: set(sorted(ps)[:2]) for t, ps in keep.items()}
    compact = h
    for path, (p, f, _, _) in byp.items():
        compact = ec._replace_nested(compact, tuple(path), (p, tuple(int(f[t]) if tuple(path) in keep.get(t, set()) else 0 for t in range(7))))
    parent = SimpleNamespace(h6=compact)
    ctx = SimpleNamespace(parents=(parent, parent), source_order=('same', 'same'))
    child = ec.ExactO7State(engine, ctx, (), 'RESOURCE_FIBRE_NATIVE_SENTINEL')
    t0 = next(t for t in range(7) if child.total_caps[t] > 0)
    def lift(edges):
        return ec.ExactLiftState(engine, 8, (child, child, child, child),
            tuple((u, v, t0, t0, 'scanner-witness-left', 'scanner-witness-right') for u, v in edges),
            'NATIVE_FIBRE_SENTINEL', 'TOPOLOGY_READ')
    path_state = lift(((0, 1), (1, 2), (2, 3)))
    star_state = lift(((0, 1), (0, 2), (0, 3)))
    assert path_state.skin == star_state.skin
    tree = rt.to_state_tree(path_state)
    assert tree == rt.to_state_tree(star_state)
    assert rt.native_digest(tree) == path_state.skin
    _ports, raw_pairs = engine.O6.load_rules()
    pairs = tuple(sorted(tuple(map(int, p)) for p in raw_pairs))
    assert {(b, a) for a, b in pairs} == set(pairs) and len(set(pairs)) == len(pairs)
    resource_counter = rt.pushed(rt.profile(tree, pairs))
    oracle = ec.ExactProjectedActionOracle(pairs)
    assert resource_counter == oracle.current_level_profile(path_state) == oracle.current_level_profile(star_state)
    assert resource_counter == mr.profile_reference(path_state, pairs) == mr.profile_reference(star_state, pairs)
    # A nested occurrence permutation is erased by the lossless normal form.
    perm_tree = (tree[0], tuple(reversed(tree[1])))
    assert rt.pushed(rt.profile(perm_tree, pairs)) == resource_counter
    # Same R, different H is a direct counterexample to extending resource-only
    # factorisation to topology/service diagnostics without an enlarged quotient.
    d_path = rs._action_aggregate_v2(path_state, list(pairs))
    d_star = rs._action_aggregate_v2(star_state, list(pairs))
    differences = sorted(k for k in set(d_path)|set(d_star) if d_path.get(k) != d_star.get(k))
    assert differences and d_path['service_signature_sha256'] != d_star['service_signature_sha256']
    write_json_atomic(Path('RESOURCE_FIBRE_AND_V2_SCOPE.json'), {'status': 'PASS',
        'same_lossless_resource_tree': True, 'same_R8_skin': path_state.skin,
        'resource_profile_equal': True, 'resource_counter_sha256': digest_counter(resource_counter),
        'resource_counter_keys': len(resource_counter), 'resource_counter_copies': sum(resource_counter.values()),
        'same_R_different_V2_service_witness': True, 'diagnostic_difference_fields': differences,
        'path_diagnostic': d_path, 'star_diagnostic': d_star,
        'scope': 'Native engineered carriers; a V2 resource-only transport counterexample, not a historical authority mismatch',
        'path_top_pairs': path_state.top_pairs, 'star_top_pairs': star_state.top_pairs,
        'scientific_consequence': 'Resource Counter factorisation does not authorize V2 topology/service transport'})
    source_pins = {name: hashlib.sha256(inspect.getsource(getattr(engine.O6, name)).encode()).hexdigest()
                   for name in ('_digest_leaf', '_digest_container', '_build_merkle', '_successor_merkle_sig')}
    # Finite collision check for the saved resource closure, without expanding
    # depth-100 trees. Payloads retain occurrence multiplicity and native tags.
    codec = {}
    def record(signature, raw):
        assert hashlib.sha256(raw).hexdigest() == signature
        assert signature not in codec or codec[signature] == raw
        codec[signature] = raw
    seen_nodes = set()
    def o6_node(node):
        if id(node) in seen_nodes:
            return
        seen_nodes.add(id(node))
        if node.site is not None:
            p, f = node.site
            raw = b'SITE|' + json.dumps([list(p), list(f)], separators=(',', ':')).encode()
        else:
            for c in node.children:
                o6_node(c)
            raw = (node.tag + '|' + '|'.join(sorted(c.base_sig for c in node.children))).encode()
        record(node.base_sig, raw)
    for state in mapper._memo.values():
        if isinstance(state, ec.ExactO7State):
            for native_root, _byp in state.owner_orbit_data:
                o6_node(native_root)
            children = list(state.owner_roots)
        else:
            children = [c.skin for c in state.children]
        record(state.skin, ('R' + str(state.level) + '|' + '|'.join(sorted(children))).encode())
    argument = Path(__file__).with_name('RESOURCE_COUNTER_PROOF.txt').read_bytes()
    assert b'Human structural induction argument' in argument and b'V2 exclusion' in argument
    write_json_atomic(Path('RESOURCE_PROOF_BINDING.json'), {'status': 'PASS',
        'normal_form': 'LOSSLESS_UNORDERED_TYPED_FREE_RESOURCE_TREE',
        'multiplicity': 'AVAILABLE_SITE_TYPE_OCCURRENCE',
        'native_codec_source_sha256': source_pins,
        'saved_resource_codec_nodes': len(codec), 'saved_resource_codec_collisions': 0,
        'human_argument_sha256': hashlib.sha256(argument).hexdigest(),
        'human_structural_induction_argument_captured': True,
        'universal_native_correctness_machine_checked': False,
        'native_conformance_scope': 'Two engineered O8 carriers over a saved mapped native compact O6 leaf',
        'hash_fibre_equality_requires_collision_free_encoding_on_admitted_domain': True,
        'native_counter_claim_is_distinct_from_V2_diagnostics': True})
    # The stale authority gates remain live; do not modify or call transport.
    stale = tr.verify_all_theorems(raise_on_stale=False)
    assert len(stale['stale_theorems']) == 4
    for theorem in stale['stale_theorems']:
        try:
            tr.require_current_theorems((theorem,))
        except tr.StaleTheoremError:
            pass
        else:
            raise AssertionError('stale theorem admitted')
    assert ss.verify_native_semantics(raise_on_change=False)['status'] == 'REOPEN_REQUIRED'
    qi.unchanged(root, binding)
    write_json_atomic(Path('QUALIFICATION_RESULT.json'), {'status': 'PASS',
        'scientific_outcome': 'R100_MAPPING_AND_RESOURCE_V2_SCOPE_SEPARATION_QUALIFIED_NO_REBIND',
        'r100_roots': 193, 'r100_mapped_nodes': mapper.checked,
        'resource_profile_equal': True, 'V2_resource_only_transport_counterexample': True,
        'stale_theorems': 4, 'production_authority': 'NONE', 'generation_allowed': False,
        'package2_status': 'OPEN_NATIVE_REFINEMENT_AND_REVISED_TRANSPORT_ROUTE_REQUIRED',
        'wall_seconds': time.monotonic() - started})


if __name__ == '__main__':
    main()
