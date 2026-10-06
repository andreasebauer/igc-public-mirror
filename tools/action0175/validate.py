"""Byte-only WP5/WP6 readiness checks. Never import a scientific producer."""
from pathlib import Path
import hashlib, io, json, zipfile

B = Path(__file__).resolve().parent
I = B / 'inputs'
sha = lambda b: hashlib.sha256(b).hexdigest()
load = lambda p: json.loads(p.read_bytes())

def one(z, suffix):
    names = [n for n in z.namelist() if n.endswith(suffix)]
    assert len(names) == 1, (suffix, names)
    return names[0]

def verify_manifest(z, suffix):
    name = one(z, suffix)
    prefix = name.rsplit('/', 1)[0] + '/' if '/' in name else ''
    files = json.loads(z.read(name))['files']
    if isinstance(files, dict):
        files = [dict(v, path=k) for k, v in files.items()]
    for f in files:
        b = z.read(prefix + f['path'])
        assert sha(b) == f['sha256']
        assert len(b) == f.get('size_bytes', f.get('bytes'))
    return len(files)

def export(path):
    with zipfile.ZipFile(path) as z:
        root = json.loads(z.read('ROOT.json'))
        out = {}
        for f in root['content']:
            b = z.read('content/' + f['sha256'] + '.blob')
            assert sha(b) == f['sha256'] and len(b) == f['bytes']
            out[f['name']] = b
        return out

def validate():
    pins = load(B / 'INPUT_PINS.json')
    for n, p in pins.items():
        b = (I / n).read_bytes()
        assert sha(b) == p['sha256'] and len(b) == p['bytes'], n
    catalog = load(I / 'CATALOG_0149.json')
    assert len(catalog['slices']) == 149
    assert sha((I / 'CATALOG_0149.json').read_bytes()) == '84641ecb46ea253c9a8cbbeaa418c6cd14bda93512a763c583ce3ed3af559f7c'
    extra = export(I / 'additional_O7.zip')
    base = export(I / 'base_O7.zip')
    occurrences = json.loads(extra['occurrences'])
    with zipfile.ZipFile(I / 'O7_seed.zip') as z:
        outer = verify_manifest(z, 'PHASE7_FRONTIER_ROOT_MANIFEST.json')
        survivors = z.read(one(z, '/graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json'))
        assert survivors == extra['survivors']
        rows = json.loads(survivors)['records']
        assert len(rows) == 205
        assert [o['record'] for o in occurrences] == rows
        assert len({o['object_id'] for o in occurrences}) == 164
        with zipfile.ZipFile(io.BytesIO(z.read(one(z, 'Infinity_Grid_O7_COMPACT_REPLAY_ROOT_v1_2026-08-29.zip')))) as inner:
            inner_count = verify_manifest(inner, 'O7_COMPACT_REPLAY_ROOT_MANIFEST.json')
            assert json.loads(inner.read(one(inner, '/OSCOUT_O7_SELECTED_O6_PROTOTYPES.json'))) == json.loads(base['selected'])
            assert json.loads(inner.read(one(inner, '/OSCOUT_O7_O6_TOPOLOGY_TWINS.json'))) == json.loads(base['twins'])
    checkpoints = 0
    previous = None
    with zipfile.ZipFile(I / 'G1_saved.zip') as z:
        for depth in range(21, 101):
            prefix = f'run/checkpoints/O{depth:05d}/'
            m = json.loads(z.read(prefix + 'CHECKPOINT_MANIFEST.json'))
            assert m['depth'] == depth
            for f in m['files']:
                b = z.read(prefix + f['path'])
                assert sha(b) == f['sha256'] and len(b) == f['size_bytes']
            if previous is not None:
                assert m['previous_checkpoint_content_sha256'] == previous
            previous = m['checkpoint_content_sha256']
            checkpoints += 1
        d = json.loads(z.read(prefix + 'SOURCE_INPUT.json'))['materialized_discovery_evidence']
        cohort = d['candidate_cohort']
        assert d['source_seed_sha256'] == pins['O7_seed.zip']['sha256']
        assert d['build_meta']['candidates'] == cohort['candidate_count'] == 193
        assert len(cohort['motif_structural_signatures']) == 193
        assert len({x['motif_id'] for x in cohort['motif_structural_signatures']}) == 193
        assert len(d['state_probes']) == len(cohort['selected_motif_ids']) == 24
        assert {x['motif_id'] for x in d['state_probes']} == set(cohort['selected_motif_ids'])
        sealed = json.loads(z.read('PHASE5_SEALED_RESULT.json'))
        assert sealed['completed_depths'] == list(range(21, 101))
        assert sealed['no_automatic_G2_promotion'] and sealed['scientific_result_finalized'] is False
        nested = one(z, 'Infinity_Grid_Algebra_Decoder_v0.28.8_TRUST_REPAIR_COMPLETE_2026-09-02.zip')
        with zipfile.ZipFile(io.BytesIO(z.read(nested))) as source:
            assert source.read(one(source, '/materialized_discovery.py')) == (I / 'materialized_discovery.py').read_bytes()
    g2_manifests = {}
    certificates = []
    for n in ['G2_final.zip', 'G2_R0.zip']:
        with zipfile.ZipFile(I / n) as z:
            g2_manifests[n] = verify_manifest(z, 'MANIFEST_SHA256.json')
            cert = json.loads(z.read(one(z, '/G2_GRADUATION_CERTIFICATE.json')))
            assert cert['science_sha256'] == '025713aab599dacbcffe397721d781e5854bc4216245a21bfd70f92c34a8a209'
            certificates.append(cert)
            historical = json.loads(z.read(one(z, '/G2_RELATION_VALUED_RESERVATION_SPEC_V1.json')))
            assert historical == load(I / 'G2_RELATION_VALUED_RESERVATION_SPEC_V1.json')
            assert z.read(one(z, '/materialized_discovery.py')) == (I / 'materialized_discovery.py').read_bytes()
    assert certificates[0] == certificates[1]
    producer = (I / 'uplift_campaign.py').read_text()
    assert 'def ensure_g1_r100_population' in producer and 'advance_to(99' in producer
    assert '193' in producer and 'rebuild_selected_states' in producer
    result = {'schema':'IG_WP5_WP6_STATIC_READINESS_V1','status':'PASS_IN_DECLARED_BYTE_BINDING_SCOPE',
        'master_release':'MASTER_DATA_V1_0149','scientific_slices':149,
        'admitted_seed_occurrences_matched':205,'distinct_contextual_objects':164,
        'outer_O7_manifest_files_checked':outer,'inner_O7_manifest_files_checked':inner_count,
        'saved_G1_checkpoint_file_hashes_checked':checkpoints * 5,'saved_G1_chain_links_checked':79,
        'saved_G1_R100_motif_signatures':193,'saved_G1_R100_selected_state_probes':24,
        'saved_G1_full_exact_carrier_cohort_available':False,
        'G2_closeout_manifest_files_checked':g2_manifests,
        'G2_historical_certificates_agree':True,'G2_current_relation_spec_matches_saved':True,
        'G1_current_materialized_producer_matches_saved':True,
        'authority_role':'HISTORICAL_EVIDENCE_ONLY_NOT_FRESH_MASTER_POPULATION_ADMISSION',
        'generation_calls':0,'new_admissions':0,'full_l0_to_g8_complete':False,
        'next_scope':'WP6_G1_SAVED_EXACT_COHORT_OR_RECIPE_CLOSURE'}
    return result

if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
