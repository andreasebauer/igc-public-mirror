import copy
import hashlib
import io
import json
import stat
import warnings
import zipfile
from pathlib import Path

import pytest

from infinity_grid.canon import canonical_bytes, canonical_sha256
from infinity_grid.storage_capture_metadata import verify_capture_metadata, CaptureMetadataError
from infinity_grid.storage_index_descriptor import content_ref


class Store:
    def __init__(self):
        self.blobs = {}
        self.reads = []

    def put(self, raw):
        ref = content_ref(raw)
        self.blobs[ref['sha256']] = raw
        return ref

    def read(self, ref, limit):
        self.reads.append(ref['sha256'])
        return self.blobs[ref['sha256']]


def archive(entries):
    out = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, raw in entries:
                z.writestr(name, raw)
    return out.getvalue()


def seal(store, capture):
    capture['job']['registration_sha256'] = canonical_sha256(
        {k: v for k, v in capture['job'].items() if k != 'registration_sha256'})
    capture['capture_id'] = canonical_sha256({k: v for k, v in capture.items() if k != 'capture_id'})
    return store.put(canonical_bytes(capture))


def fixture(tmp_path):
    from infinity_grid.v05_controller_event_loop import _source_ids
    root = tmp_path / 'source'
    files = {'infinity_grid/__init__.py': b'raise RuntimeError("DO NOT EXECUTE")\n',
             'infinity_grid/a/data.txt': b'payload', 'infinity_grid/a.txt': b'ordering',
             'README.txt': b'test-only source tree'}
    for name, raw in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    sid, pid = _source_ids(root)
    s = Store()
    source_ref = s.put(archive(list(files.items())))
    base = Path(__file__).parent / 'fixtures/capture_metadata'
    binding = json.loads((base / 'saved_runtime_binding.json').read_bytes())
    env = json.loads((base / 'saved_environment.json').read_bytes())
    # Retain actual saved observations/declarations. This miniature capture and
    # source tree are synthetic and confer no production provenance.
    binding_ref = s.put((base / 'saved_runtime_binding.json').read_bytes())
    objects = [{'role': 'engine_source', 'sha256': source_ref['sha256'],
                'size_bytes': int(source_ref['size_bytes'])}]
    for a in env['artifacts']:
        objects.append({'role': 'input:' + a['logical_name'], 'sha256': a['sha256'],
                        'size_bytes': int(binding_ref['size_bytes']) if a['logical_name'] == 'sqlite_runtime_binding' else 1})
    capture = {'schema_id': 'IG_DECODER_CAPTURE_V1',
               'workspace': {'schema_id': 'IG_DECODER_WORKSPACE_V1', 'source_sha256': sid, 'package_sha256': pid},
               'job': {'source_sha256': sid}, 'objects': objects, 'environment': env}
    return s, capture, binding, source_ref


def change_binding(s, c, b):
    ref = s.put(canonical_bytes(b))
    next(a for a in c['environment']['artifacts'] if a['logical_name'] == 'sqlite_runtime_binding')['sha256'] = ref['sha256']
    obj = next(o for o in c['objects'] if o['role'] == 'input:sqlite_runtime_binding')
    obj.update(sha256=ref['sha256'], size_bytes=int(ref['size_bytes']))


def change_archive(s, c, raw):
    ref = s.put(raw)
    c['objects'][0].update(sha256=ref['sha256'], size_bytes=int(ref['size_bytes']))


def test_source_ids_match_native_and_partial_runtime_scope(tmp_path):
    s, c, b, src = fixture(tmp_path)
    result = verify_capture_metadata(s, seal(s, c))
    assert result['source_sha256'] == c['workspace']['source_sha256']
    assert result['package_sha256'] == c['workspace']['package_sha256']
    assert result['source_members'] == 4 and len(result['observed_packages_checked']) == 7
    assert len(result['unobserved_requirements']) == 12
    assert len(result['unread_environment_artifacts']) == 3
    assert result['science_authority'] == 'NOT_GRANTED'
    for key in ('live_runtime_verified', 'observation_authenticated', 'environment_closure_verified',
                'release_root_verified', 'production_acceptance'):
        assert result[key] is False
    assert len(s.reads) == 3


def test_capture_seal_tampering(tmp_path):
    s, c, _, _ = fixture(tmp_path)
    seal(s, c)
    c['environment']['python'] = '3.12'
    with pytest.raises(CaptureMetadataError, match='CAPTURE_SEAL'):
        verify_capture_metadata(s, s.put(canonical_bytes(c)))


def test_source_identity_mismatch_even_resealed(tmp_path):
    s, c, _, _ = fixture(tmp_path)
    change_archive(s, c, archive([('infinity_grid/__init__.py', b'changed')]))
    with pytest.raises(CaptureMetadataError, match='CAPTURE_SOURCE_IDENTITY'):
        verify_capture_metadata(s, seal(s, c))


def test_job_and_package_identity_mismatch(tmp_path):
    s, c, _, _ = fixture(tmp_path)
    good = copy.deepcopy(c)
    c['job']['source_sha256'] = '0' * 64
    with pytest.raises(CaptureMetadataError, match='CAPTURE_JOB_BINDING'):
        verify_capture_metadata(s, seal(s, c))
    c = good
    c['workspace']['package_sha256'] = '0' * 64
    with pytest.raises(CaptureMetadataError, match='CAPTURE_SOURCE_IDENTITY'):
        verify_capture_metadata(s, seal(s, c))


def test_unsafe_and_skipped_archive_names(tmp_path):
    s, c, _, _ = fixture(tmp_path)
    for name in ('../escape', '/absolute', 'x\\y', 'a//b', 'a/./b', '__pycache__/x',
                 'infinity_grid/x.pyc', 'C:/file'):
        change_archive(s, c, archive([(name, b'x'), ('infinity_grid/__init__.py', b'')]))
        with pytest.raises(CaptureMetadataError, match='SOURCE_MEMBER_PATH'):
            verify_capture_metadata(s, seal(s, c))


def test_duplicate_archive_and_path_collision(tmp_path):
    s, c, _, _ = fixture(tmp_path)
    for entries, error in [([('infinity_grid/a', b'a'), ('infinity_grid/a', b'b')], 'DUPLICATE'),
                           ([('infinity_grid/a', b'a'), ('infinity_grid/a/b', b'b')], 'COLLISION')]:
        change_archive(s, c, archive(entries))
        with pytest.raises(CaptureMetadataError, match=error):
            verify_capture_metadata(s, seal(s, c))


def test_symlink_archive_refused(tmp_path):
    s, c, _, _ = fixture(tmp_path)
    info = zipfile.ZipInfo('infinity_grid/link')
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    change_archive(s, c, archive([(info, b'/etc/passwd')]))
    with pytest.raises(CaptureMetadataError, match='SOURCE_MEMBER_TYPE'):
        verify_capture_metadata(s, seal(s, c))


def test_archive_budgets_and_invalid_zip(tmp_path):
    s, c, _, _ = fixture(tmp_path)
    ref = seal(s, c)
    for kw in ({'max_expanded_bytes': 1}, {'max_members': 1}, {'max_total_bytes': 1}, {'max_members': True}):
        with pytest.raises(CaptureMetadataError):
            verify_capture_metadata(s, ref, **kw)
    change_archive(s, c, b'not a zip')
    with pytest.raises(CaptureMetadataError, match='SOURCE_ARCHIVE_INVALID'):
        verify_capture_metadata(s, seal(s, c))


def test_lying_provider_and_duplicate_json(tmp_path):
    s, c, _, _ = fixture(tmp_path)
    ref = seal(s, c)
    s.blobs[ref['sha256']] += b' '
    with pytest.raises(CaptureMetadataError, match='PROVIDER_MISMATCH'):
        verify_capture_metadata(s, ref)
    with pytest.raises(ValueError, match='DUPLICATE_JSON_KEY'):
        verify_capture_metadata(s, s.put(b'{"schema_id":1,"schema_id":2}'))


def test_artifact_binding_and_duplicate_roles(tmp_path):
    s, c, _, _ = fixture(tmp_path)
    good = copy.deepcopy(c)
    c['environment']['artifacts'][0]['sha256'] = '0' * 64
    with pytest.raises(CaptureMetadataError, match='CAPTURE_ARTIFACT_BINDING'):
        verify_capture_metadata(s, seal(s, c))
    c = good
    c['objects'].append(c['objects'][0])
    with pytest.raises(CaptureMetadataError, match='CAPTURE_OBJECT_ROW'):
        verify_capture_metadata(s, seal(s, c))


def test_runtime_python_and_authority_claims(tmp_path):
    s, c, b, _ = fixture(tmp_path)
    for key, val in [('python_version', '3.12.14'), ('runtime_qualification', 'GRANTED'),
                     ('scientific_execution', 'ALLOWED')]:
        changed = copy.deepcopy(b)
        changed['observation'][key] = val
        change_binding(s, c, changed)
        with pytest.raises(CaptureMetadataError, match='CAPTURE_RUNTIME_DECLARATION'):
            verify_capture_metadata(s, seal(s, c))


def test_sqlite_version_range_and_forged_verdict(tmp_path):
    s, c, b, _ = fixture(tmp_path)
    for key, val in [('sqlite_version', '3.99.0'), ('sqlite_source_id', 'unreviewed'),
                     ('wal_fix_gate', {'approved_for_wal_fix_gate': True})]:
        changed = copy.deepcopy(b)
        changed['observation'][key] = val
        change_binding(s, c, changed)
        with pytest.raises(CaptureMetadataError, match='CAPTURE_SQLITE_SOURCE_POLICY'):
            verify_capture_metadata(s, seal(s, c))


def test_observed_package_mismatch_and_normalized_duplicate(tmp_path):
    s, c, b, _ = fixture(tmp_path)
    b['observation']['package_versions']['pytest'] = '0.0'
    change_binding(s, c, b)
    with pytest.raises(CaptureMetadataError, match='CAPTURE_PACKAGE_VERSION'):
        verify_capture_metadata(s, seal(s, c))
    req = copy.deepcopy(c['environment']['requirements'][0])
    req['distribution'] = req['distribution'].upper()
    c['environment']['requirements'].append(req)
    with pytest.raises(CaptureMetadataError, match='CAPTURE_REQUIREMENT_DUPLICATE'):
        verify_capture_metadata(s, seal(s, c))


def test_missing_runtime_artifact_and_missing_package(tmp_path):
    s, c, _, _ = fixture(tmp_path)
    good = copy.deepcopy(c)
    c['environment']['artifacts'] = []
    with pytest.raises(CaptureMetadataError, match='CAPTURE_RUNTIME_BINDING_MISSING'):
        verify_capture_metadata(s, seal(s, c))
    c = good
    change_archive(s, c, archive([('README.txt', b'no package')]))
    with pytest.raises(CaptureMetadataError, match='SOURCE_PACKAGE_MISSING'):
        verify_capture_metadata(s, seal(s, c))
