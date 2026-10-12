"""Read-only checks of captured source bytes and declared runtime metadata.

Nothing is extracted, imported, installed or executed from the supplied archive.
Capture self hashes are content identities, not signatures. This API does not
verify a ReleaseRoot, current runtime, all environment artifacts or acceptance.
"""
import hashlib
import io
import re
import stat
import zipfile
from pathlib import PurePosixPath

from .canon import canonical_sha256
from .storage_schema import strict_loads
from .storage_catalog import _ref
from .storage_index_descriptor import content_ref


class CaptureMetadataError(ValueError):
    pass


def _require(ok, label):
    if not ok:
        raise CaptureMetadataError(label)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _source_ids(raw, expanded_limit, member_limit, selected_paths):
    rows = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            _require(0 < len(entries) <= member_limit, 'SOURCE_MEMBER_BUDGET')
            names = set()
            total = 0
            for entry in entries:
                name = entry.filename
                path = PurePosixPath(name)
                _require(name == entry.orig_filename and name == path.as_posix()
                         and not path.is_absolute() and '\\' not in name
                         and not any(p in {'..', '.', '.git', '__pycache__',
                                           '.pytest_cache', 'build', 'dist', '.engineering_tmp'}
                                     for p in path.parts)
                         and path.suffix not in {'.pyc', '.pyo'}
                         and len(path.parts) > 0 and ':' not in path.parts[0]
                         and not entry.is_dir(), 'SOURCE_MEMBER_PATH')
                _require(name not in names, 'SOURCE_DUPLICATE_MEMBER')
                names.add(name)
                mode = entry.external_attr >> 16
                _require(stat.S_IFMT(mode) in {0, stat.S_IFREG}
                         and not entry.flag_bits & 1
                         and entry.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED},
                         'SOURCE_MEMBER_TYPE')
                total += entry.file_size
                _require(total <= expanded_limit, 'SOURCE_EXPANDED_BUDGET')
            _require(any(n.startswith('infinity_grid/') for n in names), 'SOURCE_PACKAGE_MISSING')
            for name in names:
                _require(not any(p.as_posix() in names for p in PurePosixPath(name).parents
                                 if p.as_posix() != '.'), 'SOURCE_PATH_COLLISION')
            # Match sorted pathlib traversal used by both native identity algorithms.
            for entry in sorted(entries, key=lambda e: PurePosixPath(e.filename)):
                h = hashlib.sha256()
                count = 0
                with archive.open(entry) as stream:
                    while chunk := stream.read(65536):
                        count += len(chunk)
                        _require(count <= entry.file_size, 'SOURCE_MEMBER_LENGTH')
                        h.update(chunk)
                _require(count == entry.file_size, 'SOURCE_MEMBER_LENGTH')
                rows.append([entry.filename, h.hexdigest()])
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, OSError, EOFError) as exc:
        raise CaptureMetadataError('SOURCE_ARCHIVE_INVALID') from exc
    package = [[n[len('infinity_grid/'):], h] for n, h in rows if n.startswith('infinity_grid/')]
    return (canonical_sha256({'schema_id': 'IG_DECODER_ENGINEERING_SOURCE_TREE_V1', 'files': rows}),
            canonical_sha256({'schema_id': 'IG_DECODER_EXECUTABLE_SOURCE_TREE_V1', 'files': package}),
            len(rows), total, {n: h for n, h in rows if n in selected_paths})


def verify_capture_metadata(store, capture_ref, *, max_total_bytes=100663296,
                            max_expanded_bytes=134217728, max_members=20000, source_paths=()):
    """Verify a pinned capture, source ZIP and its SQLite observation artifact.

    Unobserved declared package versions are reported explicitly. The saved
    observation is checked for consistency; its truth is not authenticated.
    Other capture fields and artifacts are outside this verifier's scope.
    """
    for value, ceiling in [(max_total_bytes, 100663296),
                           (max_expanded_bytes, 134217728), (max_members, 20000)]:
        _require(type(value) is int and 1 <= value <= ceiling, 'METADATA_LIMIT')
    _require(type(source_paths) is tuple and len(source_paths) <= 64
             and all(type(p) is str and 0 < len(p) <= 256 and '\\' not in p
                     and not PurePosixPath(p).is_absolute() and '..' not in PurePosixPath(p).parts
                     and PurePosixPath(p).as_posix() == p for p in source_paths)
             and len(source_paths) == len(set(source_paths)), 'METADATA_SOURCE_PATHS')
    used = 0

    def read(ref, limit):
        nonlocal used
        _ref(ref)
        size = int(ref['size_bytes'])
        _require(size <= limit and used + size <= max_total_bytes, 'METADATA_BYTE_BUDGET')
        used += size
        raw = store.read(ref, limit)
        _require(type(raw) is bytes and content_ref(raw) == ref, 'METADATA_PROVIDER_MISMATCH')
        return raw

    capture = strict_loads(read(capture_ref, 4194304), max_bytes=4194304)
    _require(type(capture) is dict and capture.get('schema_id') == 'IG_DECODER_CAPTURE_V1',
             'CAPTURE_SCHEMA')
    _require(_sha(capture.get('capture_id')) and capture['capture_id'] == canonical_sha256(
        {k: v for k, v in capture.items() if k != 'capture_id'}), 'CAPTURE_SEAL')
    workspace = capture.get('workspace')
    _require(type(workspace) is dict and set(workspace) == {'schema_id', 'source_sha256', 'package_sha256'}
             and workspace['schema_id'] == 'IG_DECODER_WORKSPACE_V1'
             and _sha(workspace['source_sha256']) and _sha(workspace['package_sha256']), 'CAPTURE_WORKSPACE')
    job = capture.get('job')
    _require(type(job) is dict and job.get('source_sha256') == workspace['source_sha256']
             and job.get('registration_sha256') == canonical_sha256(
                 {k: v for k, v in job.items() if k != 'registration_sha256'}), 'CAPTURE_JOB_BINDING')
    objects = capture.get('objects')
    _require(type(objects) is list and 1 <= len(objects) <= 1024, 'CAPTURE_OBJECTS')
    roles = {}
    for obj in objects:
        _require(type(obj) is dict and type(obj.get('role')) is str
                 and obj['role'] not in roles and _sha(obj.get('sha256'))
                 and type(obj.get('size_bytes')) is int and obj['size_bytes'] >= 0,
                 'CAPTURE_OBJECT_ROW')
        roles[obj['role']] = {'sha256': obj['sha256'], 'size_bytes': str(obj['size_bytes'])}
    _require('engine_source' in roles, 'CAPTURE_SOURCE_MISSING')
    source_ref = roles['engine_source']
    source_id, package_id, count, expanded, selected = _source_ids(
        read(source_ref, 67108864), max_expanded_bytes, max_members, source_paths)
    _require(source_id == workspace['source_sha256'] and package_id == workspace['package_sha256'],
             'CAPTURE_SOURCE_IDENTITY')
    env = capture.get('environment')
    _require(type(env) is dict and set(env) == {'python', 'requirements', 'artifacts'}
             and type(env['python']) is str and re.fullmatch(r'3\.\d+', env['python']), 'CAPTURE_ENVIRONMENT')
    artifacts = env['artifacts']
    _require(type(artifacts) is list and len(artifacts) <= 1024, 'CAPTURE_ARTIFACTS')
    declared = {}
    for obj in artifacts:
        _require(type(obj) is dict and set(obj) == {'logical_name', 'sha256'}
                 and type(obj['logical_name']) is str and obj['logical_name']
                 and obj['logical_name'] not in declared and _sha(obj['sha256']), 'CAPTURE_ARTIFACT_ROW')
        ref = roles.get('input:' + obj['logical_name'])
        _require(ref is not None and ref['sha256'] == obj['sha256'], 'CAPTURE_ARTIFACT_BINDING')
        declared[obj['logical_name']] = ref
    _require('sqlite_runtime_binding' in declared, 'CAPTURE_RUNTIME_BINDING_MISSING')
    binding_ref = declared['sqlite_runtime_binding']
    binding = strict_loads(read(binding_ref, 1048576), max_bytes=1048576)
    _require(type(binding) is dict and set(binding) == {'schema_id', 'policy', 'observation'}
             and binding['schema_id'] == 'IG_CAPTURE_SQLITE_BINDING_V1'
             and binding['policy'] == 'REQUIRE_REVIEWED_WAL_BUILD', 'CAPTURE_RUNTIME_SCHEMA')
    observation = binding['observation']
    _require(type(observation) is dict and observation.get('schema_id') == 'IG_SQLITE_RUNTIME_OBSERVATION_V1'
             and observation.get('python_implementation') == 'CPython'
             and type(observation.get('python_version')) is str
             and re.fullmatch(re.escape(env['python']) + r'\.\d+', observation['python_version'])
             and observation.get('runtime_qualification') == 'NOT_GRANTED'
             and observation.get('scientific_execution') == 'NONE', 'CAPTURE_RUNTIME_DECLARATION')
    # Recompute the narrow source policy; do not trust the artifact's verdict.
    from .sqlite_attestation import assess_wal_build
    gate = assess_wal_build(observation.get('sqlite_version'), observation.get('sqlite_source_id'))
    _require(gate['approved_for_wal_fix_gate'] and observation.get('wal_fix_gate') == gate,
             'CAPTURE_SQLITE_SOURCE_POLICY')
    requirements = env['requirements']
    _require(type(requirements) is list and len(requirements) <= 1024, 'CAPTURE_REQUIREMENTS')
    versions = {}
    for req in requirements:
        _require(type(req) is dict and set(req) == {'distribution', 'version', 'imports'}
                 and type(req['distribution']) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', req['distribution'])
                 and type(req['version']) is str and req['version']
                 and type(req['imports']) is list and req['imports']
                 and all(type(i) is str and re.fullmatch(r'[A-Za-z_]\w*', i) for i in req['imports']), 'CAPTURE_REQUIREMENT_ROW')
        key = re.sub(r'[-_.]+', '-', req['distribution']).lower()
        _require(key not in versions, 'CAPTURE_REQUIREMENT_DUPLICATE')
        versions[key] = req['version']
    observed = observation.get('package_versions')
    _require(type(observed) is dict and 1 <= len(observed) <= 1024, 'CAPTURE_PACKAGE_OBSERVATIONS')
    checked = set()
    for name, version in observed.items():
        key = re.sub(r'[-_.]+', '-', name).lower()
        _require(key not in checked and key in versions and type(version) is str
                 and versions[key] == version, 'CAPTURE_PACKAGE_VERSION')
        checked.add(key)
    return {'status': 'CAPTURE_SOURCE_AND_RUNTIME_DECLARATIONS_VERIFIED',
            'capture_ref': capture_ref, 'capture_id': capture['capture_id'], 'source_ref': source_ref,
            'environment_sha256': canonical_sha256(env),
            'selected_source_member_sha256': selected,
            'source_sha256': source_id, 'package_sha256': package_id, 'source_members': count,
            'source_expanded_bytes': expanded, 'bytes_checked': used, 'runtime_binding_ref': binding_ref,
            'observed_packages_checked': sorted(checked),
            'unobserved_requirements': sorted(set(versions) - checked),
            'unread_environment_artifacts': sorted(set(declared) - {'sqlite_runtime_binding'}),
            'live_runtime_verified': False, 'observation_authenticated': False,
            'environment_closure_verified': False, 'release_root_verified': False,
            'production_acceptance': False, 'science_authority': 'NOT_GRANTED'}
