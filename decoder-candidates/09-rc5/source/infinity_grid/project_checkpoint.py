"""Versioned project metadata/evidence closure using exact standard ZIP archives."""
from pathlib import Path
import hashlib
import io
import json
import zipfile

from . import submission as sub
from . import portable_registry as registry
from .checkpoint_stream import archive_paths

SCHEMA = 'IG_PROJECT_DELTA_V2'


def require(value, code):
    if not value:
        raise sub.SubmissionError(code)


def require_metadata_bound(size, maximum):
    # None is an explicit new-capture policy; legacy finite bounds remain exact.
    require(maximum is None or size <= maximum, 'CHECKPOINT_PROJECT_METADATA_RAW_LIMIT')


def prepare(root, baseline_raw, limits, directory):
    """Caller owns project lock. Only capsule-referenced blobs are external; the full baseline stays pinned."""
    root, directory = Path(root), Path(directory)
    rows = registry.events(root)
    with zipfile.ZipFile(io.BytesIO(baseline_raw)) as z:
        require(len(z.namelist()) == len(set(z.namelist())), 'PROJECT_BASELINE_DUPLICATE_PATH')
        baseline = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}
    evidence = set()
    for row in rows.values():
        for field in ('capsule', 'validation', 'correctness_capsule', 'performance_capsule'):
            digest = row['value'].get(field)
            if not digest:
                continue
            capsule = json.loads(registry.blob(root, digest))
            require(capsule.get('schema_id') == 'IG_PROJECT_COMPLETION_CAPSULE_V1'
                    and isinstance(capsule.get('files'), dict), 'PROJECT_EVIDENCE_CAPSULE_SCHEMA')
            for name, sha in capsule['files'].items():
                sub._relative(name)
                # Identifier, original bytes and symlink checks occur below.
                require(isinstance(sha, str) and len(sha) == 64
                        and all(c in '0123456789abcdef' for c in sha), 'PROJECT_EVIDENCE_IDENTIFIER')
                evidence.add('objects/' + sha)
    files = {'PROJECT.json': root / 'PROJECT.json'}
    for folder in ('events', 'objects'):
        for path in sorted((root / folder).glob('*')):
            require(not path.is_symlink() and path.is_file(), 'PROJECT_EVIDENCE_FILE_TYPE')
            name = path.relative_to(root).as_posix()
            if folder == 'objects':
                registry.blob(root, path.name)
            files[name] = path
    require(set(baseline) <= set(files), 'PROJECT_BASELINE_NOT_RETAINED')
    for name, digest in baseline.items():
        require(hashlib.sha256(files[name].read_bytes()).hexdigest() == digest,
                'PROJECT_BASELINE_NOT_RETAINED')
    require(evidence <= set(files), 'PROJECT_EVIDENCE_MISSING')
    external = {n: files[n] for n in sorted(evidence)}
    metadata = {n: p for n, p in files.items() if n not in external}
    metadata_bytes = sum(p.stat().st_size for p in metadata.values())
    evidence_bytes = sum(p.stat().st_size for p in external.values())
    require_metadata_bound(metadata_bytes, limits['max_project_raw_bytes'])
    require((limits['max_project_evidence_files'] is None or len(external) <= limits['max_project_evidence_files']), 'CHECKPOINT_PROJECT_EVIDENCE_FILE_LIMIT')
    require((limits['max_project_evidence_raw_bytes'] is None or evidence_bytes <= limits['max_project_evidence_raw_bytes']), 'CHECKPOINT_PROJECT_EVIDENCE_RAW_LIMIT')
    require((limits['max_state_file_bytes'] is None or all(p.stat().st_size <= limits['max_state_file_bytes'] for p in external.values())),
            'CHECKPOINT_PROJECT_EVIDENCE_SINGLE_FILE_LIMIT')
    metadata_zip = directory / 'project-metadata.zip'
    evidence_zip = directory / 'project-evidence.zip'
    full_zip = directory / 'project-full.zip'
    archive_paths(metadata, metadata_zip)
    archive_paths(external, evidence_zip)
    archive_paths(files, full_zip)
    return {'metadata': metadata_zip.read_bytes(), 'evidence_path': evidence_zip,
            'metadata_sha256': sub._sha(metadata_zip.read_bytes()),
            'full_sha256': sub._sha(full_zip.read_bytes()),
            'metadata_raw_bytes': metadata_bytes, 'evidence_raw_bytes': evidence_bytes,
            'evidence_file_count': len(external)}


def metadata_baseline(baseline_raw, evidence_raw, limits):
    """Project exact baseline onto metadata; complete baseline stays a saved role."""
    from .preservation import _zip_files
    with zipfile.ZipFile(io.BytesIO(evidence_raw)) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'PROJECT_EVIDENCE_DUPLICATE_PATH')
        external = set(names)
        for name in external:
            rel = sub._relative(name)
            require(len(rel.parts) == 2 and rel.parts[0] == 'objects'
                    and len(rel.name) == 64 and all(c in '0123456789abcdef' for c in rel.name),
                    'PROJECT_EVIDENCE_PATH')
    with zipfile.ZipFile(io.BytesIO(baseline_raw)) as z:
        require(limits['max_project_raw_bytes'] is None or limits['max_project_evidence_raw_bytes'] is None or sum(i.file_size for i in z.infolist())
                <= limits['max_project_raw_bytes'] + limits['max_project_evidence_raw_bytes'],
                'CHECKPOINT_PROJECT_BASELINE_RAW_LIMIT')
    files = _zip_files(baseline_raw)
    metadata = {n: b for n, b in files.items() if n not in external}
    require_metadata_bound(sum(len(b) for b in metadata.values()), limits['max_project_raw_bytes'])
    return sub._archive(metadata)


def hydrate(metadata_raw, evidence_raw, project, limits):
    """Validate raw bounds before decompression; caller publishes only after hash checks."""
    from .preservation import _zip_files
    require(project.get('schema_id') == SCHEMA, 'PROJECT_EVIDENCE_PROTOCOL')
    require(limits['project_evidence_externalization'] == 1, 'PROJECT_EVIDENCE_CAPTURE_OPT_IN_REQUIRED')
    metadata = _zip_files(metadata_raw)
    require(sub._sha(metadata_raw) == project['metadata_sha256'], 'PROJECT_EVIDENCE_METADATA_HASH')
    require_metadata_bound(sum(len(v) for v in metadata.values()), limits['max_project_raw_bytes'])
    with zipfile.ZipFile(io.BytesIO(evidence_raw)) as z:
        infos = z.infolist()
        require(len(infos) == len({i.filename for i in infos}), 'PROJECT_EVIDENCE_DUPLICATE_PATH')
        require(len(infos) == project['evidence_file_count']
                and (limits['max_project_evidence_files'] is None or len(infos) <= limits['max_project_evidence_files']), 'CHECKPOINT_PROJECT_EVIDENCE_FILE_LIMIT')
        raw_bytes = sum(i.file_size for i in infos)
        require(raw_bytes == project['evidence_raw_bytes']
                and (limits['max_project_evidence_raw_bytes'] is None or raw_bytes <= limits['max_project_evidence_raw_bytes']), 'CHECKPOINT_PROJECT_EVIDENCE_RAW_LIMIT')
        for item in infos:
            rel = sub._relative(item.filename)
            require(len(rel.parts) == 2 and rel.parts[0] == 'objects'
                    and len(rel.name) == 64 and all(c in '0123456789abcdef' for c in rel.name),
                    'PROJECT_EVIDENCE_PATH')
            require(item.filename not in metadata, 'PROJECT_EVIDENCE_METADATA_OVERLAP')
            require((limits['max_state_file_bytes'] is None or item.file_size <= limits['max_state_file_bytes']), 'CHECKPOINT_PROJECT_EVIDENCE_SINGLE_FILE_LIMIT')
        for item in infos:
            raw = z.read(item.filename)
            require(sub._sha(raw) == Path(item.filename).name, 'PROJECT_EVIDENCE_OBJECT_HASH')
            metadata[item.filename] = raw
    full = sub._archive(metadata)
    require(sub._sha(full) == project['full_sha256'], 'PROJECT_EVIDENCE_FULL_HASH')
    return full
