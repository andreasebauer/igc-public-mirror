"""Preserve Q07's verified outputs without repeating its scientific body."""
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time
import zipfile

from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid.result_contracts import verify


def raw_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root, *, package=False):
    root = Path(root).resolve(strict=True)
    skip = {'__pycache__'} if package else {'.git','__pycache__','.pytest_cache','build','dist','.engineering_tmp'}
    rows = []
    for path in sorted(root.rglob('*')):
        if any(part in skip for part in path.relative_to(root).parts) or path.suffix in {'.pyc','.pyo'}:
            continue
        assert not path.is_symlink()
        if path.is_file():
            rows.append([path.relative_to(root).as_posix(), raw_sha(path)])
    schema = 'IG_DECODER_EXECUTABLE_SOURCE_TREE_V1' if package else 'IG_DECODER_ENGINEERING_SOURCE_TREE_V1'
    return canonical_sha256({'schema_id':schema,'files':rows})


def extract(archive, target):
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert len(names) == len(set(names))
        for name in names:
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert not ((bundle.getinfo(name).external_attr >> 16) & 0o170000) == 0o120000
        bundle.extractall(target)


def main():
    began = time.monotonic()
    inputs = json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    parameters = json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role, expected in parameters['input_hashes'].items():
        assert raw_sha(inputs[role]) == expected, role
    capture = json.loads(Path(inputs['origin_capture']).read_text())
    done = json.loads(Path(inputs['origin_prepared_completion']).read_text())
    registry = json.loads(Path(inputs['origin_registration']).read_text())
    assert capture['capture_id'] == parameters['origin_capture_id']
    assert canonical_sha256({k:v for k,v in capture.items() if k != 'capture_id'}) == capture['capture_id']
    assert registry == capture['job']
    assert canonical_sha256({k:v for k,v in registry.items() if k != 'registration_sha256'}) == registry['registration_sha256']
    assert done['completion_sha256'] == parameters['origin_completion_sha256']
    assert canonical_sha256({k:v for k,v in done.items() if k != 'completion_sha256'}) == done['completion_sha256']
    assert done['registration_sha256'] == registry['registration_sha256']
    assert done['source_sha256'] == registry['source_sha256'] == capture['workspace']['source_sha256']
    assert canonical_sha256(done['result']) == done['result_sha256']
    assert done['status'] == 'COMPLETED' and done['evidence_status'] == 'VERIFIED'
    assert done['result']['outcome'] == 'PROCESS_COMPLETED'
    assert done['result']['status'] == 'PASS' and done['result']['exit_code'] == 0
    assert done['scientific_outcome'] == 'CURRENT_R100_SAVED_READER_AND_BOUNDED_SERIALIZER_QUALIFIED_NO_PRODUCTION_AUTHORITY'
    for role, source_role in [('origin_engine_archive','engine_source'),('origin_project_archive','project_source')]:
        original = next(row for row in capture['objects'] if row['role'] == source_role)
        assert raw_sha(inputs[role]) == original['sha256']
        assert Path(inputs[role]).stat().st_size == original['size_bytes']
    with tempfile.TemporaryDirectory(prefix='origin-source-', dir='.') as temporary:
        source = Path(temporary)
        extract(inputs['origin_engine_archive'], source)
        extract(inputs['origin_project_archive'], source/'project')
        assert tree_digest(source) == capture['workspace']['source_sha256']
        assert tree_digest(source/'infinity_grid', package=True) == capture['workspace']['package_sha256']
        assert raw_sha(source/'project/main.py') == parameters['origin_main_sha256']
    root = Path('q07_evidence')
    root.mkdir()
    extract(inputs['origin_evidence_archive'], root)
    observed = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
    assert observed == sorted(row['path'] for row in done['evidence'])
    for row in done['evidence']:
        artifact = root/row['path']
        assert artifact.stat().st_size == row['size_bytes']
        assert raw_sha(artifact) == row['sha256']
    original_verification = json.loads((root/'RESULT_VERIFICATION.json').read_text())
    replayed_checks = verify(capture['result_contract'], done['result'], root)
    assert replayed_checks == original_verification
    assert replayed_checks['status'] == 'VERIFIED'
    binding = json.loads((root/'script/work/PRODUCER_BINDING.json').read_text())
    assert binding['producer_experiment_id'] == 'MASTER.STEP3.CURRENT.R100.Q07'
    assert binding['source_sha256'] == parameters['origin_engine_sha256']
    reader = json.loads((root/'script/work/SAVED_READER_RESULT.json').read_text())
    assert reader['complete_root_count'] == reader['public_descriptor_matches'] == 193
    assert reader['ordered_roots_equal'] and reader['exact_dag_roundtrip_equal']
    assert not reader['actual_campaign_started'] and reader['production_authority'] == 'NONE'
    for role, name in [('origin_capture','ORIGIN_CAPTURE.json'),('origin_prepared_completion','ORIGIN_PREPARED_COMPLETION.json'),('origin_registration','ORIGIN_REGISTRATION.json')]:
        shutil.copyfile(inputs[role], name)
    write_json_atomic(Path('ORIGINAL_CONTRACT_REVERIFICATION.json'), replayed_checks)
    result = {'status':'PASS', 'scientific_outcome':'Q07_VERIFIED_OUTPUTS_REUSED_NO_SCIENTIFIC_REEXECUTION',
        'scope':'PRESERVATION_ONLY_CONTINUATION', 'origin_capture_id':capture['capture_id'],
        'origin_prepared_completion_sha256':done['completion_sha256'],
        'origin_source_sha256':done['source_sha256'], 'origin_publication_status':'PAUSED_CHECKPOINT_STATE_RAW_LIMIT',
        'producer_experiment_id_retained':binding['producer_experiment_id'],
        'origin_contract_identical_reverification':True, 'all_evidence_raw_bytes_equal':True,
        'evidence_files':len(done['evidence']), 'evidence_bytes':sum(row['size_bytes'] for row in done['evidence']),
        'complete_root_count':193, 'public_descriptor_matches':193, 'dag_nodes':reader['dag_nodes'],
        'scientific_body_calls':0, 'recipe_worker_calls':0, 'selector_calls':0,
        'serializer_calls':0, 'saved_reader_calls':0, 'marker_tasks':0,
        'production_authority':'NONE', 'actual_campaign_started':False,
        'wall_seconds':time.monotonic()-began, 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json_atomic(Path('CONTINUATION_RESULT.json'), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
