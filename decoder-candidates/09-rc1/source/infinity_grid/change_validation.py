"""Native engineering handler: candidate data enters the existing test scheduler."""
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from . import submission as sub
from . import v05_controller_event_loop as loop
from .canon import canonical_sha256
from .v05_origin_guard import require_controller_execution_origin


def validate_revision(stage, runtime):
    require_controller_execution_origin('change_validation')
    from .change_sessions import _verified, _archive_files, source_version, _requirements
    from .v05_validation_runtime import run_registered_validation_nodes
    inputs = stage['input_artifacts']
    rev = _verified(inputs['change_revision'])
    session = _verified(inputs['change_session'])
    if rev['session_sha256'] != session['record_sha256']:
        raise sub.SubmissionError('CHANGE_SESSION_BINDING')
    params = stage['execution']['parameters']
    if params['revision_id'] != rev['record_sha256']:
        raise sub.SubmissionError('CHANGE_REVISION_BINDING')
    _requirements(rev['requirements'])
    raw = Path(inputs['candidate_source']).read_bytes()
    if sub._sha(raw) != rev['source_object']['sha256'] or len(raw) != rev['source_object']['size_bytes']:
        raise sub.SubmissionError('CHANGE_CANDIDATE_OBJECT_MISMATCH')
    # The work root is inside this registered controller run, never an arbitrary path.
    out = Path(runtime._chain_dir).parent/'candidate_validation'
    out.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='decoder-candidate-'))
    try:
        sub._write_files(temp, _archive_files(raw))
        expected = (rev['source_sha256'], rev['package_sha256'])
        if loop._source_ids(temp) != expected or source_version(temp) != rev['version']:
            raise sub.SubmissionError('CHANGE_CANDIDATE_SOURCE_MISMATCH')
        validation = run_registered_validation_nodes(temp, rev['requirements']['nodes'],
            workers=params['workers'], wall_seconds_max=params['wall_seconds_max'],
            output_dir=out)
        if loop._source_ids(temp) != expected:
            raise sub.SubmissionError('CHANGE_CANDIDATE_MODIFIED_DURING_TEST')
        result = {'outcome': validation['status'], 'revision_id': rev['record_sha256'],
                  'candidate_source_sha256': rev['source_sha256'], 'candidate_version': rev['version'],
                  'requirements_sha256': canonical_sha256(rev['requirements']), 'validation': validation}
        return SimpleNamespace(result=result)
    finally:
        shutil.rmtree(temp)
