import json

import pytest

from infinity_grid.preservation import _verify_terminal_state_evidence
from infinity_grid.submission import SubmissionError


def test_terminal_checkpoint_rejects_replay_state_drift():
    stage = 'runtime/runs/intent-one/chain/decoder_stage_runtime/REPLAY__L0_TO_G8__ROOT'
    status = stage + '/artifacts/REPLAY_ROOT_STATUS.json'
    state = stage + '/replay_runner/runner_state.json'
    manifest = stage + '/replay_reference_data/MANIFEST.json'
    files = {
        status: json.dumps({'runner_state_sha256': 'new', 'reference_manifest_sha256': 'reference'}).encode(),
        state: json.dumps({'state_sha256': 'old'}).encode(),
        manifest: json.dumps({'manifest_sha256': 'reference'}).encode(),
    }
    with pytest.raises(SubmissionError, match='CHECKPOINT_REPLAY_STATE_STATUS_MISMATCH'):
        _verify_terminal_state_evidence(files)
    files[state] = json.dumps({'state_sha256': 'new'}).encode()
    _verify_terminal_state_evidence(files)
    files[manifest] = json.dumps({'manifest_sha256': 'old'}).encode()
    with pytest.raises(SubmissionError, match='CHECKPOINT_REPLAY_STATE_STATUS_MISMATCH'):
        _verify_terminal_state_evidence(files)
