"""Real native preparation and recovery; no save receipts or science fabricated."""
from pathlib import Path
import json
import shutil

import pytest

from infinity_grid import portable_registry as project, submission as sub
from infinity_grid.invocation import InvocationRefused
from tests.test_decoder06_capture import _spec


def test_capture_and_project_baseline_recover_exact_original_obligations(tmp_path):
    spec = _spec(tmp_path, project=False)
    store = tmp_path / 'original'
    project.initialize(store, 'Native pre-execution recovery', spec['engine_source'])
    captured = sub.capture(store, spec)
    original = Path(captured['workspace'])
    original_binding = (original / 'PROJECT_BINDING.json').read_bytes()
    original_record = (original / 'CAPTURE.json').read_bytes()
    original_obligations = sub.required_objects(original)
    binding = json.loads(original_binding)

    retained = tmp_path / 'retained'
    shutil.copytree(store / 'objects', retained)
    capture_path = retained / 'CAPTURE.json'
    capture_path.write_bytes(original_record)
    baseline = retained / binding['baseline_object']['object_name']
    shutil.rmtree(store)

    restored_store = tmp_path / 'recovered'
    restored = restored_store / 'captures' / captured['capture_id']
    result = sub.restore_capture(capture_path, retained, restored)
    assert result['status'] == 'SAVE_REQUIRED'
    with pytest.raises(InvocationRefused, match='PROJECT_BINDING_REQUIRED'):
        project.locate(restored)

    with pytest.raises(InvocationRefused, match='PROJECT_ARCHIVE_HASH'):
        project.import_history(restored_store, baseline, '0' * 64)
    assert not (restored_store / 'coordination' / 'PROJECT.json').exists()
    project.import_history(restored_store, baseline, binding['baseline_object']['sha256'])
    project.adopt(restored_store, restored)

    assert (restored / 'CAPTURE.json').read_bytes() == original_record
    assert (restored / 'PROJECT_BINDING.json').read_bytes() == original_binding
    assert sub.required_objects(restored) == original_obligations
    assert {row['role'] for row in sub.required_objects(restored)} >= {
        'capture_record', 'engine_source', 'project_baseline', 'project_binding'
    }
    assert sub.save_status(restored)['status'] == 'SAVE_REQUIRED'
    assert not list((restored / 'durability' / 'receipts').rglob('*.json'))
    assert not (restored / 'runtime' / 'attempts').exists()
    assert not (restored / 'runtime' / 'runs').exists()

