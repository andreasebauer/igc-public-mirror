"""Explicit engineering probes; not part of the general passing test profile.

Install under tests/integration_probes/native_save_wave_probe.py only in a new
versioned, saved candidate. Select individual functions by frozen registration.
Expected fail/skip outcomes must remain fail/skip in native evidence.
"""
import json
import time
import pytest


def test_first_pass():
    print('IG_SAVE_PROBE_FIRST_PASS', flush=True)


def test_later_sentinel():
    print('IG_SAVE_PROBE_LATER_SENTINEL_EXECUTED', flush=True)


def test_expected_failure():
    pytest.fail('IG_SAVE_PROBE_PREREGISTERED_EXPECTED_FAILURE')


def test_expected_skip():
    pytest.skip('IG_SAVE_PROBE_PREREGISTERED_EXPECTED_SKIP')


def test_long_bounded_selector():
    # Fixed workload duration, not a controller runtime deadline. Native default
    # 30-second checkpoints and all backlog limits remain unchanged.
    start = time.monotonic()
    print(json.dumps({'probe':'LONG_START','duration_seconds':150}), flush=True)
    tick = 0
    while time.monotonic() - start < 150:
        time.sleep(min(1.0, max(0.0, 150 - (time.monotonic() - start))))
        tick += 1
    print(json.dumps({'probe':'LONG_END','ticks':tick,
                      'elapsed_seconds':time.monotonic()-start}), flush=True)


def test_long_durable_progress_selector(request):
    """Explicit native probe with its own changing checkpointed log.

    The native runner captures stdout until child exit. Use only a dedicated
    probe artifact; never write native phase reports, finalizations or receipts.
    """
    import os
    from infinity_grid.validation_reports import Recorder

    recorders = [p for p in request.config.pluginmanager.get_plugins()
                 if isinstance(p, Recorder)]
    assert len(recorders) == 1, 'NATIVE_RECORDER_REQUIRED'
    recorder = recorders[0]
    assert recorder.selectors == [request.node.nodeid]
    root = recorder.root.resolve(strict=True)
    folder = root / 'probe_progress'
    folder.mkdir(exist_ok=True)
    assert not folder.is_symlink() and folder.resolve().parent == root
    path = folder / 'long_durable_progress.log'
    start = time.monotonic()
    with path.open('x', encoding='utf-8') as stream:
        def emit(event, tick):
            row = {'schema_id':'IG_NATIVE_LONG_PROBE_PROGRESS_V1',
                   'binding':recorder.binding, 'selector':request.node.nodeid,
                   'event':event, 'tick':tick, 'unix':time.time(),
                   'elapsed_seconds':time.monotonic()-start}
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

        emit('START', 0)
        for tick in range(1, 16):
            target = start + 10 * tick
            while time.monotonic() < target:
                time.sleep(min(1.0, max(0.0, target-time.monotonic())))
            emit('PROGRESS', tick)
        emit('END', 15)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['tick'] for r in rows if r['event']=='PROGRESS'] == list(range(1,16))
    assert rows[-1]['event']=='END' and rows[-1]['elapsed_seconds'] >= 150
