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
