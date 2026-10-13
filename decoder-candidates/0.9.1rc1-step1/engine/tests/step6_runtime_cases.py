"""Explicitly registered demonstrations; never part of automatic test discovery."""
import time


def test_pass_for_mixed_report():
    assert 6*7==42


def test_intentional_failure_for_mixed_report():
    assert False, 'Frozen intentional failure; tests the report, not a scientific assertion.'


def test_delay_for_interruption():
    print('DELAY_STARTED',flush=True)
    time.sleep(30)
    assert True
