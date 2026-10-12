from __future__ import annotations

import multiprocessing as mp

from infinity_grid.execution import ExecutionPolicy, _choose_start_method


def test_auto_falls_back_to_spawn_when_forkserver_transport_is_blocked(monkeypatch):
    import infinity_grid.execution as execution
    # Exercise AUTO selection independently of the enclosing job's explicit policy.
    monkeypatch.delenv("IG_DECODER_START_METHOD", raising=False)
    monkeypatch.setattr(execution.mp, "get_all_start_methods", lambda: ["fork", "spawn", "forkserver"])
    monkeypatch.setattr(execution, "_forkserver_transport_available", lambda: False)
    assert _choose_start_method(ExecutionPolicy(start_method="AUTO")) == "spawn"


def test_auto_prefers_forkserver_when_transport_is_available(monkeypatch):
    import infinity_grid.execution as execution
    # Exercise AUTO selection independently of the enclosing job's explicit policy.
    monkeypatch.delenv("IG_DECODER_START_METHOD", raising=False)
    monkeypatch.setattr(execution.mp, "get_all_start_methods", lambda: ["fork", "spawn", "forkserver"])
    monkeypatch.setattr(execution, "_forkserver_transport_available", lambda: True)
    assert _choose_start_method(ExecutionPolicy(start_method="AUTO")) == "forkserver"


def test_explicit_forkserver_remains_explicit_when_transport_probe_fails(monkeypatch):
    import infinity_grid.execution as execution
    # Exercise AUTO selection independently of the enclosing job's explicit policy.
    monkeypatch.delenv("IG_DECODER_START_METHOD", raising=False)
    monkeypatch.setattr(execution.mp, "get_all_start_methods", lambda: ["fork", "spawn", "forkserver"])
    monkeypatch.setattr(execution, "_forkserver_transport_available", lambda: False)
    assert _choose_start_method(ExecutionPolicy(start_method="forkserver")) == "forkserver"


def test_legacy_fork_initializer_still_selects_fork(monkeypatch):
    import infinity_grid.execution as execution
    # Exercise AUTO selection independently of the enclosing job's explicit policy.
    monkeypatch.delenv("IG_DECODER_START_METHOD", raising=False)
    monkeypatch.setattr(execution.mp, "get_all_start_methods", lambda: ["fork", "spawn", "forkserver"])
    monkeypatch.setattr(execution, "_forkserver_transport_available", lambda: False)
    initializer = "infinity_grid.uplift_campaign:_init_worker_from_fork"
    assert _choose_start_method(ExecutionPolicy(start_method="AUTO"), initializer) == "fork"


def test_current_auto_choice_is_advertised_and_transport_usable():
    import infinity_grid.execution as execution
    method = _choose_start_method(ExecutionPolicy(start_method="AUTO"))
    assert method in mp.get_all_start_methods()
    if method == "forkserver":
        assert execution._forkserver_transport_available()
