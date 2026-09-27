from __future__ import annotations

from pathlib import Path


def test_identity_map_range_parser(tmp_path):
    import infinity_grid.v05_controller_event_loop as loop
    mapping = tmp_path / "uid_map"
    mapping.write_text("0 1000 1\n100 2000 20\n", encoding="ascii")
    assert loop._identity_is_mapped(mapping, 0)
    assert loop._identity_is_mapped(mapping, 110)
    assert not loop._identity_is_mapped(mapping, 99)
    assert not loop._identity_is_mapped(mapping, 120)


def test_root_uses_existing_same_identity_fallback_when_65534_is_unmapped(monkeypatch):
    import infinity_grid.v05_controller_event_loop as loop
    monkeypatch.setattr(loop.os, "geteuid", lambda: 0)
    monkeypatch.setattr(loop.sys, "platform", "linux")
    monkeypatch.setattr(loop, "_identity_is_mapped", lambda _path, _identity: False)
    assert loop._source_transition_worker_identity() == (None, None)


def test_root_preserves_unprivileged_worker_when_65534_is_mapped(monkeypatch):
    import infinity_grid.v05_controller_event_loop as loop
    monkeypatch.setattr(loop.os, "geteuid", lambda: 0)
    monkeypatch.setattr(loop.sys, "platform", "linux")
    monkeypatch.setattr(loop, "_identity_is_mapped", lambda _path, _identity: True)
    assert loop._source_transition_worker_identity() == (65534, 65534)


def test_unreadable_linux_maps_preserve_fail_closed_identity_choice(monkeypatch):
    import infinity_grid.v05_controller_event_loop as loop
    monkeypatch.setattr(loop.os, "geteuid", lambda: 0)
    monkeypatch.setattr(loop.sys, "platform", "linux")
    def unreadable(_path: Path, _identity: int) -> bool:
        raise OSError("blocked")
    monkeypatch.setattr(loop, "_identity_is_mapped", unreadable)
    assert loop._source_transition_worker_identity() == (65534, 65534)


def test_current_namespace_selects_supported_worker_identity():
    import infinity_grid.v05_controller_event_loop as loop
    assert loop._source_transition_worker_identity() == (None, None)
