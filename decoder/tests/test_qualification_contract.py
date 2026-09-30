"""Registered tests of qualification coverage and environment admission."""
import json
import sys
import tomllib
from pathlib import Path

import pytest
from infinity_grid import platform_support

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('platform', ['win32', 'darwin'])
def test_unsupported_platform_is_actionable(monkeypatch, platform):
    monkeypatch.setattr(sys, 'platform', platform)
    with pytest.raises(RuntimeError, match='DECODER_PLATFORM_UNSUPPORTED: use Linux'):
        platform_support.require_supported_platform()


def test_qualification_groups_cover_each_active_test_file_once():
    profile = json.loads((ROOT / 'qualification/PROFILE.json').read_text())
    selectors = [s for group in profile['groups'].values() for s in group['selectors']]
    actual = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'tests').glob('test_*.py'))
    assert sorted(selectors) == actual
    assert len(selectors) == len(set(selectors))
    assert profile['groups']['timing']['workers'] == 1
    assert profile['group_order'][-1] == 'timing'


def test_required_dependencies_and_fixture_roles_are_explicit():
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    extras = config['project']['optional-dependencies']
    assert 'networkx==3.6.1' in extras['verification']
    assert {'pytest==9.0.2', 'networkx==3.6.1', 'setuptools==84.0.0', 'pip==26.2.1'} <= set(extras['qualification'])
    profile = json.loads((ROOT / 'qualification/PROFILE.json').read_text())
    assert {r['logical_name'] for r in profile['required_fixtures']} == {
        'step2_parent_snapshot', 'saved_stage_one', 'saved_stage_one_prerun', 'saved_stage_four',
        'saved_stage_outcome', 'saved_representative_qualification', 'failed_evidence'}
    assert profile['required_fixtures'][0]['sha256'] == '713452bf109d17d4b377f9084e61383f8397c1a418012be2f41e09eff17f3bd8'

    failed = next(r for r in profile['required_fixtures'] if r['logical_name'] == 'failed_evidence')
    assert failed['sha256'] == '5f93d09e8ac71bcf7e2ed7ece86407d7dbf75acc1cfb49a5b4ac5429995b3d41'
