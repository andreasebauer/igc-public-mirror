"""Discard completed disposable regression fixtures after teardown."""
from pathlib import Path
import shutil
import pytest

@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_teardown(item, nextitem):
    path=item.funcargs.get('tmp_path')
    yield
    if path is not None:
        path=Path(path)
        allowed=Path('/tmp/ig_p0_dev154_regression').resolve()
        if path.resolve().is_relative_to(allowed) and path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
