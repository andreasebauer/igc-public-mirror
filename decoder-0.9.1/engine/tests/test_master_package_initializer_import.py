"""Real import closure and preserved explicit legacy registration regressions."""
from pathlib import Path
import os
import subprocess
import sys
import pytest
from infinity_grid import workflow_guard
from infinity_grid.invocation import InvocationRefused

def test_named_leaf_cannot_hide_forbidden_initializer_helper(tmp_path):
    package=tmp_path/'pkg';package.mkdir()
    (package/'__init__.py').write_text('from .hidden import unused\n')
    (package/'hidden.py').write_text('import os\ndef unused():\n    os.system("never-run")\n')
    leaf=package/'leaf.py';leaf.write_text('VALUE = 3\n')
    with pytest.raises(InvocationRefused) as rejected:
        workflow_guard.preflight(tmp_path,[leaf])
    assert rejected.value.code == 'PRIVATE_EXECUTION_PREFLIGHT'
    assert any(row['path']=='pkg/hidden.py' for row in rejected.value.checks)

def test_pure_nested_package_initializers_are_included_once(tmp_path):
    parent=tmp_path/'pkg';child=parent/'child';child.mkdir(parents=True)
    (parent/'__init__.py').write_text('NAME = "pure"\n')
    (child/'__init__.py').write_text('from .leaf import VALUE\n')
    leaf=child/'leaf.py';leaf.write_text('VALUE = 3\n')
    r=workflow_guard.preflight(tmp_path,[leaf])
    names=[Path(row['module_path']).relative_to(tmp_path).as_posix() for row in r['modules']]
    assert set(names)=={'pkg/__init__.py','pkg/child/__init__.py','pkg/child/leaf.py'}
    assert len(names)==len(set(names))

def test_project_package_initializer_is_checked(tmp_path):
    project=tmp_path/'project';package=project/'pkg';package.mkdir(parents=True)
    (package/'__init__.py').write_text('import subprocess\ndef hidden():\n    subprocess.run([])\n')
    leaf=package/'leaf.py';leaf.write_text('VALUE = 3\n')
    with pytest.raises(InvocationRefused) as rejected:
        workflow_guard.preflight(tmp_path,[leaf])
    assert any(row['path']=='project/pkg/__init__.py' for row in rejected.value.checks)

def test_actual_kernel_closure_includes_adapter_initializer():
    source=Path(__file__).resolve().parents[1]
    r=workflow_guard.preflight(source,[source/'infinity_grid/exact_tree_relation_kernel.py'])
    assert r['status']=='PASS'
    names={Path(x['module_path']).relative_to(source).as_posix() for x in r['modules']}
    assert 'infinity_grid/adapters/__init__.py' in names
    assert 'infinity_grid/core/__init__.py' in names
    assert 'infinity_grid/adapters/oscout_phase0.py' not in names

def _fresh(code):
    source=Path(__file__).resolve().parents[1]
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=source,env=env,capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr

def test_fresh_guarded_kernel_import_keeps_unrelated_runners_unloaded():
    _fresh('''from pathlib import Path
import sys
from infinity_grid.workflow_guard import scientific_call
with scientific_call(Path.cwd()):
    from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel
    from infinity_grid.adapters.g4_accepted import G4AcceptedAdapter
    assert G4AcceptedAdapter().relation_class_table()
assert "infinity_grid.adapters.oscout_phase0" not in sys.modules
assert "infinity_grid.legacy_adapter_registration" not in sys.modules
''')

def test_explicit_legacy_registration_preserves_runner_set():
    _fresh('''import infinity_grid.adapters
import sys
import json
from pathlib import Path
assert "infinity_grid.adapters.oscout_phase0" not in sys.modules
import infinity_grid.legacy_adapter_registration
from infinity_grid.controller import RUNNERS, RUNNER_META
from infinity_grid.v05_p7_support import current_v05_runner_classification
expected=json.loads(Path("tests/fixtures/master_legacy_runner_baseline.json").read_text())
actual={"runners":{k:v.__module__+":"+v.__qualname__ for k,v in sorted(RUNNERS.items())},"metadata":RUNNER_META,"classification":current_v05_runner_classification()}
assert actual==expected
before=dict(RUNNERS)
import infinity_grid.legacy_adapter_registration
assert RUNNERS==before
''')

def test_unrelated_legacy_registration_still_refuses_scientific_import():
    source=Path(__file__).resolve().parents[1]
    with pytest.raises(InvocationRefused) as rejected:
        workflow_guard.preflight(source,[source/'infinity_grid/legacy_adapter_registration.py'])
    assert rejected.value.code=='PRIVATE_EXECUTION_PREFLIGHT'
