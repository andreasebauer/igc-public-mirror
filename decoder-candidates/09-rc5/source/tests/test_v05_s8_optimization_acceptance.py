from pathlib import Path
import inspect,json,shutil

def test_s8_optimization_gate_is_wired_fail_closed():
    import infinity_grid.v05_engineering_jobs as e
    assert 'run_s8_optimization_acceptance_gate(source_root,candidate)' in inspect.getsource(e.execute_registered_engineering_job)

def test_s8_optimization_fixture_is_real_s7_certified_reuse_fixture():
    from importlib import resources
    obj=json.loads(resources.files('infinity_grid').joinpath('resources/v05/G6_S8_OPTIMIZATION_GATE_FIXTURE_V1.json').read_text())
    assert obj['fixture_origin']['reuse_input_sha256']=='fe6ad6352b8f0ed1358a861900a43bb11fea6143c736f89c73750d07aad6d757'
    assert len(obj['class_members'])==2

def test_s8_optimization_gate_not_applicable_when_sources_equal():
    import infinity_grid.v05_optimization_acceptance as g
    root=Path(inspect.getfile(g)).resolve().parents[1]
    assert g.run_s8_optimization_acceptance_gate(root,root)['status']=='NOT_APPLICABLE'

def _synthetic_semantic():
    return {'status':'PASS','split_found':False,'completed_prefix':124,'class_id':'fixture'}

def test_s8_optimization_gate_deterministic_pass_logic(monkeypatch,tmp_path):
    import infinity_grid.v05_optimization_acceptance as g
    root=Path(inspect.getfile(g)).resolve().parents[1]
    cand=tmp_path/'candidate';shutil.copytree(root,cand)
    monkeypatch.setattr(g,'sensitive_changes',lambda parent,candidate:['infinity_grid/g6_s8_evaluators.py'])
    sem=_synthetic_semantic()
    def fake(source,fixture):
        elapsed=0.12 if Path(source).resolve()==cand.resolve() else 0.10
        return {'semantic':sem,'elapsed_seconds':elapsed}
    monkeypatch.setattr(g,'_probe',fake)
    got=g.run_s8_optimization_acceptance_gate(root,cand)
    assert got['status']=='PASS'
    assert got['parent_semantic_sha256']==got['candidate_semantic_sha256']==got['one_worker_semantic_sha256']
    assert len(set(got['four_worker_semantic_sha256s']))==1
    assert got['paired_ratio_median'] <= got['candidate_vs_parent_max_ratio']

def test_s8_optimization_gate_rejects_deliberate_benchmark_regression(monkeypatch,tmp_path):
    import infinity_grid.v05_optimization_acceptance as g
    import pytest
    root=Path(inspect.getfile(g)).resolve().parents[1]
    cand=tmp_path/'candidate';shutil.copytree(root,cand)
    monkeypatch.setattr(g,'sensitive_changes',lambda parent,candidate:['infinity_grid/g6_s8_evaluators.py'])
    sem=_synthetic_semantic()
    def fake(source,fixture):
        elapsed=20.0 if Path(source).resolve()==cand.resolve() else 0.10
        return {'semantic':sem,'elapsed_seconds':elapsed}
    monkeypatch.setattr(g,'_probe',fake)
    with pytest.raises(g.OptimizationGateError,match='OPTIMIZATION_BENCHMARK_REGRESSION'):
        g.run_s8_optimization_acceptance_gate(root,cand)

def test_real_probe_runs_on_registered_s7_fixture_once():
    import infinity_grid.v05_optimization_acceptance as g
    root=Path(inspect.getfile(g)).resolve().parents[1]
    fixture=root/g.FIXTURE_REL
    before={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert not any('__pycache__' in n or n.endswith('.pyc') for n in before)
    got=g._probe(root,fixture)
    after={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert after == before
    assert isinstance(got['semantic'],dict)
    assert got['elapsed_seconds'] >= 0
