"""Bounded engineering fixtures; synthetic capsules do not establish save evidence."""
import io
import json
from pathlib import Path
import zipfile

import pytest
from infinity_grid import project_checkpoint as pc, result_contracts as rc
from infinity_grid import portable_registry as reg, preservation as pr, submission as sub
from infinity_grid.canon import write_json_atomic


def fixture(tmp_path, *, referenced=True):
    root = tmp_path / 'coordination'
    write_json_atomic(root / 'PROJECT.json', {'schema_id':'IG_PORTABLE_PROJECT_V1',
                                            'project_id':'ENGINEERING_ONLY','name':'UNIT'})
    baseline = reg.snapshot(root)
    a = reg.put(root, b'a' * (18 * 1024 * 1024))
    b = reg.put(root, b'b' * (18 * 1024 * 1024))
    if referenced:
        capsule = reg.put(root, sub._json_bytes({'schema_id':'IG_PROJECT_COMPLETION_CAPSULE_V1',
                    'files':{'source/a':a,'runtime/b':b},'completion_sha256':'0'*64,'job_id':'UNIT_ONLY'}))
        with reg.lock(root):
            reg.append(root,'evidence:unit',[],{'capsule':capsule},'Synthetic structural closure only')
    limits = rc.policy({'preservation':{'project_evidence_externalization':1}})
    return root, baseline, limits


def closure(tmp_path, **kw):
    root, baseline, limits = fixture(tmp_path, **kw)
    prepared = pc.prepare(root, baseline, limits, tmp_path)
    project = dict(prepared, schema_id=pc.SCHEMA)
    return root, baseline, limits, prepared, project


def test_project_over_old_limit_roundtrips_exactly_with_bounded_evidence(tmp_path):
    root, baseline, limits, prepared, project = closure(tmp_path)
    assert prepared['evidence_raw_bytes'] == 36 * 1024 * 1024
    assert prepared['metadata_raw_bytes'] < limits['max_project_raw_bytes']
    full = pc.hydrate(prepared['metadata'], prepared['evidence_path'].read_bytes(), project, limits)
    assert full == reg.snapshot(root)
    assert pr._restore_project(baseline,pr._project_delta(baseline,prepared['metadata']),
                              prepared['metadata_sha256']) == prepared['metadata']


def test_unreferenced_objects_cannot_escape_metadata_budget(tmp_path):
    root, baseline, limits = fixture(tmp_path, referenced=False)
    with pytest.raises(sub.SubmissionError,match='PROJECT_METADATA_RAW_LIMIT'):
        pc.prepare(root,baseline,limits,tmp_path)


def test_evidence_budget_refusal_happens_before_archive_creation(tmp_path):
    root, baseline, limits = fixture(tmp_path)
    limits['max_project_evidence_raw_bytes'] = 1
    with pytest.raises(sub.SubmissionError,match='PROJECT_EVIDENCE_RAW_LIMIT'):
        pc.prepare(root,baseline,limits,tmp_path)
    assert not (tmp_path/'project-evidence.zip').exists()


@pytest.mark.parametrize('kind',['wrong_bytes','outside','duplicate','overlap'])
def test_forged_evidence_is_refused(tmp_path,kind):
    root, baseline, limits, prepared, project = closure(tmp_path)
    with zipfile.ZipFile(prepared['evidence_path']) as z:
        files={n:z.read(n) for n in z.namelist()}
    name=next(iter(files))
    if kind=='wrong_bytes':files[name]=b'x'*len(files[name])
    if kind=='outside':files['../outside']=files.pop(name)
    if kind=='overlap':files['PROJECT.json']=(root/'PROJECT.json').read_bytes()
    if kind=='duplicate':
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr(name,files[name]);z.writestr(name,files[name])
        raw=stream.getvalue()
    else:raw=sub._archive(files) if kind!='outside' else None
    if kind=='outside':
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for n,b in files.items():z.writestr(n,b)
        raw=stream.getvalue()
    with pytest.raises(sub.SubmissionError):
        pc.hydrate(prepared['metadata'],raw,project,limits)


def test_historical_capture_does_not_opt_in_implicitly(tmp_path):
    root, baseline, limits, prepared, project = closure(tmp_path)
    legacy=rc.policy({})
    assert legacy['project_evidence_externalization']==0
    assert legacy['max_project_raw_bytes']==33554432
    with pytest.raises(sub.SubmissionError,match='CAPTURE_OPT_IN_REQUIRED'):
        pc.hydrate(prepared['metadata'],prepared['evidence_path'].read_bytes(),project,legacy)
    for bad in [True,-1,2,'1']:
        with pytest.raises(sub.SubmissionError):rc.policy({'preservation':{'project_evidence_externalization':bad}})


def test_external_evidence_has_distinct_native_ack_obligation():
    o={'sha256':'a'*64,'size_bytes':1};e={'sha256':'b'*64,'size_bytes':2}
    row={'capture_id':'UNIT','objects':[o,e],'source':o,'state':o,'base_objects':[],
         'project':{'base':o,'delta':o,'evidence':e}}
    roles=pr._checkpoint_obligations('c'*64,row,b'unit')
    evidence=[r for r in roles if r['role']=='checkpoint_project_evidence']
    assert len(evidence)==1 and evidence[0]['sha256']==e['sha256']
    assert evidence[0]['obligation_scope']=='CHECKPOINT_OUTBOX'


def test_capsules_already_in_baseline_keep_exact_baseline_role(tmp_path):
    root, initial, limits = fixture(tmp_path)
    baseline = reg.snapshot(root)
    prepared = pc.prepare(root, baseline, limits, tmp_path)
    projected = pc.metadata_baseline(baseline, prepared['evidence_path'].read_bytes(), limits)
    delta = pr._project_delta(projected, prepared['metadata'])
    assert pr._restore_project(projected, delta, prepared['metadata_sha256']) == prepared['metadata']
    assert pc.hydrate(prepared['metadata'], prepared['evidence_path'].read_bytes(),
                      dict(prepared, schema_id=pc.SCHEMA), limits) == baseline
    assert prepared['metadata_raw_bytes'] < limits['max_project_raw_bytes']
