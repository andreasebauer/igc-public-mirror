"""Synthetic native API fixtures; their identifiers are not remote-save evidence."""
from pathlib import Path
import copy
import hashlib
import json
import time

import pytest
from infinity_grid import preservation as pr, preservation_batch as batch
from infinity_grid import submission as sub, portable_registry
from infinity_grid.canon import canonical_sha256, write_json_atomic


def fixture(tmp_path, commits=2):
    from tests.test_decoder06_capture import _spec
    spec=_spec(tmp_path,project=False)
    store=tmp_path/'store';portable_registry.initialize(store,'BATCH_UNIT_FIXTURE',spec['engine_source'])
    capture=sub.capture(store,spec);root=Path(capture['workspace'])
    obj=pr._put(root,b'synthetic shared checkpoint bytes');previous=None
    for i in range(commits):
        row={'schema_id':pr.SCHEMA,'previous':previous,'objects':[obj],
             'created_unix':time.time(),'capture_id':capture['capture_id'],
             'source':obj,'state':obj,'project':{'base':obj,'delta':obj},'base_objects':[],
             'reason':'SYNTHETIC_BATCH_FIXTURE_'+str(i)}
        raw=sub._json_bytes(row);entry=pr._put(root,raw)
        path=pr._root(root)/'commits'/(entry['sha256']+'.json')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        previous=entry['sha256']
    write_json_atomic(pr._root(root)/'CURRENT.json',{'sha256':previous})
    state=pr.status(root)
    obligations=[{k:r[k] for k in batch.IDENTITY} for r in state['pending_objects']]
    readbacks=[]
    for sha in sorted({r['sha256'] for r in obligations}):
        path=tmp_path/('readback-'+sha);path.write_bytes((pr._root(root)/'objects'/(sha+'.bin')).read_bytes())
        readbacks.append({'sha256':sha,'kind':'RAW','path':str(path),'drive_file_id':'synthetic_batch_readback'})
    request={'schema_id':batch.SCHEMA,'capture_id':capture['capture_id'],
             'obligations':obligations,'readbacks':readbacks}
    return root,request,obj


def save(tmp_path, request):
    path=tmp_path/'batch.json';write_json_atomic(path,request);return path


def test_batch_checks_distinct_objects_once_and_closes_each_role(tmp_path,monkeypatch):
    root,request,obj=fixture(tmp_path)
    seen=[];original=pr._bytes
    def traced(workspace,row):
        seen.append(row['sha256']);return original(workspace,row)
    monkeypatch.setattr(pr,'_bytes',traced)
    result=pr.confirm_batch(root,save(tmp_path,request))
    assert len(seen)==len(set(seen))==3
    assert result['obligations_acknowledged']==10 and result['readbacks_verified']==3
    assert result['outbox']['status']=='PRESERVED'
    assert result['remote_identity_authenticated'] is False
    before={str(p):p.read_bytes() for p in (pr._root(root)/'receipts').rglob('*.json')}
    again=pr.confirm_batch(root,save(tmp_path,request))
    assert again['batch_id']==result['batch_id']
    assert {str(p):p.read_bytes() for p in (pr._root(root)/'receipts').rglob('*.json')}==before


@pytest.mark.parametrize('mode',['role','capture','duplicate','missing','wrong_bytes','extra_reference','drive_id'])
def test_invalid_batch_publishes_no_receipt_or_prepared_journal(tmp_path,mode):
    root,request,obj=fixture(tmp_path)
    if mode=='role':request['obligations'][0]['role']='wrong'
    elif mode=='capture':request['capture_id']='0'*64
    elif mode=='duplicate':request['obligations'].append(copy.deepcopy(request['obligations'][0]))
    elif mode=='missing':request['readbacks'].pop()
    elif mode=='wrong_bytes':Path(request['readbacks'][-1]['path']).write_bytes(b'wrong')
    elif mode=='extra_reference':request['readbacks'].append(copy.deepcopy(request['readbacks'][0]))
    else:request['readbacks'][0]['drive_file_id']='invalid'
    with pytest.raises(sub.SubmissionError):pr.confirm_batch(root,save(tmp_path,request))
    assert not list((pr._root(root)/'receipts').rglob('*.json'))
    assert not list((pr._root(root)/'batches').rglob('PREPARED.json'))


def test_interrupted_batch_retains_partial_and_resumes_same_receipts(tmp_path,monkeypatch):
    root,request,obj=fixture(tmp_path);path=save(tmp_path,request)
    original=batch._publish;count=0
    def interrupt(target,row):
        nonlocal count
        original(target,row)
        if 'receipts' in target.parts:
            count+=1
            if count==2:raise OSError('injected publication interruption')
    monkeypatch.setattr(batch,'_publish',interrupt)
    with pytest.raises(OSError,match='injected'):pr.confirm_batch(root,path)
    retained={str(p):p.read_bytes() for p in (pr._root(root)/'receipts').rglob('*.json')}
    assert len(retained)==2 and not list((pr._root(root)/'batches').rglob('COMPLETE.json'))
    monkeypatch.setattr(batch,'_publish',original)
    assert pr.confirm_batch(root,path)['outbox']['status']=='PRESERVED'
    assert all(Path(p).read_bytes()==raw for p,raw in retained.items())


def test_multipart_batch_has_no_raw_readback_claim_and_deduplicates_reserve(tmp_path):
    from tests.test_decoder06_transport import fixture as multipart
    root,request,obj=fixture(tmp_path)
    ref=next(r for r in request['readbacks'] if r['sha256']==obj['sha256'])
    manifest,path,parts=multipart(tmp_path/'parts-fixture',Path(ref['path']).read_bytes(),encoding='XZ_CHUNKS')
    ref.clear();ref.update(sha256=obj['sha256'],kind='MULTIPART',manifest=str(path),parts=str(parts),drive_file_id='synthetic_manifest_readback')
    # Only one logical role gets a receipt; other roles stay pending.
    request['obligations']=[next(r for r in request['obligations'] if r['sha256']==obj['sha256'])]
    request['readbacks']=[ref]
    before=pr.status(root)
    result=pr.confirm_batch(root,save(tmp_path,request))
    after=pr.status(root)
    assert len(after['pending_objects'])==len(before['pending_objects'])-1
    assert after['pending_bytes']==before['pending_bytes']-obj['size_bytes']
    receipt=json.loads(next((pr._root(root)/'receipts'/obj['sha256']).glob('*.json')).read_text())
    assert receipt['raw_readback_verified'] is False and receipt['object_bytes_verified'] is True
    assert result['outbox']['physical_objects_verified']==1


def test_corrupt_multipart_cannot_partially_acknowledge_batch(tmp_path):
    from tests.test_decoder06_transport import fixture as multipart
    root,request,obj=fixture(tmp_path)
    ref=next(r for r in request['readbacks'] if r['sha256']==obj['sha256'])
    manifest,path,parts=multipart(tmp_path/'parts-fixture',Path(ref['path']).read_bytes())
    (parts/manifest['parts'][0]['sha256']).write_bytes(b'corrupt')
    ref.clear();ref.update(sha256=obj['sha256'],kind='MULTIPART',manifest=str(path),parts=str(parts),drive_file_id='synthetic_manifest_readback')
    with pytest.raises(sub.SubmissionError):pr.confirm_batch(root,save(tmp_path,request))
    assert not list((pr._root(root)/'receipts').rglob('*.json'))


def test_new_status_rechecks_corruption_after_successful_batch(tmp_path):
    root,request,obj=fixture(tmp_path)
    pr.confirm_batch(root,save(tmp_path,request))
    (pr._root(root)/'objects'/(obj['sha256']+'.bin')).write_bytes(b'corrupt')
    with pytest.raises(sub.SubmissionError,match='OUTBOX_OBJECT_MISMATCH'):pr.status(root)


def test_batch_limits_refuse_before_publication(tmp_path,monkeypatch):
    root,request,obj=fixture(tmp_path)
    monkeypatch.setattr(batch,'MAX_READBACK_BYTES',1)
    with pytest.raises(sub.SubmissionError,match='READBACK_LIMIT'):
        pr.confirm_batch(root,save(tmp_path,request))
    assert not list((pr._root(root)/'batches').rglob('*.json'))


def test_corrupted_prepared_journal_is_not_repaired_silently(tmp_path,monkeypatch):
    root,request,obj=fixture(tmp_path);path=save(tmp_path,request);original=batch._publish
    def interrupt(target,row):
        original(target,row)
        if target.name=='PREPARED.json':raise OSError('injected')
    monkeypatch.setattr(batch,'_publish',interrupt)
    with pytest.raises(OSError):pr.confirm_batch(root,path)
    journal=next((pr._root(root)/'batches').rglob('PREPARED.json'))
    saved=json.loads(journal.read_text());saved['receipts'][0]['role']='wrong'
    write_json_atomic(journal,saved)
    monkeypatch.setattr(batch,'_publish',original)
    with pytest.raises(sub.SubmissionError,match='JOURNAL_RECEIPT'):pr.confirm_batch(root,path)
    assert not list((pr._root(root)/'receipts').rglob('*.json'))
