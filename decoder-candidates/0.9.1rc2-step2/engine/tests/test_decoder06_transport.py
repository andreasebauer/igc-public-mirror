"""Data-only transport fixtures, executed by a saved native validation job.

Fixture IDs below identify no real Drive resources and authorize no source job.
The Step 8 integration separately confirms actual downloaded Drive objects.
"""
from pathlib import Path
import copy
import hashlib
import json
import lzma
import subprocess
import sys

import pytest
from infinity_grid import save_transport as tr
from infinity_grid import submission as sub
from infinity_grid import preservation as pr
from infinity_grid.canon import canonical_sha256, write_json_atomic


def fixture(root, raw=b'one two three four\n' * 100, encoding='RAW_CHUNKS', transport=None, chunk=113):
    root.mkdir(parents=True, exist_ok=True)
    encoded = transport if transport is not None else lzma.compress(raw) if encoding == 'XZ_CHUNKS' else raw
    parts = root / 'parts'; parts.mkdir()
    rows = []
    for offset in range(0, max(1, len(encoded)), chunk):
        data = encoded[offset:offset+chunk]; sha = hashlib.sha256(data).hexdigest()
        (parts/sha).write_bytes(data)
        rows.append({'sha256': sha, 'size_bytes': len(data), 'drive_file_id': 'data_fixture_part_only'})
    manifest = {'schema_id': tr.SCHEMA, 'object': {'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)},
                'encoding': encoding, 'parts': rows}
    path = root/'manifest.json'; write_json_atomic(path, manifest)
    return manifest, path, parts


@pytest.mark.parametrize('encoding', ['RAW_CHUNKS', 'XZ_CHUNKS'])
@pytest.mark.parametrize('raw', [b'', b'0123456789'*1000])
def test_verified_reconstruction_and_receipt_keep_transport_identity(tmp_path, encoding, raw):
    manifest, path, parts = fixture(tmp_path, raw, encoding)
    output = tmp_path/'decoded'
    assert tr.unpack(path, parts, output)['sha256'] == manifest['object']['sha256']
    assert output.read_bytes() == raw
    receipt = tr.make_receipt(manifest['object'], path, parts, 'data_fixture_manifest_only')
    assert receipt['raw_readback_verified'] is False and receipt['object_bytes_verified'] is True
    assert receipt['transport']['manifest'] == manifest
    rp=tmp_path/'receipt.json'; write_json_atomic(rp, receipt)
    assert sub._valid_receipt(rp, manifest['object'])
    receipt['raw_readback_verified'] = True
    receipt['receipt_sha256'] = canonical_sha256({k:v for k,v in receipt.items() if k!='receipt_sha256'})
    write_json_atomic(rp, receipt)
    assert not sub._valid_receipt(rp, manifest['object'])


def test_corrupt_missing_reordered_parts_leave_no_output(tmp_path):
    manifest,path,parts=fixture(tmp_path, bytes(range(256))*20)
    first=parts/manifest['parts'][0]['sha256']; original=first.read_bytes()
    for mode in ('corrupt','missing','reorder'):
        current=copy.deepcopy(manifest)
        if mode=='corrupt':first.write_bytes(b'x'*len(original))
        elif mode=='missing':first.unlink()
        else:current['parts'].reverse()
        write_json_atomic(path,current)
        with pytest.raises(sub.SubmissionError):tr.unpack(path,parts,tmp_path/'decoded')
        assert not (tmp_path/'decoded').exists()
        first.write_bytes(original)


@pytest.mark.parametrize('mode', ['truncated','trailing','concatenated','overflow','wrong_hash','invalid'])
def test_xz_failures_are_refused_without_ack(tmp_path,mode):
    raw=b'bounded decoded fixture'*10000; encoded=lzma.compress(raw)
    if mode=='truncated':encoded=encoded[:-4]
    elif mode=='trailing':encoded+=b'x'
    elif mode=='concatenated':encoded+=lzma.compress(b'')
    elif mode=='invalid':encoded=b'not an XZ stream'
    manifest,path,parts=fixture(tmp_path,raw,'XZ_CHUNKS',transport=encoded)
    if mode=='overflow':manifest['object']['size_bytes']=10
    elif mode=='wrong_hash':manifest['object']['sha256']='0'*64
    write_json_atomic(path,manifest)
    with pytest.raises(sub.SubmissionError):tr.make_receipt(manifest['object'],path,parts,'data_fixture_manifest_only')


def test_manifest_binding_and_unsafe_parts_refuse(tmp_path):
    manifest,path,parts=fixture(tmp_path)
    with pytest.raises(sub.SubmissionError,match='BINDING'):
        tr.make_receipt({'sha256':'0'*64,'size_bytes':manifest['object']['size_bytes']},path,parts,'data_fixture_manifest_only')
    part=parts/manifest['parts'][0]['sha256']; outside=tmp_path/'outside'; outside.write_bytes(part.read_bytes())
    part.unlink();part.symlink_to(outside)
    with pytest.raises(sub.SubmissionError,match='UNSAFE'):tr.verify(manifest,parts)
    for key,value in [('drive_file_id',None),('sha256','../outside'),('size_bytes',True)]:
        bad=copy.deepcopy(manifest);bad['parts'][0][key]=value
        with pytest.raises(sub.SubmissionError):tr.validate_manifest(bad)


@pytest.mark.parametrize('encoding',['RAW_CHUNKS','XZ_CHUNKS'])
def test_pack_bind_and_unpack_native_cli(tmp_path,encoding):
    source=tmp_path/'input';source.write_bytes(bytes(range(256))*20)
    def cli(*args):
        run=subprocess.run([sys.executable,'-m','infinity_grid.controller','transport',*map(str,args)],capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr
        return json.loads(run.stdout)
    packed=cli('pack',source,tmp_path/'packed','--encoding',encoding,'--chunk-bytes','73')
    manifest=packed['manifest'];assert packed['status']=='TRANSPORT_PREPARED_NOT_SAVED'
    assert max(p['size_bytes'] for p in manifest['parts'])<=73
    assert all(p['drive_file_id'] is None for p in manifest['parts'])
    mapping={p['sha256']:'data_fixture_part_only' for p in manifest['parts']}
    write_json_atomic(tmp_path/'mapping.json',mapping)
    cli('bind',tmp_path/'packed/MANIFEST_DRAFT.json',tmp_path/'mapping.json',tmp_path/'bound.json')
    cli('unpack',tmp_path/'bound.json',tmp_path/'packed',tmp_path/'output')
    assert (tmp_path/'output').read_bytes()==source.read_bytes()
    with pytest.raises(sub.SubmissionError,match='DESTINATION_EXISTS'):
        tr.unpack(tmp_path/'bound.json',tmp_path/'packed',tmp_path/'output')


def test_legacy_receipt_still_valid(tmp_path):
    manifest,_,_=fixture(tmp_path)
    receipt={'schema_id':'IG_DECODER_EXPLICIT_SAVE_RECEIPT_V1','provider':'google_drive',**manifest['object'],
             'drive_file_id':'data_fixture_raw_only','raw_readback_verified':True}
    receipt['receipt_sha256']=canonical_sha256(receipt)
    path=tmp_path/'receipt.json';write_json_atomic(path,receipt)
    assert sub._valid_receipt(path,manifest['object'])


def test_native_outbox_accepts_transport_and_retains_raw_upload_note(tmp_path):
    from tests.test_decoder06_results_preservation import queued_fixture
    workspace=tmp_path/'workspace';obj,entry=queued_fixture(workspace)
    obligation = next(row for row in pr.status(workspace)['pending_objects'] if row['role'] == 'checkpoint_source')
    raw=(pr._root(workspace)/'objects'/(obj['sha256']+'.bin')).read_bytes()
    manifest,path,parts=fixture(tmp_path/'transport',raw,'XZ_CHUNKS')
    pr.note_upload(workspace,obj['sha256'],'data_fixture_raw_only')
    manifest['original_object_drive_file_id']='data_fixture_wrong_original'
    write_json_atomic(path,manifest)
    with pytest.raises(sub.SubmissionError,match='UPLOAD_ID_MISMATCH'):
        pr.confirm_transport(workspace,obj['sha256'],path,parts,'data_fixture_manifest_only', role='checkpoint_source')
    assert not list((pr._root(workspace)/'receipts').rglob('*.json'))
    manifest['original_object_drive_file_id']='data_fixture_raw_only';write_json_atomic(path,manifest)
    status=pr.confirm_transport(workspace,obj['sha256'],path,parts,'data_fixture_manifest_only', role='checkpoint_source')
    assert {r['sha256'] for r in status['pending_objects']} == {obj['sha256'], entry['sha256']}
    assert len(status['pending_objects']) == 4
    assert not any(r['role'] == 'checkpoint_source' for r in status['pending_objects'])
    assert sub._read(pr._root(workspace)/'uploads'/(obj['sha256']+'.json'))['drive_file_id']=='data_fixture_raw_only'
    assert pr._receipt(workspace,obligation)['raw_readback_verified'] is False


def test_native_outbox_transport_requires_and_accepts_exact_obligation_selector(tmp_path,monkeypatch):
    workspace=tmp_path/'workspace'
    raw=b'same physical bytes, two logical checkpoint obligations'
    obj=pr._put(workspace,raw)
    common=dict(obj,obligation_scope='CHECKPOINT_OUTBOX')
    first=dict(common,role='project_baseline',logical_name='baseline.zip',obligation_id='a'*64)
    second=dict(common,role='checkpoint_project_base',logical_name=obj['sha256'],obligation_id='b'*64)
    pending={'pending_objects':[first,second]}
    monkeypatch.setattr(pr,'status',lambda _:pending)
    manifest,path,parts=fixture(tmp_path/'transport',raw,'XZ_CHUNKS')

    with pytest.raises(sub.SubmissionError,match='OUTBOX_OBLIGATION_AMBIGUOUS'):
        pr.confirm_transport(workspace,obj['sha256'],path,parts,'data_fixture_manifest_only')
    with pytest.raises(sub.SubmissionError,match='OUTBOX_OBLIGATION_NOT_PENDING'):
        pr.confirm_transport(workspace,obj['sha256'],path,parts,'data_fixture_manifest_only',
                             obligation_id='c'*64)

    pr.confirm_transport(workspace,obj['sha256'],path,parts,'data_fixture_manifest_only',
                         role=second['role'],logical_name=second['logical_name'],
                         obligation_id=second['obligation_id'])
    receipts=list((pr._root(workspace)/'receipts'/obj['sha256']).glob('*.json'))
    assert len(receipts)==1
    receipt=sub._read(receipts[0])
    assert receipt['obligation_id']==second['obligation_id']
    assert receipt['role']==second['role']
    assert receipt['logical_name']==second['logical_name']


def test_native_capture_confirms_only_required_object(tmp_path):
    from tests.test_decoder06_capture import _spec
    capture=sub.capture(tmp_path/'store',_spec(tmp_path,project=False))
    obj=capture['pending_objects'][0];raw=Path(obj['local_object_path']).read_bytes()
    manifest,path,parts=fixture(tmp_path/'transport',raw,'XZ_CHUNKS',chunk=65536)
    before=len(capture['pending_objects'])
    after=sub.confirm_transport(capture['workspace'],obj['sha256'],path,parts,'data_fixture_manifest_only')
    assert len(after['pending_objects'])==before-1 and after['status']=='SAVE_REQUIRED'
    assert not (Path(capture['workspace'])/'runtime/attempts').exists()
    with pytest.raises(sub.SubmissionError,match='NOT_REQUIRED'):
        sub.confirm_transport(capture['workspace'],'0'*64,path,parts,'data_fixture_manifest_only')
