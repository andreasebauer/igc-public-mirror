"""Registered Drive transport fixtures; no claim of live OAuth authorization."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient
from ig_web.api import create_app
from ig_web.drive import Drive,session_url
from ig_web.models import Settings,Job
from ig_web.native import AdapterError,NativeAdapter,PIN_SOURCE
from ig_web.saves import Saves,NativeSave,process,identity
from ig_web.tracking import lane_lock
from ig_web.worker import Queue

TOKEN='synthetic-api-token-for-testing-123456'
DRIVE_ID='synthetic_drive_id_12345'

class SaveChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.q=Queue(self.root/'queue')
        self.token=self.root/'token.json';self.token.write_text(json.dumps({'access_token':'synthetic-drive-token-123456789'}));self.token.chmod(0o600)
        self.job=Job(id='J',name='Fixture',native_job_id='J',workspace=str(self.root/'workspace'))
        self.settings=Settings(**{k:str(self.root) for k in ('workspace_root','capture_store','specification_root','engine_repository')},engine_python=sys.executable,catalog=str(self.root/'catalog.json'),worker_state=str(self.q.root),drive_token_file=str(self.token))
        (self.root/'catalog.json').write_text(json.dumps({'jobs':[self.job.model_dump()]}));self.service=Saves(self.settings,self.q)
        self.client=TestClient(create_app(self.settings,TOKEN,self.q));self.headers={'Authorization':'Bearer '+TOKEN,'Idempotency-Key':'save-test-key'}
        self.bytes=b'captured fixture object';self.path=self.root/'object.bin';self.path.write_bytes(self.bytes);self.sha=hashlib.sha256(self.bytes).hexdigest()
        self.obj={'scope':'capture','obligation_id':'obligation-1','sha256':self.sha,'size_bytes':len(self.bytes),'role':'engine_source','logical_name':'engine.zip','local_object_path':str(self.path)};owner=self
        class Native:
            def __init__(s):s.rows=[dict(owner.obj)];s.confirmed=[]
            def pending(s):return list(s.rows)
            def path(s,obj):return owner.path
            def confirm(s,obj,readback,fid,directory):
                if hashlib.sha256(readback.read_bytes()).hexdigest()!=obj['sha256']:raise AssertionError('Unverified bytes')
                s.confirmed.append(identity(obj));s.rows=[o for o in s.rows if identity(o)!=identity(obj)]
        class Transport:
            def __init__(s):s.reserves=0;s.uploads=0;s.downloads=0;s.corrupt=False
            def reserve(s):s.reserves+=1;return DRIVE_ID
            def upload(s,path,state,persist):s.uploads+=1
            def download(s,fid,path,size):s.downloads+=1;path.write_bytes(b'wrong' if s.corrupt else owner.bytes)
        self.native=Native();self.transport=Transport()
    def tearDown(self):self.client.close();self.tmp.cleanup()
    def claim(self):
        self.service.submit(self.job,'save-test-key');return self.service.claim()
    def run(self,result=None):
        # Preserve unittest's runner method; project work uses execute below.
        return super().run(result)
    def execute(self):
        row=self.claim();process(self.service,row,self.transport,self.native);return row,self.service.get(row['id'])
    def test_unconfigured_is_pending_not_saved(self):
        settings=self.settings.model_copy(update={'drive_token_file':None});s=Saves(settings,self.q)
        with self.assertRaisesRegex(AdapterError,'DRIVE_NOT_CONFIGURED'):s.submit(self.job,'missing-config')
        self.assertEqual(s.history(self.job)['transport'],'UNCONFIGURED');self.assertEqual(len(self.native.pending()),1)
    def test_auth_strict_body_and_durable_duplicate(self):
        url='/api/v1/jobs/J/save-requests'
        self.assertEqual(self.client.post(url,json={}).status_code,401)
        self.assertEqual(self.client.post(url,headers=self.headers,json={'readback':'/tmp/claim'}).status_code,400)
        a=self.client.post(url,headers=self.headers,json={});b=self.client.post(url,headers=self.headers,json={});self.assertEqual(a.status_code,202);self.assertEqual(a.json(),b.json())
        s=Saves(self.settings,Queue(self.q.root));self.assertEqual(s.get(a.json()['request_id'])['status'],'queued')
    def test_different_key_active_save_refused(self):
        self.claim()
        with self.assertRaisesRegex(AdapterError,'SAVE_ALREADY_ACTIVE'):self.service.submit(self.job,'another-key')
    def test_same_key_changed_job_refused(self):
        self.claim()
        with self.assertRaisesRegex(AdapterError,'IDEMPOTENCY_CONFLICT'):self.service.submit(self.job.model_copy(update={'name':'changed'}),'save-test-key')
    def test_verified_bytes_before_native_confirmation(self):
        row,result=self.execute();self.assertEqual(result['status'],'finished');self.assertEqual(result['confirmed_obligations'],1);self.assertEqual(result['objects'][0]['phase'],'confirmed');self.assertEqual(result['native_remaining'],{'capture':0,'outbox':0});self.assertEqual(self.transport.downloads,1)
    def test_corrupt_readback_keeps_gate_pending_and_retry_same_id(self):
        self.transport.corrupt=True;row,result=self.execute();self.assertEqual(result['status'],'failed');self.assertEqual(result['error'],'SAVE_READBACK_MISMATCH');self.assertEqual(self.native.confirmed,[])
        self.transport.corrupt=False;self.service.retry(row['id']);process(self.service,self.service.claim(),self.transport,self.native)
        self.assertEqual(self.service.get(row['id'])['status'],'finished');self.assertEqual(self.transport.reserves,1);self.assertEqual(self.transport.uploads,1)
    def test_source_change_prevents_upload(self):
        self.path.write_bytes(b'changed');row,result=self.execute();self.assertEqual(result['error'],'SAVE_SOURCE_HASH_MISMATCH');self.assertEqual(self.transport.uploads,0);self.assertEqual(self.native.confirmed,[])
    def test_one_object_two_role_bound_confirmations(self):
        self.native.rows.append(dict(self.obj,scope='outbox',role='checkpoint_state',obligation_id='obligation-2'))
        row,result=self.execute();self.assertEqual(result['confirmed_obligations'],2);self.assertEqual(self.transport.uploads,1);self.assertEqual(self.transport.downloads,1)
    def test_native_refusal_not_save_success(self):
        with patch.object(self.native,'confirm',side_effect=AdapterError('NATIVE_SAVE_REFUSED')):row,result=self.execute()
        self.assertEqual(result['status'],'failed');self.assertEqual(result['confirmed_obligations'],0);self.assertEqual(len(self.native.pending()),1)
    def test_restart_marks_ambiguity_and_does_not_replay(self):
        row=self.claim();self.assertIsNone(self.service.claim());self.assertEqual(self.service.get(row['id'])['status'],'needs_reconciliation');self.assertEqual(self.transport.uploads,0)
        self.service.retry(row['id']);self.assertEqual(self.service.claim()['id'],row['id'])
    def test_retry_refused_while_worker_holds_lock(self):
        row=self.claim()
        with lane_lock(self.q,'save'):
            with self.assertRaisesRegex(AdapterError,'WORKER_LANE_ACTIVE'):self.service.retry(row['id'])
    def test_confirmation_crash_reconciles_native_state(self):
        original=self.native.confirm
        def crash(*args):original(*args);raise RuntimeError('synthetic crash after confirmation')
        with patch.object(self.native,'confirm',side_effect=crash):row,result=self.execute()
        self.assertEqual(result['status'],'failed');self.service.retry(row['id']);process(self.service,self.service.claim(),self.transport,self.native)
        self.assertEqual(self.service.get(row['id'])['status'],'finished');self.assertEqual(len(self.native.confirmed),1)
    def test_public_history_omits_session_paths_and_credentials(self):
        row=self.claim();row['detail']={'objects':[dict(self.obj,session='https://secret.example',drive_id=DRIVE_ID)],'confirmed':[]};self.service.write(row,'running')
        text=json.dumps(self.service.history(self.job))
        for secret in ('secret.example',str(self.root),'access_token','synthetic-drive-token'):self.assertNotIn(secret,text)
    def test_drive_credentials_private_and_session_host_bound(self):
        drive=Drive(self.settings);self.token.chmod(0o644)
        with self.assertRaisesRegex(AdapterError,'DRIVE_CREDENTIAL_PERMISSIONS'):drive.headers()
        for url in ('http://www.googleapis.com/upload/drive/v3/files?x=1','https://evil.example/upload/drive/v3/files?x=1','https://www.googleapis.com@evil.example/upload/drive/v3/files?x=1'):
            with self.assertRaises(AdapterError):session_url(url)
        drive.client.close()
    def test_drive_resumable_and_actual_download(self):
        received=[]
        def handle(request):
            self.assertEqual(request.headers['authorization'],'Bearer synthetic-drive-token-123456789')
            if request.url.path.endswith('generateIds'):return httpx.Response(200,json={'ids':[DRIVE_ID]})
            if request.method=='POST':return httpx.Response(200,headers={'location':'https://www.googleapis.com/upload/drive/v3/files?upload_id=fixture'})
            if request.method=='PUT':received.append(request.content);return httpx.Response(200,json={'id':DRIVE_ID})
            if request.url.params.get('alt')=='media':return httpx.Response(200,content=self.bytes)
            return httpx.Response(404)
        drive=Drive(self.settings,httpx.Client(transport=httpx.MockTransport(handle)));state={'drive_id':drive.reserve(),'sha256':self.sha};persisted=[]
        drive.upload(self.path,state,lambda:persisted.append(dict(state)));dest=self.root/'download';drive.download(DRIVE_ID,dest,len(self.bytes));self.assertEqual(received,[self.bytes]);self.assertEqual(dest.read_bytes(),self.bytes);self.assertTrue(persisted[0]['session']);drive.client.close()
    def test_expired_drive_authorization_no_upload(self):
        drive=Drive(self.settings,httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(401,content=b'secret response'))))
        with self.assertRaisesRegex(AdapterError,'DRIVE_AUTHORIZATION_REQUIRED'):drive.reserve()
        drive.client.close()
    def test_download_size_overrun_refused(self):
        drive=Drive(self.settings,httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,content=b'too long'))))
        with self.assertRaisesRegex(AdapterError,'DRIVE_READBACK_SIZE_MISMATCH'):drive.download(DRIVE_ID,self.root/'too-long',1)
        drive.client.close()
    def test_native_confirmation_exact_cli_and_pinned_environment(self):
        w=self.root/'workspace';(w/'source').mkdir(parents=True);(w/'WORKSPACE.json').write_text(json.dumps({'source_sha256':PIN_SOURCE}))
        with patch.object(NativeAdapter,'verify_source'),patch.object(NativeAdapter,'verify_runtime'):
            native=NativeSave(self.settings,self.job)
        obj=dict(self.obj,scope='outbox')
        with patch('ig_web.saves.subprocess.run',return_value=type('Result',(),{'returncode':0})()) as run:
            native.confirm(obj,self.path,DRIVE_ID,self.root)
            argv=run.call_args.args[0];self.assertEqual(argv[4:7],['preserve','confirm',str(w)]);self.assertEqual(argv[-6:],['--role',obj['role'],'--logical-name',obj['logical_name'],'--obligation-id',obj['obligation_id']]);self.assertNotIn('IG_WEB_TOKEN',run.call_args.kwargs['env'])
        with self.assertRaises(AdapterError):native.path(dict(obj,local_object_path='/etc/passwd'))
    def test_resuming_upload_queries_same_session(self):
        calls=[]
        def handle(r):
            calls.append(r)
            if r.method=='GET':return httpx.Response(404)
            if r.headers.get('content-range')=='bytes */'+str(len(self.bytes)):return httpx.Response(308,headers={'range':'bytes=0-3'})
            return httpx.Response(200,json={'id':DRIVE_ID})
        drive=Drive(self.settings,httpx.Client(transport=httpx.MockTransport(handle)));state={'drive_id':DRIVE_ID,'sha256':self.sha,'session':'https://www.googleapis.com/upload/drive/v3/files?upload_id=fixture'}
        drive.upload(self.path,state,lambda:None);self.assertEqual(calls[-1].content,self.bytes[4:]);self.assertFalse(any(r.method=='POST' for r in calls));drive.client.close()
    def test_ui_syntax_no_browser_save_attestation(self):
        p=Path(__file__).resolve().parents[1]/'ig_web/ui/saves.mjs';r=subprocess.run([shutil.which('node'),'--check',str(p)],capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr)
        for bad in ('innerHTML','localStorage','sessionStorage','access_token','readback_path'):self.assertNotIn(bad,p.read_text())
        self.assertIn('authorization is not configured',p.read_text());self.assertEqual(self.client.get('/ui/saves.mjs').status_code,200)
