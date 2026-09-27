"""Registered Run-screen engineering checks with synthetic observations."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from ig_web.api import create_app
from ig_web.models import Settings
from ig_web.native import AdapterError,Command,PIN_SOURCE,canonical_hash
from ig_web.runview import project
from ig_web.worker import Queue,command_digest

TOKEN='synthetic-test-token-not-a-secret-1234'
def observations(record=None,busy=False):
    return {'classification':'RESPONSE','native':{'workspace_locked_now':busy,'last_record':record}}
def backup(pending=False):
    return {'capture':{'classification':'RESPONSE','native':{'status':'SAVED','pending_objects':[{}] if pending else []}},'outbox':{'classification':'RESPONSE','native':{'status':'PRESERVED','pending_objects':[],'pending_checkpoints':[]}}}
def row(operation='run',status='running',created=1):return {'id':'req-fixture','operation':operation,'status':status,'created':created,'updated':created,'error':None}
class RunChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);(self.root/'registry').mkdir();(self.root/'source').mkdir()
        self.reg={'schema_id':'IG_DECODER_WORKSPACE_JOB_V1','job_id':'J','source_sha256':PIN_SOURCE,'question':{'description':'Synthetic Run fixture'},'execution':{'kind':'VALIDATION'},'input_artifacts':[]};self.reg['registration_sha256']=canonical_hash(self.reg);(self.root/'registry/J.json').write_text(json.dumps(self.reg))
        self.job={'id':'J','name':'Fixture','native_job_id':'J','workspace':str(self.root)};(self.root/'catalog.json').write_text(json.dumps({'jobs':[self.job]}))
        self.settings=Settings(**{k:str(self.root) for k in ('engine_repository','engine_python','workspace_root','specification_root','capture_store')},catalog=str(self.root/'catalog.json'));self.q=Queue(self.root/'queue');self.native=observations();self.saved=backup();self.runtime_error=None
        owner=self
        class Adapter:
            def observe(s,op,job):return owner.native if op=='status' else owner.saved['capture' if op=='pending-saves' else 'outbox']
            def command(s,op,job,reason=None):return Command(op,job.id,('python',op,job.workspace,reason if op=='pause' else job.native_job_id),str(owner.root),PIN_SOURCE)
            def verify_source(s,source):return {'status':'BYTE_EXACT_SOURCE_PASS'}
            def verify_runtime(s):
                if owner.runtime_error:raise AdapterError(owner.runtime_error)
                return {}
        self.client=TestClient(create_app(self.settings,TOKEN,self.q,Adapter()));self.headers={'Authorization':'Bearer '+TOKEN}
    def tearDown(self):self.client.close();self.temp.cleanup()
    def projection(self,record=None,busy=False,requests=None,saved=None,result='ABSENT'):
        return project('J',observations(record,busy),saved or backup(),requests or [],{'record_status':result})
    def view(self):
        r=self.client.get('/api/v1/jobs/J/run-view',headers=self.headers);self.assertEqual(r.status_code,200,r.text);return r.json()
    def action(self,v,action='start',reason='Pause fixture'):
        return self.client.post('/api/v1/jobs/J/run-actions',headers=self.headers,json={'action':action,'view_token':v['view_token'],'reason':reason})
    def test_prepared_is_requestable_not_native_ready(self):
        v=self.view();self.assertEqual(v['state']['action'],'start');self.assertEqual(v['state']['execution'],'Prepared');self.assertIn('authoritative',v['admission'])
    def test_unsaved_and_unknown_backup_block(self):
        self.assertIsNone(self.projection(saved=backup(True))['action']);self.saved={'capture':{},'outbox':{}};self.assertIsNone(self.view()['state']['action'])
    def test_runtime_gate_blocks_start(self):
        self.runtime_error='ENGINE_RUNTIME_MISMATCH';v=self.view();self.assertIsNone(v['state']['action']);self.assertEqual(v['state']['reason'],self.runtime_error)
    def test_stale_running_and_other_job_busy_block(self):
        for rec,busy in [({'job_id':'J','status':'RUNNING'},False),({'job_id':'OTHER','status':'RUNNING'},True)]:self.assertIsNone(self.projection(rec,busy)['action'])
    def test_pause_delivery_is_not_acknowledgment(self):
        v=self.projection({'job_id':'J','status':'RUNNING'},True,[row(),row('pause','finished',2)]);self.assertEqual(v['execution'],'Pause requested');self.assertIsNone(v['action'])
    def test_operator_pause_only_resume(self):
        self.assertEqual(self.projection({'job_id':'J','status':'PAUSED','reason':'SubmissionError:REQUESTED_PAUSE:fixture'})['action'],'resume')
        for reason in ('AUDIT_STOP','SubmissionError:SOURCE_CHANGED','ControllerLoopError:VALIDATION_INTERRUPTED_BEFORE_COMPLETION'):self.assertIsNone(self.projection({'job_id':'J','status':'PAUSED','reason':reason})['action'])
    def test_terminal_and_pending_completion_block_start(self):
        for result in ('PUBLISHED_RECORD','PENDING_CHECKPOINT_RECORD','UNAVAILABLE'):self.assertIsNone(self.projection(result=result)['action'])
    def test_request_uncertainty_blocks_new_start(self):
        for state in ('queued','dispatching','running','needs_reconciliation','interrupted','finished','refused'):self.assertIsNone(self.projection(requests=[row(status=state)])['action'])
    def test_lost_response_retries_one_request(self):
        v=self.view();a=self.action(v);self.assertEqual(a.status_code,202,a.text);b=self.action(v);self.assertEqual(a.json(),b.json());self.assertEqual(self.view()['requests'][0]['id'],a.json()['request_id'])
        with self.q.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0],1)
    def test_changed_view_and_unavailable_action_refused(self):
        v=self.view();self.saved=backup(True);self.assertEqual(self.action(v).status_code,409);v=self.view();self.assertEqual(self.action(v).status_code,409)
    def test_pause_uses_control_lane_and_idempotency(self):
        rec={'job_id':'J','status':'RUNNING','source_sha256':PIN_SOURCE,'registration_sha256':self.reg['registration_sha256']};self.native=observations(rec,True);v=self.view();self.assertEqual(v['state']['action'],'pause');r=self.action(v,'pause');self.assertEqual(r.status_code,202,r.text);self.assertEqual(self.q.claim('control')['id'],r.json()['request_id']);self.assertEqual(self.action(v,'pause').json()['request_id'],r.json()['request_id']);self.assertEqual(self.action(v,'pause','Different reason').status_code,409)
    def test_status_identity_mismatch_refuses_actions(self):
        self.native=observations({'job_id':'J','status':'RUNNING','registration_sha256':'wrong','source_sha256':PIN_SOURCE},True);self.assertIsNone(self.view()['state']['action'])
    def test_capture_link_requires_committed_index(self):
        cmd=Command('capture','T',('python','capture'),str(self.root),PIN_SOURCE,'0'*64);rid=self.q.submit(cmd,'capture-key',command_digest(cmd)).request_id;url='/api/v1/requests/'+rid+'/job';self.assertEqual(self.client.get(url,headers=self.headers).status_code,409);self.q.update(rid,status='finished',native=json.dumps({'capture_id':'c'*64}));self.assertEqual(self.client.get(url,headers=self.headers).status_code,409)
        with self.q.transaction() as db:db.execute('INSERT INTO jobs VALUES (?,?,?,?,?)',('J','c'*64,self.reg['registration_sha256'],json.dumps(self.job),1))
        self.assertEqual(self.client.get(url,headers=self.headers).json(),{k:self.job[k] for k in ('id','name','native_job_id')})
    def test_authenticated_typed_routes(self):
        self.assertEqual(self.client.get('/api/v1/jobs/J/run-view').status_code,401);self.assertEqual(self.client.post('/api/v1/jobs/J/run-actions',headers=self.headers,json={'action':'delete','view_token':'x'}).status_code,400)
    def test_javascript_helpers_and_safe_assets(self):
        module=Path(__file__).resolve().parents[1]/'ig_web/ui/run.mjs';script="import assert from 'node:assert/strict';const {elapsed,explanation}=await import(process.argv[1]);assert.equal(elapsed({},5),'Unknown');assert.equal(elapsed({started_unix:10,finished_unix:75},90),'1m 5s');assert.equal(elapsed({started_unix:10},9),'Unknown');assert.match(explanation('NATIVE_STOP_REQUIRES_REVIEW'),/blocked/);"
        r=subprocess.run([shutil.which('node'),'--input-type=module','-e',script,module.as_uri()],capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr)
        s=module.read_text()
        for bad in ('localStorage','sessionStorage','innerHTML'):self.assertNotIn(bad,s)
        self.assertIn('s.pending',s);self.assertIn('epoch!==s.epoch',s);self.assertIn('offset=',s);self.assertEqual(self.client.get('/ui/run.mjs').status_code,200)
