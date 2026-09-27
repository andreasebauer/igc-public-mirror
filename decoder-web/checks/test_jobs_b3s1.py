"""Registered Jobs UI/API engineering tests; fixtures are synthetic."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from ig_web.api import create_app
from ig_web.models import Settings
from ig_web.native import Command, PIN_SOURCE, canonical_hash
from ig_web.overview import overview
from ig_web.worker import Queue, command_digest

TOKEN='synthetic-test-token-not-a-secret-1234'

class JobsChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.settings=Settings(**{k:str(self.root) for k in ('engine_repository','engine_python','workspace_root','specification_root','capture_store')},catalog=str(self.root/'catalog.json'))
        self.queue=Queue(self.root/'queue');self.jobs=[];self.write_catalog()
        self.client=TestClient(create_app(self.settings,TOKEN,gateway=self.queue));self.headers={'Authorization':'Bearer '+TOKEN}
    def tearDown(self):self.client.close();self.temp.cleanup()
    def write_catalog(self):(self.root/'catalog.json').write_text(json.dumps({'jobs':self.jobs}))
    def job(self,name='Fixture',jid='JOB'):
        workspace=self.root/jid;(workspace/'registry').mkdir(parents=True)
        reg={'schema_id':'IG_DECODER_WORKSPACE_JOB_V1','job_id':jid,'source_sha256':PIN_SOURCE,'question':{'description':'Synthetic question; no science'},'execution':{'kind':'VALIDATION'},'input_artifacts':[]}
        reg['registration_sha256']=canonical_hash(reg);(workspace/'registry'/f'{jid}.json').write_text(json.dumps(reg))
        job={'id':jid,'name':name,'native_job_id':jid,'workspace':str(workspace)};self.jobs.append(job);self.write_catalog();return job
    def test_static_shell_and_api_auth(self):
        r=self.client.get('/ui/');self.assertEqual(r.status_code,200);self.assertIn('Jobs · Decoder',r.text)
        for asset in ('app.mjs','style.css'):self.assertEqual(self.client.get('/ui/'+asset).status_code,200)
        self.assertEqual(self.client.get('/api/v1/job-overview').status_code,401)
        self.assertNotIn(TOKEN,r.text);self.assertIn("frame-ancestors 'none'",r.headers['content-security-policy'])
        self.assertEqual(r.headers['cache-control'],'no-store')
    def test_empty_is_not_error(self):
        r=self.client.get('/api/v1/job-overview',headers=self.headers);self.assertEqual(r.status_code,200);self.assertEqual(r.json()['items'],[])
        (self.root/'catalog.json').unlink();self.assertEqual(self.client.get('/api/v1/job-overview',headers=self.headers).status_code,503)
    def test_search_filter_and_pagination(self):
        self.job('Alpha','A');b=self.job('Beta','B');self.job('Gamma','C')
        cmd=Command('run','B',('python','run',b['workspace'],'B'),str(self.root),PIN_SOURCE);self.queue.submit(cmd,'synthetic-key',command_digest(cmd))
        r=overview(self.settings,self.queue);self.assertEqual(r['items'][0]['id'],'B');self.assertEqual(r['items'][0]['execution'],'Queued')
        self.assertEqual(overview(self.settings,self.queue,query='alpha')['total'],1)
        self.assertEqual(overview(self.settings,self.queue,filter_by='active')['total'],1)
        self.assertEqual(len(overview(self.settings,self.queue,offset=1,limit=1)['items']),1)
    def test_registration_corruption_is_attention(self):
        j=self.job();(Path(j['workspace'])/'registry/JOB.json').write_text('{}')
        r=overview(self.settings,self.queue)['items'][0];self.assertEqual(r['execution'],'Unknown');self.assertEqual(r['category'],'attention')
    def test_rejected_result_remains_attention(self):
        self.job()
        with patch('ig_web.overview.results_observation',return_value={'record_status':'PUBLISHED_RECORD','reported':{'status':'RESULT_REJECTED','evidence_status':'REJECTED'}}):
            r=overview(self.settings,self.queue)['items'][0]
        self.assertEqual(r['category'],'attention');self.assertEqual(r['backup'],'Unknown')
    def test_running_request_is_not_native_running(self):
        j=self.job();cmd=Command('run','JOB',('python','run',j['workspace'],'JOB'),str(self.root),PIN_SOURCE)
        rid=self.queue.submit(cmd,'synthetic-key',command_digest(cmd)).request_id;self.queue.update(rid,status='running')
        r=overview(self.settings,self.queue)['items'][0];self.assertEqual(r['execution'],'Run requested');self.assertEqual(r['state_source'],'Web request')
    def test_completed_pause_does_not_hide_active_run(self):
        j=self.job();run=Command('run','JOB',('python','run',j['workspace'],'JOB'),str(self.root),PIN_SOURCE)
        rid=self.queue.submit(run,'run-key-001',command_digest(run)).request_id;self.queue.update(rid,status='running')
        pause=Command('pause','JOB',('python','preserve','pause',j['workspace'],'reason'),str(self.root),PIN_SOURCE)
        pid=self.queue.submit(pause,'pause-key-001',command_digest(pause)).request_id;self.queue.update(pid,status='finished')
        self.assertEqual(overview(self.settings,self.queue)['items'][0]['execution'],'Run requested')
    def test_route_limits(self):
        for q in ('limit=101','offset=-1','filter_by=invalid','query='+'x'*201):self.assertEqual(self.client.get('/api/v1/job-overview?'+q,headers=self.headers).status_code,400)
    def test_browser_projection_logic(self):
        module=Path(__file__).resolve().parents[1]/'ig_web/ui/app.mjs'
        script="""import assert from 'node:assert/strict';
const {validateOverview,nativeExecution,backupState,timeLabel}=await import(process.argv[1]);
assert.throws(()=>validateOverview({items:[],total:-1,counts:{}}));
assert.equal(nativeExecution({classification:'RESPONSE',native:{workspace_locked_now:false,last_record:{job_id:'J',status:'RUNNING'}}},'J'),'Needs reconciliation');
assert.equal(nativeExecution({classification:'RESPONSE',native:{workspace_locked_now:true,last_record:{job_id:'OTHER',status:'RUNNING'}}},'J'),'Workspace busy · job unconfirmed');
assert.equal(backupState({}),'Unknown');
assert.equal(backupState({capture:{classification:'RESPONSE',native:{pending_objects:[]}},outbox:{classification:'RESPONSE',native:{status:'PRESERVED',pending_objects:[],pending_checkpoints:[]}}}),'Native reports preserved');
assert.equal(backupState({capture:{classification:'RESPONSE',native:{pending_objects:[{}]}},outbox:{classification:'RESPONSE',native:{status:'PRESERVED',pending_objects:[],pending_checkpoints:[]}}}),'Backup pending');
assert.equal(timeLabel(null),'Activity time unknown');
console.log('B3S1 JavaScript projection assertions PASS; Node '+process.version);
"""
        node=shutil.which('node');self.assertIsNotNone(node,'Node toolchain must be installed for registered UI checks')
        result=subprocess.run([node,'--input-type=module','-e',script,module.as_uri()],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);print(result.stdout)
    def test_no_client_persistence_or_html_injection(self):
        root=Path(__file__).resolve().parents[1]/'ig_web/ui';script=(root/'app.mjs').read_text()
        for unsafe in ('localStorage','sessionStorage','innerHTML','insertAdjacentHTML'):self.assertNotIn(unsafe,script)
        self.assertIn('Execution may still continue',script);self.assertIn('state.epoch',script)
        html=(root/'index.html').read_text();self.assertIn('viewport-fit=cover',html);self.assertIn('aria-pressed',html)
