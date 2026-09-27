"""Registered synthetic engineering checks; not scientific result qualification."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from ig_web.api import create_app
from ig_web.models import Settings
from ig_web.native import AdapterError, Command, PIN_SOURCE, canonical_hash
from ig_web.worker import Queue, command_digest
from ig_web.tracking import atomic_record, catalogue, file_hash, finalize, lane_lock, reconcile


class TrackingChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.queue=Queue(self.root/'queue')
        self.settings=Settings(**{k:str(self.root) for k in ('engine_repository','engine_python','workspace_root','specification_root','capture_store')},catalog=str(self.root/'catalog.json'),worker_state=str(self.queue.root))
        (self.root/'catalog.json').write_text('{"jobs":[],"tasks":[],"inputs":[]}')
        self.command=Command('run','one',('python','-B','-m','infinity_grid.controller','run',str(self.root/'workspace'),'JOB'),str(self.root),PIN_SOURCE)

    def tearDown(self): self.temp.cleanup()

    def submit(self,command=None,key='unique-key'):
        command=command or self.command
        return self.queue.submit(command,key,command_digest(command))

    def receipt(self,row,native=None,rc=0):
        directory=self.queue.root/row['id'];directory.mkdir(exist_ok=True)
        (directory/'stdout').write_text(json.dumps(native or {'status':'SYNTHETIC_FINISHED'}))
        (directory/'stderr').write_text('')
        atomic_record(directory/'exit.json',{'schema':'IG_WEB_EXIT_V1','request_id':row['id'],'digest':row['digest'],'exit_code':rc,'stdout_sha256':file_hash(directory/'stdout'),'stderr_sha256':file_hash(directory/'stderr')})
        return directory

    def test_simultaneous_retry_one_identity(self):
        def send(_): return Queue(self.queue.root).submit(self.command,'unique-key',command_digest(self.command)).request_id
        with ThreadPoolExecutor(max_workers=4) as pool: ids=list(pool.map(send,range(12)))
        self.assertEqual(len(set(ids)),1)
        self.assertEqual(len(self.queue.events(ids[0])['items']),1)

    def test_alias_cannot_duplicate_native_run(self):
        self.submit()
        alias=Command('run','another-web-name',self.command.argv,self.command.cwd,PIN_SOURCE)
        with self.assertRaisesRegex(AdapterError,'JOB_ALREADY_ACTIVE'): self.submit(alias,'different-key')

    def test_exit_receipt_recovers_database_gap(self):
        rid=self.submit().request_id;row=self.queue.claim('execution');self.receipt(row)
        self.assertEqual(reconcile(self.settings,self.queue,rid)['status'],'finished')
        self.assertEqual(Queue(self.queue.root).get(rid)['exit_code'],0)
        self.assertEqual(self.queue.events(rid)['items'][-1]['kind'],'EXIT_RECONCILED')

    def test_unknown_dispatch_never_replayed(self):
        rid=self.submit().request_id;self.queue.claim('execution')
        result=reconcile(self.settings,self.queue,rid)
        self.assertEqual(result['status'],'needs_reconciliation')
        self.assertEqual(result['error'],'EXIT_RECEIPT_ABSENT')
        with self.assertRaisesRegex(AdapterError,'LANE_NEEDS_RECONCILIATION'): self.queue.claim('execution')

    def test_live_lane_refuses_reconciliation(self):
        rid=self.submit().request_id;self.queue.claim('execution')
        with lane_lock(self.queue,'execution'):
            with self.assertRaisesRegex(AdapterError,'WORKER_LANE_ACTIVE'): reconcile(self.settings,self.queue,rid)
        self.assertEqual(self.queue.get(rid)['status'],'dispatching')

    def test_corrupted_output_blocks_reconciliation(self):
        rid=self.submit().request_id;row=self.queue.claim('execution');d=self.receipt(row)
        (d/'stdout').write_text('{"status":"changed"}')
        with self.assertRaisesRegex(AdapterError,'RECONCILIATION_REFUSED'): reconcile(self.settings,self.queue,rid)
        self.assertEqual(self.queue.get(rid)['status'],'needs_reconciliation')

    def test_wrong_receipt_request_blocks(self):
        rid=self.submit().request_id;row=self.queue.claim('execution');d=self.receipt(row)
        p=d/'exit.json';r=json.loads(p.read_text());r['request_id']='another';p.write_text(json.dumps(r))
        with self.assertRaisesRegex(AdapterError,'RECONCILIATION_REFUSED'): reconcile(self.settings,self.queue,rid)

    def fixture_capture(self):
        spec={'job_id':'CAPTURE.TEST','execution':{'kind':'VALIDATION','nodes':['tests/synthetic.py']},'question':{'outcomes':['PASS']},'resources':{},'inputs':[],'environment':{'python':'3.12','requirements':[],'artifacts':[]},'output_contract':{'claim':'VALIDATION'}}
        raw=json.dumps(spec).encode()
        import hashlib
        command=Command('capture','task',('synthetic',),str(self.root),PIN_SOURCE,hashlib.sha256(raw).hexdigest())
        self.submit(command);row=self.queue.claim('execution');d=self.queue.root/row['id'];d.mkdir();(d/'specification.json').write_bytes(raw)
        job={'schema_id':'IG_DECODER_WORKSPACE_JOB_V1','job_id':spec['job_id'],'source_sha256':PIN_SOURCE,**{k:spec[k] for k in ('execution','question','resources')},'input_artifacts':[]}
        job['registration_sha256']=canonical_hash(job)
        rec={'schema_id':'IG_DECODER_CAPTURE_V1','job':job,'workspace':{'source_sha256':PIN_SOURCE},'environment':spec['environment'],'output_contract':spec['output_contract'],'repeat_id':None}
        rec['capture_id']=canonical_hash(rec);workspace=self.root/'captures'/rec['capture_id'];(workspace/'registry').mkdir(parents=True);(workspace/'source').mkdir()
        for name,value in [('CAPTURE.json',rec),('WORKSPACE.json',rec['workspace']),('registry/'+job['job_id']+'.json',job)]: (workspace/name).write_text(json.dumps(value))
        self.receipt(row,{'workspace':str(workspace),'capture_id':rec['capture_id'],'job_id':job['job_id']})
        return row,workspace

    def test_capture_index_atomic_and_persistent(self):
        row,workspace=self.fixture_capture()
        with patch('ig_web.tracking.NativeAdapter.verify_source',return_value={'status':'SYNTHETIC'}): result=reconcile(self.settings,self.queue,row['id'])
        self.assertEqual(result['status'],'finished')
        jobs=catalogue(self.settings,Queue(self.queue.root)).jobs
        self.assertEqual(len(jobs),1);self.assertEqual(jobs[0].workspace,str(workspace))
        reconcile(self.settings,self.queue,row['id']);self.assertEqual(len(catalogue(self.settings,self.queue).jobs),1)

    def test_capture_wrong_registration_cannot_index(self):
        row,workspace=self.fixture_capture();(workspace/'registry/CAPTURE.TEST.json').write_text('{}')
        with patch('ig_web.tracking.NativeAdapter.verify_source',return_value={}):
            with self.assertRaisesRegex(AdapterError,'RECONCILIATION_REFUSED'): reconcile(self.settings,self.queue,row['id'])
        self.assertEqual(catalogue(self.settings,self.queue).jobs,[])

    def test_events_logs_auth_and_bounds(self):
        rid=self.submit().request_id;row=self.queue.claim('execution');self.receipt(row)
        token='synthetic-test-token-not-a-secret-1234'
        with TestClient(create_app(self.settings,token,gateway=self.queue)) as client:
            base='/api/v1/requests/'+rid
            self.assertEqual(client.get(base+'/events').status_code,401)
            headers={'Authorization':'Bearer '+token}
            self.assertEqual(client.get(base+'/events?limit=101',headers=headers).status_code,400)
            r=client.get(base+'/logs/stdout?limit=4',headers=headers).json();self.assertEqual(r['next_offset'],4)
            self.assertEqual(client.get(base+'/logs/arbitrary',headers=headers).status_code,404)
            self.assertEqual(client.get(base+'/logs/stdout?offset=-1',headers=headers).status_code,400)

    def test_log_symlink_refused(self):
        rid=self.submit().request_id;row=self.queue.claim('execution');d=self.receipt(row)
        (d/'stdout').unlink();(d/'stdout').symlink_to(self.root/'catalog.json')
        with self.assertRaises(AdapterError): self.queue.logs(rid,'stdout')

    def test_native_completion_is_diagnostic_only(self):
        rid=self.submit().request_id;self.queue.claim('execution')
        (self.root/'catalog.json').write_text(json.dumps({'jobs':[{'id':'one','name':'one','native_job_id':'JOB','workspace':str(self.root/'workspace')}]}))
        with patch('ig_web.results.results_observation',return_value={'record_status':'PUBLISHED_RECORD','reported':{'status':'COMPLETED'}}):
            r=reconcile(self.settings,self.queue,rid)
        self.assertEqual(r['status'],'needs_reconciliation')
        self.assertEqual(self.queue.events(rid)['items'][-1]['detail']['native_result']['record_status'],'PUBLISHED_RECORD')

    def test_independent_process_retry_one_identity(self):
        script = """import json,sys
from ig_web.native import Command
from ig_web.worker import Queue,command_digest
c=json.loads(sys.argv[2]);c['argv']=tuple(c['argv']);command=Command(**c)
print(Queue(sys.argv[1]).submit(command,'independent-key',command_digest(command)).request_id)
"""
        children=[subprocess.Popen([sys.executable,'-B','-c',script,str(self.queue.root),json.dumps(asdict(self.command))],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(3)]
        ids=[]
        try:
            for child in children:
                out,err=child.communicate(timeout=20)
                self.assertEqual(child.returncode,0,err);ids.append(out.strip())
            self.assertEqual(len(set(ids)),1)
        finally:
            for child in children:
                if child.poll() is None: child.kill(); child.wait()

    def test_separate_process_crash_after_exit_receipt(self):
        rid=self.submit().request_id;row=self.queue.claim('execution')
        script="""import json,sys,os
from pathlib import Path
from ig_web.tracking import atomic_record,file_hash
row=json.loads(sys.argv[2]);d=Path(sys.argv[1])/row['id'];d.mkdir()
(d/'stdout').write_text('{"status":"SYNTHETIC_FINISHED"}');(d/'stderr').write_text('')
atomic_record(d/'exit.json',{'schema':'IG_WEB_EXIT_V1','request_id':row['id'],'digest':row['digest'],'exit_code':0,'stdout_sha256':file_hash(d/'stdout'),'stderr_sha256':file_hash(d/'stderr')})
os._exit(17)
"""
        result=subprocess.run([sys.executable,'-B','-c',script,str(self.queue.root),json.dumps(row)],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,17,result.stderr)
        self.assertEqual(self.queue.get(rid)['status'],'dispatching')
        self.assertEqual(reconcile(self.settings,Queue(self.queue.root),rid)['status'],'finished')
