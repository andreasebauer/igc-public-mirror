"""Registered engineering checks for preservation transport; no science activation."""
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

from fastapi.testclient import TestClient
from ig_web.api import create_app
from ig_web.exports import archive,export_path,issue_ticket
from ig_web.models import Job,Settings
from ig_web.native import AdapterError,NativeAdapter,PIN_SOURCE,canonical_hash
from ig_web.tracking import atomic_record,file_hash,finalize,reconcile
from ig_web.worker import Queue,command_digest,prepare

TOKEN='synthetic-token-not-a-secret-123456789'

class ExportChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.q=Queue(self.root/'queue')
        self.job=Job(id='J',name='Fixture',native_job_id='J',workspace=str(self.root/'workspace'))
        (self.root/'workspace/source').mkdir(parents=True);(self.root/'workspace/WORKSPACE.json').write_text(json.dumps({'source_sha256':PIN_SOURCE}))
        (self.root/'catalog.json').write_text(json.dumps({'jobs':[self.job.model_dump()]}))
        self.settings=Settings(**{k:str(self.root) for k in ('engine_repository','workspace_root','specification_root','capture_store')},engine_python=sys.executable,catalog=str(self.root/'catalog.json'),worker_state=str(self.q.root))
        self.pin=patch.object(NativeAdapter,'verify_source',return_value={'status':'SYNTHETIC'});self.pin.start()
        self.adapter=NativeAdapter(self.settings);self.client=TestClient(create_app(self.settings,TOKEN,self.q,self.adapter));self.headers={'Authorization':'Bearer '+TOKEN,'Idempotency-Key':'export-test-0001'}
    def tearDown(self):self.client.close();self.pin.stop();self.temp.cleanup()
    def submit(self,op='export-full',key='export-test-0001'):
        command=self.adapter.command(op,self.job,**({'export_id':canonical_hash(key)} if op.startswith('export-') else {}))
        accepted=self.q.submit(command,key,command_digest(command))
        with closing(self.q.connect()) as db:row=dict(db.execute('SELECT * FROM requests WHERE id=?',(accepted.request_id,)).fetchone())
        return row
    def receipt(self,row,rc=0,native=None):
        path=export_path(str(self.q.root),canonical_hash(row['key']));path.parent.mkdir(exist_ok=True);path.write_bytes(b'fixture archive bytes')
        result=native or {'status':'EXPORTED_LOCAL','path':str(path),'sha256':file_hash(path),'drive_save_confirmed':False}
        folder=self.q.root/row['id'];folder.mkdir();(folder/'stdout').write_text(json.dumps(result) if rc==0 else 'partial output');(folder/'stderr').write_text(json.dumps(result) if rc else '')
        atomic_record(folder/'exit.json',{'schema':'IG_WEB_EXIT_V1','request_id':row['id'],'digest':row['digest'],'exit_code':rc,'stdout_sha256':file_hash(folder/'stdout'),'stderr_sha256':file_hash(folder/'stderr')})
        return path
    def finished(self):
        row=self.submit();path=self.receipt(row);finalize(self.settings,self.q,row);return row,path
    def test_exact_native_commands(self):
        self.assertEqual(self.adapter.command('snapshot',self.job).argv[-3:],('preserve','snapshot',self.job.workspace))
        for op in ('export-full','export-slim'):
            argv=self.adapter.command(op,self.job,export_id='a'*64).argv
            self.assertEqual(argv[4:7],('export',self.job.workspace,str(self.q.root/'exports'/('a'*64+'.zip'))))
            self.assertEqual('--slim' in argv,op=='export-slim')
    def test_auth_and_typed_action(self):
        base='/api/v1/jobs/J/preserve/'
        self.assertEqual(self.client.post(base+'snapshot',json={}).status_code,401)
        self.assertEqual(self.client.post(base+'snapshot',headers=self.headers,json={'path':'/tmp/untrusted'}).status_code,400)
        self.assertEqual(self.client.post(base+'delete',headers=self.headers,json={}).status_code,400)
        self.assertEqual(self.client.post(base+'snapshot',headers=self.headers,json={}).status_code,202)
    def test_duplicate_survives_reopen_and_conflicts(self):
        a=self.client.post('/api/v1/jobs/J/preserve/export-full',headers=self.headers,json={})
        b=self.client.post('/api/v1/jobs/J/preserve/export-full',headers=self.headers,json={})
        self.assertEqual(a.json(),b.json());self.assertEqual(Queue(self.q.root).get(a.json()['request_id'])['status'],'queued')
        self.assertEqual(self.client.post('/api/v1/jobs/J/preserve/export-slim',headers=self.headers,json={}).status_code,409)
    def test_history_recovers_requests(self):
        row=self.submit();r=self.client.get('/api/v1/jobs/J/preserve-requests',headers=self.headers)
        self.assertEqual(r.json()['items'][0]['id'],row['id']);self.assertEqual(self.client.get('/api/v1/jobs/J/preserve-requests').status_code,401)
    def test_worker_reconstructs_path_and_refuses_changed_command(self):
        row=self.submit('export-slim')
        with patch.object(NativeAdapter,'verify_runtime',return_value={}):
            argv,cwd,env=prepare(self.settings,row,self.root)
            self.assertEqual(argv[-2],str(export_path(str(self.q.root),canonical_hash(row['key']))));self.assertEqual(cwd,str(self.root/'workspace/source'))
            row['key']='changed-key'
            with self.assertRaisesRegex(AdapterError,'COMMAND_CHANGED'):prepare(self.settings,row,self.root)
    def test_ambiguous_dispatch_not_replayed(self):
        row=self.submit();self.q.claim('execution')
        with self.assertRaisesRegex(AdapterError,'LANE_NEEDS_RECONCILIATION'):self.q.claim('execution')
        self.assertEqual(reconcile(self.settings,self.q,row['id'])['status'],'needs_reconciliation')
    def test_refusal_retains_native_busy_reason(self):
        row=self.submit();self.receipt(row,2,{'code':'WORKSPACE_BUSY'});v=finalize(self.settings,self.q,row)
        self.assertEqual(v['status'],'refused');self.assertEqual(v['native']['code'],'WORKSPACE_BUSY')
        with self.assertRaisesRegex(AdapterError,'EXPORT_NOT_READY'):archive(self.q,row['id'])
    def test_finalize_requires_archive_hash(self):
        row=self.submit();p=self.receipt(row);p.write_bytes(b'changed')
        with self.assertRaisesRegex(AdapterError,'EXPORT_HASH_MISMATCH'):finalize(self.settings,self.q,row)
        self.assertNotEqual(self.q.get(row['id'])['status'],'finished')
    def test_finalized_archive_keeps_save_false(self):
        row,p=self.finished();actual,record=archive(self.q,row['id']);self.assertEqual(actual,p)
        self.assertFalse(self.q.get(row['id'])['native']['drive_save_confirmed']);self.assertEqual(self.q.get(row['id'])['scientific_outcome'],'NOT_INFERRED')
    def test_symlink_and_path_injection_refused(self):
        with self.assertRaises(AdapterError):export_path(str(self.q.root),'../bad')
        row,p=self.finished();other=self.root/'other';p.rename(other);p.symlink_to(other)
        with self.assertRaises(AdapterError):archive(self.q,row['id'])
    def test_native_output_path_is_not_authority(self):
        row=self.submit();self.receipt(row,native={'status':'EXPORTED_LOCAL','path':'/etc/passwd','sha256':'0'*64,'drive_save_confirmed':False})
        with self.assertRaisesRegex(AdapterError,'EXPORT_RECEIPT_MISMATCH'):finalize(self.settings,self.q,row)
    def test_ticket_auth_single_use_and_streamed_bytes(self):
        row,p=self.finished();url='/api/v1/requests/'+row['id']+'/download-ticket'
        self.assertEqual(self.client.post(url,json={}).status_code,401)
        ticket=self.client.post(url,headers=self.headers,json={}).json();self.assertNotIn(TOKEN,ticket['url'])
        response=self.client.get(ticket['url']);self.assertEqual(response.status_code,200,response.text);self.assertEqual(response.content,p.read_bytes());self.assertIn('attachment',response.headers['content-disposition']);self.assertEqual(response.headers['cache-control'],'no-store')
        self.assertEqual(self.client.get(ticket['url']).status_code,404)
    def test_expired_and_unknown_tickets(self):
        row,p=self.finished();ticket=issue_ticket(self.q,row['id'])
        with self.q.transaction() as db:db.execute('UPDATE downloads SET expires=0')
        self.assertEqual(self.client.get(ticket['url']).status_code,404);self.assertEqual(self.client.get('/downloads/'+'a'*43).status_code,404)
    def test_download_rechecks_bytes_after_ticket(self):
        row,p=self.finished();ticket=issue_ticket(self.q,row['id']);p.write_bytes(b'changed')
        self.assertEqual(self.client.get(ticket['url']).status_code,409)
    def test_ui_syntax_and_no_credential_persistence(self):
        path=Path(__file__).resolve().parents[1]/'ig_web/ui/preserve.mjs'
        r=subprocess.run([shutil.which('node'),'--check',str(path)],capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr)
        text=path.read_text()
        for bad in ('innerHTML','localStorage','sessionStorage','URL.createObjectURL','.blob('):self.assertNotIn(bad,text)
        self.assertIn('Reconcile last request',text);self.assertIn('epoch!==s.epoch',text);self.assertEqual(self.client.get('/ui/preserve.mjs').status_code,200)
