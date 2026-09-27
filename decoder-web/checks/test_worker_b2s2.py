"""Synthetic worker engineering tests; execute only via native VALIDATION."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from ig_web.models import Settings
from ig_web.native import AdapterError, Command, NativeAdapter, PIN_SOURCE
from ig_web.worker import Queue, command_digest, execute, prepare, process_identity


class WorkerChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.q = Queue(self.root/'queue')
        self.settings = Settings(**{k:str(self.root) for k in
            ('engine_repository','engine_python','workspace_root','specification_root','capture_store')},
            catalog=str(self.root/'catalog.json'))
        self.command = Command('run','job',('not-executed',),str(self.root),PIN_SOURCE)

    def tearDown(self):
        self.temp.cleanup()

    def submit(self,command=None,key='test-key-0001'):
        command = command or self.command
        return self.q.submit(command,key,command_digest(command))

    def test_durable_idempotency_and_conflict(self):
        first = self.submit()
        reopened = Queue(self.root/'queue')
        self.assertEqual(reopened.submit(self.command,'test-key-0001',command_digest(self.command)),first)
        changed = Command('pause','job',('different',),str(self.root),PIN_SOURCE)
        with self.assertRaisesRegex(AdapterError,'IDEMPOTENCY_CONFLICT'):
            self.submit(changed)
        with self.assertRaisesRegex(AdapterError,'JOB_ALREADY_ACTIVE'):
            self.submit(key='test-key-0002')

    def test_separate_control_and_ambiguous_restart(self):
        self.submit()
        run = self.q.claim('execution')
        self.assertEqual(run['operation'],'run')
        pause = Command('pause','job',('reason',),str(self.root),PIN_SOURCE)
        self.submit(pause,key='pause-key')
        self.assertEqual(self.q.claim('control')['operation'],'pause')
        with self.assertRaisesRegex(AdapterError,'LANE_NEEDS_RECONCILIATION'):
            self.q.claim('execution')
        self.assertEqual(self.q.get(run['id'])['status'],'needs_reconciliation')

    def synthetic_process(self,code,operation='run'):
        cmd = Command(operation,'job',('synthetic',),str(self.root),PIN_SOURCE)
        accepted = self.submit(cmd)
        row = self.q.claim('execution')
        env = NativeAdapter(self.settings).environment()
        with (self.root/'lock').open('w') as lock, patch('ig_web.worker.prepare',return_value=(
                [sys.executable,'-I','-B','-c',code],str(self.root),env)):
            execute(self.settings,self.q,row,lock.fileno())
        return self.q.get(accepted.request_id),self.q.root/accepted.request_id

    def test_process_output_and_secret_isolation(self):
        with patch.dict(os.environ,{'IG_WEB_TOKEN':'must-not-leak','PYTHONPATH':'must-not-leak'}):
            result,folder=self.synthetic_process("import os,json; print(json.dumps({'token':os.getenv('IG_WEB_TOKEN'),'cwd':os.getcwd()}))")
        self.assertEqual(result['status'],'finished')
        self.assertIsNone(result['native']['token'])
        self.assertEqual(result['native']['cwd'],str(self.root))
        self.assertEqual(result['scientific_outcome'],'NOT_INFERRED')
        self.assertGreater((folder/'stdout').stat().st_size,0)
        self.assertEqual(result['exit_code'],0)
        self.assertIsNotNone(result['pid'])

    def test_native_refusal_retained(self):
        result,folder=self.synthetic_process("import sys; print('partial stdout'); print('{\"code\":\"SAVE_REQUIRED\"}',file=sys.stderr); sys.exit(2)")
        self.assertEqual(result['status'],'refused')
        self.assertEqual(result['native']['code'],'SAVE_REQUIRED')
        self.assertEqual(result['exit_code'],2)
        self.assertIn('partial stdout',(folder/'stdout').read_text())

    def test_unparseable_requires_reconciliation(self):
        result,folder=self.synthetic_process("print('not JSON')")
        self.assertEqual(result['status'],'needs_reconciliation')
        self.assertEqual(result['classification'],'UNPARSEABLE')

    def test_capture_is_not_blindly_repeated(self):
        result,folder=self.synthetic_process("print('{\"status\":\"CAPTURED\"}')",'capture')
        self.assertEqual(result['status'],'needs_reconciliation')
        with self.assertRaisesRegex(AdapterError,'LANE_NEEDS_RECONCILIATION'):
            self.q.claim('execution')

    def test_dispatch_revalidates_catalogue(self):
        self.submit()
        row=self.q.claim('execution')
        Path(self.settings.catalog).write_text('{"jobs":[],"tasks":[],"inputs":[]}')
        with self.assertRaisesRegex(AdapterError,'JOB_NOT_FOUND'):
            prepare(self.settings,row,self.root)
        with (self.root/'lock').open('w') as lock:
            execute(self.settings,self.q,row,lock.fileno())
        result=self.q.get(row['id'])
        self.assertEqual(result['status'],'refused')
        self.assertEqual(result['error'],'JOB_NOT_FOUND')
        self.assertIsNone(result['pid'])

    def test_private_queue_and_process_identity(self):
        public=self.root/'public'; public.mkdir(mode=0o755); public.chmod(0o755)
        with self.assertRaises(ValueError): Queue(public)
        # Native validation hides parts of /proc; qualify parsing and explicit
        # missing-identity behavior without claiming a live host readiness check.
        fields = ['0'] * 20; fields[19] = '123456'
        with patch.object(Path,'read_text',side_effect=[str(os.getpid())+' (self) '+' '.join(fields),'9 (name with spaces) '+ ' '.join(fields),'boot-id\n']):
            self.assertEqual(process_identity(9),'boot-id:123456')
        with patch.object(Path,'read_text',side_effect=OSError('restricted')):
            self.assertIsNone(process_identity(9))
        env=NativeAdapter(self.settings).environment(source=self.root/'captured')
        self.assertEqual(env['PYTHONPATH'],str(self.root/'captured'))
        self.assertEqual(env['PYTHONNOUSERSITE'],'1')

    def test_namespace_mismatch_refuses_even_if_numeric_pid_exists(self):
        with patch.object(Path,'read_text',return_value=str(os.getpid()+1000)+' (different namespace) '+ ' '.join(['0']*20)) as read:
            self.assertIsNone(process_identity(os.getpid()))
            self.assertEqual(read.call_count,1)

    def test_wrong_numeric_record_and_malformed_identity_refused(self):
        own=str(os.getpid())+' (self) '+' '.join(['0']*20)
        for value in ('not-a-pid (bad) fields','77 (wrong target) '+' '.join(['0']*20)):
            with patch.object(Path,'read_text',side_effect=[own,value]):
                self.assertIsNone(process_identity(9))
