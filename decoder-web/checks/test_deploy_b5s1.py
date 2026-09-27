"""Registered host-packet checks. Does not install or start host services."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from ig_web.models import Settings
from ig_web.native import AdapterError,NativeAdapter
from ig_web.service import credential,readiness,main

class DeploymentChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        path=Path(__file__).resolve().parents[1]/'deploy/render.py';spec=importlib.util.spec_from_file_location('host_render',path);self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.commit='a'*40;self.files=self.module.render(self.commit,self.root/'packet')
        (self.root/'decoder-web').mkdir();(self.root/'decoder-web/requirements.lock.txt').write_text('fastapi==0.141.1\n')
        for name in ('workspaces','store','specifications','queue'):(self.root/name).mkdir(mode=0o700)
        (self.root/'catalog.json').write_text('{"jobs":[],"tasks":[],"inputs":[]}')
        self.settings=Settings(engine_repository=str(self.root),engine_python='/usr/bin/python3',workspace_root=str(self.root/'workspaces'),capture_store=str(self.root/'store'),specification_root=str(self.root/'specifications'),worker_state=str(self.root/'queue'),catalog=str(self.root/'catalog.json'))
    def tearDown(self):self.tmp.cleanup()
    def check(self,**kwargs):
        with patch.object(NativeAdapter,'verify_source',return_value={'status':'FIXTURE'}),patch.object(NativeAdapter,'verify_runtime',return_value={}),patch('ig_web.service.importlib.metadata.version',return_value=kwargs.get('version','0.141.1')),patch('ig_web.worker.process_identity',return_value=kwargs.get('identity','fixture-identity')):
            return readiness(self.settings)
    def test_exact_commit_required(self):
        for bad in ('main','latest','a'*39,'a'*40+';id','../release'):
            with self.assertRaises(ValueError):self.module.render(bad,self.root/'bad')
    def test_packet_does_not_overwrite(self):
        with self.assertRaises(ValueError):self.module.render(self.commit,self.root/'packet')
        link=self.root/'link';link.symlink_to(self.root/'packet',target_is_directory=True)
        with self.assertRaises(ValueError):self.module.render(self.commit,link/'child')
    def test_persistent_paths_outside_release_and_separate_runtimes(self):
        s=json.loads(self.files['config.json']);self.assertTrue(s['engine_python'].endswith('/engine-venv/bin/python'));self.assertEqual(s['worker_state'],'/var/lib/infinity-grid/web-state')
        for key in ('workspace_root','capture_store','specification_root','catalog'):self.assertTrue(s[key].startswith('/var/lib/infinity-grid/'))
        self.assertIsNone(s['drive_token_file']);self.assertEqual(s['allowed_hosts'],['127.0.0.1','localhost'])
    def test_four_services_and_api_only_credential(self):
        for role in ('api','execution','control','save'):
            s=self.files['ig-decoder-'+role+'.service'];self.assertIn('User=igdecoder',s);self.assertIn('--role '+role,s);self.assertIn('/web-venv/bin/python',s)
            self.assertEqual('LoadCredential=' in s,role=='api');self.assertNotIn('IG_WEB_TOKEN=',s)
    def test_no_service_imposed_science_deadline(self):
        for name,s in self.files.items():
            if name.endswith('.service'):
                for value in ('RuntimeMaxSec=infinity','TimeoutStopSec=infinity','SendSIGKILL=no','KillMode=mixed'):self.assertIn(value,s)
    def test_installer_syntax_no_activation_or_data_deletion(self):
        r=subprocess.run(['sh','-n',str(self.root/'packet/install.sh')],capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr)
        s=self.files['install.sh']
        for bad in ('systemctl start','systemctl enable','rm -rf','git reset --hard'):self.assertNotIn(bad,s)
        self.assertIn('--require-hashes',s);self.assertIn('--no-deps --no-build-isolation',s);self.assertIn('verify_source.py',s);self.assertIn(self.commit,s)
    def test_readiness_does_not_claim_science_or_drive(self):
        r=self.check();self.assertEqual(r['status'],'HOST_RUNTIME_CHECK_PASS');self.assertEqual(r['science_qualification'],'NOT_INFERRED');self.assertEqual(r['drive_authorization'],'NOT_CHECKED')
    def test_runtime_mismatch_refuses(self):
        with self.assertRaisesRegex(AdapterError,'WEB_RUNTIME_MISMATCH'):self.check(version='wrong')
    def test_missing_process_identity_refuses(self):
        with self.assertRaisesRegex(AdapterError,'PROCESS_IDENTITY_UNAVAILABLE'):self.check(identity=None)
    def test_symlink_persistent_directory_refuses(self):
        (self.root/'store').rmdir();(self.root/'store').symlink_to(self.root/'workspaces',target_is_directory=True)
        with self.assertRaises(AdapterError):self.check()
    def test_credentials_private_bounded_and_validated(self):
        p=self.root/'web-token';p.write_text('x'*48);p.chmod(0o600)
        with patch.dict(os.environ,{'CREDENTIALS_DIRECTORY':str(self.root)}):
            self.assertEqual(credential(),'x'*48);p.chmod(0o644)
            with self.assertRaisesRegex(AdapterError,'WEB_CREDENTIAL_PERMISSIONS'):credential()
            p.chmod(0o600);p.write_text('x'*4097)
            with self.assertRaisesRegex(AdapterError,'WEB_CREDENTIAL_INVALID'):credential()
    def test_worker_exec_does_not_inherit_api_token(self):
        cfg=self.root/'config.json';cfg.write_text(self.settings.model_dump_json())
        with patch.dict(os.environ,{'CREDENTIALS_DIRECTORY':str(self.root),'IG_WEB_TOKEN':'synthetic'}),patch('sys.argv',['service','--config',str(cfg),'--role','control']),patch('ig_web.service.readiness',return_value={}),patch('ig_web.service.os.execv') as execute:
            main();self.assertNotIn('CREDENTIALS_DIRECTORY',os.environ);self.assertNotIn('IG_WEB_TOKEN',os.environ);self.assertEqual(execute.call_args.args[1][-2:],['--lane','control'])
