"""Registered SCRIPT entrypoint. Synthetic transport checks, not science."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from ig_web.api import create_app
from ig_web.models import Settings, NativeObservation, Job
from ig_web.native import AdapterError, Command, NativeAdapter, PIN_SOURCE, canonical_hash, contained, read_json

TOKEN = 'synthetic-test-token-not-a-secret-1234'

class FakeNative:
    def verify_source(self):
        return {'status': 'SYNTHETIC'}
    def command(self, operation, job=None, task=None, reason=None):
        return Command(operation, job.id if job else task.id, ('not-executed',), '/synthetic', PIN_SOURCE)
    def observe(self, operation, job):
        return NativeObservation(observed_at='synthetic', exit_code=2, native={'code': 'SAVE_REQUIRED'},
                                 stdout='', stderr='{"code":"SAVE_REQUIRED"}', classification='REFUSED')

class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(**{k: str(self.root) for k in (
            'engine_repository', 'engine_python', 'workspace_root', 'specification_root', 'capture_store')},
            catalog=str(self.root/'catalog.json'))
        self.catalog = {'jobs':[{'id':'j1','name':'Fixture','native_job_id':'FIXTURE','workspace':'job'}],
                        'tasks':[{'id':'t1','name':'Fixture','specification':'spec.json','specification_sha256':'a'*64}], 'inputs':[]}
        self.write_catalog()
        self.client = TestClient(create_app(self.settings, TOKEN, adapter=FakeNative()))
        self.headers = {'Authorization':'Bearer '+TOKEN, 'Idempotency-Key':'synthetic-key-001'}
    def tearDown(self):
        self.client.close()
        self.temp.cleanup()
    def write_catalog(self):
        Path(self.settings.catalog).write_text(json.dumps(self.catalog))
    def test_auth_and_schema(self):
        for path in ('engine','jobs','tasks','inputs','openapi.json'):
            self.assertEqual(self.client.get('/api/v1/'+path).status_code,401)
            r=self.client.get('/api/v1/'+path,headers=self.headers)
            self.assertEqual(r.status_code,200,r.text)
            self.assertEqual(r.headers['cache-control'],'no-store')
        schema=self.client.get('/api/v1/openapi.json',headers=self.headers).json()
        self.assertTrue(schema['paths']['/api/v1/jobs']['get']['security'])
        self.assertEqual(self.client.post('/api/v1/jobs/j1/run',json={}).status_code,401)
    def test_validation_and_worker(self):
        for path,body in (('captures',{'task_id':'t1'}),('jobs/j1/run',{}),('jobs/j1/pause',{'reason':'fixture'})):
            r=self.client.post('/api/v1/'+path,json=body,headers=self.headers)
            self.assertEqual(r.status_code,503,r.text)
            self.assertEqual(r.json()['error']['code'],'BACKGROUND_WORKER_NOT_CONFIGURED')
        self.assertEqual(self.client.post('/api/v1/jobs/j1/run',json={},headers={'Authorization':'Bearer '+TOKEN}).status_code,400)
        self.assertEqual(self.client.post('/api/v1/jobs/j1/run',json={'shell':'no'},headers=self.headers).status_code,400)
        self.assertEqual(self.client.get('/api/v1/jobs?limit=101',headers=self.headers).status_code,400)
        self.assertEqual(self.client.get('/api/v1/jobs/absent',headers=self.headers).status_code,404)
    def test_size_and_host(self):
        r=self.client.post('/api/v1/captures',content=b'x'*65537,headers=self.headers)
        self.assertEqual(r.status_code,413)
        r=self.client.get('/api/v1/jobs',headers={**self.headers,'Host':'unapproved.example'})
        self.assertEqual(r.status_code,400)
    def test_native_refusal_preserved(self):
        r=self.client.get('/api/v1/jobs/j1/status',headers=self.headers).json()
        self.assertEqual(r['classification'],'REFUSED')
        self.assertEqual(r['exit_code'],2)
        self.assertEqual(r['native']['code'],'SAVE_REQUIRED')
        self.assertIn('SAVE_REQUIRED',r['stderr'])
    def test_catalog_failures(self):
        self.catalog['jobs']*=2; self.write_catalog()
        self.assertEqual(self.client.get('/api/v1/jobs',headers=self.headers).status_code,503)
        Path(self.settings.catalog).unlink()
        self.assertEqual(self.client.get('/api/v1/jobs',headers=self.headers).status_code,503)
    def test_paths_and_environment(self):
        (self.root/'file').write_text('{}')
        (self.root/'link').symlink_to(self.root/'file')
        for path in ('../escape','link'):
            with self.assertRaises(AdapterError): contained(self.root,path,directory=False)
        native=NativeAdapter(self.settings)
        with patch.dict(os.environ,{'IG_WEB_TOKEN':'secret','PYTHONPATH':'bad'}):
            env=native.environment()
            self.assertNotIn('IG_WEB_TOKEN',env)
            self.assertNotEqual(env['PYTHONPATH'],'bad')
        with self.assertRaises(AdapterError): native.verify_source()
        (self.root/'file').write_text('{"x":NaN}')
        with self.assertRaises(AdapterError): read_json(self.root/'file')
    def test_result_states_and_corruption(self):
        root=self.root/'job'; (root/'registry').mkdir(parents=True)
        reg={'job_id':'FIXTURE','source_sha256':'b'*64,'execution':{'kind':'SCRIPT'},'input_artifacts':[]}
        reg['registration_sha256']=canonical_hash(reg)
        (root/'registry/FIXTURE.json').write_text(json.dumps(reg))
        url='/api/v1/jobs/j1/results'
        self.assertEqual(self.client.get(url,headers=self.headers).json()['record_status'],'ABSENT')
        rid='job-'+reg['registration_sha256'][:32]
        request={'schema_id':'IG_DECODER_PASSIVE_REQUEST_V1','request_id':rid,'registered_job_id':'FIXTURE',
                 'requested_operation_id':'SCRIPT','parent_source_sha256':reg['source_sha256'],'input_artifacts':[]}
        result={'outcome':'NEGATIVE'}
        done={'schema_id':'IG_DECODER_WORKSPACE_COMPLETION_V1','request_id':rid,
              'request_sha256':canonical_hash(request),'registration_sha256':reg['registration_sha256'],
              'source_sha256':reg['source_sha256'],'result':result,'result_sha256':canonical_hash(result),
              'status':'COMPLETED','execution_status':'FINISHED','evidence_status':'VERIFIED','scientific_outcome':'NEGATIVE'}
        for folder,state in [('prepared_completions','PENDING_CHECKPOINT_RECORD'),('completed','PUBLISHED_RECORD')]:
            path=root/'runtime/intake'/folder/(rid+'.json'); path.parent.mkdir(parents=True)
            done.pop('completion_sha256',None); done['completion_sha256']=canonical_hash(done)
            path.write_text(json.dumps(done))
            r=self.client.get(url,headers=self.headers)
            self.assertEqual(r.status_code,200,r.text)
            body=r.json(); self.assertEqual(body['record_status'],state)
            self.assertEqual(body['reported']['scientific_outcome'],'NEGATIVE')
            self.assertEqual(body['evidence_verification'],'NOT_RUN')
            self.assertEqual(body['preservation'],'UNKNOWN'); self.assertFalse(body['reusable'])
        done['scientific_outcome']='invented'; path.write_text(json.dumps(done))
        self.assertEqual(self.client.get(url,headers=self.headers).status_code,409)
        path.unlink(); path.symlink_to(root/'missing')
        self.assertEqual(self.client.get(url,headers=self.headers).status_code,400)
    def test_observation_parsing(self):
        native=NativeAdapter(self.settings)
        command=Command('status','j1',('not-executed',),'/synthetic',PIN_SOURCE)
        job=Job(**self.catalog['jobs'][0])
        with patch.object(native,'command',return_value=command):
            for output,kind in [(('','{"code":"REFUSAL"}',2),'REFUSED'),(('not json','',0),'UNPARSEABLE'),(('{"status":"pending"}','',0),'RESPONSE')]:
                with patch.object(native,'_read_process',return_value=output):
                    self.assertEqual(native.observe('status',job).classification,kind)

    def test_native_fixture_projection(self):
        from ig_web.results import results_observation
        fixture = Path(__file__).parent / 'web_native_fixture'
        settings = self.settings.model_copy(update={'workspace_root': str(fixture.parent)})
        job = Job(id='native-fixture', name='Recorded engineering fixture',
                  native_job_id='ENGINEERING.DEV84.REVISION5.FIXTURE.FOUR', workspace=fixture.name)
        observed = results_observation(settings, job)
        self.assertEqual(observed['record_status'], 'PUBLISHED_RECORD')
        self.assertEqual(observed['completion_sha256'], 'c5cc1f707e8933e2c4c8c748f62c9c05f2c3416054303d808675e8c439c0009b')
        self.assertEqual(observed['evidence_verification'], 'NOT_RUN')
        self.assertFalse(observed['reusable'])

if __name__ == '__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'scope':'SYNTHETIC_ADAPTER_CHECKS_ONLY','tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful()}
    Path('CHECK_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
