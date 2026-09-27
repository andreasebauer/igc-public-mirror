"""Native registered synthetic checks for durable preparation; no science execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from ig_web.api import create_app
from ig_web.models import Settings
from ig_web.native import Command,PIN_SOURCE
from ig_web.worker import Queue

TOKEN='synthetic-test-token-not-a-secret-1234'
class PrepareChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.settings=Settings(**{k:str(self.root) for k in ('engine_repository','engine_python','workspace_root','specification_root','capture_store')},catalog=str(self.root/'catalog.json'))
        self.queue=Queue(self.root/'queue');self.headers={'Authorization':'Bearer '+TOKEN,'Idempotency-Key':'review-key-001'}
        self.spec={'schema_id':'IG_DECODER_CAPTURE_SPEC_V1','question':{'description':'Synthetic fixture','outcomes':['PASS','FAIL'],'stopping_rule':'One registered check'},'resources':{'workers':1,'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'},'output_contract':{'required_artifacts':[]},'inputs':[]}
        self.catalog={'tasks':[],'inputs':[]};self.write()
        class Adapter:
            def command(s,operation,task=None):return Command(operation,task.id,('python','capture','store',task.specification),str(self.root),PIN_SOURCE,task.specification_sha256)
        self.client=TestClient(create_app(self.settings,TOKEN,gateway=self.queue,adapter=Adapter()))
    def tearDown(self):self.client.close();self.tmp.cleanup()
    def write(self):
        p=self.root/'spec.json';p.write_text(json.dumps(self.spec));self.catalog['tasks']=[{'id':'T','name':'Fixture','specification':str(p),'specification_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}];(self.root/'catalog.json').write_text(json.dumps(self.catalog))
    def draft(self):
        r=self.client.post('/api/v1/drafts',headers=self.headers,json={'task_id':'T'});self.assertEqual(r.status_code,201,r.text);return r.json()
    def capture(self,d):return self.client.post('/api/v1/drafts/'+d['id']+'/capture',headers=self.headers,json={'review_sha256':d['review_sha256']})
    def test_review_auth_and_no_path(self):
        self.assertEqual(self.client.get('/api/v1/drafts').status_code,401)
        d=self.draft();self.assertNotIn(str(self.root),json.dumps(d));self.assertFalse(d['review']['editable']);self.assertTrue(d['review']['can_prepare'])
    def test_draft_creation_idempotency_after_catalog_removed(self):
        d=self.draft();(self.root/'catalog.json').unlink();self.assertEqual(self.draft(),d)
    def test_draft_key_conflict(self):
        self.draft();r=self.client.post('/api/v1/drafts',headers=self.headers,json={'task_id':'OTHER'});self.assertEqual(r.status_code,409)
    def test_capture_exactly_one_durable_request(self):
        d=self.draft()
        with ThreadPoolExecutor(max_workers=4) as pool: responses=list(pool.map(lambda _:self.capture(d),range(4)))
        self.assertTrue(all(r.status_code==202 for r in responses));self.assertEqual(len({r.json()['request_id'] for r in responses}),1)
        (self.root/'catalog.json').unlink();self.assertEqual(self.capture(d).json(),responses[0].json())
        with self.queue.connect() as db:self.assertEqual(db.execute('SELECT count(*) FROM requests').fetchone()[0],1)
    def test_changed_task_requires_new_review(self):
        d=self.draft();self.spec['resources']['workers']=2;self.write();self.assertEqual(self.capture(d).status_code,409)
    def test_missing_and_changed_input_block_capture(self):
        p=self.root/'input';p.write_bytes(b'fixture');sha=hashlib.sha256(p.read_bytes()).hexdigest();self.spec['inputs']=[{'logical_name':'data','path':str(p),'sha256':sha}];self.write()
        d=self.draft();self.assertFalse(d['review']['can_prepare']);self.assertEqual(self.capture(d).status_code,409)
        self.catalog['inputs']=[{'id':'I','name':'Synthetic input','sha256':sha,'available':True}];self.write();self.headers['Idempotency-Key']='review-key-002';d=self.draft();self.assertTrue(d['review']['can_prepare']);p.write_bytes(b'changed');self.assertEqual(self.capture(d).status_code,409)
    def test_symlink_input_not_verified(self):
        p=self.root/'input';p.write_bytes(b'x');link=self.root/'link';link.symlink_to(p);sha=hashlib.sha256(b'x').hexdigest();self.spec['inputs']=[{'logical_name':'data','path':str(link),'sha256':sha}];self.catalog['inputs']=[{'id':'I','name':'Input','sha256':sha,'available':True}];self.write();self.assertFalse(self.draft()['review']['can_prepare'])
    def test_wrong_review_and_unknown_fields_refused(self):
        d=self.draft();d['review_sha256']='0'*64;self.assertEqual(self.capture(d).status_code,409)
        self.assertEqual(self.client.post('/api/v1/drafts',headers=self.headers,json={'task_id':'T','workers':4}).status_code,400)
    def test_restart_recovery_listing(self):
        d=self.draft();r=self.capture(d).json();from ig_web.drafts import Drafts
        restored=Drafts(Queue(self.root/'queue')).get(d['id']);self.assertEqual(restored['request'],r)
        self.assertEqual(self.client.get('/api/v1/drafts',headers=self.headers).json()['items'][0]['id'],d['id'])
    def test_prepare_assets_no_execution_or_token_storage(self):
        ui=Path(__file__).resolve().parents[1]/'ig_web/ui';s=(ui/'prepare.mjs').read_text()
        for forbidden in ('localStorage','sessionStorage','innerHTML','/run','/pause'):self.assertNotIn(forbidden,s)
        self.assertIn('review_sha256',s);self.assertIn('state.epoch',s);self.assertIn('crypto.randomUUID()',s)
        self.assertEqual(self.client.get('/ui/prepare.mjs').status_code,200)
