"""Registered Results view checks. Synthetic records plus a retained native fixture."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from fastapi.testclient import TestClient
from ig_web.api import create_app
from ig_web.models import Job,Settings
from ig_web.native import AdapterError,canonical_hash
from ig_web.resultsview import observe,result_facts,save_summary

TOKEN='synthetic-test-token-not-a-secret-1234'
class ResultsChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);(self.root/'registry').mkdir()
        self.reg={'job_id':'J','question':{'description':'Synthetic results fixture'},'source_sha256':'b'*64,'execution':{'kind':'VALIDATION'},'input_artifacts':[]};self.reg['registration_sha256']=canonical_hash(self.reg);(self.root/'registry/J.json').write_text(json.dumps(self.reg))
        self.job=Job(id='J',name='Fixture',native_job_id='J',workspace=str(self.root));(self.root/'catalog.json').write_text(json.dumps({'jobs':[self.job.model_dump()]}))
        self.settings=Settings(**{k:str(self.root) for k in ('engine_repository','engine_python','workspace_root','specification_root','capture_store')},catalog=str(self.root/'catalog.json'))
        self.capture={'classification':'RESPONSE','native':{'job_id':'J','status':'SAVED','pending_objects':[]}};self.outbox={'classification':'RESPONSE','native':{'status':'PRESERVED','pending_objects':[],'pending_checkpoints':[]}};owner=self
        class Adapter:
            def observe(s,operation,job):
                if operation not in ('pending-saves','preservation'):raise AssertionError('Unexpected native operation')
                return owner.capture if operation=='pending-saves' else owner.outbox
        self.adapter=Adapter();self.client=TestClient(create_app(self.settings,TOKEN,adapter=self.adapter));self.headers={'Authorization':'Bearer '+TOKEN}
    def tearDown(self):self.client.close();self.temp.cleanup()
    def completion(self,outcome='NEGATIVE',evidence='VERIFIED',folder='completed',status='COMPLETED'):
        rid='job-'+self.reg['registration_sha256'][:32];request={'schema_id':'IG_DECODER_PASSIVE_REQUEST_V1','request_id':rid,'registered_job_id':'J','requested_operation_id':'VALIDATION','parent_source_sha256':self.reg['source_sha256'],'input_artifacts':[]};result={'status':'PASS','measurement':0}
        done={'schema_id':'IG_DECODER_WORKSPACE_COMPLETION_V1','request_id':rid,'request_sha256':canonical_hash(request),'registration_sha256':self.reg['registration_sha256'],'source_sha256':self.reg['source_sha256'],'result':result,'result_sha256':canonical_hash(result),'status':status,'execution_status':'FINISHED','evidence_status':evidence,'scientific_outcome':outcome};done['completion_sha256']=canonical_hash(done);p=self.root/'runtime/intake'/folder/(rid+'.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(done));return p
    def view(self):
        r=self.client.get('/api/v1/jobs/J/results-view',headers=self.headers);self.assertEqual(r.status_code,200,r.text);return r.json()
    def test_negative_outcome_verified_evidence_pending_save(self):
        self.completion();self.outbox['native']['pending_objects']=[{'sha256':'c'*64}];v=self.view();self.assertEqual(v['facts']['scientific_outcome']['value'],'NEGATIVE');self.assertEqual(v['facts']['evidence']['reported_status'],'VERIFIED');self.assertEqual(v['save']['status'],'PENDING');self.assertEqual(v['facts']['evidence']['artifact_reverification'],'NOT_RUN');self.assertFalse(v['facts']['reusable'])
    def test_rejected_evidence_can_have_saved_workspace(self):
        self.completion('NEGATIVE','REJECTED',status='RESULT_REJECTED');v=self.view();self.assertEqual(v['facts']['evidence']['reported_status'],'REJECTED');self.assertEqual(v['save']['status'],'NATIVE_REPORTS_PRESERVED')
    def test_completion_does_not_invent_scientific_pass(self):
        self.completion(None);v=self.view();self.assertIsNone(v['facts']['scientific_outcome']['value']);self.assertEqual(v['facts']['execution_reported'],'FINISHED')
    def test_pending_publication_keeps_partial_report(self):
        self.completion(folder='prepared_completions');v=self.view();self.assertEqual(v['facts']['publication'],'PENDING_CHECKPOINT');self.assertEqual(v['facts']['scientific_outcome']['value'],'NEGATIVE')
    def test_absent_result_not_error_or_pass(self):
        v=self.view();self.assertEqual(v['facts']['record_status'],'ABSENT');self.assertEqual(v['facts']['evidence']['reported_status'],'UNKNOWN');self.assertIsNone(v['facts']['scientific_outcome']['value'])
    def test_corrupt_record_unknown_without_hiding_save(self):
        p=self.completion();d=json.loads(p.read_text());d['scientific_outcome']='ALTERED';p.write_text(json.dumps(d));v=self.view();self.assertEqual(v['facts']['record_status'],'UNAVAILABLE');self.assertEqual(v['result']['error'],'RESULT_RECORD_HASH_MISMATCH');self.assertEqual(v['facts']['evidence']['reported_status'],'UNKNOWN');self.assertEqual(v['save']['status'],'NATIVE_REPORTS_PRESERVED')
    def test_native_read_refusal_does_not_become_zero(self):
        self.capture={'classification':'REFUSED','native':{'code':'SAVE_REQUIRED'}};v=self.view();self.assertEqual(v['save']['status'],'UNKNOWN');self.assertIsNone(v['save']['capture_pending'])
    def test_capture_job_mismatch_unknown_save(self):
        self.capture['native']['job_id']='OTHER';v=self.view();self.assertEqual(v['save']['status'],'UNKNOWN');self.assertEqual(v['preservation']['capture']['error'],'CAPTURE_JOB_MISMATCH')
    def test_no_checkpoint_not_preserved(self):
        self.outbox['native']['status']='NO_CHECKPOINT';self.assertEqual(self.view()['save']['status'],'NO_CHECKPOINT')
    def test_missing_pending_list_unknown(self):
        self.outbox['native'].pop('pending_checkpoints');v=self.view();self.assertEqual(v['save']['status'],'UNKNOWN');self.assertIsNone(v['save']['checkpoints_pending'])
    def test_read_only_route_and_auth(self):
        self.assertEqual(self.client.get('/api/v1/jobs/J/results-view').status_code,401);self.assertEqual(self.client.post('/api/v1/jobs/J/results-view',headers=self.headers,json={}).status_code,405);self.assertEqual(self.client.get('/api/v1/jobs/absent/results-view',headers=self.headers).status_code,404)
    def test_record_symlink_refused(self):
        p=self.completion();p.unlink();p.symlink_to(self.root/'missing');self.assertEqual(self.view()['facts']['record_status'],'UNAVAILABLE')
    def test_unknown_evidence_enum_not_verified(self):
        self.completion(evidence='UNRECOGNIZED');self.assertEqual(self.view()['facts']['evidence']['reported_status'],'UNKNOWN')
    def test_retained_native_fixture(self):
        fixture=Path(__file__).parent/'web_native_fixture';settings=self.settings.model_copy(update={'workspace_root':str(fixture.parent)});job=Job(id='fixture',name='Native record',native_job_id='ENGINEERING.DEV84.REVISION5.FIXTURE.FOUR',workspace=fixture.name);self.capture['native']['job_id']=job.native_job_id;v=observe(settings,self.adapter,job);self.assertEqual(v['result']['completion_sha256'],'c5cc1f707e8933e2c4c8c748f62c9c05f2c3416054303d808675e8c439c0009b');self.assertEqual(v['facts']['record_status'],'PUBLISHED_RECORD');self.assertEqual(v['facts']['evidence']['artifact_reverification'],'NOT_RUN')
    def test_javascript_projection_and_safe_markup(self):
        module=Path(__file__).resolve().parents[1]/'ig_web/ui/results.mjs';script="""import assert from 'node:assert/strict';const {presentResults}=await import(process.argv[1]);const v={job:{id:'J'},observed_at:1,facts:{record_status:'PENDING_CHECKPOINT_RECORD',scientific_outcome:{value:'NEGATIVE'},evidence:{reported_status:'REJECTED'}},save:{status:'PENDING'}};let p=presentResults(v);assert.equal(p.scientific,'NEGATIVE');assert.equal(p.evidence,'Native reports rejected');assert.equal(p.save,'Save pending');assert.match(p.publication,/Partial/);for(const value of [false,0]){v.facts.scientific_outcome.value=value;assert.equal(presentResults(v).scientific,JSON.stringify(value));}v.facts.scientific_outcome.value=null;assert.equal(presentResults(v).scientific,'Not reported');assert.throws(()=>presentResults({}));"""
        r=subprocess.run([shutil.which('node'),'--input-type=module','-e',script,module.as_uri()],capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr);s=module.read_text()
        for bad in ('innerHTML','localStorage','sessionStorage',"method:'POST'"):self.assertNotIn(bad,s)
        self.assertIn('epoch!==s.epoch',s);self.assertIn('may be stale',s);self.assertEqual(self.client.get('/ui/results.mjs').status_code,200)
