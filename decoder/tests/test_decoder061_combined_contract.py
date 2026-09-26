import json
import tempfile
import unittest
from pathlib import Path

from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid.execution_policy import LEGACY, NO_DEADLINE, automatic_deadlines, name
from infinity_grid import preservation, submission


class CombinedContractTests(unittest.TestCase):
    def _workspace(self):
        temp=tempfile.TemporaryDirectory(); store=Path(temp.name)/'store'; root=store/'captures'/'x'; root.mkdir(parents=True)
        raw=b'same bytes'; digest=submission._sha(raw)
        job={'job_id':'TEST','resources':{}}
        body={'schema_id':submission.CAPTURE_SCHEMA,'decoder_version':'0.6.1.dev3','workspace':{'source_sha256':'a','package_sha256':'b'},
              'job':job,'objects':[
                 {'role':'project_baseline','sha256':digest,'size_bytes':len(raw),'object_name':digest+'.bin'},
                 {'role':'input:baseline','sha256':digest,'size_bytes':len(raw),'object_name':digest+'.bin'}]}
        rec=dict(body,capture_id=canonical_sha256(body)); root.name
        write_json_atomic(root/'CAPTURE.json',rec); write_json_atomic(root/'WORKSPACE.json',rec['workspace'])
        write_json_atomic(root/'registry/TEST.json',job)
        readback=Path(temp.name)/'readback'; readback.write_bytes(raw)
        return temp,root,digest,readback

    def test_policy_is_explicit_and_legacy_default_is_preserved(self):
        self.assertEqual(name({}),LEGACY); self.assertTrue(automatic_deadlines({}))
        self.assertEqual(name({'execution_policy':NO_DEADLINE}),NO_DEADLINE)
        self.assertFalse(automatic_deadlines({'execution_policy':NO_DEADLINE}))

    def test_same_digest_keeps_distinct_capture_obligations(self):
        temp,root,digest,readback=self._workspace()
        try:
            rows=[x for x in submission.required_objects(root) if x['sha256']==digest]
            self.assertEqual(len(rows),2); self.assertEqual(len({x['obligation_id'] for x in rows}),2)
            with self.assertRaises(submission.SubmissionError): submission.confirm_save(root,digest,readback,'drive_fixture_001')
            for row in rows:
                submission.confirm_save(root,digest,readback,'drive_fixture_001',role=row['role'],logical_name=row['logical_name'])
            self.assertFalse(any(x['sha256']==digest for x in submission.save_status(root)['pending_objects']))
        finally: temp.cleanup()

    def test_ambiguous_digest_transport_cannot_bypass_roles(self):
        temp,root,digest,_=self._workspace()
        try:
            with self.assertRaises(submission.SubmissionError) as caught:
                submission.confirm_transport(root,digest,'missing-manifest','missing-parts','drive_fixture_001')
            self.assertIn('SAVE_OBLIGATION_AMBIGUOUS',str(caught.exception))
        finally: temp.cleanup()

    def test_checkpoint_obligations_are_role_bound(self):
        obj={'sha256':'a'*64,'size_bytes':7}
        row={'capture_id':'b'*64,'objects':[obj],'base_objects':[
             dict(obj,role='project_baseline',object_name='base.bin'),
             dict(obj,role='input:baseline',object_name='base.bin')],
             'source':obj,'project':{'base':obj,'delta':obj},'state':obj}
        obligations=preservation._checkpoint_obligations('c'*64,row,b'commit')
        same=[x for x in obligations if x['sha256']=='a'*64]
        self.assertGreaterEqual(len(same),2)
        self.assertEqual(len(same),len({x['obligation_id'] for x in same}))


if __name__=='__main__': unittest.main()
