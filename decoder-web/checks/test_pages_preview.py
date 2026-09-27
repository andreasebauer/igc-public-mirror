"""Static preview checks registered through native VALIDATION."""
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]/'decoder-preview'

class PreviewChecks(unittest.TestCase):
    def test_markup_labels_samples_and_disables_live_actions(self):
        class Parser(HTMLParser):
            def __init__(s):super().__init__();s.buttons=[];s.inputs=[]
            def handle_starttag(s,tag,attrs):
                if tag=='button':s.buttons.append(dict(attrs))
                if tag=='input':s.inputs.append(dict(attrs))
        text=(ROOT/'index.html').read_text();parser=Parser();parser.feed(text)
        self.assertIn('Sample jobs only',text);self.assertIn('not connected',text)
        self.assertEqual(sum('disabled' in b for b in parser.buttons),6)
        self.assertFalse(any(i.get('type')=='password' for i in parser.inputs))
    def test_jobs_and_review_are_local_projections(self):
        module=ROOT/'preview.mjs';script="import assert from 'node:assert/strict';const m=await import(process.argv[1]);assert.equal(m.jobs.length,3);assert.equal(m.filterJobs('comparison').length,1);assert.equal(m.filterJobs('missing').length,0);assert.equal(m.taskReview('comparison').name,'Registered comparison');assert(m.jobs.every(j=>j.id.startsWith('SAMPLE.')));"
        r=subprocess.run([shutil.which('node'),'--input-type=module','-e',script,module.as_uri()],capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr)
    def test_no_backend_or_credential_storage(self):
        s=(ROOT/'preview.mjs').read_text()
        for bad in ('/api/','Authorization','localStorage','sessionStorage','innerHTML',"method:'POST'"):self.assertNotIn(bad,s)
        self.assertIn('No capture is created',s)
    def test_manifest_scoped_relative_assets(self):
        m=json.loads((ROOT/'manifest.webmanifest').read_text());self.assertEqual(m['scope'],'./');self.assertEqual(m['start_url'],'./');self.assertEqual(m['display'],'standalone')
        for icon in m['icons']:self.assertTrue((ROOT/icon['src']).is_file())
        for name in ('style.css','preview.css','preview.mjs','sw.js'):self.assertTrue((ROOT/name).is_file())
    def test_worker_cache_namespace_and_syntax(self):
        p=ROOT/'sw.js';r=subprocess.run([shutil.which('node'),'--check',str(p)],capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr);s=p.read_text()
        self.assertIn('key.startsWith(PREFIX)',s);self.assertIn("event.request.method!=='GET'",s);self.assertIn('url.pathname.startsWith(scope.pathname)',s)
    def test_shared_pages_build_preserves_microscope_root(self):
        s=(ROOT/'pages-workflow.txt').read_text();self.assertIn('cp -a microscope/. _pages/',s);self.assertIn('cp -a decoder-preview/. _pages/decoder/',s);self.assertIn('path: _pages',s);self.assertIn("'decoder-preview/**'",s)
