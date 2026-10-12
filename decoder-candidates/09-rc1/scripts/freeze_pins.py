from pathlib import Path
import json,hashlib,sys
h=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(h/'source'))
from infinity_grid.v05_controller_event_loop import _source_ids
from infinity_grid import submission as sub
from infinity_grid.isolated_runtime import verified_runtime
p=json.loads((h/'results/CAPTURE_PENDING.json').read_text());record=sub.capture_record(p['workspace']);runtime=verified_runtime()
source_sha,package_sha=_source_ids(Path(p['workspace'])/'source')
assert (source_sha,package_sha)==(record['workspace']['source_sha256'],record['workspace']['package_sha256'])
assert sub._tree(h/'source')=={n:v for n,v in sub._tree(Path(p['workspace'])/'source').items() if not n.startswith('project/')}
assert sub._tree(h/'project')==sub._tree(Path(p['workspace'])/'source/project')
out={'version':'0.9.0rc1','capture_id':record['capture_id'],'source_sha256':source_sha,'package_sha256':package_sha,'capture_saved':False,'native_attempts':0,'runtime_root':str(runtime),'runtime_manifest_sha256':hashlib.sha256((h/'runtime/V5_MANIFEST.json').read_bytes()).hexdigest(),'runtime_archive_sha256':hashlib.sha256((h/'runtime/IG_RUNTIME_CPY3135_SQLITE3513.tar.gz').read_bytes()).hexdigest(),'project_source_sha256':hashlib.sha256(sub._archive(sub._tree(h/'project'))).hexdigest(),'frozen_capture_source_exact':True}
(h/'results/CANDIDATE_PINS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
