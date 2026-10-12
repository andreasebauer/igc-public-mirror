from pathlib import Path
import json, sys

root = Path(__file__).resolve().parent
before = root.parent / 'successor175/source'
after = root / 'source'
changed = sorted(str(p.relative_to(after)) for p in after.rglob('*')
                 if p.is_file() and p.read_bytes() != (before / p.relative_to(after)).read_bytes())
assert changed == ['tests/test_p0_global_identity.py'], changed
sys.path.insert(0, str(after))
from infinity_grid.v05_controller_event_loop import _source_ids
sid, pid = _source_ids(after)
old_sid, old_pid = _source_ids(before)
assert pid == old_pid
proof = {'changed_files': changed, 'parent_source_sha256':old_sid,
 'successor_source_sha256':sid, 'production_package_sha256':pid,
 'production_package_unchanged':True, 'scope':'Recovery/test fixture binding only'}
(root/'SOURCE_DIFF.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
