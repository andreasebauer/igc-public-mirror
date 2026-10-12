from pathlib import Path
import hashlib,json,sys

HERE=Path(__file__).resolve().parent
source=HERE/'source'
before=HERE.parent/'successor176/source'
changed=sorted(str(p.relative_to(source)) for p in source.rglob('*')
 if p.is_file() and p.read_bytes()!=(before/p.relative_to(source)).read_bytes())
if changed!=['qualification/PROFILE.json','tests/test_qualification_contract.py']:
 raise RuntimeError('SOURCE_DIFF_OUTSIDE_PROFILE_SCOPE:'+repr(changed))
p=json.loads((source/'qualification/PROFILE.json').read_text())
selectors=[s for g in p['groups'].values() for s in g['selectors']]
actual=sorted(f.relative_to(source).as_posix() for f in (source/'tests').glob('test_*.py'))
if sorted(selectors)!=actual or len(set(selectors))!=len(selectors):
 raise RuntimeError('PROFILE_INVENTORY_MISMATCH')
sys.path.insert(0,str(source))
from infinity_grid.v05_controller_event_loop import _source_ids
sid,pid=_source_ids(source)
old_sid,old_pid=_source_ids(before)
if pid!=old_pid:raise RuntimeError('PRODUCTION_PACKAGE_CHANGED')
result={'status':'PASS_ADMINISTRATIVE_PROFILE_RECONCILIATION','source_sha256':sid,
 'parent_source_sha256':old_sid,'package_sha256':pid,'changed_files':changed,
 'profile_sha256':hashlib.sha256((source/'qualification/PROFILE.json').read_bytes()).hexdigest(),
 'selectors':len(selectors),'active_test_files':len(actual),'each_once':True,
 'group_order':p['group_order'],'required_fixture_names':[r['logical_name'] for r in p['required_fixtures']],
 'native_execution':False,'full_profile_qualified':False}
(HERE/'PROFILE_RECONCILIATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
