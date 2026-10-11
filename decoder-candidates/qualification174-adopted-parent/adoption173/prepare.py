from pathlib import Path
import sys,json,hashlib,shutil
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'successor171/source'
sys.path.insert(0,str(SOURCE))
from infinity_grid import submission as sub,portable_registry as reg,v05_controller_event_loop as loop
from infinity_grid.canon import write_json_atomic,canonical_sha256
ROOT=Path('/tmp/ig_parent_adoption173_r1')
def main():
    if ROOT.exists():raise RuntimeError('ROOT_EXISTS')
    if shutil.disk_usage('/tmp').free<10*1024**3:raise RuntimeError('MINIMUM_DISK')
    parent=HERE.parent/'successor171/step2_parent_snapshot.zip'
    proof=json.loads((HERE.parent/'successor171/PARENT_FIXTURE_RECOVERY.json').read_text())
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=proof['sha256']:raise RuntimeError('PARENT_HASH')
    write_json_atomic(HERE/'PREREGISTRATION.json',{'schema':'IG_LEGACY_PARENT_PROJECT_ADOPTION_V1','parent':proof,'source':loop._source_ids(SOURCE),'steps':['restore verified parent working copy','initialize project with exact historical parent source','adopt restored copy','save/readback pending baseline and binding roles','export full project and cold restore','verify original completion and bytes without execution'],'stop':'First mismatch; no legacy dispatch or release promotion','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    work=ROOT/'store/captures/15d7dc505e4de37146cbdda99fe1da6b72ab1c6ce743243f40dfeb05940d5867'
    result=loop.restore_workspace(parent,work,proof['sha256'])
    before={str(p.relative_to(work)):sub._sha(p.read_bytes()) for p in (work/'runtime/intake/completed').glob('*.json')}
    for p in (work/'runtime/intake/completed').glob('*.json'):
        row=json.loads(p.read_text())
        if row['completion_sha256']!=proof['completion_sha256'] or canonical_sha256({k:v for k,v in row.items() if k!='completion_sha256'})!=row['completion_sha256']:raise RuntimeError('COMPLETION_HASH')
    write_json_atomic(HERE/'RESTORED.json',{'workspace':str(work),'result':result,'completion_bytes_before':before})
    reg.initialize(work,'Explicit recovered Step2 parent adoption173',work/'source')
    status=reg.adopt(work,work)
    after={str(p.relative_to(work)):sub._sha(p.read_bytes()) for p in (work/'runtime/intake/completed').glob('*.json')}
    if after!=before:raise RuntimeError('COMPLETION_CHANGED')
    write_json_atomic(HERE/'ADOPTED_PENDING.json',status)
    print(json.dumps(status))
if __name__=='__main__':main()
