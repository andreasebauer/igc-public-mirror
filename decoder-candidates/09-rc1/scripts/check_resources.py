from pathlib import Path
import sys,subprocess,json,tempfile
h=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(h/'source'))
from infinity_grid.runtime_capacity import process_tree_snapshot,shared_capacity_snapshot
from infinity_grid.execution import HostCpuLeaseManager,CpuLeaseError
runtime='/tmp/ig_decoder09_v5_runtime/python-bundled-host'
with tempfile.TemporaryDirectory(prefix='ig-namespace-lease-') as tmp:
 child_code="import sys;sys.path.insert(0,"+repr(str(h/'source'))+");from infinity_grid.execution import HostCpuLeaseManager;m=HostCpuLeaseManager("+repr(tmp)+",budget=1);lease=m.acquire(1,owner='BOUNDED_CHILD',wait=False);buf=bytearray(64*1024*1024);print('READY',flush=True);sys.stdin.readline();lease.release()"
 child=subprocess.Popen([runtime,'-c',child_code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:
  assert child.stdout.readline().strip()=='READY'
  snap=process_tree_snapshot();m=HostCpuLeaseManager(tmp,budget=1);assert m.status()['used']==1
  try:m.acquire(1,owner='BOUNDED_PARENT',wait=False)
  except CpuLeaseError:pass
  else:raise AssertionError('cross-process lease stolen')
  assert len(snap['processes'])>=2 and snap['aggregate_rss_bytes']>64*1024*1024
  child.communicate('\n',timeout=10);assert child.returncode==0 and m.status()['used']==0
 finally:
  if child.poll() is None:child.kill();child.wait()
 result={'status':'PASS','scope':'Bounded parent-child resource and namespace lease check','snapshot':snap,'capacity':shared_capacity_snapshot(),'cross_process_lease_exclusion':True,'release_verified':True}
 (h/'results/RESOURCE_ACCOUNTING.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
