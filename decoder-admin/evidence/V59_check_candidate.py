"""Bounded standalone checks; no native qualification claim."""
from pathlib import Path
import hashlib,json,os,subprocess,time
D=Path(__file__).parent
R=Path('/tmp/ig_decoder_dev138_20260930')
runtime='/tmp/ig_runtime_v55_fresh_20260930/python-fixed-host'
cases=['early','active','already_exited','missing_ack','malformed_ack']
plan={'scope':'Standalone helper regressions, not native controller qualification','cases':cases,'source_identity':json.loads((D/'SOURCE_IDENTITY.json').read_text()),'stop_on_first_failure':True,'native_tests_executed':0}
with (D/'CHECK_PREREGISTRATION.json').open('x') as f:json.dump(plan,f,indent=2)
results=[]
env=dict(os.environ,PYTHONPATH=str(R/'decoder'),PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1')
for case in cases:
    start=time.monotonic()
    r=subprocess.run([runtime,'-B',str(R/'decoder/tests/test_validation_cancellation.py'),case],env=env,capture_output=True,text=True,timeout=20)
    results.append({'case':case,'returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'stdout':r.stdout,'stderr':r.stderr})
    (D/'CHECK_RESULTS.json').write_text(json.dumps({'scope':plan['scope'],'results':results,'all_pass':len(results)==5 and all(x['returncode']==0 for x in results)},indent=2)+'\n')
    print(case,r.returncode,flush=True)
    if r.returncode:raise SystemExit(r.returncode)
v=subprocess.run(['python3','-B','decoder-admin/decoder.py','verify-source'],cwd=R,capture_output=True,text=True)
(D/'SOURCE_VERIFICATION.txt').write_text(v.stdout+v.stderr)
raise SystemExit(v.returncode)
