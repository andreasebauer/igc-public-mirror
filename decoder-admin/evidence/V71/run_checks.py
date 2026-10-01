"""V71 standalone regression gate, exact dev146; no native completion claim."""
from pathlib import Path
import hashlib,json,os,subprocess,time,runpy,sys
D=Path(__file__).resolve().parent
R=Path('/tmp/ig_decoder_dev146_20261001');S=R/'decoder'
RT=Path('/tmp/ig_runtime_v55_fresh_20260930')
def write(n,v):
    with (D/n).open('x') as f:json.dump(v,f,indent=2)
def source():
    r=subprocess.run(['python3','-B',str(R/'decoder-admin/decoder.py'),'verify-source'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    return json.loads(r.stdout)
verify=runpy.run_path(str(R/'decoder-admin/decoder.py'))['verify_runtime']
files=['tests/test_change_preservation_rebind.py::test_recovery_retains_reviewed_core_byte_boundaries','tests/test_qualification_contract.py']
plan={'scope':'Standalone pytest coverage of native runtime components; NOT native controller qualification','source':source(),'runtime':str(RT),'files':files,'stop_on_first_failure':True,'launcher_fix':'Insert candidate into sys.path after wrapper starts; set PYTHONPATH inside Python for test child interpreters','native_completions':0}
write('CHECK_PREREGISTRATION.json',plan)
write('PRE_RUNTIME.json',verify(RT))
bootstrap="import sys,os;sys.path.insert(0,"+repr(str(S))+");os.environ['PYTHONPATH']="+repr(str(S))+";import pytest;raise SystemExit(pytest.main(sys.argv[1:]))"
cmd=[str(RT/'python-fixed-host'),'-B','-c',bootstrap,'-q','-x','-p','no:cacheprovider','--basetemp='+str(D/'pytest_tmp'),'--junitxml='+str(D/'RESULTS.xml'),*files]
write('COMMAND.json',{'argv':cmd,'cwd':str(S)})
start=time.monotonic()
with (D/'PYTEST.txt').open('x') as out:
    r=subprocess.run(cmd,cwd=S,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),timeout=180)
write('RESULT.json',{'returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'native_completions':0})
write('POST_SOURCE.json',source());write('POST_RUNTIME.json',verify(RT))
print((D/'PYTEST.txt').read_text());sys.exit(r.returncode)
