from pathlib import Path
import json,hashlib,resource,shutil,sys
root=Path(__file__).resolve().parents[1];reg=json.loads((root/'review165/REGISTRATION_V2.json').read_text())
for p,h in reg['pins'].items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
assert shutil.disk_usage(root).free>=10*1024**3
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
sys.path.insert(0,str(root/'decoder165'))
import pytest
raise SystemExit(pytest.main(['-q','-x','tests','--basetemp',str(root/'review165/test_tmp')]))
