"""Read-only verification of preregistered source, inputs and native import closure."""
from pathlib import Path
import sys,json,hashlib,py_compile
B=Path(__file__).resolve().parent
sys.path.insert(0,'/tmp/ig_engine0204');sys.path.insert(0,str(B))
def run():
    from infinity_grid.workflow_guard import preflight
    p=json.loads((B/'PREREGISTRATION.json').read_text());s=json.loads((B/'SPEC.json').read_text())
    for name,digest in p['source_sha256'].items():
        path=Path('/tmp/ig_engine0204/infinity_grid/uplift_structural.py') if name.startswith('engine/') else B/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, name
        py_compile.compile(str(path),doraise=True)
    for row in s['inputs']:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==p['input_sha256'][row['logical_name']]
    cases=json.loads((B/'CASES.json').read_text())
    assert len(cases['cases'])==62 and [x['ordinal'] for x in cases['cases']]==list(range(62))
    guard=preflight(B,[B/'project/handler.py',B/'project/worker.py'])
    from project.historical import maturation_parallel as mp,regime_scanner as rs
    assert mp.rs is rs
    result={'status':'PASS_REGISTERED_SOURCE_AND_NATIVE_PREFLIGHT','official_cases_executed':0,'master_slices':152,'new_admissions':0,'G2_promotion':False,'native_guard':guard}
    (B/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],flush=True)
if __name__=='__main__':run()
