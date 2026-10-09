"""Record native refusal details and contrast the earlier local import closure."""
from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid.workflow_guard import preflight_job
from infinity_grid import submission
from infinity_grid.invocation import InvocationRefused
try:preflight_job({'source':J/'source','job':submission.capture_record(J)['job']})
except InvocationRefused as e:
 out={k:v for k,v in vars(e).items() if k!='args'};out.update(status='STOPPED_NATIVE_IMPORT_PREFLIGHT',official_cases_executed=0,scientific_source_changed=False,master_slices=152,new_admissions=0,G2_promotion=False)
 (B/'PREFLIGHT_DIAGNOSIS.json').write_text(json.dumps(out,indent=2,default=str)+'\n');print(json.dumps(out,default=str))
