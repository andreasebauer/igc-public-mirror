"""Request native pause without signalling processes or dispatching workloads."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('workspace',type=Path);p.add_argument('reason');a=p.parse_args()
 sys.path.insert(0,str(ROOT/'decoder'))
 from infinity_grid.v05_controller_event_loop import validate_workspace_job
 from infinity_grid.submission import capture_record
 from infinity_grid.preservation import request_pause
 w=a.workspace.resolve(strict=True);record=capture_record(w)
 validate_workspace_job(w,record['job']['job_id'],check_loaded=False)
 print(json.dumps(request_pause(w,a.reason),indent=2))
if __name__=='__main__':main()
