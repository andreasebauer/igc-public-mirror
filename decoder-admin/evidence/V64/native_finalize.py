from pathlib import Path
import json,sys
D=Path(__file__).parent;c=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(c['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr
from infinity_grid.v05_controller_event_loop import validate_workspace_job,verified_completion
s=pr.status(w);assert not s['pending_objects'] and s['pending_bytes']==0 and not s['pending_checkpoints']
done=verified_completion(validate_workspace_job(w,c['job_id'],check_loaded=False));assert done and done['status']=='COMPLETED' and pr.terminal_completion_proof(w,done)
nodes=done['result']['result']['nodes'];assert len(nodes)==18 and all(n['status']=='PASS' for n in nodes)
for name,obj in [('TERMINAL_STATUS',s),('VERIFIED_COMPLETION',done),('NATIVE_EXPORT',pr.export_checkpoint(w,D/'V64_TERMINAL_CHECKPOINT.zip',slim=True))]:
 with (D/(name+'.json')).open('x') as f:json.dump(obj,f,indent=2)
print(json.dumps({'nodes':len(nodes),'all_pass':True,'completion_sha256':done['completion_sha256'],'pending_roles':0}))
