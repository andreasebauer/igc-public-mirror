"""Persist real transport intents before the orchestrator performs Drive writes."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'candidate/operational'))
from p0_control import SaveJournal,read,write,require
pending=read(HERE/'results/LIVE_PENDING_OBJECTS.json')
journal=HERE/'results/LIVE_SAVE_JOURNAL.json'
require(not journal.exists(),'UPLOAD_JOURNAL_EXISTS_RECONCILE_NO_BLIND_RETRY')
intents=[]
def upload(path,name):
    intents.append({'path':str(path),'provider_name':name})
    raise InterruptedError('DURABLE_INTENT_READY_FOR_REAL_PROVIDER_WRITE')
def forbidden(*args):raise RuntimeError('NO_PROVIDER_CALL_EXPECTED_AT_INTENT_PREPARATION')
service=SaveJournal(journal,upload,forbidden,forbidden)
for obj in pending['native_plan']['physical_objects']:
    try:service.save(obj['path'])
    except InterruptedError:pass
    else:raise RuntimeError('EXPECTED_INTENT_BOUNDARY_NOT_OBSERVED')
write(HERE/'results/UPLOAD_INTENTS.json',intents)
