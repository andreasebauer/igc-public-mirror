from pathlib import Path
import sys,json,time
D=Path(__file__).parent;s=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation_batch as batch
original=batch._verify_raw;first=True
def verify(path,expected):
 global first
 if first:
  first=False;(D/'ACK_LOCKED.json').write_text(json.dumps({'unix':time.time()}));start=time.monotonic()
  while not (D/'ACK_RELEASE.json').exists():
   if time.monotonic()-start>60:raise RuntimeError('ACK_GATE_GUARD')
   time.sleep(.02)
 return original(path,expected)
batch._verify_raw=verify
try:
 result=batch.confirm_batch(w,D/'PRE_RUN_BATCH.json');(D/'GATED_ACK.json').write_text(json.dumps({'unix':time.time(),'result':result},indent=2))
except BaseException as exc:
 (D/'ACK_ERROR.json').write_text(json.dumps({'unix':time.time(),'error':repr(exc)}));raise
