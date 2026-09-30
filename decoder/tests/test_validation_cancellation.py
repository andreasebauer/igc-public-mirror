"""Run coordinator checks in dedicated processes with no unrelated children."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from infinity_grid.validation_cancellation import WorkerCancellation, ValidationCancelled, cancel_workers


def _reap():
    end=time.monotonic()+5
    while True:
        try:pid,_=os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:return
        if not pid:
            if time.monotonic()>end:raise RuntimeError('DESCENDANT_REMAINS')
            time.sleep(.01)


def _case(mode):
    r,w=os.pipe()
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM})
    pid=os.fork()
    if pid==0:
        os.close(r)
        try:
            if ctypes.CDLL(None).prctl(36,1,0,0,0)!=0:raise RuntimeError('SUBREAPER')
            if mode=='early':time.sleep(.15)
            with WorkerCancellation() as c:
                signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGTERM})
                if mode in ('early','active'):
                    try:c.communicate([sys.executable,'-B','-c','import os,time;os.fork();time.sleep(30)'],cwd='/tmp',env=dict(os.environ),reap_descendants=_reap)
                    except ValidationCancelled:pass
                    else:raise RuntimeError('CANCELLATION_NOT_OBSERVED')
                _reap()
                packet=[] if mode=='malformed_ack' else {'cleanup_quiescent':mode!='missing_ack'}
                os.write(w,json.dumps(packet).encode())
            os.close(w);os._exit(0)
        except BaseException as exc:
            os.write(w,json.dumps({'cleanup_quiescent':False,'error':repr(exc)}).encode());os._exit(2)
    signal.pthread_sigmask(signal.SIG_SETMASK,mask);os.close(w)
    if mode!='early':time.sleep(.3)
    pending={pid:(0,r,[],None)}
    try:cancel_workers(pending)
    except RuntimeError as exc:
        assert mode in ('missing_ack','malformed_ack')
        assert 'VALIDATION_CLEANUP_UNCONFIRMED' in str(exc)
    else:assert mode not in ('missing_ack','malformed_ack')
    assert not pending
    try:os.waitpid(pid,os.WNOHANG)
    except ChildProcessError:pass
    else:raise AssertionError('WORKER_NOT_REAPED')


def _isolated(mode):
    r=subprocess.run([sys.executable,'-B',__file__,mode],capture_output=True,text=True,timeout=15)
    assert r.returncode==0,r.stderr


def test_cancel_before_worker_ready():_isolated('early')
def test_cancel_active_selector_and_child():_isolated('active')
def test_cleanup_already_exited_worker():_isolated('already_exited')
def test_missing_cleanup_ack_refused():_isolated('missing_ack')
def test_malformed_cleanup_ack_refused():_isolated('malformed_ack')


if __name__=='__main__':
    _case(sys.argv[1])
