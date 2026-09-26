"""Fresh-process runner for one registered validation selector."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def main(argv=None)->int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--log-root',required=True)
    parser.add_argument('--binding',required=True)
    parser.add_argument('--selector',required=True)
    args=parser.parse_args(argv)
    candidate=Path(args.candidate).resolve(strict=True)
    log_root=Path(args.log_root).resolve(strict=True)
    if not candidate.is_dir() or not log_root.is_dir():return 3
    # This process, rather than its long-lived worker, owns every descendant
    # created by the selected test.
    from .v05_validation_runtime import _enable_validation_subreaper,_wait_for_validation_descendants
    _enable_validation_subreaper()
    from .validation_reports import Recorder
    import pytest
    tmp=Path(tempfile.mkdtemp(prefix='ig-decoder-validation-node-'))
    os.chdir(candidate);os.umask(0o077)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['PYTHONPATH']=str(candidate)
    os.environ['TMPDIR']=str(tmp);os.environ['TEMP']=str(tmp);os.environ['TMP']=str(tmp)
    recorder=Recorder(log_root,args.binding,[args.selector])
    code=int(pytest.main(['-q','-s','-p','no:cacheprovider',args.selector],plugins=[recorder]))
    _wait_for_validation_descendants()
    recorder.finalize()
    return code


if __name__=='__main__':
    raise SystemExit(main())
