"""Host service entrypoint; readiness is distinct from science qualification."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from .models import Settings
from .native import AdapterError,NativeAdapter,contained,read_json


def readiness(settings):
    adapter=NativeAdapter(settings)
    source=adapter.verify_source();adapter.verify_runtime()
    lock=Path(settings.engine_repository)/'decoder-web/requirements.lock.txt'
    for line in lock.read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        name,version=line.split('==')
        if importlib.metadata.version(name)!=version:raise AdapterError('WEB_RUNTIME_MISMATCH')
    for value in (settings.workspace_root,settings.capture_store,settings.specification_root,settings.worker_state):
        if not value:raise AdapterError('PERSISTENT_DIRECTORY_REQUIRED')
        contained(Path(value),value,directory=True)
        if not os.access(value,os.W_OK|os.X_OK):raise AdapterError('PERSISTENT_DIRECTORY_NOT_WRITABLE')
    from .worker import Queue,process_identity
    if process_identity(os.getpid()) is None:raise AdapterError('PROCESS_IDENTITY_UNAVAILABLE')
    queue=Queue(settings.worker_state)
    from .tracking import catalogue
    catalogue(settings,queue)
    return {'status':'HOST_RUNTIME_CHECK_PASS','source':source,'science_qualification':'NOT_INFERRED',
            'drive_authorization':'NOT_CHECKED','device_access':'NOT_CHECKED'}


def credential():
    directory=os.environ.get('CREDENTIALS_DIRECTORY')
    if not directory:raise AdapterError('WEB_CREDENTIAL_UNAVAILABLE')
    root=Path(directory)
    if not root.is_absolute():raise AdapterError('WEB_CREDENTIAL_UNAVAILABLE')
    path=contained(root,'web-token',directory=False)
    if path.stat().st_mode & 0o077:raise AdapterError('WEB_CREDENTIAL_PERMISSIONS')
    with path.open('rb') as stream:raw=stream.read(4097)
    try:token=raw.decode('ascii').strip()
    except UnicodeError as exc:raise AdapterError('WEB_CREDENTIAL_INVALID') from exc
    if len(raw)>4096 or len(token)<32 or any(c.isspace() for c in token):raise AdapterError('WEB_CREDENTIAL_INVALID')
    return token


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--role',required=True,choices=['api','execution','control','save'])
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    settings=Settings.model_validate(read_json(Path(args.config)));os.umask(0o077)
    status=readiness(settings)
    if args.check:print(json.dumps(status));return
    if args.role=='api':
        from .api import create_app
        from .worker import Queue
        import uvicorn
        uvicorn.run(create_app(settings,credential(),Queue(settings.worker_state)),host='127.0.0.1',port=8765,
                    access_log=False,proxy_headers=False)
    else:
        module='ig_web.saves' if args.role=='save' else 'ig_web.worker'
        argv=[sys.executable,'-B','-m',module,'--config',args.config]
        if args.role!='save':argv+=['--lane',args.role]
        # The bearer credential is not needed by any worker.
        os.environ.pop('CREDENTIALS_DIRECTORY',None);os.environ.pop('IG_WEB_TOKEN',None)
        os.execv(sys.executable,argv)


if __name__=='__main__':main()
