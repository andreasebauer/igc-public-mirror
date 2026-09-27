"""Native exports, verified local bytes and short-lived single-use downloads."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import secrets
import time

from .native import AdapterError, canonical_hash, contained
from .tracking import file_hash

OPERATIONS = ('snapshot', 'export-full', 'export-slim')


def export_path(root, export_id):
    if not root or not isinstance(export_id,str) or not re.fullmatch('[0-9a-f]{64}',export_id):
        raise AdapterError('INVALID_EXPORT_LOCATION',400)
    root=Path(root)
    contained(root,str(root),directory=True)
    path=root/'exports'/(export_id+'.zip')
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise AdapterError('PATH_NOT_ALLOWED',400)
    return path


def verify_archive(queue,row,native):
    path=export_path(str(queue.root),canonical_hash(row['key']))
    if (native.get('status')!='EXPORTED_LOCAL' or native.get('path')!=str(path)
            or native.get('drive_save_confirmed') is not False):
        raise AdapterError('EXPORT_RECEIPT_MISMATCH',409)
    path=contained(queue.root,str(path),directory=False)
    if file_hash(path)!=native.get('sha256'):
        raise AdapterError('EXPORT_HASH_MISMATCH',409)
    return path


def archive(queue,rid):
    with closing(queue.connect()) as db:
        row=db.execute('SELECT * FROM requests WHERE id=?',(rid,)).fetchone()
    if row is None: raise AdapterError('REQUEST_NOT_FOUND',404)
    if row['operation'] not in ('export-full','export-slim') or row['status']!='finished':
        raise AdapterError('EXPORT_NOT_READY',409)
    return verify_archive(queue,row,json.loads(row['native'])),dict(row)


def history(queue,job):
    if not hasattr(queue,'connect'): raise AdapterError('BACKGROUND_WORKER_NOT_CONFIGURED')
    with closing(queue.connect()) as db:
        rows=db.execute("SELECT id FROM requests WHERE target=? AND operation IN "
                        "('snapshot','export-full','export-slim') ORDER BY created DESC LIMIT 50",(job.id,)).fetchall()
    return {'items':[queue.get(r['id']) for r in rows]}


def issue_ticket(queue,rid):
    path,row=archive(queue,rid)
    ticket=secrets.token_urlsafe(32)
    with queue.transaction() as db:
        db.execute('CREATE TABLE IF NOT EXISTS downloads (digest TEXT PRIMARY KEY, request_id TEXT, expires REAL)')
        db.execute('DELETE FROM downloads WHERE expires<?',(time.time(),))
        db.execute('INSERT INTO downloads VALUES (?,?,?)',(hashlib.sha256(ticket.encode()).hexdigest(),rid,time.time()+120))
    return {'url':'/downloads/'+ticket,'expires_in_seconds':120,'sha256':json.loads(row['native'])['sha256'],
            'size_bytes':path.stat().st_size,'drive_save_confirmed':False}


def download_app(queue):
    # Separate ASGI app: only this capability route is outside API bearer auth.
    # Never put the bearer token in a URL. No directory browsing or arbitrary paths.
    from starlette.applications import Starlette
    from starlette.responses import FileResponse,JSONResponse
    from starlette.routing import Route
    def serve(request):
        ticket=request.path_params['ticket']
        if not re.fullmatch('[A-Za-z0-9_-]{43}',ticket):
            return JSONResponse({'error':'DOWNLOAD_EXPIRED_OR_USED'},status_code=404)
        try:
            with queue.transaction() as db:
                db.execute('CREATE TABLE IF NOT EXISTS downloads (digest TEXT PRIMARY KEY, request_id TEXT, expires REAL)')
                digest=hashlib.sha256(ticket.encode()).hexdigest()
                row=db.execute('SELECT * FROM downloads WHERE digest=?',(digest,)).fetchone()
                db.execute('DELETE FROM downloads WHERE digest=?',(digest,))
            if row is None or row['expires']<time.time(): raise AdapterError('DOWNLOAD_EXPIRED_OR_USED',404)
            path,record=archive(queue,row['request_id'])
            return FileResponse(path,media_type='application/zip',filename='decoder-'+record['id']+'.zip')
        except AdapterError as exc:
            return JSONResponse({'error':exc.code},status_code=exc.status)
    return Starlette(routes=[Route('/{ticket}',serve,methods=['GET'])])
