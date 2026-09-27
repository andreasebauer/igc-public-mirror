"""Bounded Drive blob transport. Credentials belong to the host, not the browser."""
import os
from pathlib import Path
import re
from urllib.parse import urlsplit
import httpx
from .native import AdapterError,contained,read_json

BASE='https://www.googleapis.com/drive/v3'
UPLOAD='https://www.googleapis.com/upload/drive/v3/files'
CHUNK=8*1024*1024


def file_id(value):
    if not isinstance(value,str) or not re.fullmatch('[A-Za-z0-9_-]{10,200}',value):
        raise AdapterError('INVALID_DRIVE_FILE_ID',409)
    return value


def session_url(value):
    p=urlsplit(value)
    if (p.scheme!='https' or p.netloc!='www.googleapis.com' or p.path!='/upload/drive/v3/files'
            or p.fragment or not p.query):raise AdapterError('INVALID_UPLOAD_SESSION',409)
    return value


class Drive:
    def __init__(self,settings,client=None):
        self.settings=settings
        self.client=client or httpx.Client(timeout=60,follow_redirects=False,trust_env=False)

    def headers(self):
        if not self.settings.drive_token_file:raise AdapterError('DRIVE_NOT_CONFIGURED')
        p=Path(self.settings.drive_token_file)
        p=contained(p.parent,str(p),directory=False)
        if p.stat().st_mode & 0o077:raise AdapterError('DRIVE_CREDENTIAL_PERMISSIONS')
        data=read_json(p,16384);token=data.get('access_token')
        if not isinstance(token,str) or not re.fullmatch(r'[A-Za-z0-9._~+/-]{20,4096}=*',token):
            raise AdapterError('DRIVE_CREDENTIAL_INVALID')
        return {'Authorization':'Bearer '+token}

    def request(self,method,url,**kwargs):
        try:r=self.client.request(method,url,headers={**self.headers(),**kwargs.pop('headers',{})},**kwargs)
        except httpx.HTTPError as exc:raise AdapterError('DRIVE_CONNECTION_UNCERTAIN') from exc
        if r.status_code in (401,403):raise AdapterError('DRIVE_AUTHORIZATION_REQUIRED')
        return r

    @staticmethod
    def ok(r,codes=(200,201)):
        if r.status_code not in codes:raise AdapterError('DRIVE_HTTP_'+str(r.status_code))
        return r

    def reserve(self):
        return file_id(self.ok(self.request('GET',BASE+'/files/generateIds',params={'count':1,'space':'drive','type':'files'})).json()['ids'][0])

    def upload(self,path,state,persist):
        # A reserved ID is durably stored before create. Retrying never allocates
        # a second remote object for an ambiguous transfer.
        fid=file_id(state['drive_id']);size=path.stat().st_size
        metadata=self.request('GET',BASE+'/files/'+fid,params={'fields':'id,size,trashed'})
        if metadata.status_code==200:
            data=metadata.json()
            if data.get('trashed') or int(data.get('size',-1))!=size:raise AdapterError('DRIVE_OBJECT_CONFLICT',409)
            return  # Actual byte hash, not metadata, must pass readback next.
        self.ok(metadata,(404,))
        url=state.get('session')
        if url:
            r=self.request('PUT',session_url(url),headers={'Content-Length':'0','Content-Range':'bytes */'+str(size)},content=b'')
            if r.status_code in (200,201):return
            if r.status_code in (404,410):url=None
            elif r.status_code!=308:self.ok(r)
        if not url:
            body={'id':fid,'name':'decoder-'+state['sha256']+'.bin','mimeType':'application/octet-stream'}
            if self.settings.drive_folder_id:body['parents']=[self.settings.drive_folder_id]
            r=self.request('POST',UPLOAD,params={'uploadType':'resumable','fields':'id'},json=body,
                           headers={'X-Upload-Content-Type':'application/octet-stream','X-Upload-Content-Length':str(size)})
            self.ok(r);url=session_url(r.headers.get('location',''));state['session']=url;persist()
            offset=0
        else:
            value=r.headers.get('range','')
            if value and not re.fullmatch(r'bytes=0-\d+',value):raise AdapterError('DRIVE_RANGE_INVALID')
            offset=int(value.split('-')[-1])+1 if value else 0
        if not 0<=offset<=size:raise AdapterError('DRIVE_RANGE_INVALID')
        with path.open('rb') as stream:
            stream.seek(offset)
            while offset<size:
                data=stream.read(min(CHUNK,size-offset))
                if not data:raise AdapterError('SOURCE_CHANGED',409)
                r=self.request('PUT',url,content=data,headers={'Content-Length':str(len(data)),
                    'Content-Range':f'bytes {offset}-{offset+len(data)-1}/{size}','Content-Type':'application/octet-stream'})
                if r.status_code in (200,201):
                    if offset+len(data)!=size:raise AdapterError('DRIVE_EARLY_COMPLETION')
                    return
                self.ok(r,(308,));expected=offset+len(data)
                if r.headers.get('range')!=f'bytes=0-{expected-1}':raise AdapterError('DRIVE_RANGE_INVALID')
                offset=expected
        # Empty objects, or a fully uploaded session awaiting final response.
        r=self.request('PUT',url,content=b'',headers={'Content-Length':'0','Content-Range':f'bytes */{size}'})
        self.ok(r)

    def download(self,fid,destination,size):
        try:
            with self.client.stream('GET',BASE+'/files/'+file_id(fid),params={'alt':'media'},headers=self.headers()) as r:
                self.ok(r)
                with destination.open('xb') as out:
                    received=0
                    for block in r.iter_bytes(1024*1024):
                        received+=len(block)
                        if received>size:raise AdapterError('DRIVE_READBACK_SIZE_MISMATCH',409)
                        out.write(block)
                    out.flush();os.fsync(out.fileno())
                if received!=size:raise AdapterError('DRIVE_READBACK_SIZE_MISMATCH',409)
        except httpx.HTTPError as exc:raise AdapterError('DRIVE_DOWNLOAD_FAILED') from exc
