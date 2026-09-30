"""Read-only /proc monitor. Never signals or waits on native processes."""
import hashlib,json,os,time
from pathlib import Path

def process(pid):
    try:
        p=Path('/proc')/str(pid); raw=(p/'stat').read_text(); fields=raw[raw.rfind(')')+2:].split()
        return dict(pid=int(raw.split(' ',1)[0]),ppid=int(fields[1]),start_ticks=int(fields[19]),state=fields[0],cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
    except (FileNotFoundError,ProcessLookupError):return None

def identity(p):return (p['pid'],p['start_ticks'])

def inventory(root):
    result={}
    for p in sorted(Path(root).rglob('*')):
        if p.is_symlink():raise RuntimeError('EVIDENCE_SYMLINK')
        if p.is_file():
            try:
                before=p.stat();data=p.read_bytes();after=p.stat()
            except FileNotFoundError:raise RuntimeError('EVIDENCE_CHANGED_DURING_INVENTORY')
            if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise RuntimeError('EVIDENCE_CHANGED_DURING_INVENTORY')
            result[str(p.relative_to(root))]={'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'mode':after.st_mode&0o777}
    return result

class Monitor:
    def __init__(self,owner_pid,workspace):
        # Pass self inside the controller; external callers must pass /proc namespace PID.
        self.owner=process(owner_pid)
        if self.owner is None:raise RuntimeError('OWNER_MISSING')
        self.workspace=str(Path(workspace).resolve(strict=True));self.known={identity(self.owner):self.owner};self.rows=[]
    def sample(self):
        table={int(p.name):process(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()}
        table={k:v for k,v in table.items() if v is not None}
        # Preserve identities across reparenting; never identify a reused PID as a child.
        selected={pid for pid,p in table.items() if identity(p) in self.known}
        changed=True
        while changed:
            add={pid for pid,p in table.items() if p['ppid'] in selected}-selected
            changed=bool(add);selected.update(add)
        matched=[]
        for pid in selected:
            p=table[pid];self.known.setdefault(identity(p),p);matched.append(p)
        row={'unix':time.time(),'monotonic':time.monotonic(),'processes':sorted(matched,key=lambda p:p['pid'])}
        self.rows.append(row);return row
    def save(self,path):
        p=Path(path).resolve()
        if p.is_relative_to(Path(self.workspace)):raise RuntimeError('EXTERNAL_OUTPUT_REQUIRED')
        with p.open('x') as f:json.dump({'owner':self.owner,'known':list(self.known.values()),'samples':self.rows,'limitations':'Polling can miss short-lived descendants. Registered identities remain tracked after reparenting; no child supervision.'},f,indent=2)
