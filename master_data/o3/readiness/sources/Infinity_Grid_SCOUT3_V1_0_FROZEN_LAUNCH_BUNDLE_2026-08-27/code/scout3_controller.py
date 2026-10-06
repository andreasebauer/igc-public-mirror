#!/usr/bin/env python3
import os, sys, json, time, hashlib, subprocess, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
CAP=16; ACTION_CAP=32; HORIZON=64

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def log(msg):
 s=f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n"
 with open(ROOT/'logs/controller.log','a') as f:f.write(s)

def verify_frozen():
 m=json.load(open(ROOT/'provenance/FROZEN_MANIFEST.json'))
 bad=[]
 for rel,expect in m['files'].items():
  p=ROOT/rel
  if not p.exists() or sha(p)!=expect:bad.append(rel)
 if bad: raise SystemExit('INTEGRITY_FAIL_FROZEN_MANIFEST '+repr(bad))

def expected(r):
 base=[f'r{r:03d}_selected_carriers.json.gz',f'r{r:03d}_observer.json',f'r{r:03d}_metrics.json',f'r{r:03d}_evaluation.json']
 if r==0 or r%8==0:base.append(f'r{r:03d}_d2control.json')
 return [ROOT/'checkpoints'/x for x in base]

def run_cmd(tag,cmd):
 out=ROOT/'logs'/f'{tag}.stdout';err=ROOT/'logs'/f'{tag}.stderr'
 env=os.environ.copy();env['PYTHONHASHSEED']='0'
 t=time.time()
 with open(out,'w') as fo,open(err,'w') as fe:
  cp=subprocess.run(cmd,stdout=fo,stderr=fe,env=env)
 dt=time.time()-t
 if cp.returncode!=0:raise SystemExit(f'INTEGRITY_FAIL_COMMAND {tag} rc={cp.returncode}')
 if err.stat().st_size:raise SystemExit(f'INTEGRITY_FAIL_STDERR {tag} bytes={err.stat().st_size}')
 log(f'{tag} PASS wall={dt:.3f}s')

def snapshot(r):
 zpath=ROOT/'snapshots'/f'SCOUT3_V1_0_r{r:03d}_COMPACT.zip'
 tmp=Path(str(zpath)+'.tmp')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for rel in ['00_READ_FIRST/README.txt','spec/SCOUT3_V1_0_FROZEN_SPEC.txt','provenance/FROZEN_MANIFEST.json','provenance/SALVAGE_AND_PREFLIGHT_REPORT.txt','docs/PLAIN_LANGUAGE.txt']:
   z.write(ROOT/rel,rel)
  for p in sorted((ROOT/'checkpoints').glob('*')):
   z.write(p,p.relative_to(ROOT).as_posix())
  for p in sorted((ROOT/'logs').glob('controller*')):z.write(p,p.relative_to(ROOT).as_posix())
 os.replace(tmp,zpath)
 (ROOT/'snapshots'/f'SCOUT3_V1_0_r{r:03d}_COMPACT.zip.sha256').write_text(sha(zpath)+'  '+zpath.name+'\n')
 log(f'snapshot r{r} sha256={sha(zpath)}')

def main():
 (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'checkpoints').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True);(ROOT/'snapshots').mkdir(exist_ok=True)
 pidfile=ROOT/'logs/controller.pid'
 if pidfile.exists():
  try:
   old=int(pidfile.read_text().strip());os.kill(old,0)
   if old!=os.getpid():raise SystemExit(f'DUPLICATE_CONTROLLER pid={old}')
  except ProcessLookupError:pass
 pidfile.write_text(str(os.getpid())+'\n')
 (ROOT/'logs/controller.ppid').write_text(str(os.getppid())+'\n')
 log(f'START pid={os.getpid()} ppid={os.getppid()} cap={CAP} action_cap={ACTION_CAP} horizon={HORIZON}')
 verify_frozen();log('frozen manifest PASS')
 py=sys.executable;runner=str(ROOT/'code/scout3_o3_roadmap_v1_0.py')
 # bootstrap
 ex=expected(0);present=[p.exists() for p in ex]
 if any(present) and not all(present):raise SystemExit('INTEGRITY_FAIL_PARTIAL_r000')
 if not all(present):run_cmd('r000_bootstrap',[py,runner,'bootstrap','--root',str(ROOT)])
 # climb
 last=0
 for r in range(1,HORIZON+1):
  ex=expected(r);present=[p.exists() for p in ex]
  if any(present) and not all(present):raise SystemExit(f'INTEGRITY_FAIL_PARTIAL_r{r:03d}')
  if not all(present):run_cmd(f'r{r:03d}',[py,runner,'step','--root',str(ROOT),'--r3',str(r),'--cap',str(CAP),'--action-cap',str(ACTION_CAP)])
  last=r
  ev=json.load(open(ROOT/'checkpoints'/f'r{r:03d}_evaluation.json'))
  st=ev.get('status','')
  if r in (32,64):snapshot(r)
  if st in ('O3_SCOUT_TRIGGER_V1_0','RELATION_CAPACITY_SATURATION') or st.startswith('INTEGRITY_FAIL'):
   log(f'STOP_CONDITION r={r} status={st}');break
 run_cmd('finalize',[py,runner,'finalize','--root',str(ROOT)])
 res=json.load(open(ROOT/'results/SCOUT3_V1_0_RESULT.json'))
 (ROOT/'results/SCIENCE_SHA256.txt').write_text(res['science_sha256']+'\n')
 log(f"FINISH status={res['status']} last_r3={res['last_r3']} science_sha256={res['science_sha256']}")
if __name__=='__main__':main()
