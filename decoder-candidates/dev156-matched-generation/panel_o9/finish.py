import pathlib,json,hashlib,subprocess
R=pathlib.Path(__file__).resolve().parents[1];P=R/'panel_o9';runtime='/tmp/ig_runtime_v55_fresh_20260930/python-bundled-host'
def hashes():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.glob('*_BACKBONE_*.json')}
r=json.loads((P/'RESULT.json').read_text());assert r['executed_backbone_builds']==14
(P/'INITIAL_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');before=hashes();(P/'PRE_RESUME_HASHES.json').write_text(json.dumps(before,indent=2)+'\n')
subprocess.run(['python3',str(P/'audit.py')],check=True)
with (P/'resume.log').open('w') as f:subprocess.run([runtime,str(P/'run.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
r=json.loads((P/'RESULT.json').read_text());assert r['executed_backbone_builds']==0 and r['reused_backbone_checkpoints']==14
assert before==hashes();(P/'RESUME_CHECK.json').write_text(json.dumps({'unchanged':True,'checkpoint_files':14,'new_backbone_builds':0,'reused':14},indent=2)+'\n')
subprocess.run(['python3',str(P/'audit.py')],check=True)
subprocess.run(['python3',str(R/'package_panel_o9.py')],check=True)
