import pathlib,json,hashlib,subprocess
R=pathlib.Path(__file__).resolve().parents[1];P=R/'integration157';runtime='/tmp/ig_o9_cold_20261011/python-bundled-host'
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'checkpoints').glob('*.json')}
r=json.loads((P/'RESULT.json').read_text());assert r['status']=='PAUSED' and r['executed_chains']==2
(P/'PAUSED_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');before=hashes();assert len(before)==2
with (P/'resume_missing.log').open('w') as f:subprocess.run([runtime,str(P/'run.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
r=json.loads((P/'RESULT.json').read_text());assert r['executed_chains']==191 and r['reused_chains']==2
assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in before.items());(P/'RECOVERY_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');before=hashes();assert len(before)==193
with (P/'resume_complete.log').open('w') as f:subprocess.run([runtime,str(P/'run.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
r=json.loads((P/'RESULT.json').read_text());assert r['executed_chains']==0 and r['reused_chains']==193 and r['new_reservation_steps']==0;assert before==hashes()
(P/'RESUME_CHECK.json').write_text(json.dumps({'paused_completed_chains_preserved':2,'recovery_executed_missing':191,'complete_resume_reused':193,'new_steps_complete_resume':0,'all_checkpoint_bytes_unchanged':True},indent=2)+'\n');print('PASS_NATIVE_CANDIDATE_PAUSED_RECOVERY')
