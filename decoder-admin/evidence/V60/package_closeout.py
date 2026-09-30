from pathlib import Path
import json,shutil,hashlib,zipfile,subprocess
D=Path(__file__).parent;B=D/'native_b';R=Path('/tmp/ig_decoder_dev138_20260930')
read=lambda p:json.loads(p.read_text())
assert read(D/'RESULT.json')['returncode']==0
assert len(read(B/'NATIVE_RESULT.json')['result']['result']['nodes'])==5
assert all(x['status']=='PASS' for x in read(B/'NATIVE_RESULT.json')['result']['result']['nodes'])
assert read(B/'RESTORE_VERIFICATION.json')['exact_completion_equal']
assert not read(B/'ACK_RESULT.json')['outbox']['pending_objects']
verification=subprocess.run(['python3','-B','decoder-admin/decoder.py','verify-source'],cwd=R,capture_output=True,text=True)
assert verification.returncode==0; (D/'FINAL_SOURCE_VERIFICATION.txt').write_text(verification.stdout)
ledger=read(R/'decoder-admin/RC_LEDGER.json')
assert len(ledger['rows'])==23 and all(r['rc_acceptance_status']=='OPEN' for r in ledger['rows'])
ledger['latest_candidate'].update(tests_executed=5,standalone_focused_passes=49,native_completions=1,status='UNQUALIFIED',reason='V60B five native regression passes and exact completion restore; fresh live cleanup gate and full RC pending')
ledger['latest_source_repair'].update(native_tests_executed=5,latest_followup='V60B bounded native PASS; V55 scenario not requalified')
ledger['latest_cleanup_regression']={'gate':'V60B','capture_id':read(B/'CAPTURE_SAVE_STATUS.json')['capture_id'],'completion_sha256':read(B/'VERIFIED_COMPLETION.json')['completion_sha256'],'native_passes':5,'standalone_passes':49,'distinct_test_warning':'Five native selectors are included in the 49 standalone cases','restore_dispatches':0,'exact_completion_restore':True,'full_RC':'OPEN'}
(R/'decoder-admin/RC_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
(R/'DECODER_READ_FIRST.txt').write_text('DEV143 V60B BOUNDED REGRESSION PASS — UNQUALIFIED — 2026-10-01\n49 standalone checks; five native selectors; exact completion restored with zero dispatch.\nFresh live cleanup scenario, current-source full RC and independent-host recovery pending.\nV55 remains failed lifecycle cleanup. All 23 RC rows OPEN. No science replay.\nRead decoder-admin/evidence/V60_REPORT.txt.\nVerify source: python3 -B decoder-admin/decoder.py verify-source\n')
e=R/'decoder-admin/evidence/V60';e.mkdir()
for root in [D,B]:
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix in {'.py','.txt','.json','.xml'}:
            target=e/(('native_b_' if root==B else '')+p.name);shutil.copy2(p,target)
shutil.copy2(D/'READ_FIRST_V60_REGRESSION_2026-10-01.txt',R/'decoder-admin/evidence/V60_REPORT.txt')
files=[]
for root in [D,B]:
    for p in sorted(root.iterdir()):
        if p.is_file() and (p.suffix in {'.py','.txt','.json','.xml'} or p.name=='V60_TERMINAL_CHECKPOINT.zip'):files.append(p)
manifest=[{'path':p.relative_to(D).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files]
(D/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(D/'V60_REGRESSION_RECOVERY_2026-10-01.zip','x',zipfile.ZIP_DEFLATED) as z:
    for p in files+[D/'BUNDLE_MANIFEST.json']:z.write(p,p.relative_to(D))
print('V60 closeout packaged; source unchanged; all 23 RC rows OPEN')
