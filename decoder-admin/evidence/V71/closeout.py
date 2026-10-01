from pathlib import Path
import json,shutil
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev146_20261001');A=R/'decoder-admin';read=lambda p:json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert read(D/'RESULT.json')['returncode']==0;source=read(D/'SOURCE_IDENTITY.json');source['drive_id']='1XmfuMyCjhLvkGS9B5RzPBAjNgacbRYaR'
result={'gate':'V71_DEV146_REVIEW_SUCCESSOR','status':'BOUNDED_STANDALONE_PASS','tests_passed':5,'tests_failed':0,'native_completions':0,'source_sha256':source['source_sha256'],'engine_implementation_unchanged':True,'historical_pins_unchanged':True,'native_pause_operator':'Existing request_pause API exposed through decoder-admin/request_pause.py; fresh active-pause qualification pending','full_rc':'OPEN'};write(D/'V71_RESULT.json',result)
report='''V71 DEV146 REVIEWED-PIN SUCCESSOR — 5 TARGETED PASS — 2026-10-01

Reviewed actual diffs from dev140 baseline across controller, preservation and
validation runtime:89 additions31 deletions. Changes are refusal-before-checkpoint,
ordinary rwx mode restoration and owned worker cancellation/cleanup acknowledgments.
Reviewed associated validation_cancellation.py and bounded V62/V63/V64 evidence.
Baseline file hashes exactly match dev140 amendment. New successor records reasons
and exact identities; all historical pin files unchanged. No engine implementation
changed in dev146. Five source files change:version, build metadata, profile
identity/status, pin test successor block, new pin amendment JSON.

Source archive plus review/protocol saved and byte-verified before execution.
Previously failing core-pin test and four profile-contract cases:5 PASS,0 FAIL.
Standalone regression only, not native qualification or full RC. Source and
reviewed runtime verified before/after. Initial administrative launch failed
before pytest because wrapper PYTHONHOME was inherited by host Python3.12;
corrected by running orchestrator under host Python, which launches tests through
reviewed3.13 wrapper. No tests dispatched by failed launch; no native replay.

OPERATOR CORRECTION
V70 used external signals/session termination. Existing request_pause is the
proper native route: capture-bound nonce, polled acknowledgment, owned cleanup,
PAUSED checkpoint. Added decoder-admin/request_pause.py; no new signalling code.
Fresh active-pause native validation remains REQUIRED before another full run.
Never send a pause or resume request to the frozen V70 failed capture to make
its stale RUNNING record look clean. V70 evidence/restoration remains immutable.

NEXT
Use a fresh saved dev146 native capture to verify public request_pause reaches
active validation, cleanup acknowledgment, PAUSED/refusal checkpoint and exact
restoration without rerunning recovery workloads. Then regenerate exact-dev146
fixtures as required; dev145 five fixtures remain separately historical and are
not silently promoted to dev146. Full qualification remains open; all23 RC rows
OPEN. No science replay, native completion, independent-host or release claim.
Worktree /tmp/ig_decoder_dev146_20261001
Runtime /tmp/ig_runtime_v55_fresh_20260930/python-fixed-host; -B,PYTHONOPTIMIZE=0.
Administrative run_checks.py uses host python3; its test subprocess uses wrapper.
Review CORE_REVIEW.txt, REVIEWED_DIFF.patch, PIN_REVIEW.json and SOURCE_IDENTITY.
'''
(D/'READ_FIRST_V71_DEV146_2026-10-01.txt').write_text(report+'\n'+json.dumps(source,indent=2)+'\n'+json.dumps(result,indent=2)+'\n')
c=read(A/'CATALOG.json');c.setdefault('historical_sources',{})['dev145']=c['source'];c['source']=source;c.setdefault('historical_candidate_fixtures',{})['dev145']=c['candidate_fixtures'];c['candidate_fixtures']={};c['unresolved_fixtures']=sorted(c['historical_candidate_fixtures']['dev145']);write(A/'CATALOG.json',c)
l=read(A/'RC_LEDGER.json');l.setdefault('historical_candidates',{})['dev145']=l['latest_candidate'];l['latest_candidate']={**source,'status':'UNQUALIFIED_BOUNDED_V71_PASS','tests_executed':5};l['latest_source_repair']=result;write(A/'RC_LEDGER.json',l)
(R/'DECODER_READ_FIRST.txt').write_text('DEV146 V71 REVIEW SUCCESSOR — 5 TARGETED PASS, OVERALL UNQUALIFIED\nEngine implementation unchanged; reviewed pin successor added. V70 failure immutable.\nNext: fresh saved native request_pause validation, then exact-candidate fixtures/full RC.\nRead decoder-admin/evidence/V71_REPORT.txt and V71 review evidence. All23 RC rows OPEN.\n')
(R/'decoder-import/README.txt').write_text('Dev146 normalized source. Verify with python3 -B decoder-admin/decoder.py verify-source.\nSource Drive ID 1XmfuMyCjhLvkGS9B5RzPBAjNgacbRYaR. Exact hashes in source-manifest.json.\n');(R/'decoder-import/VERIFICATION.txt').write_text(json.dumps(source,indent=2)+'\n')
e=A/'evidence/V71';e.mkdir()
for p in D.iterdir():
 if p.is_file() and p.suffix in {'.json','.txt','.py','.patch','.xml'}:shutil.copy2(p,e/p.name)
shutil.copy2(D/'READ_FIRST_V71_DEV146_2026-10-01.txt',A/'evidence/V71_REPORT.txt');print(json.dumps(result))
