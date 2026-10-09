"""Manifested handoff for registration/preflight only; retain scientific stop evidence."""
from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=json.loads((B/'AUDIT.json').read_text());assert a['official_cases_executed']==0
    passed=a['dry_status']=='PASS_ONE_CASE_DRY_PREFLIGHT'
    report=W/'IG_MASTER152_G2_BOUND62_PREFLIGHT0235_2026-10-09.txt'
    text='Checkpoint 0235: bounded historical S1 deterministic Q2 registration and preflight.\n\nMaster: 152. New admissions: 0. G2 promotion: false. Historical V1 replay depth: 100.\n\nExactly 62 cases (first two saved unordered pairs, all 31 operators each) were frozen before fresh execution. Case zero is the declared one-case dry probe and overlaps the official set; no blind holdout claim is made. Official cases executed: zero.\n\nSource and input SHA256 checks and native scientific import preflight passed. The recipe uses the frozen historical regime_scanner._build_lift at level 101 with a single forced bridge and new explicit probe labels. Its deterministic reservation operation is separate from later relation-valued G2 semantics. No historical exact G2 construction identity or G2 graduation is claimed.\n\n'
    if passed:
        r=json.loads((B/'DRY_RESULT.json').read_text())
        text+='The dry case passed projected outcome, both D4 hashes, exact bridge reservation witnesses, reserved-child identities, capacity accounting, all public Q2 successor skins, repaired outcome and whole-carrier swap Q2 comparison. Operational reservation witnesses are stored separately from the strict public Q2 payload. One primary G2 realization and one swapped realization were constructed. G1 candidate generation: zero.\n\nElapsed dry seconds: '+str(r['elapsed_seconds'])+'. Peak RSS bytes: '+str(r['peak_RSS_bytes'])+'; registered memory budget: 4294967296. These measurements apply to one dry case, not all 62.\n\n'
    else:text+='The dry probe stopped at the first mismatch. See STOPPED.txt and DRY_RUN.log. No criteria or scientific source were changed after the stop. Official execution remains blocked.\n\n'
    text+='The independent saved-payload audit checks registration, hashes and scope; it is not a second scientific replay or cold native qualification.\n\nPreregistration SHA256: '+sha(B/'PREREGISTRATION.json')+'\nAudit SHA256: '+sha(B/'AUDIT.json')+'\n\nRecovery: verify every MANIFEST.json entry. Recover the sealed exact G1 input from the bundled seal Drive receipt; preserve its bootstrap and public-population hashes. Recover the unchanged engine and pinned CPython 3.13.5 from the bundled runtime and prior handoff receipts. Rebase only local paths in SPEC.json after checking all hashes. Scientific project source and comparison criteria are frozen.\n\nNext: '+a['next_scope']+'.\n'
    report.write_text(text)
    s=json.loads((W/'CURRENT_STATUS.json').read_text());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_checkpoint=235,current_phase='G2_BOUND62_PREFLIGHT_READY' if passed else 'G2_BOUND62_DRY_PREFLIGHT_STOPPED',master_slices=152,new_admissions=0,checkpoint0235=a,next_scope=a['next_scope'],code_mirror=json.loads((B/'CODE_MIRROR.json').read_text()))
    (B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2)+'\n')
    files=[p for p in B.rglob('*') if p.is_file() and p.suffix in ('.json','.py','.log','.txt') and p.name not in ('DELIVERABLES.json','SAVE_RECEIPT.json')]+[report]
    files += [W/'continuation0231/HISTORICAL_V1_INTERFACE_POPULATION.json',W/'continuation0231/RUNTIME_MANIFEST.json',W/'continuation0232/SEAL_RECEIPT.json',W/'continuation0232/SEAL_DRIVE_SAVE.json',W/'continuation0233/CATALOG_0152.json',W/'continuation0233/SCOPED_ADMISSION.json',W/'continuation0233/SAVE_RECEIPT.json',W/'continuation0234/REVIEW.json',W/'continuation0234/SAVE_RECEIPT.json']
    manifest={str(p.relative_to(W)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)}
    out=W/'IG_MASTER152_G2_BOUND62_PREFLIGHT0235_HANDOFF_2026-10-09.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,str(p.relative_to(W)))
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for n,v in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] and len(z.read(n))==v['bytes']
    d={k:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for k,p in [('report',report),('handoff',out)]}
    (B/'DELIVERABLES.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
if __name__=='__main__':main()
