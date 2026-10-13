from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from importlib.resources import files
from pathlib import Path

from . import __version__
from .paths import resolve_root
from .store import ArtifactStore
from .datasets import DatasetStore
from .catalog import Catalogue
from .protocols import ProtocolRegistry
from .planning import create_plan
from .controller import Controller
from .checkpoints import recover_stale_lock
from .doctor import doctor
from .claims import list_claims
from .releases import ReleaseGenerator, verify_release_archive
from .canon import write_json_atomic, canonical_sha256
from .verification import verify_run
from .lifecycle import ExecutionModeManager
from .records import source_sha256, runtime_sha256
from .hashing import sha256_file
from .conveyor import (
    ScoutChainController, freeze_chain_plan, validate_chain_plan,
    verify_bootstrap_parent, validate_eligible_parent_certificate,
    load_verified_parent_capsule, rebuild_chain_state,
)
from .adapters.scout_level import normalize_parent_baseline
from .graph_store import GraphStore
from .graph_replay import GraphReplayEngine, minimum_preservation_set, computation_closure, evidence_closure
from .graph_export import GraphExporter, GraphImporter, verify_graph_export_archive
from .graph_gc import gc_dry_run
from .graph_doctor import graph_doctor
from .graph_adapters import ingest_existing_records, ingest_o7_unresolved_case
from .jumpstart import JumpstartRuntime, JumpstartPlanner, JumpstartProfileStore, build_profile
from .replay_contracts import ReplayContractStore
from .scientific_tests import ScientificTestStore
from .contracts import (
    failure_record, load_envelope, payload_registry, release_identity, verify_envelope,
)
from .current_state import load_current_state, reconcile_legacy_context, verify_current_state, write_current_state
from .migration import migrate_json_file_to_envelope, migrate_unschematized_v026_file_to_envelope, scan_legacy_json_tree
from .records import utc_now
import infinity_grid.adapters  # noqa: F401


def _dump(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def _load_json_arg(raw):
    p = Path(raw)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(raw)


def _pairs(items):
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE: {item}")
        if value.lower() == "true": parsed = True
        elif value.lower() == "false": parsed = False
        else:
            try: parsed = int(value)
            except ValueError: parsed = value
        out[key] = parsed
    return out


def build_parser():
    p = argparse.ArgumentParser(prog="ig", description=f"Infinity Grid Algebra Decoder / Unified Runtime v{__version__}")
    p.add_argument("--root")
    p.add_argument("--science-mode", choices=["auto", "hold", "graduated"], default="auto",
                   help="Assert execution capability; never overrides frozen graduation records.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sp = p.add_subparsers(dest="cmd", required=True)

    i = sp.add_parser("init"); i.add_argument("root_arg", nargs="?")
    d = sp.add_parser("doctor"); d.add_argument("--graph", action="store_true"); d.add_argument("--target", action="append", default=[])

    ctr = sp.add_parser("contracts")
    ctrsp = ctr.add_subparsers(dest="contracts_cmd", required=True)
    ctrsp.add_parser("registry")
    ctv = ctrsp.add_parser("validate"); ctv.add_argument("path", type=Path); ctv.add_argument("--envelope", action="store_true")
    ctw = ctrsp.add_parser("wrap"); ctw.add_argument("source", type=Path); ctw.add_argument("output", type=Path); ctw.add_argument("--profile", default="LEGACY_V0_26_PRESERVED")
    ctm = ctrsp.add_parser("migrate-v026"); ctm.add_argument("source", type=Path); ctm.add_argument("output", type=Path); ctm.add_argument("--profile", default="LEGACY_V0_26_PRESERVED")
    cts = ctrsp.add_parser("scan-legacy"); cts.add_argument("tree", type=Path); cts.add_argument("--output", type=Path)

    sta = sp.add_parser("state")
    stasp = sta.add_subparsers(dest="state_cmd", required=True)
    stv = stasp.add_parser("verify"); stv.add_argument("path", type=Path)
    sts = stasp.add_parser("show"); sts.add_argument("path", type=Path)
    strc = stasp.add_parser("reconcile-legacy"); strc.add_argument("context_dir", type=Path); strc.add_argument("output", type=Path); strc.add_argument("--as-of-utc")

    imp = sp.add_parser("import")
    isp = imp.add_subparsers(dest="import_cmd", required=True)
    ia = isp.add_parser("artifact"); ia.add_argument("path", type=Path); ia.add_argument("--role", default="SOURCE_ARCHIVE")
    ids = isp.add_parser("dataset"); ids.add_argument("path", type=Path); ids.add_argument("--role", required=True)
    igi = isp.add_parser("graph"); igi.add_argument("archive", type=Path)

    cat = sp.add_parser("catalog"); csp = cat.add_subparsers(dest="catalog_cmd", required=True); csp.add_parser("rebuild")
    pr = sp.add_parser("protocol"); psp = pr.add_subparsers(dest="protocol_cmd", required=True); psp.add_parser("list"); ps = psp.add_parser("show"); ps.add_argument("protocol_id")
    pl = sp.add_parser("plan"); plsp = pl.add_subparsers(dest="plan_cmd", required=True)
    pc = plsp.add_parser("create")
    for name in ("protocol", "protocol-label", "subject", "input-dataset", "input-role", "runner", "authority", "origin", "scope", "disposition"):
        pc.add_argument("--" + name, required=True)
    pc.add_argument("--input-flag", action="append", default=[]); pc.add_argument("--run-id"); pc.add_argument("--output", required=True, type=Path)
    pv = plsp.add_parser("verify"); pv.add_argument("plan", type=Path)
    req = sp.add_parser("request"); reqsp = req.add_subparsers(dest="request_cmd", required=True)
    rqs = reqsp.add_parser("submit"); rqs.add_argument("--intake-root", required=True, type=Path); rqs.add_argument("--request", required=True, type=Path)
    rn = sp.add_parser("run"); rn.add_argument("plan", type=Path)
    rs = sp.add_parser("resume"); rs.add_argument("run_id")
    st = sp.add_parser("status"); st.add_argument("run_id", nargs="?")
    ver = sp.add_parser("verify"); vsp = ver.add_subparsers(dest="verify_cmd", required=True)
    va = vsp.add_parser("artifact"); va.add_argument("sha256"); va.add_argument("--size", type=int)
    vr = vsp.add_parser("run"); vr.add_argument("run_id")
    vre = vsp.add_parser("release"); vre.add_argument("archive", type=Path)
    vg = vsp.add_parser("graph-export"); vg.add_argument("archive", type=Path)
    rel = sp.add_parser("release"); rsp = rel.add_subparsers(dest="release_cmd", required=True); rc = rsp.add_parser("create"); rc.add_argument("run_id"); rc.add_argument("--mode", choices=["compact", "standalone"], required=True)
    cl = sp.add_parser("claim"); csp2 = cl.add_subparsers(dest="claim_cmd", required=True); csp2.add_parser("list"); cs = csp2.add_parser("show"); cs.add_argument("claim_id")
    lk = sp.add_parser("lock"); lsp = lk.add_subparsers(dest="lock_cmd", required=True); lr = lsp.add_parser("recover"); lr.add_argument("run_id"); lr.add_argument("--operator-override-reason")
    gr = sp.add_parser("graduation"); gsp = gr.add_subparsers(dest="graduation_cmd", required=True); gsp.add_parser("status"); gi = gsp.add_parser("install"); gi.add_argument("record", type=Path)

    up = sp.add_parser("uplift", help="G-layer structural uplift architecture and experiments")
    usp = up.add_subparsers(dest="uplift_cmd", required=True)
    usp.add_parser("show", help="Show frozen uplift architecture and S0/S1 implementation spec")
    urs = usp.add_parser("s0-s1", help="Run frozen G2:S0 carrier interfaces and G2:S1 pair census")
    urs.add_argument("--output", required=True, type=Path)
    urs.add_argument("--workers", type=int, default=4)
    usp.add_parser("registry", help="List native registered G-uplift scientific experiments")
    unr = usp.add_parser("native-run", help="Run one registered native G-uplift stage inside the persistent campaign engine")
    unr.add_argument("experiment_id")
    unr.add_argument("--output", required=True, type=Path)
    unr.add_argument("--workers", default="AUTO")
    unr.add_argument("--lease-root")
    unr.add_argument("--input", action="append", default=[], help="KEY=PATH binding for registered stage input")

    gp = sp.add_parser("graph"); gsp = gp.add_subparsers(dest="graph_cmd", required=True)
    gshow = gsp.add_parser("show"); gshow.add_argument("reference"); gshow.add_argument("--version")
    groots = gsp.add_parser("roots"); groots.add_argument("targets", nargs="+")
    gclosure = gsp.add_parser("closure"); gclosure.add_argument("targets", nargs="+"); gclosure.add_argument("--evidence", action="store_true")
    gvalidate = gsp.add_parser("validate"); gvalidate.add_argument("--target", action="append", default=[])
    gsp.add_parser("ingest-existing")
    go7 = gsp.add_parser("ingest-o7")
    go7.add_argument("--preregistration", required=True, type=Path)
    go7.add_argument("--engine", required=True, type=Path)
    go7.add_argument("--repair", required=True, type=Path)
    go7.add_argument("--postrun-analysis", required=True, type=Path)

    rsv = sp.add_parser("resolve"); rsv.add_argument("reference"); rsv.add_argument("--version")
    rpl = sp.add_parser("replay"); rpl.add_argument("target"); rpl.add_argument("--clean", action="store_true"); rpl.add_argument("--offline", action="store_true"); rpl.add_argument("--workspace", type=Path)
    dep = sp.add_parser("depends"); dep.add_argument("target")
    why = sp.add_parser("why"); why.add_argument("target")
    roots = sp.add_parser("roots"); roots.add_argument("targets", nargs="+")
    gc = sp.add_parser("gc"); gc.add_argument("--dry-run", action="store_true", default=True); gc.add_argument("--target", action="append", required=True)
    ex = sp.add_parser("export"); ex.add_argument("targets", nargs="+"); ex.add_argument("--mode", choices=["compact", "standalone", "publication"], required=True); ex.add_argument("--output", required=True, type=Path); ex.add_argument("--forbidden-inference", action="append", default=[])
    gim = sp.add_parser("graph-import"); gim.add_argument("archive", type=Path)

    js = sp.add_parser("jumpstart"); jsp = js.add_subparsers(dest="jumpstart_cmd", required=True)
    jpl = jsp.add_parser("plan"); jpl.add_argument("target")
    jpr = jsp.add_parser("prepare"); jpr.add_argument("target"); jpr.add_argument("--workspace", type=Path); jpr.add_argument("--clean", action="store_true")
    jsm = jsp.add_parser("smoke"); jsm.add_argument("target"); jsm.add_argument("--workspace", type=Path); jsm.add_argument("--clean", action="store_true"); jsm.add_argument("--timeout", type=int)
    jla = jsp.add_parser("launch"); jla.add_argument("target"); jla.add_argument("--workspace", type=Path); jla.add_argument("--clean", action="store_true"); jla.add_argument("--foreground", action="store_true")
    jst = jsp.add_parser("status"); jst.add_argument("target"); jst.add_argument("--workspace", type=Path)
    jrg = jsp.add_parser("register-profile"); jrg.add_argument("profile", type=Path)
    jsp.add_parser("profiles")
    jct = jsp.add_parser("contract"); jct.add_argument("target")
    jsp.add_parser("contracts")
    jsp.add_parser("coverage")

    tst = sp.add_parser("test"); tsp = tst.add_subparsers(dest="test_cmd", required=True)
    tl = tsp.add_parser("list"); tl.add_argument("--level")
    ts = tsp.add_parser("show"); ts.add_argument("test_id")
    tr = tsp.add_parser("replay"); tr.add_argument("test_id"); tr.add_argument("--workspace", type=Path); tr.add_argument("--no-clean", action="store_true")
    tsp.add_parser("coverage")

    sci = sp.add_parser("science", help="Generic Entity/Regime/Test scientific architecture")
    scsp = sci.add_subparsers(dest="science_cmd", required=True)
    scsp.add_parser("registry")
    sf = scsp.add_parser("frontier")
    sfsp = sf.add_subparsers(dest="science_frontier_cmd", required=True)
    sfs = sfsp.add_parser("show"); sfs.add_argument("--path", type=Path)
    sfv = sfsp.add_parser("verify"); sfv.add_argument("--path", type=Path)
    ssh = scsp.add_parser("shadow", help="Non-promoting scientific shadow observers")
    sshsp = ssh.add_subparsers(dest="science_shadow_cmd", required=True)
    sshsp.add_parser("k4-validate", help="Validate frozen n=7 H1->H2 K4 prediction gate")
    sko = sshsp.add_parser("k4-observe", help="Observe K4 closure count for an explicit simple owner graph")
    sko.add_argument("graph", type=Path, help="JSON with {n, edges}")
    srp = scsp.add_parser("runplan")
    srpsp = srp.add_subparsers(dest="science_runplan_cmd", required=True)
    srv = srpsp.add_parser("validate"); srv.add_argument("plan", type=Path); srv.add_argument("--frontier", type=Path)
    src = srpsp.add_parser("compile"); src.add_argument("plan", type=Path); src.add_argument("--frontier", type=Path)
    sre = srpsp.add_parser("execute-compat"); sre.add_argument("plan", type=Path); sre.add_argument("--frontier", type=Path)
    srl = srpsp.add_parser("execute-live", help="Crash-safe resumable Regime Maturation run")
    srl.add_argument("plan", type=Path); srl.add_argument("--frontier", type=Path); srl.add_argument("--output", required=True, type=Path); srl.add_argument("--stop-after-depth", type=int); srl.add_argument("--acknowledge-review", action="store_true", help="Explicitly acknowledge the latest REVIEW_REQUIRED gate and allow one subsequent advance until a new gate fires")
    srs = srpsp.add_parser("status", help="Show compact maturation run operational status")
    srs.add_argument("output", type=Path)
    srr = srpsp.add_parser("external-request", help="Show the external durability request for one committed depth")
    srr.add_argument("output", type=Path); srr.add_argument("depth", type=int)
    sra = srpsp.add_parser("ack-external", help="Record a connector-produced external durability receipt")
    sra.add_argument("output", type=Path); sra.add_argument("depth", type=int); sra.add_argument("receipt", type=Path)
    srca = srpsp.add_parser("audit-checkpoints", help="Perform an explicit full committed-checkpoint chain audit")
    srca.add_argument("output", type=Path)

    ora = sp.add_parser("oracle"); osp = ora.add_subparsers(dest="oracle_cmd", required=True)
    osp.add_parser("show")
    ov = osp.add_parser("verify"); ov.add_argument("--corpus-root", type=Path)

    fr = sp.add_parser("frontier"); fsp = fr.add_subparsers(dest="frontier_cmd", required=True)
    fshow = fsp.add_parser("show-spec"); fshow.add_argument("name", choices=["protocol","read-functions","retention","o8-bp0","grrl-application","o8-graduation","o9-bp0","auto-advance","pregeometry","meta-grammar","o10-plus","regime-scanner","regime-service-validation","fixed-grammar-transport","native-semantics","theorem-premises","exact-carrier-unblinding","unified-execution"])
    fo7 = fsp.add_parser("o7-read-probe")
    fo7.add_argument("--o7-graduation-root", required=True, type=Path)
    fo7.add_argument("--post-o7-root", required=True, type=Path)
    fo7.add_argument("--o7-runtime-root", required=True, type=Path)
    fo7.add_argument("--output", required=True, type=Path)
    fo8 = fsp.add_parser("o8-bp0")
    fo8.add_argument("--o7-graduation-root", required=True, type=Path)
    fo8.add_argument("--o7-runtime-root", required=True, type=Path)
    fo8.add_argument("--o7-read-probe", required=True, type=Path)
    fo8.add_argument("--output", required=True, type=Path)
    fog = fsp.add_parser("o8-graduate")
    fog.add_argument("--o7-graduation-root", required=True, type=Path)
    fog.add_argument("--o8-bp0", required=True, type=Path)
    fog.add_argument("--frontier-authority-root", required=True, type=Path)
    fog.add_argument("--output", required=True, type=Path)
    fo9 = fsp.add_parser("o9-bp0")
    fo9.add_argument("--o7-graduation-root", required=True, type=Path)
    fo9.add_argument("--o7-runtime-root", required=True, type=Path)
    fo9.add_argument("--o8-graduation", required=True, type=Path)
    fo9.add_argument("--frontier-authority-root", required=True, type=Path)
    fo9.add_argument("--output", required=True, type=Path)
    faa = fsp.add_parser("auto-advance")
    faa.add_argument("--o7-graduation-root", required=True, type=Path)
    faa.add_argument("--o7-runtime-root", required=True, type=Path)
    faa.add_argument("--o8-bp0", required=True, type=Path)
    faa.add_argument("--frontier-authority-root", required=True, type=Path)
    faa.add_argument("--output", required=True, type=Path)
    fpg = fsp.add_parser("pregeometry-scout")
    fpg.add_argument("--o7-graduation-root", required=True, type=Path)
    fpg.add_argument("--post-o7-root", required=True, type=Path)
    fpg.add_argument("--grrl-root", required=True, type=Path)
    fpg.add_argument("--phase8-root", required=True, type=Path)
    fpg.add_argument("--output", required=True, type=Path)
    frs = fsp.add_parser("regime-scan")
    frs.add_argument("--phase8-seed", required=True, type=Path)
    frs.add_argument("--output", required=True, type=Path)
    frs.add_argument("--through", type=int, default=32)
    frs.add_argument("--ignore-earned-laws", action="store_true", help="replay historical discovery without suppressing already-earned scanner laws")
    frv = fsp.add_parser("regime-service-validate")
    frv.add_argument("--phase8-seed", required=True, type=Path)
    frv.add_argument("--output", required=True, type=Path)
    frv.add_argument("--through", type=int, default=13)
    ftr = fsp.add_parser("fixed-grammar-transport")
    ftr.add_argument("--output", required=True, type=Path)
    ftr.add_argument("--through", type=int, default=1000)
    ftr.add_argument("--phase8-seed", type=Path, help="optional full historical authority archive; embedded pinned authority pack is default")
    ftr.add_argument("--snapshot-interval", type=int)
    ftr.add_argument("--reset", action="store_true")
    fts = fsp.add_parser("fixed-grammar-status"); fts.add_argument("--output", required=True, type=Path)
    fsp.add_parser("native-sentinel")
    fsp.add_parser("theorem-status")
    fsp.add_parser("exact-reopen-status")
    fer = fsp.add_parser("exact-reopen-audit")
    fer.add_argument("--phase8-seed", type=Path)
    fer.add_argument("--output", required=True, type=Path)
    fer.add_argument("--reason", required=True)
    fer.add_argument("--through", type=int, default=9)
    fer.add_argument("--allow-full-panel", action="store_true")

    sch = sp.add_parser("science-chain", help="Controlled S-series plus adaptive preregistered R-series chaining")
    schsp = sch.add_subparsers(dest="science_chain_cmd", required=True)
    schv = schsp.add_parser("validate"); schv.add_argument("registration", type=Path)
    schf = schsp.add_parser("freeze"); schf.add_argument("registration", type=Path)
    schr = schsp.add_parser("run"); schr.add_argument("registration_or_chain")
    schrs = schsp.add_parser("resume"); schrs.add_argument("chain_id")
    schst = schsp.add_parser("status"); schst.add_argument("chain_id")

    sc = sp.add_parser("scout-chain"); ssp = sc.add_subparsers(dest="scout_chain_cmd", required=True)
    sf = ssp.add_parser("freeze"); sf.add_argument("--parent-pin", required=True, type=Path); sf.add_argument("--parent-bundle", type=Path)
    mx = sf.add_mutually_exclusive_group(required=True); mx.add_argument("--through", type=int); mx.add_argument("--max-levels", type=int)
    sf.add_argument("--budgets", required=True); sf.add_argument("--wheel-sha", required=True); sf.add_argument("--primitive-file", type=Path); sf.add_argument("--primitive-sha", required=True); sf.add_argument("--chain-id")
    ss = ssp.add_parser("start"); ss.add_argument("--plan", required=True, type=Path)
    sst = ssp.add_parser("status"); sst.add_argument("--chain-id", required=True)
    spa = ssp.add_parser("pause"); spa.add_argument("--chain-id", required=True)
    sr = ssp.add_parser("resume"); sr.add_argument("--chain-id", required=True)
    sv = ssp.add_parser("verify"); sv.add_argument("--chain-id", required=True)
    srel = ssp.add_parser("release"); srel.add_argument("--chain-id", required=True)
    return p

def _mode(paths,requested):
    m=ExecutionModeManager(paths); active=m.active()
    if requested=='auto': return active
    if requested=='hold': return m.hold_record()
    if requested=='graduated':
        if active.get('mode')!='GRADUATED_SCIENCE': raise RuntimeError('graduated mode requested but no successful frozen graduation record is active')
        return active


def _conveyor_resource_dir() -> Path:
    return Path(str(files('infinity_grid').joinpath('resources/conveyor')))


def _chain_freeze(root,a):
    root.ensure(); store=ArtifactStore(root.store)
    pin_sha=sha256_file(a.parent_pin)
    pin=json.loads(a.parent_pin.read_text(encoding='utf-8'))
    bootstrap_material={}

    if pin.get('schema_id') == 'IG_SCOUT_CONVEYOR_PARENT_CERTIFICATE_V0_1':
        validate_eligible_parent_certificate(pin)
        if a.parent_bundle is None:
            raise RuntimeError('later-level chain freeze requires --parent-bundle')
        cap=load_verified_parent_capsule(a.parent_bundle)
        if cap['parent_file_sha256'] != pin_sha or canonical_sha256(cap['parent']) != canonical_sha256(pin):
            raise RuntimeError('parent certificate and parent capsule disagree')
        start=int(pin['level'])
        tmpdir=root.workspace/f'chain-freeze-parent-L{start}'
        shutil.rmtree(tmpdir,ignore_errors=True); tmpdir.mkdir(parents=True,exist_ok=True)
        sel=tmpdir/'SELECTED_RELATION.json'; sel.write_bytes(cap['selected_relation_bytes'])
        base=tmpdir/'PARENT_NORMALIZED_BASELINE.json'; base.write_bytes(cap['normalized_baseline_bytes'])
        sel_art=store.put_file(sel,logical_role='SCOUT_BOOTSTRAP_SELECTED_RELATION',source_name=sel.name)
        nb_art=store.put_file(base,logical_role='SCOUT_BOOTSTRAP_NORMALIZED_BASELINE',source_name=base.name)
        bootstrap_material={
            'selected_relation_artifact_sha256':sel_art['sha256'],
            'normalized_baseline_artifact_sha256':nb_art['sha256'],
            'parent_capsule_sha256':cap['capsule_sha256'],
            'parent_certificate_file_sha256':pin_sha,
            'parent_certificate_sha256':pin['certificate_sha256'],
        }
    else:
        expected_bundle=None
        if a.parent_bundle is not None: expected_bundle='bb445d7e9a630398831d7fdb6295859173857d5225f48ad12ae3ad626f88e1ef'
        pin=verify_bootstrap_parent(parent_pin_path=a.parent_pin,expected_pin_file_sha256=pin_sha,closeout_bundle_path=a.parent_bundle,expected_closeout_bundle_sha256=expected_bundle)
        if pin_sha!='bd8b607fc7113dbebaba81da8bf153eea32d2a149a7fcad86da9e95f37a3abec': raise RuntimeError('Phase-0 bootstrap pin identity mismatch')
        closeout=a.parent_pin.parent.parent
        sel=closeout/'06_RESULTS/SELECTED_RELATION.json'; base=closeout/'06_RESULTS/L16_FORWARD_BASELINE.json'; cls=closeout/'06_RESULTS/L16_FORWARD_CLASSIFICATION.json'
        if sha256_file(sel)!=pin['root_material']['file_sha256']: raise RuntimeError('bootstrap selected relation file mismatch')
        sel_art=store.put_file(sel,logical_role='SCOUT_BOOTSTRAP_SELECTED_RELATION',source_name='SELECTED_RELATION.json')
        baseline=json.loads(base.read_text()); classification=json.loads(cls.read_text()); norm=normalize_parent_baseline(level=16,baseline=baseline,classification=classification)
        tmp=root.workspace/'conveyor-bootstrap-normalized-L16.json'; write_json_atomic(tmp,norm); nb_art=store.put_file(tmp,logical_role='SCOUT_BOOTSTRAP_NORMALIZED_BASELINE',source_name='PARENT_NORMALIZED_BASELINE.json')
        start=16
        bootstrap_material={'selected_relation_artifact_sha256':sel_art['sha256'],'normalized_baseline_artifact_sha256':nb_art['sha256']}

    if a.primitive_file is not None:
        if sha256_file(a.primitive_file)!=a.primitive_sha: raise RuntimeError('primitive file SHA mismatch')
        prim_art=store.put_file(a.primitive_file,logical_role='PRIMITIVE_DATA',source_name='primitive.pkl')
    else:
        v=store.verify(a.primitive_sha)
        if v.get('status')!='PASS': raise RuntimeError('primitive artifact is not installed; supply --primitive-file')
        prim_art={'sha256':a.primitive_sha}
    budgets=_load_json_arg(a.budgets); through=a.through if a.through is not None else start+int(a.max_levels)
    if through<=start: raise ValueError('finite chain must include at least one target level')
    rid={'source_sha256':source_sha256(),'wheel_sha256':a.wheel_sha,'runtime_sha256':runtime_sha256(),'package_version':__version__}
    chain_id=a.chain_id or f"scout-chain-l{start}-to-l{through}-{canonical_sha256({'pin':pin_sha,'through':through,'budgets':budgets,'runtime':rid})[:16]}"
    plan=freeze_chain_plan(chain_id=chain_id,bootstrap_parent_pin=pin,bootstrap_parent_pin_sha256=pin_sha,runtime_identity=rid,primitive_identity={'sha256':a.primitive_sha,'artifact_sha256':prim_art['sha256']},resource_dir=_conveyor_resource_dir(),max_target_level=through,finite_limits=budgets,bootstrap_material=bootstrap_material)
    cdir=ScoutChainController(root).chain_dir(chain_id); (cdir/'parents').mkdir(exist_ok=True)
    write_json_atomic(cdir/'chain_plan.json',plan); write_json_atomic(cdir/'bootstrap_parent.json',pin)
    if pin.get('schema_id') == 'IG_SCOUT_CONVEYOR_PARENT_CERTIFICATE_V0_1':
        parent_name=f"L{start}-ELIGIBLE_AUTO_PARENT.json"
        write_json_atomic(cdir/'parents'/parent_name,pin)
        write_json_atomic(cdir/'current_parent.json',{'status':'COMMITTED','level':start,'certificate_sha256':pin['certificate_sha256'],'certificate_path':parent_name})
        shutil.copy2(a.parent_bundle,cdir/'bootstrap_parent_capsule.zip')
    write_json_atomic(cdir/'chain_status.json',{'state':'FROZEN','chain_plan_sha256':plan['chain_plan_sha256'],'start_level':start,'next_level':start+1,'science_executed':False})
    hdir=cdir/'handoff'; (hdir/'wheelhouse').mkdir(parents=True,exist_ok=True)
    active_ptr=root.store/'graduation'/'active.json'
    if not active_ptr.is_file(): raise RuntimeError('runtime graduation record is not installed')
    active=json.loads(active_ptr.read_text()); grad_path=root.store/'graduation'/f"{active.get('graduation_sha256')}.json"
    if not grad_path.is_file(): raise RuntimeError('active runtime graduation record missing')
    shutil.copy2(grad_path,hdir/'UNIFIED_RUNTIME_GRADUATION_RECORD.json')
    conv_path=root.store/'conveyor'/'graduation.json'
    if not conv_path.is_file(): raise RuntimeError('conveyor graduation record is not installed')
    shutil.copy2(conv_path,hdir/'CONVEYOR_GRADUATION_RECORD.json')
    wheels=[q for q in (root.app/'wheelhouse').glob('*.whl') if sha256_file(q)==a.wheel_sha]
    if len(wheels)!=1: raise RuntimeError('exact frozen runtime wheel unavailable in app/wheelhouse')
    shutil.copy2(wheels[0],hdir/'wheelhouse'/wheels[0].name)
    controller=ScoutChainController(root); dry_packet,_=controller.derive_and_freeze_next_packet(plan,cdir)
    files=[]
    for q in sorted(x for x in hdir.rglob('*') if x.is_file() and x.name!='HANDOFF_MANIFEST.json'):
        files.append({'path':q.relative_to(hdir).as_posix(),'sha256':sha256_file(q),'size_bytes':q.stat().st_size})
    handoff={'schema_id':'IG_SCOUT_CONVEYOR_EXECUTION_HANDOFF_V0_19_5','chain_id':chain_id,'chain_plan_sha256':plan['chain_plan_sha256'],'dry_packet_id':dry_packet['packet_id'],'dry_packet_sha256':dry_packet['packet_sha256'],'files':files,'science_executed':False}
    write_json_atomic(hdir/'HANDOFF_MANIFEST.json',handoff)
    return {'status':'PASS','chain_id':chain_id,'plan':str(cdir/'chain_plan.json'),'chain_plan_sha256':plan['chain_plan_sha256'],'state':'FROZEN','start_level':start,'next_level':start+1,'science_executed':False,'dry_packet_sha256':dry_packet['packet_sha256'],'handoff_manifest_sha256':sha256_file(hdir/'HANDOFF_MANIFEST.json')}


def _chain_release(root,chain_id):
    cdir=ScoutChainController(root).chain_dir(chain_id); out=root.releases/f'{chain_id}-closeout.zip'; out.parent.mkdir(parents=True,exist_ok=True)
    files=sorted(p for p in cdir.rglob('*') if p.is_file())
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            rel=p.relative_to(cdir).as_posix(); info=zipfile.ZipInfo(rel,date_time=(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644<<16; z.writestr(info,p.read_bytes())
    return {'status':'PASS','archive':str(out),'sha256':sha256_file(out),'files':len(files)}


def _historical_main(argv=None):
    from .invocation import InvocationRefused
    raise InvocationRefused('REGISTERED_COMMAND_REQUIRED', 'historical-cli')

    args = build_parser().parse_args(argv)
    from .v05_route_closure import classify_external_cli, PASSIVE_SUBMIT, REJECT, rejection_record
    route_decision=classify_external_cli(args)
    if route_decision==PASSIVE_SUBMIT:
        from .v05_passive_intake import submit_passive_request
        request_obj=json.loads(args.request.read_text(encoding="utf-8"))
        _dump(submit_passive_request(args.intake_root,request_obj)); return 0
    if route_decision==REJECT:
        sub=None
        for attr in ("request_cmd","verify_cmd","protocol_cmd","claim_cmd","graduation_cmd","uplift_cmd","graph_cmd","contracts_cmd","state_cmd","test_cmd","jumpstart_cmd","science_cmd","frontier_cmd","science_chain_cmd","scout_chain_cmd","release_cmd","import_cmd"):
            value=getattr(args,attr,None)
            if value is not None: sub=value; break
        _dump(rejection_record(args.cmd if sub is None else f"{args.cmd}:{sub}")); return 4
    root = resolve_root(args.root)
    try:
        if args.cmd == "init":
            root = resolve_root(args.root_arg or args.root).ensure()
            ProtocolRegistry().install_into_store(root)
            out = {"status": "PASS", "catalogue": Catalogue(root).initialize(), "graph": GraphStore(root).validate_graph()}
            _dump(out); return 0

        root.ensure()
        mode = _mode(root, args.science_mode)
        if args.cmd == "contracts":
            if args.contracts_cmd == "registry":
                out = {"status": "PASS", "release": release_identity(), "registry": payload_registry()}
            elif args.contracts_cmd == "validate":
                if args.envelope:
                    out = verify_envelope(load_envelope(args.path))
                else:
                    from .contracts import classify_json_file
                    out = classify_json_file(args.path)
            elif args.contracts_cmd == "wrap":
                out = migrate_json_file_to_envelope(args.source, args.output, semantic_profile_id=args.profile)
            elif args.contracts_cmd == "migrate-v026":
                out = migrate_unschematized_v026_file_to_envelope(args.source, args.output, semantic_profile_id=args.profile)
            else:
                report = scan_legacy_json_tree(args.tree, baseline={"decoder_version": "0.26.0"})
                if args.output:
                    write_json_atomic(args.output, report)
                out = report
        elif args.cmd == "state":
            if args.state_cmd == "verify":
                out = verify_current_state(load_current_state(args.path))
            elif args.state_cmd == "show":
                out = load_current_state(args.path)
            else:
                state, report = reconcile_legacy_context(args.context_dir, gate1_release=release_identity(), as_of_utc=args.as_of_utc or utc_now())
                write_current_state(args.output, state)
                report["output"] = str(args.output)
                out = report
        elif args.cmd == "doctor":
            base = doctor(root)
            out = {"status": base["status"], "base": base}
            if args.graph:
                targets = [GraphStore(root).resolve(t)["node_id"] for t in args.target] if args.target else None
                gd = graph_doctor(root, targets)
                out["graph"] = gd
                out["status"] = "PASS" if base["status"] == "PASS" and gd["status"] == "PASS" else "FAIL"
        elif args.cmd == "import":
            if args.import_cmd == "artifact": out = ArtifactStore(root.store).put_file(args.path, logical_role=args.role)
            elif args.import_cmd == "dataset": out = DatasetStore(ArtifactStore(root.store)).import_directory(args.path, logical_role=args.role)
            else: out = GraphImporter(root).import_archive(args.archive)
        elif args.cmd == "catalog": out = Catalogue(root).rebuild()
        elif args.cmd == "protocol":
            reg = ProtocolRegistry(); out = reg.list() if args.protocol_cmd == "list" else reg.get(args.protocol_id)
        elif args.cmd == "plan":
            if args.plan_cmd == "create":
                ev = {"origin": args.origin, "scope": args.scope, "disposition": args.disposition, "authority": args.authority, "protocol_label": args.protocol_label}
                plan = create_plan(protocol_id=args.protocol, protocol_label=args.protocol_label, subject=_load_json_arg(args.subject), input_dataset_sha256=args.input_dataset, input_role=args.input_role, runner=args.runner, evidence=ev, input_flags=_pairs(args.input_flag), run_id=args.run_id)
                write_json_atomic(args.output, plan); out = {"status": "PASS", "plan": str(args.output), "plan_sha256": plan["plan_sha256"], "run_id": plan["run_id"]}
            else:
                plan = json.loads(args.plan.read_text()); sha, _ = Controller(root).validate_plan(plan, execution_mode=mode); out = {"status": "PASS", "plan_sha256": sha, "run_id": plan["run_id"]}
        elif args.cmd == "run": out = Controller(root).run(json.loads(args.plan.read_text()), execution_mode=mode)
        elif args.cmd == "resume": out = Controller(root).resume(args.run_id, execution_mode=mode)
        elif args.cmd == "status":
            if args.run_id:
                p = root.runs / args.run_id / "run.json"; out = json.loads(p.read_text()) if p.is_file() else {"status": "NOT_FOUND", "run_id": args.run_id}
            else: out = {"status": "PASS", "root": str(root.root), "execution_mode": mode, "runs": len(list(root.runs.glob("*/run.json")))}
        elif args.cmd == "verify":
            if args.verify_cmd == "artifact": out = ArtifactStore(root.store).verify(args.sha256, args.size)
            elif args.verify_cmd == "run": out = verify_run(root, args.run_id)
            elif args.verify_cmd == "release": out = verify_release_archive(args.archive)
            else: out = verify_graph_export_archive(args.archive)
        elif args.cmd == "release": out = ReleaseGenerator(root).create(args.run_id, args.mode)
        elif args.cmd == "claim":
            claims = list_claims(root); out = claims if args.claim_cmd == "list" else [c for c in claims if c["claim_id"] == args.claim_id]
        elif args.cmd == "lock":
            override = {"allow_recovery": True, "reason": args.operator_override_reason, "operator_action": "CLI_OVERRIDE"} if args.operator_override_reason else None
            out = recover_stale_lock(root.runs / args.run_id, operator_override=override)
        elif args.cmd == "graduation":
            mgr = ExecutionModeManager(root); out = mgr.active() if args.graduation_cmd == "status" else mgr.install_graduation(json.loads(args.record.read_text()))
        elif args.cmd == "uplift":
            from .uplift_architecture import uplift_contract, nomenclature_contract
            from .uplift_structural import implementation_spec, run_g2_s0_s1
            from .uplift_campaign import list_registered_uplift_experiments, run_native_campaign_stage
            if args.uplift_cmd == "show":
                out = {"status":"PASS", "architecture":uplift_contract(), "nomenclature":nomenclature_contract(), "s0_s1_spec":implementation_spec(), "native_registry":list_registered_uplift_experiments()}
            elif args.uplift_cmd == "registry":
                out = list_registered_uplift_experiments()
            elif args.uplift_cmd == "native-run":
                bindings={}
                for item in args.input:
                    k,sep,v=item.partition('=')
                    if not sep or not k or not v: raise ValueError(f'expected KEY=PATH: {item}')
                    bindings[k]=Path(v)
                workers=args.workers
                if isinstance(workers,str) and workers.upper()!='AUTO': workers=int(workers)
                out = run_native_campaign_stage(run_root=args.output,experiment_id=args.experiment_id,inputs=bindings,requested_workers=workers,lease_root=args.lease_root)
            else:
                out = run_g2_s0_s1(args.output, workers=args.workers)
        elif args.cmd == "graph":
            graph = GraphStore(root)
            if args.graph_cmd == "show": out = graph.resolve(args.reference, version=args.version)
            elif args.graph_cmd == "roots": out = minimum_preservation_set(graph, args.targets)
            elif args.graph_cmd == "closure": out = evidence_closure(graph, args.targets[0]) if args.evidence and len(args.targets) == 1 else computation_closure(graph, args.targets)
            elif args.graph_cmd == "validate": out = graph.validate_graph(targets=[graph.resolve(t)["node_id"] for t in args.target] if args.target else None)
            elif args.graph_cmd == "ingest-existing": out = ingest_existing_records(root)
            else: out = ingest_o7_unresolved_case(graph, preregistration_bundle=args.preregistration, qualified_engine_bundle=args.engine, repair_bundle=args.repair, postrun_analysis_bundle=args.postrun_analysis)
        elif args.cmd == "resolve": out = GraphStore(root).resolve(args.reference, version=args.version)
        elif args.cmd == "replay": out = GraphReplayEngine(root).replay(args.target, force_rebuild=args.clean, workspace=args.workspace)
        elif args.cmd == "depends": out = computation_closure(GraphStore(root), [args.target])
        elif args.cmd == "why": out = evidence_closure(GraphStore(root), args.target)
        elif args.cmd == "roots": out = minimum_preservation_set(GraphStore(root), args.targets)
        elif args.cmd == "gc": out = gc_dry_run(root, args.target)
        elif args.cmd == "export": out = GraphExporter(root).create(args.targets, args.mode, args.output, forbidden_inferences=args.forbidden_inference)
        elif args.cmd == "graph-import": out = GraphImporter(root).import_archive(args.archive)
        elif args.cmd == "jumpstart":
            jr = JumpstartRuntime(root)
            if args.jumpstart_cmd == "plan": out = jr.planner.plan(args.target)
            elif args.jumpstart_cmd == "prepare": out = jr.prepare(args.target, workspace=args.workspace, clean=args.clean)
            elif args.jumpstart_cmd == "smoke": out = jr.smoke(args.target, workspace=args.workspace, clean=args.clean, timeout_seconds=args.timeout)
            elif args.jumpstart_cmd == "launch": out = jr.launch(args.target, workspace=args.workspace, clean=args.clean, detach=not args.foreground)
            elif args.jumpstart_cmd == "status": out = jr.status(args.target, workspace=args.workspace)
            elif args.jumpstart_cmd == "register-profile": out = JumpstartProfileStore(root).register(json.loads(args.profile.read_text(encoding="utf-8")), GraphStore(root))
            elif args.jumpstart_cmd == "profiles": out = {"status":"PASS", "profiles": JumpstartProfileStore(root).list()}
            elif args.jumpstart_cmd == "contract":
                node = jr.planner._resolve_target(args.target)
                out = ReplayContractStore(root).resolve_for_target(node["node_id"])
            elif args.jumpstart_cmd == "contracts":
                out = {"status":"PASS", "contracts": ReplayContractStore(root).list()}
            else:
                ps = JumpstartProfileStore(root).list(); cs = ReplayContractStore(root)
                rows=[]
                for prof in sorted(ps, key=lambda x:x["profile_id"]):
                    c=cs.resolve_for_target(prof["target_node_id"])
                    rows.append({"profile_id":prof["profile_id"],"target_alias":prof["target_alias"],"target_node_id":prof["target_node_id"],"replay_class":c["replay_class"],"authority_class":c["authority_class"],"historical_identity":c["historical_identity"],"recomputes":c["recomputes"],"does_not_recompute":c["does_not_recompute"],"limitations":c["limitations"]})
                out={"schema_id":"IG_JUMPSTART_REPLAY_COVERAGE_V1","status":"PASS","profiles":len(rows),"rows":rows}
        elif args.cmd == "test":
            st = ScientificTestStore(root)
            if args.test_cmd == "list": out={"status":"PASS","tests":st.list(level=args.level)}
            elif args.test_cmd == "show": out=st.show(args.test_id)
            elif args.test_cmd == "replay": out=st.replay(args.test_id,workspace=args.workspace,clean=not args.no_clean)
            else: out=st.coverage()
        elif args.cmd == "science":
            from .scientific_architecture import ScientificProtocolRegistry
            from .research_frontier import load_research_frontier, verify_research_frontier, current_frontier_status
            from .run_plan_v2 import load_run_plan_v2, validate_run_plan_v2, compile_run_plan_v2, execute_o7_compatibility_run_plan, execute_o7_resumable_maturation_run_plan
            from .run_operations import summarize_run_status, open_existing_run_operations
            if args.science_cmd == "registry":
                out = ScientificProtocolRegistry().verification_result()
            elif args.science_cmd == "frontier":
                if args.science_frontier_cmd == "show":
                    out = current_frontier_status() if args.path is None else {"status":"PASS","frontier":load_research_frontier(args.path),"frontier_science_sha256":canonical_sha256(load_research_frontier(args.path))}
                else:
                    fr_obj = load_research_frontier(args.path) if args.path is not None else load_research_frontier()
                    out = verify_research_frontier(fr_obj)
            elif args.science_cmd == "shadow":
                from .collective_closure_shadow import k4_shadow_observation, validate_k4_shadow_prediction
                if args.science_shadow_cmd == "k4-validate":
                    out = validate_k4_shadow_prediction()
                else:
                    graph = json.loads(args.graph.read_text(encoding="utf-8"))
                    out = k4_shadow_observation(int(graph["n"]), graph["edges"])
            else:
                if args.science_runplan_cmd == "status":
                    out = summarize_run_status(args.output)
                elif args.science_runplan_cmd == "external-request":
                    ops = open_existing_run_operations(args.output)
                    out = ops.write_external_request(args.depth)
                elif args.science_runplan_cmd == "ack-external":
                    ops = open_existing_run_operations(args.output)
                    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
                    out = ops.record_external_ack(args.depth, receipt)
                elif args.science_runplan_cmd == "audit-checkpoints":
                    ops = open_existing_run_operations(args.output)
                    depths = ops.full_checkpoint_audit()
                    out = {"status":"PASS","run_id":ops.run_id,"verified_depths":depths,"verified_count":len(depths)}
                else:
                    plan = load_run_plan_v2(args.plan)
                    fr_obj = load_research_frontier(args.frontier) if args.frontier is not None else load_research_frontier()
                    if args.science_runplan_cmd == "validate": out = validate_run_plan_v2(plan, frontier=fr_obj)
                    elif args.science_runplan_cmd == "compile": out = compile_run_plan_v2(plan, frontier=fr_obj)
                    elif args.science_runplan_cmd == "execute-live": out = execute_o7_resumable_maturation_run_plan(plan, frontier=fr_obj, output=args.output, stop_after_depth=args.stop_after_depth, acknowledge_review=args.acknowledge_review)
                    else: out = execute_o7_compatibility_run_plan(plan, frontier=fr_obj)
        elif args.cmd == "oracle":
            from .oracle import oracle_manifest, verify_embedded_oracle, verify_corpus
            if args.oracle_cmd == "show": out = oracle_manifest()
            elif args.corpus_root: out = verify_corpus(args.corpus_root)
            else: out = verify_embedded_oracle()
        elif args.cmd == "frontier":
            from .frontier import load_frontier_spec, run_o7_topology_read_probe, run_o8_bp0, run_o8_graduation, run_o9_bp0, run_o_frontier_auto_advance
            from .pregeometry import load_pregeometry_spec, run_pregeometry_scout
            from .regime_scanner import load_regime_scanner_spec, run_o_regime_scanner
            from .regime_scanner_validation import load_depth_service_validation_spec, run_depth_erasing_service_validation
            from .fixed_grammar_transport import load_streaming_regime_spec, run_fixed_grammar_transport, fixed_grammar_transport_status
            from .semantic_sentinel import action_registry, verify_native_semantics
            from .theorem_registry import theorem_premise_registry, verify_all_theorems
            from .exact_reopen_auditor import reopen_status, run_exact_reopen_audit
            from .exact_carrier_unblinding import load_exact_carrier_unblinding_spec
            from .execution import load_unified_parallel_runtime_spec
            if args.frontier_cmd == "show-spec":
                if args.name in {"pregeometry","meta-grammar","o10-plus"}: out = load_pregeometry_spec(args.name)
                elif args.name == "regime-scanner": out = load_regime_scanner_spec()
                elif args.name == "regime-service-validation": out = load_depth_service_validation_spec()
                elif args.name == "fixed-grammar-transport": out = load_streaming_regime_spec()
                elif args.name == "native-semantics": out = action_registry()
                elif args.name == "theorem-premises": out = theorem_premise_registry()
                elif args.name == "exact-carrier-unblinding": out = load_exact_carrier_unblinding_spec()
                elif args.name == "unified-execution": out = load_unified_parallel_runtime_spec()
                else: out = load_frontier_spec(args.name)
            elif args.frontier_cmd == "o7-read-probe": out = run_o7_topology_read_probe(args.o7_graduation_root, args.post_o7_root, args.o7_runtime_root, args.output)
            elif args.frontier_cmd == "o8-bp0": out = run_o8_bp0(args.o7_graduation_root, args.o7_runtime_root, args.o7_read_probe, args.output)
            elif args.frontier_cmd == "o8-graduate": out = run_o8_graduation(args.o7_graduation_root, args.o8_bp0, args.frontier_authority_root, args.output)
            elif args.frontier_cmd == "o9-bp0": out = run_o9_bp0(args.o7_graduation_root, args.o7_runtime_root, args.o8_graduation, args.frontier_authority_root, args.output)
            elif args.frontier_cmd == "pregeometry-scout": out = run_pregeometry_scout(o7_graduation_root=args.o7_graduation_root, post_o7_root=args.post_o7_root, grrl_root=args.grrl_root, phase8_root=args.phase8_root, output=args.output)
            elif args.frontier_cmd == "regime-scan": out = run_o_regime_scanner(args.phase8_seed, args.output, max_level=args.through, honor_earned_laws=not args.ignore_earned_laws)
            elif args.frontier_cmd == "regime-service-validate": out = run_depth_erasing_service_validation(args.phase8_seed, args.output, through=args.through)
            elif args.frontier_cmd == "fixed-grammar-transport": out = run_fixed_grammar_transport(args.output, through=args.through, phase8_seed=args.phase8_seed, snapshot_interval=args.snapshot_interval, reset=args.reset)
            elif args.frontier_cmd == "fixed-grammar-status": out = fixed_grammar_transport_status(args.output)
            elif args.frontier_cmd == "native-sentinel": out = verify_native_semantics(raise_on_change=False)
            elif args.frontier_cmd == "theorem-status": out = verify_all_theorems(raise_on_stale=False)
            elif args.frontier_cmd == "exact-reopen-status": out = reopen_status()
            elif args.frontier_cmd == "exact-reopen-audit": out = run_exact_reopen_audit(phase8_seed=args.phase8_seed, output=args.output, reopen_reason=args.reason, through=args.through, allow_full_panel=args.allow_full_panel)
            else: out = run_o_frontier_auto_advance(args.o7_graduation_root, args.o7_runtime_root, args.o8_bp0, args.frontier_authority_root, args.output)
        elif args.cmd == "science-chain":
            from .v05_chain import CHAIN_SCHEMA_V1, ScientificChainController, validate_chain_registration
            from .v05_stage_registry import register_controller_only_stage_handlers
            cc = ScientificChainController(root.runs / "scientific-chains", decoder_paths=root)
            # New scientific execution is controller-only.  Historical private
            # executors are loaded lazily only when replaying an already-frozen
            # V1 chain; they are never registered for a new V2 route.
            register_controller_only_stage_handlers(cc)

            def _register_legacy_v1_if_frozen(chain_id: str) -> None:
                rp = root.runs / "scientific-chains" / str(chain_id) / "chain_registration.json"
                if not rp.is_file():
                    return
                frozen = json.loads(rp.read_text(encoding="utf-8"))
                if frozen.get("schema_id") != CHAIN_SCHEMA_V1:
                    return
                from .g6_stage_executors import register_g6_chain_executors
                from .g6_s1_repair import register_g6_s1_repair_executor
                from .g6_s2_repaired import register_g6_s2_repaired_executor
                from .g6_s3_repaired import register_g6_s3_repaired_executor
                register_g6_chain_executors(cc)
                register_g6_s1_repair_executor(cc)
                register_g6_s2_repaired_executor(cc)
                register_g6_s3_repaired_executor(cc)

            if args.science_chain_cmd == "validate":
                reg = validate_chain_registration(json.loads(args.registration.read_text(encoding="utf-8")))
                out = {"status":"PASS","chain_id":reg["chain_id"],"registration_sha256":reg["registration_sha256"],"stages":len(reg["stages"])}
            elif args.science_chain_cmd == "freeze":
                reg = cc.freeze(json.loads(args.registration.read_text(encoding="utf-8")))
                out = {"status":"PASS","chain_id":reg["chain_id"],"registration_sha256":reg["registration_sha256"],"chain_dir":str(cc.chain_dir(reg["chain_id"]))}
            elif args.science_chain_cmd == "run":
                raw = args.registration_or_chain
                pth = Path(raw)
                if pth.is_file():
                    reg_obj = json.loads(pth.read_text(encoding="utf-8"))
                    _register_legacy_v1_if_frozen(str(reg_obj.get("chain_id", "")))
                    out = cc.run(reg_obj)
                else:
                    _register_legacy_v1_if_frozen(raw)
                    out = cc.run(raw)
            elif args.science_chain_cmd == "resume":
                _register_legacy_v1_if_frozen(args.chain_id)
                out = cc.resume(args.chain_id)
            else:
                out = cc.status(args.chain_id)
        elif args.cmd == "scout-chain":
            cc = ScoutChainController(root)
            if args.scout_chain_cmd == "freeze": out = _chain_freeze(root, args)
            elif args.scout_chain_cmd == "start": out = cc.start(json.loads(args.plan.read_text()))
            elif args.scout_chain_cmd == "status":
                cdir = cc.chain_dir(args.chain_id); plan = json.loads((cdir / "chain_plan.json").read_text()); out = rebuild_chain_state(chain_plan=plan, events_path=cdir / "events.jsonl", parent_certificates_dir=cdir / "parents"); out["execution_enabled"] = cc.graduation_record() is not None
            elif args.scout_chain_cmd == "pause":
                cdir = cc.chain_dir(args.chain_id); write_json_atomic(cdir / "pause.request.json", {"requested": True, "reason": "OPERATOR", "chain_id": args.chain_id}); out = {"status": "PASS", "state": "PAUSE_REQUESTED_SAFE_BOUNDARY", "chain_id": args.chain_id}
            elif args.scout_chain_cmd == "resume":
                cdir = cc.chain_dir(args.chain_id); (cdir / "pause.request.json").unlink(missing_ok=True); out = {"status": "PASS", "chain_id": args.chain_id, "note": "resume request accepted"}
            elif args.scout_chain_cmd == "verify":
                cdir = cc.chain_dir(args.chain_id); plan = json.loads((cdir / "chain_plan.json").read_text()); validate_chain_plan(plan); out = rebuild_chain_state(chain_plan=plan, events_path=cdir / "events.jsonl", parent_certificates_dir=cdir / "parents")
            else: out = _chain_release(root, args.chain_id)
        else: raise RuntimeError(f"unhandled command {args.cmd}")
        _dump(out)
        status = out.get("status", "PASS") if isinstance(out, dict) else "PASS"
        return 0 if status in {"PASS", "COMPLETE", "COMPLETE_VALID", "AUTO_ADVANCE_AUTHORIZED", "COMPLETE_LIMIT_REACHED", "PAUSED_OPERATOR", "PAUSED_OPERATIONAL", "PAUSED_SCIENTIFIC", "COMPLETE_RESOURCE_LIMIT", "UNRESOLVED", "SCIENTIFIC_STOP", "O8_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_THEOREM_ACCELERATED_V2_8"} else 2
    except Exception as exc:
        _dump(failure_record(error=exc, operation=getattr(args, "cmd", "cli"), context={"decoder_version": __version__}))
        return 2


def main(argv=None):
    from .controller import main as native_main
    return native_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
