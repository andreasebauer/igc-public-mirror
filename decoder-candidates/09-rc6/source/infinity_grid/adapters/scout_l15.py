from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from ..controller import register_runner
from ..canon import canonical_sha256, write_json_atomic
from ..spectroscope import run_scout_spectroscope, scientific_projection

FROZEN_CALIBRATION_SOURCE_SHA256 = "547498ded6c19b914447f2f005c3e0719e7eea2004cc4b3a0c997772eb62fc3e"
CALIBRATION_ACCELERATOR_ID = "SCOUT_L15_FARTHEST_MATH_DIST_EXACT_SCIENCE_EQUIVALENCE_V1"


def _read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def _science(o): return {k:v for k,v in o.items() if k!='cost'}
def _run(script:Path,cwd:Path,args=None):
    env=dict(os.environ); env['PYTHONHASHSEED']='0'; cmd=[sys.executable,str(script)]+list(args or [])
    p=subprocess.run(cmd,cwd=str(cwd),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    return p

def _classification(b):
    out={'status':'PASS','level':15,'evidence_label':'SCOUT_OBSERVED','authoritative':False,'classification_semantics':'Release-level grouping of already frozen Scout maturation/promotion labels; not a new scientific test.','observed':[],'persistent':[],'reorganized_or_refined':[],'destroyed_or_relieved':[],'escalated_for_review':[],'unresolved':[],'promotion_summary':b.get('promotion_summary',{}),'prohibitions':b.get('prohibitions',[])}
    for name,item in b['relations'].items():
        states=set(item.get('states',[])); promo=item.get('promotion')
        if 'SCOUT_NEW' in states:out['observed'].append(name)
        if states & {'PERSISTS','EXPANDS'}:out['persistent'].append(name)
        if states & {'REFINES','REORGANIZES','COMBINES','CLOSES'}:out['reorganized_or_refined'].append(name)
        if 'DESTROYS_OR_RELIEVES' in states:out['destroyed_or_relieved'].append(name)
        if promo in {'HIGH_PRIORITY_MATURATION_REVIEW','HIGH_PRIORITY_OFFICIAL_TEST'}:out['escalated_for_review'].append(name)
        if 'UNRESOLVED_DEPTH_SCOPE' in states or promo=='UNRESOLVED_DEPTH_SCOPE':out['unresolved'].append(name)
    for k in ['observed','persistent','reorganized_or_refined','destroyed_or_relieved','escalated_for_review','unresolved']:out[k]=sorted(set(out[k]))
    return out

def _copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def accelerated_calibration_source(source_path: Path) -> tuple[str, dict]:
    """Return a semantics-preserving speedup of the frozen Scout calibration script.

    Only the farthest-point distance kernel is replaced: the Python generator sum of
    squared coordinate differences becomes math.dist.  Selection ordering is checked
    against the frozen science artifact, so any tie/order change fails the replay.
    """
    raw = Path(source_path).read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != FROZEN_CALIBRATION_SOURCE_SHA256:
        raise RuntimeError(f'frozen Scout calibration source identity mismatch: {observed}')
    src = raw.decode('utf-8')
    old_import = 'import ast,hashlib,json,pickle,time,resource'
    if src.count(old_import) != 1:
        raise RuntimeError('unexpected Scout calibration import surface')
    src = src.replace(old_import, old_import + ',math')
    old = """    ch=[s]; used={s}; md=[sum((a-b)**2 for a,b in zip(nv[i],nv[s])) for i in range(len(rows))];md[s]=-1\n    while len(ch)<n:\n        k=max((i for i in range(len(rows)) if i not in used),key=lambda i:(md[i],rows[i]['sha256']))\n        ch.append(k);used.add(k);md[k]=-1\n        for i in range(len(rows)):\n            if md[i]>=0:md[i]=min(md[i],sum((a-b)**2 for a,b in zip(nv[i],nv[k])))\n"""
    new = """    ch=[s]; used={s}; md=[math.dist(nv[i],nv[s]) for i in range(len(rows))];md[s]=-1\n    while len(ch)<n:\n        k=max((i for i in range(len(rows)) if i not in used),key=lambda i:(md[i],rows[i]['sha256']))\n        ch.append(k);used.add(k);md[k]=-1\n        nk=nv[k]\n        for i in range(len(rows)):\n            if md[i]>=0:\n                d=math.dist(nv[i],nk)\n                if d<md[i]:md[i]=d\n"""
    if src.count(old) != 1:
        raise RuntimeError('unexpected Scout calibration farthest kernel')
    src = src.replace(old, new)
    patched_sha = hashlib.sha256(src.encode('utf-8')).hexdigest()
    return src, {
        'accelerator_id': CALIBRATION_ACCELERATOR_ID,
        'frozen_source_sha256': observed,
        'accelerated_source_sha256': patched_sha,
        'science_acceptance_rule': 'FULL_NON_COST_CALIBRATION_ARTIFACT_MUST_EQUAL_FROZEN_EXPECTED',
        'optimization_scope': 'DISTANCE_KERNEL_ONLY',
        'new_third_party_dependency': False,
    }

def execute(fixture:Path,root:Path):
    # WIDE
    wide=root/'wide'
    for q in ['code','inputs','results','evidence','checkpoints','metrics','logs']:(wide/q).mkdir(parents=True,exist_ok=True)
    _copy(fixture/'code/l15_scout_wide_reference.py',wide/'code/l15_scout_wide.py'); _copy(fixture/'inputs/GLOBAL_L14_PANEL_CHILDREN.json',wide/'inputs/GLOBAL_L14_PANEL_CHILDREN.json'); _copy(fixture/'inputs/primitive.pkl',wide/'inputs/primitive.pkl')
    p=_run(wide/'code/l15_scout_wide.py',wide); (wide/'logs/stdout.txt').write_text(p.stdout); (wide/'logs/stderr.txt').write_text(p.stderr)
    if p.returncode:raise RuntimeError(f'wide failed rc={p.returncode}: {p.stderr[-2000:]}')
    wide_obs=_read(wide/'results/L15_SCOUT_WIDE_RESULT.json'); wide_exp=_read(fixture/'expected/L15_SCOUT_WIDE_RESULT.json')
    # CALIBRATION.  The frozen algorithm is preserved, with a source-hash-guarded
    # standard-library distance-kernel acceleration.  Full non-cost output must equal
    # the frozen calibration artifact below, otherwise this route fails closed.
    cal=root/'calibration'
    for q in ['code','inputs','results','evidence','metrics','logs']:(cal/q).mkdir(parents=True,exist_ok=True)
    patched, accel_meta = accelerated_calibration_source(fixture/'code/l15_scout_calibration_port.py')
    (cal/'code/l15_scout_calibration.py').write_text(patched,encoding='utf-8')
    write_json_atomic(cal/'evidence/V05_CALIBRATION_ACCELERATOR.json',accel_meta)
    _copy(fixture/'inputs/GLOBAL_L14_PANEL_CHILDREN.json',cal/'inputs/GLOBAL_L14_PANEL_CHILDREN.json'); _copy(fixture/'inputs/primitive.pkl',cal/'inputs/primitive.pkl'); _copy(wide/'results/L15_SCOUT_WIDE_RESULT.json',cal/'inputs/L15_SCOUT_WIDE_RESULT.json'); _copy(wide/'evidence/SEALED_LANE_SELECTION.json',cal/'inputs/SEALED_LANE_SELECTION.json'); _copy(wide/'evidence/SELECTED_RELATION.json',cal/'inputs/SELECTED_RELATION.json')
    p=_run(cal/'code/l15_scout_calibration.py',cal); (cal/'logs/stdout.txt').write_text(p.stdout); (cal/'logs/stderr.txt').write_text(p.stderr)
    if p.returncode:raise RuntimeError(f'calibration failed rc={p.returncode}: {p.stderr[-2000:]}')
    cal_obs=_read(cal/'results/L15_SCOUT_CALIBRATION_RESULT.json'); cal_exp=_read(fixture/'expected/L15_SCOUT_CALIBRATION_RESULT.json')
    # SPECTROSCOPE
    spec=root/'spectroscope'; spec_obs=run_scout_spectroscope(wide/'evidence/SELECTED_RELATION.json',spec,15); spec_exp=_read(fixture/'expected/SCOUT_SPECTROSCOPE_RESULT.json')
    # Normalize through JSON exactly as the frozen v0.16 integration surface did.
    # This removes Python-only integer dictionary-key distinctions without changing science.
    spec_obs_normalized=json.loads(json.dumps(spec_obs,sort_keys=True))
    # BASELINE
    base=root/'baseline'; (base/'code').mkdir(parents=True,exist_ok=True); (base/'results').mkdir(parents=True,exist_ok=True); _copy(fixture/'code/scout_baseline.py',base/'code/scout_baseline.py')
    args=['--wide',str(wide/'results/L15_SCOUT_WIDE_RESULT.json'),'--calibration',str(cal/'results/L15_SCOUT_CALIBRATION_RESULT.json'),'--spectroscope',str(spec/'SCOUT_SPECTROSCOPE_RESULT.json'),'--out',str(base/'results'),'--level','15']
    p=_run(base/'code/scout_baseline.py',base,args); (base/'stdout.txt').write_text(p.stdout); (base/'stderr.txt').write_text(p.stderr)
    if p.returncode:raise RuntimeError(f'baseline failed rc={p.returncode}: {p.stderr[-2000:]}')
    base_obs=_read(base/'results/SCOUT_BASELINE.json'); base_exp=_read(fixture/'expected/SCOUT_BASELINE.json')
    # DEPTH
    depth=root/'depth_compact'
    for q in ['code','inputs','evidence','logs']:(depth/q).mkdir(parents=True,exist_ok=True)
    _copy(fixture/'code/replay_depth_compact.py',depth/'code/replay_depth_compact.py'); _copy(wide/'evidence/SELECTED_RELATION.json',depth/'inputs/L15_SELECTED_RELATION.json'); _copy(fixture/'evidence/BOUNDED_L14_L13_EDGES.json',depth/'evidence/BOUNDED_L14_L13_EDGES.json')
    p=_run(depth/'code/replay_depth_compact.py',depth); (depth/'logs/stdout.txt').write_text(p.stdout); (depth/'logs/stderr.txt').write_text(p.stderr)
    if p.returncode:raise RuntimeError(f'depth failed rc={p.returncode}: {p.stderr[-2000:]}')
    try:depth_obs=json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:raise RuntimeError(f'depth output parse failed: {e}')
    depth_exp=_read(fixture/'expected/L15_SCOUT_DEPTH_RESULT.json')
    classification=_classification(base_obs); class_exp=_read(fixture/'expected/L15_SCOUT_CLASSIFICATION.json')
    depth_expected_projection={
      'C0_classes':depth_exp.get('profile_refinement',{}).get('C0_classes'),
      'C1_classes':depth_exp.get('profile_refinement',{}).get('C1_classes'),
      'C2_classes':depth_exp.get('profile_refinement',{}).get('C2_classes'),
      'L14_used':depth_exp.get('bounded_depth_relation',{}).get('L14_objects'),
      'L15':depth_exp.get('bounded_depth_relation',{}).get('L15_objects'),
      'all_used_L14_single_source':'BOUNDED_DEPTH_ROOT_COLLAPSE_ONE_L13_SOURCE_PER_L14_ROOT' in depth_exp.get('roadmap',{}).get('maturation_labels',[]),
    }
    comparisons={
      'wide_science_fields_equal':_science(wide_obs)==_science(wide_exp),
      'calibration_science_fields_equal':_science(cal_obs)==_science(cal_exp),
      'spectroscope_science_fields_equal':scientific_projection(spec_obs_normalized)==scientific_projection(spec_exp),
      'baseline_science_fields_equal':_science(base_obs)==_science(base_exp),
      'depth_compact_replay_pass':depth_obs.get('status')=='PASS',
      'depth_compact_science_fields_equal':depth_obs.get('replay')==depth_expected_projection,
      'classification_equal':classification==class_exp,
    }
    principal={'status':'PASS' if all(comparisons.values()) else 'FAIL','target':'L15_ROADMAP_SCOUT_VERTICAL_SLICE','evidence_label':'SCOUT_OBSERVED','authoritative':False,'authority':'RECONNAISSANCE_ONLY','level':15,'comparisons':comparisons,
      'science_hashes':{'wide':canonical_sha256(_science(wide_obs)),'calibration':canonical_sha256(_science(cal_obs)),'spectroscope':canonical_sha256(scientific_projection(spec_obs_normalized)),'baseline':canonical_sha256(_science(base_obs)),'depth':canonical_sha256(_science(depth_obs)),'classification':canonical_sha256(classification)},
      'generation':wide_obs.get('generation'),'selection':wide_obs.get('selection'),'bounded_relation':wide_obs.get('bounded_relation'),'promotion_summary':base_obs.get('promotion_summary'),'classification':classification,'depth_scope':cal_obs.get('depth_ancestry'),'spectroscope_recognition':spec_obs.get('recognition_summary'),'prohibitions':sorted(set(wide_obs.get('prohibitions',[])+spec_obs.get('prohibitions',[]))),'handoff':'L16_DEFERRED_UNTIL_PHASE1_GRADUATION','v05_migration':{'calibration_accelerator':accel_meta}}
    return principal,{'wide':wide/'results/L15_SCOUT_WIDE_RESULT.json','calibration':cal/'results/L15_SCOUT_CALIBRATION_RESULT.json','spectroscope':spec/'SCOUT_SPECTROSCOPE_RESULT.json','baseline':base/'results/SCOUT_BASELINE.json','classification':None}


def _journal_payload(principal: dict) -> dict:
    # Cost-free scientific/result payload only.  Operational execution artifacts remain
    # in staging and are not trusted as the durable scientific task identity.
    return {'principal': principal}


@register_runner('adapter.scout_l15', v05_operation='SCOUT_L15_REFERENCE_REPLAY', call_style='context')
def stage_runner(*,context,plan,stage):
    ds=stage['params']['fixture_dataset_sha256']; fixture=context.materialize_dataset(ds,'fixture')
    journal=context.task_journal(); task_id='scout-l15-reference-replay'
    rec=journal.load(task_id); reused = rec is not None
    if rec is None:
        principal,_=execute(fixture,context.staging_path('execution'))
        if principal['status']!='PASS':raise RuntimeError(f"Scout mismatch: {principal['comparisons']}")
        journal.commit(task_id,_journal_payload(principal))
        rec=journal.load(task_id)
    principal=rec['payload']['principal']
    out=context.staging_path('L15_SCOUT_PRINCIPAL_RESULT.json'); write_json_atomic(out,principal)
    js=journal.summary()
    return {'outputs':{'L15_SCOUT_PRINCIPAL_RESULT.json':out},'stage_result':{'status':'PASS','target':'VS_SCOUT_L15','protocol_label':'SCOUT_OBSERVED','authority':'RECONNAISSANCE_ONLY','l16_status':'DEFERRED_NOT_RUN','science_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'logical_tasks':{'tasks_total':1,'tasks_reused':1 if reused else 0,'tasks_committed_this_attempt':0 if reused else 1,'journal':js}}}
