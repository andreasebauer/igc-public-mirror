from __future__ import annotations
"""Fail-closed engineering acceptance for G6:S8 performance-sensitive changes."""
import hashlib,json,statistics,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from .canon import canonical_sha256
S8_SENSITIVE_PATHS=('infinity_grid/g6_s8_intrinsic_descriptor.py','infinity_grid/g6_s8_evaluators.py','infinity_grid/v05_stage_runtime.py')
FIXTURE_REL='infinity_grid/resources/v05/G6_S8_OPTIMIZATION_GATE_FIXTURE_V1.json'
class OptimizationGateError(RuntimeError):pass
def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def sensitive_changes(parent:Path,candidate:Path)->list[str]:
    return [rel for rel in S8_SENSITIVE_PATHS if not (parent/rel).is_file() or not (candidate/rel).is_file() or _sha(parent/rel)!=_sha(candidate/rel)]
_PROBE='''import json,sys,time,hashlib\nfrom pathlib import Path\nroot=Path(sys.argv[1]);fixture=json.load(open(sys.argv[2]));sys.path.insert(0,str(root))\nfrom infinity_grid import g6_s8_evaluators as ev\nclass View:\n def call(self,name,*args):\n  if name=="AUTHORITY_BASIS":return {r:{"seed":r} for r in fixture["basis_refs"]}\n  if name=="PUBLIC_READ":\n   t=args[0]\n   if "n" not in t:return {"legal":True,"descriptor":"CAPS7_PLUS_H_CLASS_BAG","caps7":[1,0,0,0,0,0,0],"H_class_bag":{str(t.get("seed","B")):1}}\n   return {"legal":True,"descriptor":"CAPS7_PLUS_H_CLASS_BAG","caps7":[t["n"],len(t["edges"]),0,0,0,0,0],"H_class_bag":{x:t["H_classes"].count(x) for x in sorted(set(t["H_classes"]))}}\n  if name=="EXACT_RELATION_PROFILE":\n   tree,seed,op=args;raw=json.dumps([tree,seed,op],sort_keys=True,separators=(",",":")).encode();return {"exact_outcome_count":int(hashlib.sha256(raw).hexdigest()[:8],16)%97}\n  if name=="EXACT_RELATION_PROFILE_FAMILY":\n   tree,seeds,ops=args;vals=[]\n   for seed in seeds:\n    for op in ops:\n     raw=json.dumps([tree,seed,op],sort_keys=True,separators=(",",":")).encode();vals.append(int(hashlib.sha256(raw).hexdigest()[:8],16)%97)\n   return {"exact_outcome_counts":vals}\n  if name=="EXACT_RELATION":\n   tree,seed,op=args\n   if tree.get("n",0)<=1:return {"children":[]}\n   n=tree["n"]-1;edges=[e for e in tree["edges"] if e[0]<n and e[1]<n];ops=tree["edge_operators"][:len(edges)];child={"n":n,"H_classes":tree["H_classes"][:n],"edges":edges,"edge_operators":ops}\n   return {"children":[child]}\n  raise RuntimeError(name)\nev.current_kernel_view=lambda:View()\npayload={"future_depth":3,"outer_execution_context_index":0,"recursive_prefix_schedule":[1,8,32,124],"s7_class_id":fixture["class_id"],"class_members":fixture["class_members"],"basis_refs":fixture["basis_refs"],"operator_basis":fixture["operator_basis"],"max_recursive_states":250000,"max_exact_relation_calls":250000,"max_worker_rss_bytes":2**40}\nstarted=time.perf_counter();out=ev.s8_s7_class_recursive_prefix_comparator_evaluator(payload);elapsed=time.perf_counter()-started\nsemantic={k:v for k,v in out.items() if k!="metrics"}\nprint(json.dumps({"semantic":semantic,"elapsed_seconds":elapsed},sort_keys=True))'''
def _probe(source:Path,fixture:Path)->dict[str,Any]:
    p=subprocess.run([sys.executable,'-I','-S','-c',_PROBE,str(source),str(fixture)],cwd=str(source),env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8','PYTHONNOUSERSITE':'1'},capture_output=True,text=True,timeout=60)
    if p.returncode!=0:raise OptimizationGateError('OPTIMIZATION_PROBE_FAILED:'+p.stderr[-1000:])
    return json.loads(p.stdout.strip().splitlines()[-1])
def run_s8_optimization_acceptance_gate(parent_root:str|Path,candidate_root:str|Path)->dict[str,Any]:
    parent=Path(parent_root).resolve();candidate=Path(candidate_root).resolve();changed=sensitive_changes(parent,candidate)
    if not changed:return {'schema_id':'IG_G6_S8_OPTIMIZATION_ACCEPTANCE_V1','status':'NOT_APPLICABLE','changed_paths':[],'scientific_effect':'NONE'}
    fixture=candidate/FIXTURE_REL
    if not fixture.is_file():raise OptimizationGateError('OPTIMIZATION_FIXTURE_MISSING')
    fobj=json.loads(fixture.read_text());expected={'schema_id','fixture_origin','class_id','class_members','basis_refs','operator_basis','benchmark'}
    if set(fobj)!=expected or fobj['schema_id']!='IG_G6_S8_OPTIMIZATION_GATE_FIXTURE_V1':raise OptimizationGateError('OPTIMIZATION_FIXTURE_SCHEMA')
    pr=_probe(parent,fixture);cr=_probe(candidate,fixture)
    if canonical_sha256(pr['semantic'])!=canonical_sha256(cr['semantic']):raise OptimizationGateError('OPTIMIZATION_PARENT_CANDIDATE_MISMATCH')
    one=_probe(candidate,fixture)
    with ThreadPoolExecutor(max_workers=4) as ex: four=list(ex.map(lambda _:_probe(candidate,fixture),range(4)))
    one_sha=canonical_sha256(one['semantic']);four_shas=[canonical_sha256(x['semantic']) for x in four]
    if any(x!=one_sha for x in four_shas):raise OptimizationGateError('OPTIMIZATION_1V4_SEMANTIC_MISMATCH')
    nrep=int(fobj['benchmark']['repeats'])
    parent_reps=[];reps=[];paired_ratios=[]
    for _ in range(nrep):
        pv=float(_probe(parent,fixture)['elapsed_seconds']);cv=float(_probe(candidate,fixture)['elapsed_seconds'])
        parent_reps.append(pv);reps.append(cv);paired_ratios.append(cv/max(pv,0.001))
    med=float(statistics.median(reps));parent_elapsed=float(statistics.median(parent_reps));ratio_med=float(statistics.median(paired_ratios));rel=float(fobj['benchmark']['candidate_vs_parent_max_ratio']);absmax=float(fobj['benchmark']['absolute_median_seconds_max'])
    # Absolute ceiling catches genuinely unusable candidates.  Paired median ratio
    # catches source regressions while tolerating transient host contention/outliers.
    if med>absmax or ratio_med>rel:raise OptimizationGateError('OPTIMIZATION_BENCHMARK_REGRESSION')
    return {'schema_id':'IG_G6_S8_OPTIMIZATION_ACCEPTANCE_V1','status':'PASS','changed_paths':changed,'fixture_sha256':_sha(fixture),'parent_semantic_sha256':canonical_sha256(pr['semantic']),'candidate_semantic_sha256':canonical_sha256(cr['semantic']),'one_worker_semantic_sha256':one_sha,'four_worker_semantic_sha256s':four_shas,'parent_elapsed_seconds':parent_elapsed,'parent_samples_seconds':parent_reps,'candidate_median_seconds':med,'candidate_samples_seconds':reps,'paired_ratio_median':ratio_med,'paired_ratios':paired_ratios,'candidate_vs_parent_max_ratio':rel,'absolute_median_seconds_max':absmax,'scientific_effect':'NONE'}
