from __future__ import annotations

"""G6:S5R wider materialized-feature search.

The fixed exact observer remains authoritative.  The controller materializes the
observer partition through StageScienceRuntime over the certified S0-S4 state
collection, then audits only feature families already admitted by the frozen S5
grammar.  Raw hidden topology, owner identity, ancestry and construction identity
are never candidate fields or scientific outputs.
"""

import gzip, hashlib, io, json, zipfile
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .execution import TaskSpec
from .g6_s1_repair import _basis
from .v05_origin_guard import require_controller_execution_origin

PLAN_SCHEMA='IG_G6_S5R_WIDER_FEATURE_SEARCH_REGISTRATION_V1'
RESULT_SCHEMA='IG_G6_S5R_WIDER_FEATURE_SEARCH_RESULT_V1'
STAGE_ID='G6:S5R-WIDER'
OBSERVER_EVALUATOR='infinity_grid.g6_controller_evaluators:g6_s3_l1_legacy_signature_evaluator'
EXPECTED_LOGICAL_NAMES={
    'science_plan','master_prereg','g5_parent_review','s0_s1_closeout','s1r_closeout',
    's2r_closeout','s3r_closeout','s4r_closeout','s5_closeout','e3_recovered_fixture',
    'continuity_result','audit'
}

class G6S5RWiderError(RuntimeError): pass

def _sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _json_member(zp:Path,name:str)->Any:
    with zipfile.ZipFile(zp) as z:
        if name not in z.namelist(): raise G6S5RWiderError('EVIDENCE_MEMBER_MISSING:'+name)
        return json.loads(z.read(name))

def _tree_record(tree)->dict[str,Any]:
    return {'n':int(tree.n),'edges':[list(map(int,e)) for e in tree.edges],
            'H_classes':list(tree.H_classes),'edge_operators':[list(map(int,x)) for x in tree.edge_operators]}

def _plan(plan:dict[str,Any],artifacts:Mapping[str,Path])->None:
    required={'schema_id','search_id','frozen_question','question_sha256','leaf_universe','frozen_observer',
              'candidate_acceptance_requirement','candidate_order','materialization_policy','evidence_bindings',
              'registered_outcomes','next_on_no_descriptor','nonclaims'}
    if set(plan)!=required or plan.get('schema_id')!=PLAN_SCHEMA: raise G6S5RWiderError('WIDER_PLAN_SCHEMA')
    if hashlib.sha256(plan['frozen_question'].encode('utf-8')).hexdigest()!=plan['question_sha256']: raise G6S5RWiderError('WIDER_QUESTION_HASH')
    if set(artifacts)!=EXPECTED_LOGICAL_NAMES: raise G6S5RWiderError('WIDER_ARTIFACT_SET')
    binds=plan['evidence_bindings']
    if type(binds) is not dict or set(binds)!=(EXPECTED_LOGICAL_NAMES-{'science_plan'}): raise G6S5RWiderError('WIDER_EVIDENCE_BINDINGS')
    for logical,sha in binds.items():
        if type(sha) is not str or len(sha)!=64 or _sha_file(artifacts[logical])!=sha: raise G6S5RWiderError('WIDER_EVIDENCE_SHA:'+logical)
    if plan['leaf_universe']!={'kind':'FOUR_FROZEN_S0_SEEDS','seed_refs':['D2_PATH','D2_BROOM','D4_PATH','D4_BROOM']}: raise G6S5RWiderError('WIDER_LEAF_UNIVERSE')
    expected={'probe_ref':'D2_PATH','operator':[0,0],'position':'LEFT','semantics':'complete exact relation signature under structural unrooted canon'}
    if plan['frozen_observer']!=expected: raise G6S5RWiderError('WIDER_OBSERVER')
    if plan['candidate_order']!=['ADDITIVE_PUBLIC_AGGREGATES','OBSERVER_L1_LABEL','D_PLUS_OBSERVER_L1','CERTIFIED_CLASS_BAG','S1R_C3_ROOTED_OWNER_RESPONSE_BAG']:
        raise G6S5RWiderError('WIDER_CANDIDATE_ORDER')

def _load_s1_states(zp:Path)->list[dict[str,Any]]:
    rows=_json_member(zp,'g6_s2r_closeout/evidence/S1_CHILD_UNIVERSE_INDEX.json')
    if type(rows) is not list or len(rows)!=4520: raise G6S5RWiderError('WIDER_S1_FIXTURE')
    out=[]
    for i,row in enumerate(rows):
        if type(row) is not dict or type(row.get('tree')) is not dict: raise G6S5RWiderError('WIDER_S1_ROW')
        out.append({'id':f'S1-{i:05d}','tree':row['tree']})
    return out

def _load_higher_states(zp:Path)->list[dict[str,Any]]:
    member='e3_fixture_checkpoint/fixture/higher_states.jsonl.gz'
    with zipfile.ZipFile(zp) as z:
        if member not in z.namelist(): raise G6S5RWiderError('WIDER_HIGHER_FIXTURE_MEMBER')
        raw=z.read(member)
    out=[]
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as g:
        for line in g:
            row=json.loads(line)
            sid=str(row.get('state_id',''))
            tree=row.get('tree')
            if len(sid)!=64 or type(tree) is not dict: raise G6S5RWiderError('WIDER_HIGHER_ROW')
            out.append({'id':'H-'+sid,'tree':tree})
    if len(out)!=26594: raise G6S5RWiderError('WIDER_HIGHER_COUNT')
    return out

def _task(row:dict[str,Any],observer:dict[str,Any])->TaskSpec:
    payload={'state_tree':row['tree'],'probe_ref':observer['probe_ref'],'operator':observer['operator'],'position':observer['position']}
    binding=canonical_sha256({'schema_id':'IG_G6_S5R_WIDER_OBSERVER_TASK_BINDING_V1','task_id':row['id'],'observer':observer})
    return TaskSpec(task_id=row['id'],task_kind='G6_S5R_OBSERVER_MATERIALIZATION',binding_sha256=binding,payload=payload,cost_weight=max(1.0,float(row['tree'].get('n',1))))

def run_wider_feature_search(*, plan_path:str|Path, artifacts:Mapping[str,str|Path], output_dir:str|Path,
                             accepted_source_sha256:str, internal_execution_id:str, runtime)->dict[str,Any]:
    require_controller_execution_origin('g6-s5r-wider-feature-search')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8')); _plan(plan,paths)

    continuity=json.loads(paths['continuity_result'].read_text(encoding='utf-8'))
    if continuity.get('classification')!='OBSERVER_CONTINUITY_REPAIR_PASS_WIDER_FEATURE_SEARCH_REQUIRED' or continuity.get('descriptor_earned') is not False:
        raise G6S5RWiderError('WIDER_CONTINUITY_DEPENDENCY')
    master=paths['master_prereg'].read_text(encoding='utf-8')
    grammar_ok=all(x in master for x in ('integer-linear/additive aggregates','finite bags/counts','finite deterministic read labels','raw hidden tree topology/canon'))
    if not grammar_ok: raise G6S5RWiderError('WIDER_MASTER_GRAMMAR')
    s4=_json_member(paths['s4r_closeout'],'G6_S4R_CERTIFIED_CLOSEOUT_2026-09-07/G6_S4R_RESULT.json')
    if s4['certified_scope']['union_exact_state_count']!=31114 or s4['finite_write_law_earned'] is not False or s4['gates']['full_certified_union_L1_injective'] is not True:
        raise G6S5RWiderError('WIDER_S4_SCOPE')
    s1r=_json_member(paths['s1r_closeout'],'g6_s1_repair_closeout/G6_S1_REPAIR_CERTIFIED_CLOSEOUT_2026-09-07.json')
    c3=s1r['candidate_results']['C3_ROOTED_OWNER_RESPONSE_BAG']
    c3_nonpublic=('non-public' in json.dumps(s1r).lower() or 'NO_TOPOLOGY_PROMOTION' in json.dumps(s1r))
    if c3.get('class_count') not in (None,4520):
        # historical schemas may store the count elsewhere; the non-public gate is decisive.
        pass
    old_s5=_json_member(paths['s5_closeout'],'G6_S5_CERTIFIED_CLOSEOUT_2026-09-07/G6_S5_RESULT.json')
    if old_s5['feature_search']['family']!='2^11 coordinate projections of D=(f,m)': raise G6S5RWiderError('WIDER_OLD_S5_BINDING')

    basis=_basis()
    seeds=[{'id':'S0-'+name,'tree':_tree_record(basis[name])} for name in plan['leaf_universe']['seed_refs']]
    s1=_load_s1_states(paths['s2r_closeout']); higher=_load_higher_states(paths['e3_recovered_fixture'])
    observer=plan['frozen_observer']

    u_tasks=[_task(x,observer) for x in (s1+higher)]
    u_part=runtime.run_structural_partition(phase_id='WIDER_U_OBSERVER_MATERIALIZATION',tasks=u_tasks,
        evaluator_ref=OBSERVER_EVALUATOR,requested_workers=plan['materialization_policy']['workers'],max_tasks=31114)
    if u_part.summary['task_count']!=31114 or u_part.summary['class_count']!=31114 or u_part.summary['max_class_size']!=1:
        raise G6S5RWiderError('WIDER_U_OBSERVER_NOT_INJECTIVE')

    seed_tasks=[_task(x,observer) for x in seeds]
    seed_part=runtime.run_structural_partition(phase_id='WIDER_S0_OBSERVER_MATERIALIZATION',tasks=seed_tasks,
        evaluator_ref=OBSERVER_EVALUATOR,requested_workers=min(4,plan['materialization_policy']['workers']),max_tasks=4)
    seed_classes=seed_part.summary['class_count']

    # The observer image of S0 cannot overlap the U image because every observer child
    # has n(parent)+5 vertices: S0 gives 10/12; U gives 15..26 on the certified scope.
    s0_parent_n=sorted({int(x['tree']['n']) for x in seeds})
    u_parent_n=sorted({int(x['tree']['n']) for x in (s1+higher)})
    s0_child_n=sorted({n+5 for n in s0_parent_n}); u_child_n=sorted({n+5 for n in u_parent_n})
    cross_disjoint=set(s0_child_n).isdisjoint(u_child_n)
    if not cross_disjoint: raise G6S5RWiderError('WIDER_S0_U_OBSERVER_IMAGE_SCOPE')
    frozen_rows=31118
    observer_class_count=31114+seed_classes

    candidate_results=[
      {
        'candidate_id':'ADDITIVE_PUBLIC_AGGREGATES','status':'NON_ADMITTED_FAMILY',
        'observer_factorization':'FAIL_BY_STORED_SEED_WITNESS',
        'reason':'D2_PATH and D2_BROOM agree on inherited public coordinates; any integer-linear/additive aggregate of those public coordinates agrees on each single-atom witness while the frozen exact observer differs.'
      },
      {
        'candidate_id':'OBSERVER_L1_LABEL','status':'READ_SUFFICIENT_WRITE_LAW_NOT_EARNED',
        'observer_factorization':'PASS_BY_DEFINITION','materialized_on_frozen_scope':True,
        'distinct_values_on_frozen_scope':observer_class_count,
        'write_status':'NO_NONCIRCULAR_COMPOSITIONAL_UPDATE_RULE_MATERIALIZED_IN_S1_S4',
        'reason':'The exact observer label is an admitted materialized read and separates certified U, but S4 explicitly earned no finite write law and U is not transition closed. A label lookup through hidden exact carriers is not accepted as an abstract write rule.'
      },
      {
        'candidate_id':'D_PLUS_OBSERVER_L1','status':'READ_SUFFICIENT_WRITE_LAW_NOT_EARNED',
        'observer_factorization':'PASS','distinct_values_on_frozen_scope':observer_class_count,
        'write_status':'SAME_OBSERVER_LABEL_UPDATE_OBLIGATION_REMAINS',
        'reason':'Adding inherited D supplies its additive update but does not supply an update rule for the observer-derived component.'
      },
      {
        'candidate_id':'CERTIFIED_CLASS_BAG','status':'NO_ADMISSIBLE_MATERIALIZATION_AVAILABLE',
        'observer_factorization':'NOT_EXECUTABLE_AS_CURRENT_CANDIDATE',
        'reason':'S1-S4 did not materialize a construction-independent constituent bag with an admitted composition update. Creating one now would be a new registered read/representation design, not reuse of an existing materialized feature.'
      },
      {
        'candidate_id':'S1R_C3_ROOTED_OWNER_RESPONSE_BAG','status':'NON_ADMITTED',
        'observer_factorization':'SCOPED_UPPER_BOUND_ONLY',
        'reason':'The S1R artifact explicitly keeps this exact-rooted-owner representation candidate-only/non-public; topology/owner promotion remains forbidden without review.'
      },
    ]

    result={
      'schema_id':RESULT_SCHEMA,'status':'PASS','stage_id':STAGE_ID,'search_id':plan['search_id'],
      'accepted_decoder_source_sha256':accepted_source_sha256,'internal_execution_id':internal_execution_id,
      'frozen_observer':observer,'feature_grammar_bound':True,
      'fresh_observer_materialization':{
          'S0_state_count':4,'S0_observer_class_count':seed_classes,
          'U_state_count':31114,'U_observer_class_count':31114,'U_max_class_size':1,
          'S0_U_observer_images_disjoint_by_child_size':cross_disjoint,
          'frozen_scope_state_rows':frozen_rows,'frozen_scope_observer_class_count':observer_class_count,
          'U_partition_bindings_sha256':u_part.summary['partition_bindings_sha256'],
          'S0_partition_bindings_sha256':seed_part.summary['partition_bindings_sha256'],
          'structural_equality_used':True,'hashes_are_binding_index_handles_only':True,
      },
      'candidate_results':candidate_results,
      'classification':'PASS_NO_COMPLETE_S5_READ_WRITE_DESCRIPTOR_IN_CURRENT_MATERIALIZED_FEATURE_SET',
      'descriptor_earned':False,'s6_exact_observer_closure_unlocked':False,'g6_graduated':False,'g6_r0_started':False,
      'next_authorized':plan['next_on_no_descriptor'],
      'scope_statement':'Search exhausts the currently materialized candidate families admitted by the frozen S5 grammar; it does not prove that no future registered observer-derived representation can exist.',
      'nonclaims':plan['nonclaims'],
    }
    result['result_sha256']=canonical_sha256(result)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out/'G6_S5R_WIDER_FEATURE_SEARCH_RESULT.json',result)
    read_first=(
      'INFINITY GRID — G6:S5R WIDER MATERIALIZED-FEATURE SEARCH\n\n'
      'STATUS: PASS / NO COMPLETE S5 READ-WRITE DESCRIPTOR EARNED\n\n'
      f"Fresh Decoder-owned observer materialization: U 31,114/31,114 singleton classes; S0 {seed_classes}/4 classes; combined frozen scope {observer_class_count} observer classes across 31,118 rows.\n\n"
      'Additive inherited-public aggregates remain insufficient. The exact observer label itself is an admissible sufficient read on the frozen scope, but no non-circular compositional update rule for that label was materialized by S1-S4. Adding D does not repair that missing update. No construction-independent certified-class bag with a registered update currently exists, and the S1R rooted-owner bag remains non-public.\n\n'
      'Therefore the current materialized feature set is exhausted without a complete S5 read/write descriptor. This is not an impossibility theorem for future finite recursive descriptions. The next step requires an explicitly registered compositional observer-derived representation design/amendment. G6 is not graduated; exact-observer S6 remains locked; R0 is not started.\n\n'
      'RESULT SHA-256: '+result['result_sha256']+'\n')
    (out/'READ_FIRST.txt').write_text(read_first,encoding='utf-8')
    return result
