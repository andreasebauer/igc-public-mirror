from __future__ import annotations
import ast, contextlib, hashlib, json
from pathlib import Path
from ..controller import register_runner
from ..canon import write_json_atomic


from ..l2_tables import _load_npy, Tables

from ..core.l2 import (
    need_satisfied as _vn,
    destination_class as _dc,
    ordered_states as _ordered_states,
    j3 as _J3,
    bridge_records as _bridge,
    composite_abc as _composite,
    relation_digests as _digests,
    j3 as _j3_core,
    bridge_records as _bridge_core,
)



def _task_payload_for_b(EA, EC, b):
    LA=[a for a in EA if _bridge_core(a,0,b,0)]
    LC=[c for c in EC if _bridge_core(b,1,c,0)]
    rows=[]
    for a in LA:
        for c in LC:
            rows.append((a[0]|b[0]|c[0],(a[1][1],a[1][2],b[1][2],c[1][1],c[1][2]),min(a[2],b[2],c[2]),(a[3],b[3],c[3]),int(a[4] and b[4] and c[4])))
    return {
        'b_repr':repr(b),
        'left_links':len(LA),
        'right_links':len(LC),
        'joined_paths':len(LA)*len(LC),
        'rows_repr':[repr(r) for r in sorted(set(rows), key=repr)],
    }

def run_cleanroom_resumable(fixture:Path, journal, phase=None):
    def ph(cat, component):
        return phase(cat, component=component) if phase is not None else contextlib.nullcontext()
    with ph('CENSUS','base_l2.tables_and_j3_events'):
        T=Tables(fixture); golden=json.loads((fixture/'metadata/golden_cases.json').read_text())['step10_small']
        classes=tuple(golden['classes']); tids=tuple(T.rep[c] for c in classes)
        EA=_j3_core(T,tids[0]); EB=_j3_core(T,tids[1]); EC=_j3_core(T,tids[2])
    out=set(); left=right=joined=0; reused=committed=0
    with ph('COMPOSITION','base_l2.per_b_composition'):
        for idx,b in enumerate(sorted(EB,key=repr)):
            task_id=f'b-{idx:04d}'
            rec=journal.load(task_id)
            if rec is None:
                payload=_task_payload_for_b(EA,EC,b)
                with ph('CHECKPOINT_IO','base_l2.logical_task_commit'):
                    summary=journal.commit(task_id,payload)
                committed+=0 if summary.reused else 1
                rec=journal.load(task_id)
            else:
                reused+=1
            payload=rec['payload']
            if payload.get('b_repr')!=repr(b):
                raise RuntimeError(f'logical task binding mismatch for {task_id}')
            left+=int(payload['left_links']); right+=int(payload['right_links']); joined+=int(payload['joined_paths'])
            with ph('MERGE','base_l2.merge_task_rows'):
                for rr in payload['rows_repr']:
                    out.add(ast.literal_eval(rr))
    with ph('CANONICALIZATION','base_l2.relation_dedup_and_digests'):
        rel=frozenset(out); max_score=max(r[2] for r in rel); selected=[r for r in rel if r[2]==max_score]; sr,ln=_digests(rel)
    diag={'J3_A_events':len(EA),'J3_B_events':len(EB),'J3_C_events':len(EC),'left_compatible_event_links':left,'right_compatible_event_links':right,'internally_compatible_three_event_paths':joined}
    result={'status':'PASS','target':'step10-small','implementation':'PURE_STDLIB_CLEANROOM_FROM_FROZEN_L2_DATA_RESUMABLE_V05','classes':list(classes),'representative_tids':list(tids),'boundary_records':len(rel),'selected_tied_max_records':len(selected),'selected_max_score':max_score,'diagnostics':diag,'sha256_sorted_repr':sr,'expected_sha256_sorted_repr':golden['sha256_sorted_repr'],'sha256_line_serialization':ln,'expected_sha256_line_serialization':golden['sha256_line_serialization'],'scope_note':'bounded canonical A-B-C case; not the full historical production census'}
    result['status']='PASS' if len(rel)==golden['boundary_records'] and diag['internally_compatible_three_event_paths']==golden['internally_compatible_three_event_paths'] and sr==golden['sha256_sorted_repr'] and ln==golden['sha256_line_serialization'] else 'FAIL'
    return result, {'tasks_total':len(EB),'tasks_reused':reused,'tasks_committed_this_attempt':committed,'journal':journal.summary()}

def run_cleanroom(fixture:Path):
    T=Tables(fixture); golden=json.loads((fixture/'metadata/golden_cases.json').read_text())['step10_small']
    classes=tuple(golden['classes']); tids=tuple(T.rep[c] for c in classes); rel,diag=_composite(T,*tids)
    max_score=max(r[2] for r in rel); selected=[r for r in rel if r[2]==max_score]; sr,ln=_digests(rel)
    result={'status':'PASS','target':'step10-small','implementation':'PURE_STDLIB_CLEANROOM_FROM_FROZEN_L2_DATA','classes':list(classes),'representative_tids':list(tids),'boundary_records':len(rel),'selected_tied_max_records':len(selected),'selected_max_score':max_score,'diagnostics':diag,'sha256_sorted_repr':sr,'expected_sha256_sorted_repr':golden['sha256_sorted_repr'],'sha256_line_serialization':ln,'expected_sha256_line_serialization':golden['sha256_line_serialization'],'scope_note':'bounded canonical A-B-C case; not the full historical production census'}
    result['status']='PASS' if len(rel)==golden['boundary_records'] and diag['internally_compatible_three_event_paths']==golden['internally_compatible_three_event_paths'] and sr==golden['sha256_sorted_repr'] and ln==golden['sha256_line_serialization'] else 'FAIL'
    return result

@register_runner('adapter.base_l2_step10', v05_operation='BASE_L2_STEP10_REPLAY', call_style='context')
def stage_runner(*,context,plan,stage):
    phase=getattr(context,'phase',None)
    mat = phase('MATERIALIZATION',component='base_l2.fixture_materialization') if phase else contextlib.nullcontext()
    with mat:
        ds=stage['params']['fixture_dataset_sha256']; fixture=context.materialize_dataset(ds,'fixture')
    replay = phase('REPLAY',component='base_l2.step10_replay') if phase else contextlib.nullcontext()
    with replay:
        if hasattr(context,'task_journal'):
            result, task_meta=run_cleanroom_resumable(fixture,context.task_journal(),phase=phase)
        else:
            result=run_cleanroom(fixture); task_meta={'tasks_total':1,'tasks_reused':0,'tasks_committed_this_attempt':1,'journal':None}
    ser = phase('SERIALIZATION',component='base_l2.principal_json') if phase else contextlib.nullcontext()
    with ser:
        out=context.staging_path('BASE_L2_PRINCIPAL_RESULT.json'); write_json_atomic(out,result)
    if result['status']!='PASS': raise RuntimeError('Base L2 clean-room reproduction mismatch')
    coverage=context.telemetry_coverage() if hasattr(context,'telemetry_coverage') else None
    return {'outputs':{'BASE_L2_PRINCIPAL_RESULT.json':out},'stage_result':{'status':'PASS','target':'VS_BASE_L2','science_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'logical_tasks':task_meta,'worker_telemetry_coverage':coverage}}
