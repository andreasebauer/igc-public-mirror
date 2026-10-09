"""Bounded S1 deterministic binary-lift probe, separate from later G2 relations."""
from pathlib import Path
import json,gzip,hashlib
from infinity_grid.canon import canonical_sha256
def read_bound(path,digest):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('INPUT_IDENTITY')
    return json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)
def prepare(payload):
    from .historical import maturation_parallel as mp
    bootstrap=read_bound(payload['bootstrap_path'],payload['bootstrap_sha256'])
    population=read_bound(payload['population_path'],payload['population_sha256'])
    cases=read_bound(payload['cases_path'],payload['cases_sha256'])
    engine,states=mp._states_from_dag(bootstrap['dag'])
    by_ref={s.construction_digest:s for s in states};pop={r['carrier_ref']:r for r in population['interfaces']}
    if len(by_ref)!=193 or set(by_ref)!=set(pop):raise ValueError('INPUT_COHORT')
    return engine,by_ref,pop,cases
def check_case(engine,by_ref,pop,case,operators,source_binding):
    from .historical import regime_scanner as rs
    from infinity_grid.uplift_structural import pair_connection_record,post_reservation_public_continuation_projection,one_reservation_successor_skin_projection,repaired_pair_connection_record_v2_from_public_hashes
    saved=case['record'];left=by_ref[saved['left_carrier_ref']];right=by_ref[saved['right_carrier_ref']]
    a,b=map(int,saved['connection_operator_ref'].rsplit(':',1)[1].split('>'))
    projected=pair_connection_record(pop[left.construction_digest],pop[right.construction_digest],a,b,bridge_pairs=operators,source_authority_sha256=source_binding)
    if projected['outcome_science_sha256']!=saved['projected_outcome_science_sha256']:raise ValueError('PROJECTED_OUTCOME')
    d4l=post_reservation_public_continuation_projection(left,a);d4r=post_reservation_public_continuation_projection(right,b)
    if d4l['science_sha256']!=saved['d4_left_post_reservation_public_continuation_sha256'] or d4r['science_sha256']!=saved['d4_right_post_reservation_public_continuation_sha256']:raise ValueError('D4_COMPARE')
    lane='G2_S1_HISTORICAL_DETERMINISTIC_PROBE';motif=f'G2:S1:BOUND_BINARY:{a}>{b}'
    assembly=rs._build_lift(engine,101,[left,right],[(0,1)],operators,lane,motif,force_pair=(a,b))
    if assembly is None or len(assembly.top_edges_full)!=1:raise ValueError('BINARY_REALIZATION')
    lsucc,lw=left.reserve_external(a);rsucc,rw=right.reserve_external(b)
    if assembly.top_edges_full!=((0,1,a,b,lw,rw),):raise ValueError('BRIDGE_WITNESS')
    if tuple(c.construction_digest for c in assembly.children)!=(lsucc.construction_digest,rsucc.construction_digest):raise ValueError('RESERVED_CHILDREN')
    expected_caps=list(left.total_caps);other=list(right.total_caps)
    expected_caps=[x+y for x,y in zip(expected_caps,other)];expected_caps[a]-=1;expected_caps[b]-=1
    if list(assembly.total_caps)!=expected_caps:raise ValueError('CAPACITY_ACCOUNTING')
    q2=one_reservation_successor_skin_projection(assembly)
    if q2['science_sha256']!=saved['q2_one_reservation_successor_skins_sha256']:raise ValueError('Q2_COMPARE')
    repaired=repaired_pair_connection_record_v2_from_public_hashes(projected,left_continuation_sha256=d4l['science_sha256'],right_continuation_sha256=d4r['science_sha256'],q2_projection=q2,source_authority_sha256=source_binding)
    if repaired['outcome_science_sha256']!=saved['outcome_science_sha256']:raise ValueError('REPAIRED_OUTCOME')
    witness_rows=[]
    for row in q2['rows']:
        t=row['endpoint_type']
        if row['available']:
            succ,witness=assembly.reserve_external(t);caps=list(assembly.total_caps);caps[t]-=1
            if list(succ.total_caps)!=caps or str(succ.skin)!=row['successor_boundary_resource_skin_sha256']:raise ValueError('Q2_WITNESS')
            witness_rows.append(dict(endpoint_type=t,operational_witness=witness,successor_construction_digest=succ.construction_digest))
    swapped=rs._build_lift(engine,101,[right,left],[(0,1)],operators,lane,f'G2:S1:BOUND_BINARY:{b}>{a}',force_pair=(b,a))
    if one_reservation_successor_skin_projection(swapped)!=q2:raise ValueError('WHOLE_CARRIER_SWAP_Q2')
    return dict(ordinal=case['ordinal'],source_case_sha256=canonical_sha256(case),public_projection=q2,projected_outcome_sha256=projected['outcome_science_sha256'],repaired_outcome_sha256=repaired['outcome_science_sha256'],d4_left_sha256=d4l['science_sha256'],d4_right_sha256=d4r['science_sha256'],operational_witness=dict(assembly_construction_digest=assembly.construction_digest,level=101,lane=lane,motif_id=motif,reserved_child_refs=[lsucc.construction_digest,rsucc.construction_digest],bridge_edge=assembly.top_edges_full[0],Q2_reservation_witnesses=witness_rows),whole_carrier_swap_q2_exact=True,scope='BOUNDED_HISTORICAL_S1_DETERMINISTIC_Q2_PROBE',historical_exact_G2_identity_claimed=False,G2_relation_semantics_claimed=False,G2_promotion=False)
def evaluate(payload):
    engine,by_ref,pop,cases=prepare(payload);outputs=[]
    operators=[tuple(x) for x in cases['bridge_pairs']]
    for case in cases['cases']:
        data=check_case(engine,by_ref,pop,case,operators,payload['source_binding'])
        outputs.append(dict(identity=data,state=data))
    return dict(states=outputs,metrics=dict(official_cases_checked=len(outputs),G1_candidate_generation=0,G2_primary_realizations=len(outputs),G2_swap_checks=len(outputs)))
