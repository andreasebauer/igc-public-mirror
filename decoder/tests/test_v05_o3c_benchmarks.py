from __future__ import annotations

import json, statistics, time
from pathlib import Path
from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel
from infinity_grid.g5_capabilities import _g5_s1_carriers
from infinity_grid.g6_s5r_crw_kernel import exact_observer_state, decode_exact_observer_state, write_exact_observer_states


def _kernel(cached:bool):
    n=64 if cached else 0
    return ExactTreeRelationKernel(
      scope_identity='O3C:BENCH:'+('CACHED' if cached else 'BASELINE'),
      max_cache_entries=128,max_cache_bytes=32*1024*1024,
      max_relation_cache_entries=n,max_relation_cache_bytes=(32*1024*1024 if cached else 0),
      max_observer_q_cache_entries=(256 if cached else 0),max_observer_q_cache_bytes=(32*1024*1024 if cached else 0),
      max_observer_decode_cache_entries=(256 if cached else 0),max_observer_decode_cache_bytes=(32*1024*1024 if cached else 0))


def _emit(name,base,opt,k):
    payload={'benchmark':name,'baseline_seconds':base,'optimized_seconds':opt,
             'speedup':(base/opt if opt>0 else None),'metrics':k.metrics()}
    print('IG_BENCHMARK_JSON:'+json.dumps(payload,sort_keys=True),flush=True)


def test_o3c_benchmark_s4_relation_reuse():
    b=_g5_s1_carriers(); left,right=b['D4_PATH'],b['D4_BROOM']; op=(0,0); reps=24
    k0=_kernel(False); t=time.perf_counter(); a=[k0.relation(left,right,op).canons for _ in range(reps)]; base=time.perf_counter()-t
    k1=_kernel(True); t=time.perf_counter(); c=[k1.relation(left,right,op).canons for _ in range(reps)]; opt=time.perf_counter()-t
    assert a==c and k1.metrics()['relation_cache_hits']>=reps-1
    _emit('S4_EQUIVALENT_RELATION_REUSE',base,opt,k1)


def test_o3c_benchmark_s5_q_decode_reuse():
    b=_g5_s1_carriers(); parent=b['D4_BROOM']; probe=b['D2_PATH']; op=(0,0); reps=8
    def run(k):
        rows=[]
        for _ in range(reps):
            q=exact_observer_state(parent,probe,op,kernel=k); d,_=decode_exact_observer_state(q,probe,op,kernel=k)
            rows.append((q,k.prepare(d).unrooted_canon))
        return rows
    k0=_kernel(False); t=time.perf_counter(); a=run(k0); base=time.perf_counter()-t
    k1=_kernel(True); t=time.perf_counter(); c=run(k1); opt=time.perf_counter()-t
    assert a==c and k1.metrics()['observer_q_cache_hits']>=reps-1 and k1.metrics()['observer_decode_cache_hits']>=reps-1
    _emit('S5_Q_DECODE_REUSE',base,opt,k1)


def test_o3c_benchmark_s6_write_child_reuse():
    b=_g5_s1_carriers(); left,right=b['D4_BROOM'],b['D4_PATH']; probe=b['D2_PATH']; op=(0,0); obs=(0,0); reps=3
    def one(k):
        ql=exact_observer_state(left,probe,obs,kernel=k); qr=exact_observer_state(right,probe,obs,kernel=k)
        abstract=write_exact_observer_states(ql,qr,op,probe=probe,observer_operator=obs,kernel=k)
        raw=k.relation(left,right,op); projected=[]
        for child in raw.children:
            q=exact_observer_state(child,probe,obs,kernel=k); d,_=decode_exact_observer_state(q,probe,obs,kernel=k)
            projected.append((q,k.prepare(d).unrooted_canon))
        return abstract,tuple(projected)
    k0=_kernel(False); t=time.perf_counter(); a=[one(k0) for _ in range(reps)]; base=time.perf_counter()-t
    k1=_kernel(True); t=time.perf_counter(); c=[one(k1) for _ in range(reps)]; opt=time.perf_counter()-t
    assert a==c
    m=k1.metrics(); assert m['relation_cache_hits']>0 and m['observer_q_cache_hits']>0 and m['observer_decode_cache_hits']>0
    _emit('S6_WRITE_CHILD_REUSE',base,opt,k1)


def test_o3c_benchmark_s7d2_profile_batch_a28():
    """Execution-only benchmark for A27 profile batching on one real child."""
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel
    from infinity_grid.v05_kernel_service_providers import (
        _basis_records, _exact_relation_service,
        _exact_relation_profile_service, _exact_relation_profile_batch_service,
    )
    basis=_basis_records()
    configure_relation_kernel(scope_identity='A28:CHILD:GEN', max_relation_cache_entries=0, max_relation_cache_bytes=0)
    ops=tuple(get_relation_kernel().authority.operators)
    child=None
    for op in ops:
        rel=_exact_relation_service(basis['D4_PATH'],basis['D2_BROOM'],op)
        if rel['children']:
            child=rel['children'][0]
            break
    assert child is not None
    refs=tuple(sorted(basis))
    reps=3

    configure_relation_kernel(scope_identity='A28:SCALAR', max_relation_cache_entries=0, max_relation_cache_bytes=0)
    t=time.perf_counter(); scalar_rows=[]
    for _ in range(reps):
        row=[]
        for ref in refs:
            seed=basis[ref]
            for op in ops:
                row.append(int(_exact_relation_profile_service(child,seed,op)['exact_outcome_count']))
        scalar_rows.append(tuple(row))
    scalar_s=time.perf_counter()-t

    configure_relation_kernel(scope_identity='A28:BATCH', max_relation_cache_entries=0, max_relation_cache_bytes=0)
    t=time.perf_counter(); batch_rows=[]
    for _ in range(reps):
        row=[]
        for ref in refs:
            got=_exact_relation_profile_batch_service(child,basis[ref],ops)
            row.extend(int(x) for x in got['exact_outcome_counts'])
        batch_rows.append(tuple(row))
    batch_s=time.perf_counter()-t
    assert scalar_rows==batch_rows
    payload={
      'benchmark':'S7D2_PROFILE_BATCH_A28',
      'repetitions':reps,
      'logical_profiles_per_repetition':124,
      'scalar_service_calls_per_repetition':124,
      'batch_service_calls_per_repetition':4,
      'scalar_seconds':scalar_s,
      'batch_seconds':batch_s,
      'speedup':(scalar_s/batch_s if batch_s>0 else None),
      'scalar_seconds_per_124_vector':scalar_s/reps,
      'batch_seconds_per_124_vector':batch_s/reps,
      'batch_kernel_metrics':get_relation_kernel().metrics(),
    }
    print('IG_BENCHMARK_JSON:'+json.dumps(payload,sort_keys=True),flush=True)


def test_o3c_benchmark_s7d2_compact_family_a30():
    """Accepted-full-parent comparison and the A30 >=10x continuation gate."""
    from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel, tree_from_record
    from infinity_grid.g6_s7_evaluators import (
        _multiset_signature, _s7_factor_swap_representative_contexts,
    )
    from infinity_grid.v05_kernel_service_providers import _basis_records

    fixture_path=Path(__file__).with_name('fixtures')/'g6_s7d2_a30_certified_carriers.json'
    fixture=json.loads(fixture_path.read_text(encoding='utf-8'))
    parents=tuple(tree_from_record(row['state']) for row in fixture['selected'][:3])
    basis_records=_basis_records(); refs=tuple(sorted(basis_records))
    seeds=tuple(tree_from_record(basis_records[ref]) for ref in refs)
    operators=tuple(ExactTreeRelationKernel().authority.operators)
    contexts=_s7_factor_swap_representative_contexts(refs,operators)
    outer_indices=(0,37,93)

    def kernel(scope):
        return ExactTreeRelationKernel(
          scope_identity=scope,max_cache_entries=4096,max_cache_bytes=512*1024*1024,
          max_relation_cache_entries=0,max_relation_cache_bytes=0)

    def run(parent,outer_index,mode,scope):
        k=kernel(scope); ref,outer_operator,position=contexts[outer_index]
        assert position=='LEFT'
        outer=k.relation(parent,seeds[refs.index(ref)],outer_operator)
        blocks=[]
        for child in outer.children:
            if mode=='family':
                profiles=k.relation_profile_family(child,seeds,operators)
            else:
                profiles=tuple(
                    profile for seed in seeds
                    for profile in k.relation_profile_batch(child,seed,operators))
            counts=tuple(int(profile.exact_outcome_count) for profile in profiles)
            blocks.append(('S7_CHILD_D1_BRANCH_COUNT_PREFIX',counts))
        signature=('S7_D2_OUTER_SUCCESSOR_SIG1_FACTOR_SWAP_REP_PREFIX_MULTISET',
                   124,_multiset_signature(blocks))
        return signature,len(outer.children),k.metrics()

    cases=[]
    # Warm immutable tables and imports outside the measured region.
    kernel('A30:BENCH:WARM').prepare(parents[0])
    for parent_index,parent in enumerate(parents):
        for outer_index in outer_indices:
            started=time.perf_counter()
            accepted,nchildren,_accepted_metrics=run(
                parent,outer_index,'accepted',f'A30:BENCH:ACCEPTED:{parent_index}:{outer_index}')
            accepted_s=time.perf_counter()-started
            started=time.perf_counter()
            compact,nchildren_compact,compact_metrics=run(
                parent,outer_index,'family',f'A30:BENCH:FAMILY:{parent_index}:{outer_index}')
            compact_s=time.perf_counter()-started
            assert compact==accepted
            assert nchildren_compact==nchildren
            cases.append({
              'parent_index':parent_index,'outer_context_index':outer_index,
              'outer_child_count':nchildren,'accepted_batch_seconds':accepted_s,
              'compact_family_seconds':compact_s,
              'speedup':accepted_s/compact_s if compact_s>0 else None,
              'compact_family_ambiguous_candidates':int(
                  compact_metrics['relation_profile_family_ambiguous_candidates']),
              'compact_family_exact_graft_signatures':int(
                  compact_metrics['relation_profile_family_exact_graft_signatures']),
            })
    speedups=[row['speedup'] for row in cases]
    median_speedup=statistics.median(speedups)
    payload={
      'benchmark':'S7D2_COMPACT_EXACT_PROFILE_FAMILY_A30',
      'scientific_effect':'NONE_EXECUTION_ONLY',
      'accepted_baseline':'A27_EXACT_RELATION_PROFILE_BATCH',
      'candidate':'A30_EXACT_RELATION_PROFILE_FAMILY',
      'fixture':'3_CERTIFIED_A6_PARENTS_X_3_FROZEN_OUTER_COORDINATES',
      'exact_full_parent_signature_equality':True,
      'case_count':len(cases),'cases':cases,
      'median_speedup':median_speedup,
      'minimum_speedup':min(speedups),'maximum_speedup':max(speedups),
      'required_median_speedup':10.0,
      'gate':'PASS' if median_speedup>=10.0 else 'FAIL',
    }
    print('IG_BENCHMARK_JSON:'+json.dumps(payload,sort_keys=True),flush=True)
    assert median_speedup>=10.0
