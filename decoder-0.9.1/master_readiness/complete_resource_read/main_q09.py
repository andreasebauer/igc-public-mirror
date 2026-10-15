"""Prospective native complete resource-read and actual reservation qualification."""
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

from infinity_grid import complete_resource_read as read
from infinity_grid import exact_carrier_unblinding as ec, materialized_counter_reference as ref
from infinity_grid import maturation_parallel as mp, regime_scanner as rs
from infinity_grid import qualified_discovery_inputs as qi
from infinity_grid import semantic_sentinel as ss, theorem_registry as tr
from infinity_grid.canon import canonical_sha256, write_json_atomic


def digest(counter):
    return canonical_sha256(sorted((repr(k),v) for k,v in counter.items()))


def main():
    started=time.monotonic()
    inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role,sha in params['input_hashes'].items():
        assert hashlib.sha256(Path(inputs[role]).read_bytes()).hexdigest()==sha
    contract=json.loads(Path(inputs['read_contract']).read_text());assert contract==read.make_contract()
    baseline=ss.verify_native_semantics(raise_on_change=False)
    theorems=tr.verify_all_theorems(raise_on_stale=False)
    assert baseline['status']=='REOPEN_REQUIRED' and len(theorems['stale_theorems'])==4
    with tempfile.TemporaryDirectory(prefix='ig_counter_q09_') as directory:
        root=Path(directory)
        with zipfile.ZipFile(inputs['saved_inputs']) as z:
            names=z.namelist();assert len(names)==len(set(names))
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
            z.extractall(root)
        binding=json.loads(Path(inputs['saved_binding']).read_text());qi.preflight(root,binding)
        dag=json.loads((root/'PREFIX_DAG.json').read_text())
        index=json.loads((root/'PREFIX_INDEX.json').read_text())['roots']
        engine,states=mp._states_from_dag(dag)
        pairs=sorted(tuple(p) for p in engine.O6.load_rules()[1])
        mapper=ref.SavedScannerActionMapper();session=read.CompleteResourceRead(engine,pairs,contract)
        oracle=ec.ExactProjectedActionOracle(pairs)
        # Six source anchors selected prospectively by frozen DAG occurrence count,
        # then least overlay count and construction identity. No read outcomes used.
        source_counts=Counter(n['source_id'] for n in dag['nodes'].values() if n['kind']=='O7')
        sources=sorted(source_counts,key=lambda s:(-source_counts[s],s))[:6]
        o7states={}
        def visit(s):
            if isinstance(s,rs.O7State):o7states[s.construction_digest]=s
            else:
                for c in s.children:
                    if id(c) not in seen:
                        seen.add(id(c));visit(c)
        seen=set()
        for s in states:
            if id(s) not in seen:seen.add(id(s));visit(s)
        anchors=[min((s for s in o7states.values() if s.source_id==sid),
                     key=lambda s:(sum(s.reserve_counts),s.construction_digest)) for sid in sources]
        saved_rows=[];owner_checks=0
        for s in anchors:
            exact=mapper.map(s);node=session.state(exact)
            for owner,h in enumerate(exact.owner_h6s):
                local=session.from_h6(h)
                for t in range(7):
                    independent=ref.o7_owner_reference(exact,owner,t)
                    got=session.pushed_reservation(local,t)
                    assert got==independent==oracle._o7_owner_projected_counter(exact,owner,t)
                    owner_checks+=1
            for t in range(7):
                got=session.pushed_reservation(node,t)
                assert got==oracle.projected_reservation_counter(exact,t)
                saved_rows.append({'source_id':s.source_id,'construction_digest':s.construction_digest,
                    'type':t,'counter_sha256':digest(got),'keys':len(got),'site_occurrences':sum(got.values())})
        write_json_atomic(Path('SAVED_COUNTERS.json'),{'status':'PASS','source_anchors':6,
            'selection_rule':'TOP_SIX_DAG_O7_SOURCE_COUNTS_THEN_LEAST_OVERLAY_THEN_CONSTRUCTION_ID',
            'owner_type_comparisons':owner_checks,'reservation_counters':len(saved_rows),'rows':saved_rows})
        # New exhaustive 3^3 compact-resource fixtures. API-valid resources only;
        # not claimed to be historically earned or native construction histories.
        first=mapper.map(anchors[0]);p=next(iter(first.owner_orbit_data[0][1].values()))[0]
        micro=[];chosen=None
        for capacities in itertools.product(range(3),repeat=3):
            block=tuple((p,(x,0,0,0,0,0,0)) for x in capacities)
            h=block
            for _ in range(4):h=(h,)
            ctx=SimpleNamespace(parents=(SimpleNamespace(h6=h),SimpleNamespace(h6=h)),source_order=('read-fixture','read-fixture'))
            exact=ec.ExactO7State(engine,ctx,(),'RESOURCE_READ_MICRO')
            node=session.state(exact)
            for t in range(7):
                assert session.pushed_reservation(node,t)==ref.reservation_reference(exact,t)
            got=session.pushed_profile(node)
            assert got==ref.profile_reference(exact,pairs)==oracle.current_level_profile(exact)
            active=sum(x>0 for x in capacities)
            assert sum(got.values())==active*active
            assert sum(session.reserve(node,0).values())==2*active
            micro.append({'capacities':capacities,'positive_sites_per_owner':active,
                'capacity_slots_per_owner':sum(capacities),'profile_sha256':digest(got),
                'pair_occurrences':sum(got.values()),'lossless_successor_keys':len(session.profile(node))})
            if capacities==(2,1,0):chosen=exact
        ladder=[];lift=chosen
        for level in range(8,14):
            lift=ec.ExactLiftState(engine,level,(lift,lift),(), 'RESOURCE_READ_MICRO','REPEATED_OCCURRENCES')
            node=session.state(lift)
            for t in range(7):
                assert session.pushed_reservation(node,t)==oracle.projected_reservation_counter(lift,t)
            got=session.pushed_profile(node)
            assert got==oracle.current_level_profile(lift)
            # Two repeated children; positive-site choices double on every lift.
            expected=(4*(2**(level-8)))**2
            assert sum(got.values())==expected
            ladder.append({'level':level,'profile_sha256':digest(got),'site_choice_pairs':sum(got.values()),'keys':len(got)})
        write_json_atomic(Path('COMPLETE_COUNTER_CONTROLS.json'),{'status':'PASS','micro_domains':27,
            'micro_reservation_comparisons':189,'micro_full_profiles':27,'lift_levels':6,
            'capacity_slots_not_site_choices':True,'equal_children_keep_occurrence_weight':True,
            'projection_collisions_added':True,'rows':micro,'ladder':ladder,
            'native_historical_survivor_claim':False})
        # Preregistered saved selected root per level. Resolve each actual sampler
        # witness constructively and compare complete lossless post-resource values.
        selected=[(row,s) for row,s in zip(index,states) if row['domain']=='selected']
        representatives=[min((s for row,s in selected if row['level']==level),key=lambda s:s.construction_digest)
                         for level in range(7,14)]
        witness_rows=[];calls=set()
        def profile(frame,event,arg):
            if event=='call':calls.add((str(frame.f_globals.get('__name__','')),frame.f_code.co_name))
        assert sys.getprofile() is None
        sys.setprofile(profile)
        try:
            for before in representatives:
                exact=mapper.map(before)
                for t in range(7):
                    if before.total_caps[t]<=0:continue
                    native_post,witness=before.reserve_external(t)
                    constructive,address=read.witnessed_reservation(before,exact,t,witness)
                    mapped_post=mapper.map(native_post)
                    assert session.state(constructive)==session.state(mapped_post)
                    assert constructive.skin==mapped_post.skin==native_post.skin
                    expected=list(before.total_caps);expected[t]-=1
                    assert tuple(expected)==constructive.total_caps==native_post.total_caps
                    assert constructive.typed_edges==mapped_post.typed_edges==native_post.typed_edges
                    witness_rows.append({'level':before.level,'source_construction_digest':before.construction_digest,
                        'post_construction_digest':native_post.construction_digest,'type':t,'native_witness':witness,
                        'resolved_site_address':address,'post_resource_skin':native_post.skin})
        finally:sys.setprofile(None)
        forbidden=[list(c) for c in sorted(calls) if c[0].endswith('fixed_grammar_transport') or
                   c[1] in ('_scan_level','_action_aggregate_v2','_read_probe','_resource_future_signature','_build_next_panel','_farthest_select')]
        assert not forbidden
        assert len(witness_rows)==49
        write_json_atomic(Path('NATIVE_RESERVATION_WITNESSES.json'),{'status':'PASS','checks':49,
            'levels':list(range(7,14)),'selection_rule':'ONE_LOWEST_CONSTRUCTION_DIGEST_SAVED_SELECTED_ROOT_PER_LEVEL_ALL_TYPES',
            'complete_counter_is_sampler':False,'constructive_site_choice_membership':True,
            'lossless_post_resources_equal':True,'edges_unchanged':True,'rows':witness_rows,
            'scope':'Actual external reservations only; no exact incidence-action or historical top-edge witness certification.'})
        refusals=[]
        def refuse(name,fn):
            try:fn()
            except read.ResourceReadError as exc:refusals.append({'name':name,'reason':str(exc)})
            else:raise AssertionError('missing refusal:'+name)
        for t in (-1,7,True,'0'):
            refuse('type:'+str(t),lambda t=t:session.reserve(node,t))
        for free in ((-1,0,0,0,0,0,0),(True,0,0,0,0,0,0),(0,)):
            refuse('free:'+str(free),lambda free=free:session.site(p,free))
        refuse('scanner_not_mapped',lambda:session.state(anchors[0]))
        refuse('unknown_node',lambda:session.reserve(-1,0))
        refuse('leaf_not_parent',lambda:session.profile(session.site(p,(1,0,0,0,0,0,0))))
        bad=deepcopy(contract);bad['unit']='CAPACITY_SLOTS';bad['science_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='science_sha256'})
        refuse('resealed_wrong_unit',lambda:read.CompleteResourceRead(engine,pairs,bad))
        refuse('duplicate_Bridge',lambda:read.CompleteResourceRead(engine,pairs+pairs[:1],contract))
        refuse('wrong_witness',lambda:read.witnessed_reservation(anchors[0],mapper.map(anchors[0]),0,(0,-1)))
        qi.unchanged(root,binding)
        node_count=len(session.nodes)
        # Tables retain lossless payload keys for audit; no cryptographic hash is
        # assumed to recover tree identity or used as the memoization key.
        write_json_atomic(Path('LOSSLESS_RESOURCE_NODES.json'),{'status':'PASS','internal_identity':'LOSSLESS_INTERNED_TAGGED_MULTISET_PAYLOAD_NOT_DIGEST','nodes':session.export_nodes()})
        session.close();refuse('closed_session',lambda:session.reserve(node,0))
        assert len(refusals)==14
        write_json_atomic(Path('REFUSALS.json'),{'status':'PASS','count':len(refusals),'checks':refusals})
        write_json_atomic(Path('READ_TRACE.json'),{'status':'PASS','forbidden_calls':forbidden,
            'profile_restored':sys.getprofile() is None,'native_sampler_used_only_for_witness_checks':True,
            'measured_calls':[list(c) for c in sorted(calls)],'complete_counter_kernel_uses_sampler':False})
    for ident in theorems['stale_theorems']:
        try:tr.require_current_theorems((ident,))
        except tr.StaleTheoremError:pass
        else:raise AssertionError('stale theorem admitted')
    assert ss.verify_native_semantics(raise_on_change=False)==baseline
    assert tr.verify_all_theorems(raise_on_stale=False)==theorems
    write_json_atomic(Path('QUALIFICATION_RESULT.json'),{'status':'PASS',
        'scientific_outcome':'COMPLETE_RESOURCE_READ_AND_NATIVE_RESERVATION_MEMBERSHIP_FINITE_QUALIFIED',
        'saved_source_anchors':6,'saved_owner_type_comparisons':owner_checks,'saved_reservation_counters':42,
        'micro_full_profiles':27,'lift_profiles':6,'native_witness_checks':49,'refusal_checks':14,
        'lossless_nodes':node_count,'stale_theorems':4,'production_authority':'NONE','generation_allowed':False,
        'global_native_refinement_theorem':False,'matched_longitudinal_evidence_provided':False,
        'package2_status':'OPEN_PRODUCTION_READ_CONSUMER_AND_MATCHED_LONGITUDINAL_INTEGRATION',
        'wall_seconds':time.monotonic()-started})


if __name__=='__main__':main()
