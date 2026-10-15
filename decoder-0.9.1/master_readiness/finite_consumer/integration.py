"""Explicit finite consumer: original saved diagnostics, complete reads, new motifs.

Distinct populations stay distinct. This candidate does not replace old RunPlan
admission, inherit theorem claims, or generate master data.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

from infinity_grid import complete_resource_read as read, direct_v2_observation as direct
from infinity_grid import o7_science_compat as legacy, regime_scanner as rs
from infinity_grid.canon import canonical_sha256
from infinity_grid.scientific_architecture import (
    ObservationView, ScientificProtocolRegistry, TestExecution, build_finding,
    thaw_json, validate_scientific_artifact,
)

PROTOCOL='IG_FINITE_NATIVE_CONSUMER@1.0.0'
ENTITY='IG_NATIVE_FINITE_BOUND_POPULATION@1.0.0'
REGIME='IG_NATIVE_FINITE_SOURCE_SCOPE@1.0.0'
COUNTER='COMPLETE_SITE_RESERVATION_COUNTERS'
BACKBONE='NATIVE_MATCHED_BACKBONE_V2'
COUNTER_TEST='IG_TEST_NATIVE_COMPLETE_RESOURCE_READ@1.0.0'
REPRESENTATION='IG_TEST_NATIVE_MATCHED_REPRESENTATION@1.0.0'
BOUNDARY='IG_TEST_NATIVE_MATCHED_TEMPORAL_BOUNDARY@1.0.0'
LANES={
 'branching':(legacy.TEST_BRANCHING,('type_pair_support','owner_pair_support','legal_action_labels','action_orbits')),
 'symmetry':(legacy.TEST_SYMMETRY,('automorphism_size','owner_orbits')),
 'lineage':(legacy.TEST_LINEAGE,('factor_fiber_size',)),
 'topology_services':(legacy.TEST_TOPOLOGY,('topology_canon_sha256','degree','beta','diameter','radius',
     'articulations','bridges','triangles','service_classes','service_signature_sha256')),
}
LANE_REFS={lane:'IG_TEST_NATIVE_MATCHED_'+lane.upper()+'@1.0.0' for lane in LANES}


class ConsumerError(RuntimeError):pass


def require(value,reason):
    if not value:raise ConsumerError(reason)


def sealed(value):
    result=deepcopy(value);result['science_sha256']=canonical_sha256(result);return result


def validate_seal(value):
    require(type(value) is dict and value.get('science_sha256')==
        canonical_sha256({k:v for k,v in value.items() if k!='science_sha256'}),'EVIDENCE_SEAL_CHANGED')


def make_contract():
    return sealed({'schema_id':'IG_FINITE_NATIVE_CONSUMER_CONTRACT_V1','protocol':PROTOCOL,
      'consumer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'package_payload_sha256':direct.source_snapshot(),
      'complete_read_contract_sha256':read.make_contract()['science_sha256'],
      'finite_admission':json.loads(Path(__file__).with_name('ADMISSION.json').read_text()),
      'direct_contract_sha256':direct.make_contract()['science_sha256'],
      'populations':{'saved_diagnostics':'QUALIFIED_SAVED_PANELS_ONLY',
        'complete_read':'SIX_PREREGISTERED_SAVED_O7_SOURCE_ANCHORS_ONLY',
        'matched_backbone':'SEVEN_FROZEN_MOTIFS_NEWLY_BUILT_FROM_SAVED_SELECTED_CENTERS_O8_TO_O13'},
      'new_qualification_construction_allowed':True,'master_generation_allowed':False,
      'backbone_motifs':direct.normalized(rs.BACKBONE_MOTIFS),
      'center_selection':'MIN_DISTANCE_TO_MEDIAN_TOTAL_CAPACITY_THEN_NATIVE_CONSTRUCTION_ORDER',
      'matched_levels':list(range(8,14)), 'temporal_windows':[[n-1,n,n+1] for n in range(9,13)],
      'temporal_rule':'AT_LEAST_THREE_OF_FOUR_CHANGED_FAMILIES_REMAIN_CHANGED_VS_SAME_PREVIOUS_BASELINE_AT_NEXT_DEPTH',
      'family_fields':{lane:list(fields) for lane,(_old,fields) in LANES.items()},
      'last_depth_policy':'STOP_AT_13_NO_NEXT_DEPTH_INFERENCE_FOR_13',
      'exact_individual_ancestry_matched':False,'matching_basis':'FROZEN_MOTIF_IDS_AND_SELECTED_CENTER_RULE',
      'inherited_laws':'NOT_CONSUMED','legacy_regime_admission':'NOT_GRANTED',
      'production_authority':'NONE','generation_allowed':False})


def compile_case(contract,kind,level,capabilities):
    require(contract==make_contract(),'CONSUMER_SOURCE_OR_SCOPE_CHANGED')
    require(type(level) is int and type(capabilities) in (list,tuple) and capabilities and
            all(type(x) is str for x in capabilities) and len(set(capabilities))==len(capabilities),'CASE_REQUEST_REQUIRED')
    domains={'saved_diagnostics':(range(7,14),{'SEALED_SAVED_DIAGNOSTIC_TESTS'}),
             'complete_read':((7,),{COUNTER}), 'matched_backbone':(range(8,14),{BACKBONE})}
    require(kind in domains,'CASE_POPULATION_REQUIRED')
    levels,caps=domains[kind]
    require(level in levels and set(capabilities)<=caps,'CASE_CAPABILITY_POPULATION_OR_LEVEL_MISMATCH')
    return {'protocol':PROTOCOL,'population':kind,'level':level,'capabilities':list(capabilities),
            'contract_sha256':contract['science_sha256'],'production_authority':'NONE'}


def spec_for(ref,old,capabilities,question,meta=False):
    spec=deepcopy(ScientificProtocolRegistry().test(old))
    identity,version=ref.rsplit('@',1)
    spec.update(test_id=identity,version=version,status='NATIVE_FINITE_QUALIFICATION_ONLY',
      scientific_question=question,required_capabilities=capabilities,
      input_mode='FINDING_DEPENDENCIES' if meta else 'OBSERVATION_VIEW',
      shared_projection_refs=[old],scope_class='RECONNAISSANCE' if meta else 'BOUNDED_EXACT',
      claim_ceiling='RECONNAISSANCE' if meta else 'BOUNDED_EXACT',
      dependency_test_refs=list(LANE_REFS.values()) if meta else [],
      sampling_policy='Prospectively frozen finite population and motif matching; no outcome selection.',
      observer_definition='Finite current native observations; no inherited theorem or original Regime admission.',
      default_cadence='Frozen captured finite dispatch.',
      reopen_conditions=['Source, read set, population, matching or unit changes.'])
    spec['admissibility']={'entity_class_refs':[ENTITY],'regime_refs':[REGIME],
      'constraints':['Only admitted finite bound population; no historical backbone recovery or master production authority.']}
    spec['computation']={'kind':'NATIVE_MODULE','entrypoint_ref':'integration',
      'deterministic_science_projection':'Explicit finite source/population-bound native consumer.'}
    spec['outcome_rules']={'null':'No positive change inference, or insufficient persistent-family threshold.',
      'positive':'Finite matched diagnostic variation; temporal candidate only after declared next-depth check.',
      'counterexample':'Changed source, malformed dependency, unit or population mismatch refuses dispatch.'}
    validate_scientific_artifact(spec);return spec


def execution(spec,view,evidence,source,status='NULL'):
    ref=spec['test_id']+'@'+spec['version']
    finding=build_finding(test_spec=spec,entity_instance_ref=view.entity_instance_ref,
      regime_ref=REGIME,regime_depth=view.regime_depth,source_science_sha256=source,
      evidence_payload=evidence,scientific_status=status,
      summary='Captured finite population observation only; no inherited law, universal invariance, or production promotion.')
    return TestExecution(ref,evidence,canonical_sha256(evidence),finding,view.artifact())


def as_json(exe):
    return {'test_ref':exe.test_ref,'evidence_payload':exe.evidence_payload,
      'evidence_sha256':exe.evidence_sha256,'finding':dict(exe.finding),
      'observation_view':dict(exe.observation_view) if exe.observation_view else None}


def consume_saved(dispatches,domain,level,contract,parent_binding):
    compile_case(contract,'saved_diagnostics',level,['SEALED_SAVED_DIAGNOSTIC_TESTS'])
    require([domain,level] in contract['finite_admission']['saved_diagnostic_cases'],
            'SAVED_PACKET_POPULATION_NOT_BOUND')
    require(parent_binding==contract['finite_admission']['parents']['q08'],'SAVED_PARENT_NOT_BOUND')
    matches=[row for row in dispatches if row['domain']==domain and row['level']==level]
    require(len(matches)==1,'SAVED_PACKET_POPULATION_NOT_BOUND')
    row=deepcopy(matches[0]);validate_seal(row)
    require(len(row['executions'])==10,'SAVED_TEST_DOMAIN_CHANGED')
    for exe in row['executions']:
        require(exe['evidence_sha256']==canonical_sha256(exe['evidence_payload']) and
          exe['finding']['test_ref']==exe['test_ref'],'SAVED_TEST_EVIDENCE_CHANGED')
    return sealed({'population':'saved_diagnostics','domain':domain,'level':level,
      'producing_native_completion':deepcopy(parent_binding),'original_dispatch':row,
      'reused_without_test_reexecution':True,'consumer_contract_sha256':contract['science_sha256']})


def consume_complete(session,exact,source_id,contract):
    compile_case(contract,'complete_read',exact.level,[COUNTER])
    require(source_id in contract['finite_admission']['counter_anchor_ids'],'COUNTER_ANCHOR_NOT_BOUND')
    node=session.state(exact);rows=[]
    for t in range(7):
        counter=session.pushed_reservation(node,t)
        rows.append({'type':t,'successors':sorted(counter.items()),'site_occurrences':sum(counter.values())})
    payload=sealed({'unit':read.UNIT,'source_construction_digest':source_id,'resource_skin':exact.skin,
      'rows':rows,'source_read_contract_sha256':session.contract['science_sha256'],
      'scope':'COMPLETE_RESERVATION_COUNTERS_NOT_PAIR_ACTION_COPIES'})
    spec=spec_for(COUNTER_TEST,legacy.TEST_GRAMMAR,[COUNTER],'What are the complete typed site-reservation successor Counters of this saved O7 anchor?')
    view=ObservationView(entity_instance_ref='saved-O7:'+source_id,entity_class_ref=ENTITY,regime_ref=REGIME,
      regime_depth=7,test_ref=COUNTER_TEST,required_capabilities=[COUNTER],capability_payloads={COUNTER:payload},
      source_science_sha256=payload['science_sha256'],provider_ref=PROTOCOL)
    evidence=thaw_json(view.read(COUNTER))
    return execution(spec,view,evidence,payload['science_sha256'])


def construct_backbone(engine,previous,level,contract,pairs):
    compile_case(contract,'matched_backbone',level,[BACKBONE])
    require(previous and all(s.level==level-1 and s.engine is engine and
      s.execution_policy==rs.NATIVE_EXECUTION_POLICY for s in previous),'SAVED_CENTER_POPULATION_REQUIRED')
    median=statistics.median(sum(s.total_caps) for s in previous)
    center=min(previous,key=lambda s:(abs(sum(s.total_caps)-median),rs.execution_order_key(s)))
    states=rs._build_backbone(engine,level,center,pairs)
    signature=direct.normalized(rs._backbone_signature(states,pairs))
    require(signature['diagnostic_schema']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2 and
      {r['motif_id'] for r in signature['rows']}=={'BACKBONE:'+m[0] for m in rs.BACKBONE_MOTIFS},'MATCHED_MOTIF_DOMAIN_CHANGED')
    return states,sealed({'schema_id':'IG_NATIVE_FINITE_MATCHED_PACKET_V1','level':level,
      'center_source_construction_digest':center.construction_digest,
      'input_population':sorted(s.construction_digest for s in previous),
      'new_qualification_construction':True,'historical_recovery':False,
      'matching_basis':contract['matching_basis'],'exact_individual_ancestry_matched':False,
      'states':[{'motif_id':s.motif_id,'construction_digest':s.construction_digest,'resource_skin':s.skin} for s in states],
      'matched_backbone':signature,'consumer_contract_sha256':contract['science_sha256']})


def matched_findings(packet,contract,previous=None):
    validate_seal(packet);level=packet['level'];compile_case(contract,'matched_backbone',level,[BACKBONE])
    require(packet['consumer_contract_sha256']==contract['science_sha256'],'MATCHED_CONTEXT_CHANGED')
    results={};payload=packet['matched_backbone'];source=packet['science_sha256']
    for lane,(old,fields) in LANES.items():
        ref=LANE_REFS[lane];spec=spec_for(ref,old,[BACKBONE],'How does this frozen motif family change on the admitted native depth pair?')
        view=ObservationView(entity_instance_ref='new-motifs:O'+str(level),entity_class_ref=ENTITY,
          regime_ref=REGIME,regime_depth=level,test_ref=ref,required_capabilities=[BACKBONE],
          capability_payloads={BACKBONE:payload},source_science_sha256=source,provider_ref=PROTOCOL)
        observed=thaw_json(view.read(BACKBONE))
        projection=[{'motif_id':r['motif_id'],**{k:r[k] for k in fields}} for r in observed['rows']]
        evidence={'matching_basis':contract['matching_basis'],'projection':projection}
        status='NULL' if previous is None else ('INVARIANT' if evidence==previous[ref].evidence_payload else 'OBSERVED_VARIATION')
        results[ref]=execution(spec,view,evidence,source,status)
    # Unchanged representation projection function, under an explicit new Test scope.
    spec=spec_for(REPRESENTATION,legacy.TEST_REPRESENTATION,['BOUNDARY_INTERFACE'],
      'What is the actual newly constructed finite matched-backbone representation packet?')
    view=ObservationView(entity_instance_ref='new-motifs:O'+str(level),entity_class_ref=ENTITY,
      regime_ref=REGIME,regime_depth=level,test_ref=REPRESENTATION,required_capabilities=['BOUNDARY_INTERFACE'],
      capability_payloads={'BOUNDARY_INTERFACE':{'matched_backbone':payload,'inherited_law_events':[]}},
      source_science_sha256=source,provider_ref=PROTOCOL)
    representation=execution(spec,view,legacy._representation(view),source)
    return results,representation


def temporal_window(packets,findings,center,contract):
    require(contract==make_contract(),'CONSUMER_SOURCE_OR_SCOPE_CHANGED')
    require(type(center) is int and center in range(9,13),'FROZEN_TEMPORAL_WINDOW_REQUIRED')
    levels=(center-1,center,center+1);deps=list(LANE_REFS.values())
    for level in levels:
        require(level in packets and level in findings,'TEMPORAL_CONTEXT_MISSING')
        packet=packets[level];validate_seal(packet)
        require(packet['level']==level and packet['consumer_contract_sha256']==contract['science_sha256'],
          'TEMPORAL_SOURCE_OR_LEVEL_MISMATCH')
        require(set(findings[level])==set(deps),'DEPENDENCY_TEST_IDENTITIES_REQUIRED')
        for ref,exe in findings[level].items():
            f=exe.finding
            require(exe.test_ref==ref==f['test_ref'] and exe.evidence_sha256==canonical_sha256(exe.evidence_payload)
              and f['regime_ref']==REGIME and f['regime_depth']==level and f['operational_status']=='PASS'
              and f['source_science_sha256']==packet['science_sha256'],'SEALED_BOUND_DEPENDENCY_REQUIRED')
            preimage={'test_ref':ref,'entity_instance_ref':f['entity_instance_ref'],'regime_ref':f['regime_ref'],
              'regime_depth':level,'source_science_sha256':f['source_science_sha256'],'evidence_sha256':exe.evidence_sha256}
            require(f['finding_id']==canonical_sha256(preimage),'FINDING_ID_CHANGED')
            lane=next(l for l,r in LANE_REFS.items() if r==ref)
            fields=LANES[lane][1]
            expected=[{'motif_id':row['motif_id'],**{k:row[k] for k in fields}}
                      for row in packet['matched_backbone']['rows']]
            require(exe.evidence_payload=={'matching_basis':contract['matching_basis'],'projection':expected},
                    'DEPENDENCY_PROJECTION_PACKET_MISMATCH')
    left,cur,right=(findings[level] for level in levels)
    changed=[r for r in deps if left[r].evidence_sha256!=cur[r].evidence_sha256]
    persistent=[r for r in changed if left[r].evidence_sha256!=right[r].evidence_sha256]
    candidate=len(changed)>=3 and len(persistent)>=3
    evidence={'schema_id':'IG_NATIVE_MATCHED_TEMPORAL_EVIDENCE_V1','window':list(levels),
      'dependency_test_refs':deps,'changed_families':changed,'persistent_changed_families':persistent,
      'next_depth_observed':True,'minimum_family_threshold':3,'candidate':candidate,
      'matching_basis':contract['matching_basis'],'exact_individual_ancestry_matched':False,
      'packet_sha256':[packets[l]['science_sha256'] for l in levels],
      'dependency_finding_ids':{str(l):{r:findings[l][r].finding['finding_id'] for r in deps} for l in levels},
      'promotion_authorized':False,'inherited_Regime_reopened':False}
    spec=spec_for(BOUNDARY,legacy.TEST_BOUNDARY_SENTINEL,[],
      'Do at least three frozen motif diagnostic families change and remain changed at the next admitted depth?',meta=True)
    source=canonical_sha256(evidence)
    finding=build_finding(test_spec=spec,entity_instance_ref='new-motifs:O'+str(center),regime_ref=REGIME,
      regime_depth=center,source_science_sha256=source,evidence_payload=evidence,
      scientific_status='CANDIDATE' if candidate else 'NULL',
      summary='Finite frozen-motif next-depth check only; no exact-ancestry boundary, earned law or automatic promotion.')
    return TestExecution(BOUNDARY,evidence,source,finding,None)
