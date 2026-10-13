from __future__ import annotations

"""Registered G3:S0 whole-G2 carrier interface extraction.

This stage freezes the baseline G3 unit/observer only.  It deliberately keeps the
G2:R0 incidence topology as hidden challenge evidence and does not promote it into
G3 state.  The next stage, G3:S1, is responsible for asking whether a legal higher-
level context can actually read that topology.
"""
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import json

from .canon import canonical_sha256
from .g2_relation import compose_binary_relation
from .uplift_r0 import _choose_growth_carrier_and_operator, flattened_g1_incidence, _tree_canon, _graph_metrics

class G3S0Error(RuntimeError): pass

def _resource(name:str)->Path:
    return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def _load(name:str)->dict[str,Any]:
    return json.loads(_resource(name).read_text(encoding='utf-8'))

def phase0_spec()->dict[str,Any]:
    obj=_load('G3_PHASE0_ARCHITECTURE_FREEZE_V1.json')
    if obj.get('schema_id')!='IG_G3_PHASE0_ARCHITECTURE_FREEZE_V1': raise G3S0Error('bad G3 Phase0 spec')
    chk={k:v for k,v in obj.items() if k!='science_sha256'}
    if canonical_sha256(chk)!=obj.get('science_sha256'): raise G3S0Error('G3 Phase0 spec hash mismatch')
    return obj

def s0_spec()->dict[str,Any]:
    obj=_load('G3_S0_INTERFACE_EXTRACTION_SPEC_V1.json')
    if obj.get('schema_id')!='IG_G3_S0_INTERFACE_EXTRACTION_SPEC_V1': raise G3S0Error('bad G3 S0 spec')
    chk={k:v for k,v in obj.items() if k!='science_sha256'}
    if canonical_sha256(chk)!=obj.get('science_sha256'): raise G3S0Error('G3 S0 spec hash mismatch')
    if obj.get('phase0_spec_sha256')!=phase0_spec()['science_sha256']: raise G3S0Error('G3 S0/Phase0 binding mismatch')
    return obj

def verify_authority(*, graduation_certificate:Mapping[str,Any], r0_result:Mapping[str,Any])->dict[str,Any]:
    p=phase0_spec()['authority']
    if graduation_certificate.get('status')!='PASS' or graduation_certificate.get('classification')!=p['g2_graduation_classification']:
        raise G3S0Error('G2 graduation authority mismatch')
    if graduation_certificate.get('science_sha256')!=p['g2_graduation_certificate_science_sha256']:
        raise G3S0Error('G2 graduation certificate science hash mismatch')
    if r0_result.get('status')!='PASS' or r0_result.get('classification')!=p['g2_r0_classification']:
        raise G3S0Error('G2:R0 authority mismatch')
    if r0_result.get('science_sha256')!=p['g2_r0_science_sha256']:
        raise G3S0Error('G2:R0 science hash mismatch')
    if not r0_result.get('g2_graduated_before_r0') or r0_result.get('promotion') is not False:
        raise G3S0Error('G2:R0 authority flags invalid')
    out={'schema_id':'IG_G3_S0_AUTHORITY_V1','status':'PASS','g2_graduation_science_sha256':graduation_certificate['science_sha256'],'g2_r0_science_sha256':r0_result['science_sha256']}
    out['science_sha256']=canonical_sha256(out)
    return out

def _iface(st:Any)->dict[str,Any]:
    obj={'schema_id':'IG_G2_CAPS7_STATE_V1','coordinates':[int(x) for x in st.total_caps]}
    obj['science_sha256']=canonical_sha256(obj)
    return obj

def _record(st:Any)->dict[str,Any]:
    inc=flattened_g1_incidence(st)
    pairs=[(int(e[0]),int(e[1])) for e in inc['edges']]
    gm=_graph_metrics(inc['node_count'],pairs)
    return {
      'g2_carrier_ref':str(st.construction_digest),
      'public_interface':_iface(st),
      'hidden_challenge_diagnostic':{
        'visibility':'NOT_G3_PUBLIC_INTERFACE',
        'g1_unit_count':int(inc['node_count']),
        'topology_canon':_tree_canon(int(inc['node_count']),pairs),
        'metrics':gm,
        'typed_edges':inc['edges'],
      }
    }

def run_g3_s0(*, engine:Any, graduation_certificate:Mapping[str,Any], r0_result:Mapping[str,Any])->dict[str,Any]:
    spec=s0_spec(); auth=verify_authority(graduation_certificate=graduation_certificate,r0_result=r0_result)
    carriers=engine.ensure_g1_r100_population()
    bridge_pairs=sorted({tuple(map(int,x)) for x in engine.session.bridge_pairs})
    seed,op=_choose_growth_carrier_and_operator(carriers,bridge_pairs); a,b=map(int,op)
    states=[seed]
    for n in range(2,5):
        nxt={}
        for st in states:
            for out in compose_binary_relation(engine.session.engine,100+n,st,seed,a,b,lane='G2_R0_TREE_GROWTH',motif_id=f'G2:R0:TREE:{n}:{a}>{b}'):
                nxt.setdefault(str(out.construction_digest),out)
        if not nxt: raise G3S0Error(f'G3:S0 witness reproduction exhausted at n={n}')
        states=[nxt[k] for k in sorted(nxt)]
    rows=[_record(x) for x in states]
    by_ref={r['g2_carrier_ref']:r for r in rows}
    w=r0_result['caps7_topology_separation_witness']
    ra=str(w['topology_A']['example_construction_digest']); rb=str(w['topology_B']['example_construction_digest'])
    if ra not in by_ref or rb not in by_ref: raise G3S0Error('exact R0 path/star witness not reproduced')
    A=by_ref[ra]; B=by_ref[rb]
    if A['public_interface']['science_sha256']!=B['public_interface']['science_sha256']:
        raise G3S0Error('R0 challenge pair unexpectedly differs at CAPS7 public interface')
    if A['hidden_challenge_diagnostic']['topology_canon']==B['hidden_challenge_diagnostic']['topology_canon']:
        raise G3S0Error('R0 challenge pair topology collapsed')
    if A['public_interface']['coordinates']!=list(w['shared_caps7']) or B['public_interface']['coordinates']!=list(w['shared_caps7']):
        raise G3S0Error('R0 challenge pair CAPS7 coordinates mismatch')
    result={
      'schema_id':'IG_G3_S0_INTERFACE_EXTRACTION_RESULT_V1','status':'PASS',
      'classification':'G3_BASELINE_UNIT_INTERFACE_FROZEN_CAPS7_ONLY_TOPOLOGY_COLLISION_CORPUS_EARNED_S1_UNLOCKED',
      'g3_started':True,'g3_graduated':False,'g3_s1_unlocked':True,
      'authority':auth,'phase0_spec_sha256':phase0_spec()['science_sha256'],'s0_spec_sha256':spec['science_sha256'],
      'g3_unit_definition':phase0_spec()['g3_unit_definition'],
      'baseline_public_observer':'CAPS7_ONLY_INHERITED_FROM_GRADUATED_G2',
      'challenge_corpus':{'g1_unit_count':4,'exact_carrier_count':len(rows),'records':rows},
      'certified_collision_pair':{
        'A':A,'B':B,'same_public_interface':True,'different_incidence_topology':True,
        'meaning':'The pair is deliberately indistinguishable at the frozen G3:S0 public observer. S0 does not claim that any G3 context can distinguish it.'
      },
      'topology_visibility':'HIDDEN_CHALLENGE_EVIDENCE_ONLY_NOT_PROMOTED',
      'next_authorized_stage':'G3:S1',
      'nonclaims':spec['nonclaims'] + ['NO_G3_CONTEXT_DISTINGUISHABILITY_RESULT_YET']
    }
    result['science_sha256']=canonical_sha256(result)
    return result
