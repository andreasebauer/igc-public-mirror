"""Bind frozen Q2/observer members to verified historical G1 roots; no execution."""
import argparse, collections, json, re, resource
from pathlib import Path
from bind_scopes import sha, digest, load, write, require

def check_case(case,expected,roots,interfaces):
    require(case['ordinal']==expected['ordinal'],'MEMBER_ORDINAL')
    require(digest(case)==expected['source_case_sha256'],'SOURCE_CASE_HASH')
    r=case['record'];left=r['left_carrier_ref'];right=r['right_carrier_ref']
    require(left in roots and right in roots,'MEMBER_ROOT_JOIN')
    require(r['left_interface_sha256']==interfaces[left]['interface_sha256'] and r['right_interface_sha256']==interfaces[right]['interface_sha256'],'PUBLIC_INTERFACE_JOIN')
    require(r['outcome_science_sha256']==expected['outcome_science_sha256'],'OUTCOME_JOIN')
    require(case['stored_realized_public_sha256']==expected['realized_public_sha256'],'OBSERVER_JOIN')
    match=re.fullmatch(r'G1_PUBLIC_BRIDGE_RELATION_V1:([0-6])>([0-6])',r['connection_operator_ref'])
    require(match is not None,'OPERATOR_REFERENCE')
    return {'ordinal':case['ordinal'],'left_historical_root':left,'right_historical_root':right,
      'endpoint_types':list(map(int,match.groups())),'outcome_science_sha256':r['outcome_science_sha256'],
      'historical_observer_sha256':expected['realized_public_sha256'],'source_case_sha256':expected['source_case_sha256'],
      'left_anchor_sources':roots[left]['anchor_sources'],'right_anchor_sources':roots[right]['anchor_sources'],
      'migration_status':'NONZERO_OVERLAY_AND_EXACT_SUCCESSOR_WITNESSES_PENDING'}

def main(root,output):
    reg=load(output/'MEMBER_PREREGISTRATION.json')
    for n,pin in reg['inputs'].items():require(sha(root/n)==pin['sha256'],'INPUT_PIN:'+n)
    q=root/'recovery/q2_registration';manifest=load(q/'MANIFEST.json')
    for n,pin in manifest.items():require(sha(q/n)==pin['sha256'] and (q/n).stat().st_size==pin['bytes'],'Q2_BUNDLE_MANIFEST:'+n)
    o=root/'recovery/observer_registration/continuation0285';prereg=load(o/'PREREGISTRATION.json')
    plan=load(q/'continuation0240/PLAN.json');qreg=load(q/'continuation0240/PREREGISTRATION.json')
    require(sha(q/'continuation0240/PLAN.json')==qreg['plan_sha256'],'Q2_PLAN_PIN')
    d=load(output/'G1_DEPENDENCY_REGISTER.json');roots={r['historical_root']:r for r in d['roots']}
    population=load(root/'recovery/g1/HISTORICAL_V1_INTERFACE_POPULATION.json')
    interfaces={i['carrier_ref']:i for i in population['interfaces']};require(set(roots)==set(interfaces),'ALL193_INTERFACE_ROOT_JOIN')
    rows=[];bridge=None;ordinals=set();class_hist=collections.Counter();chunks=[]
    byidx={x['chunk_index']:x for x in plan['chunks']}
    for specrow in prereg['specifications']:
        idx=specrow['chunk_index'];spec=o/specrow['spec_path'];require(sha(spec)==specrow['spec_sha256'],'OBSERVER_SPEC_PIN')
        expect=o/'expected'/spec.name;require(sha(expect)==specrow['expectations_sha256'],'OBSERVER_EXPECTATION_PIN')
        s=load(spec);qr=byidx[idx];cp=q/'continuation0240'/qr['cases_path'];require(sha(cp)==qr['cases_sha256'],'Q2_CASE_PIN')
        casefile=load(cp);cases=casefile['cases'];expected=load(expect)['cases'];require(len(cases)==len(expected)==specrow['case_count'],'CHUNK_CENSUS')
        require(s['execution']['parameters']['bindings']['cases']==sha(cp),'OBSERVER_TO_Q2_CASE_PIN')
        alphabet=casefile['bridge_pairs']
        require(len(alphabet)==31 and len({tuple(x) for x in alphabet})==31,'BRIDGE_ALPHABET_CENSUS')
        if bridge is None:bridge=alphabet
        require(alphabet==bridge,'BRIDGE_ORDER_CHANGED')
        cc=collections.Counter()
        for case,e in zip(cases,expected):
            row=check_case(case,e,roots,interfaces);require(row['ordinal'] not in ordinals,'DUPLICATE_ORDINAL');ordinals.add(row['ordinal'])
            require(row['endpoint_types'] in bridge,'OPERATOR_NOT_IN_ALPHABET');row['chunk_index']=idx;rows.append(row);cc[row['outcome_science_sha256']]+=1
        require(set(cc)==set(qr['class_keys']) and len(cc)==specrow['class_count'],'CLASS_SET_JOIN')
        require(all(c in (2,3) for c in cc.values()),'INCOMPLETE_CLASS')
        class_hist.update(cc.values());chunks.append({'index':idx,'case_count':len(cases),'class_count':len(cc),'cases_sha256':sha(cp),'spec_sha256':sha(spec),'expectations_sha256':sha(expect)})
    require(len(rows)==7058 and len(ordinals)==7058 and sum(class_hist.values())==3523,'TOTAL_MEMBER_CENSUS')
    require(not (ordinals & set(plan['earned_member_ordinals'])),'PRIOR_PILOT_OVERLAP')
    touched={r[k] for r in rows for k in ('left_historical_root','right_historical_root')}
    write(output/'Q2_OBSERVER_MEMBER_ROUTES.json',{'schema':'IG_MATCHED_Q2_OBSERVER_G1_ROUTES_V1','rows':rows,'chunks':chunks,
      'bridge_pairs_in_exact_frozen_order':bridge,'scope':'7058 frozen members; static source bytes, historical references and expectations only; no candidate output comparison'})
    summary={'status':'PASS_ALL_FROZEN_MEMBER_ROOT_AND_OBSERVER_JOINS','members':len(rows),'classes':sum(class_hist.values()),
      'class_size_histogram':dict(class_hist),'chunks':len(chunks),'distinct_G1_roots_referenced':len(touched),
      'exact_G1_interface_root_set':193,'Q2_payload_or_operational_successor_witnesses_recovered':False,
      'new_scientific_execution':0,'full_580351_population_compared':False,'V3_equivalence':'NOT_ESTABLISHED'}
    write(output/'MEMBER_SUMMARY.json',summary);print(json.dumps(summary))

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
