"""Verify member checks refuse a changed source, missing root and wrong observer."""
import argparse,copy,json
from pathlib import Path
from bind_members import check_case
from bind_scopes import load,digest,require,write

def main(root,output):
    q=root/'recovery/q2_registration/continuation0240/chunks/000.json';o=root/'recovery/observer_registration/continuation0285/expected/000.json'
    case=load(q)['cases'][0];expected=load(o)['cases'][0]
    roots={r['historical_root']:r for r in load(output/'G1_DEPENDENCY_REGISTER.json')['roots']}
    interfaces={r['carrier_ref']:r for r in load(root/'recovery/g1/HISTORICAL_V1_INTERFACE_POPULATION.json')['interfaces']}
    check_case(case,expected,roots,interfaces)
    negatives=[]
    for kind,label in [('source','SOURCE_CASE_HASH'),('root','MEMBER_ROOT_JOIN'),('observer','OBSERVER_JOIN')]:
        c=copy.deepcopy(case);e=copy.deepcopy(expected)
        if kind=='source':c['record']['left_carrier_ref']='0'*64
        elif kind=='root':c['record']['left_carrier_ref']='0'*64;e['source_case_sha256']=digest(c)
        else:c['stored_realized_public_sha256']='0'*64;e['source_case_sha256']=digest(c)
        try:check_case(c,e,roots,interfaces)
        except RuntimeError as err:
            require(str(err)==label,'NEGATIVE_WRONG_FAILURE');negatives.append({'case':kind,'refused':True,'reason':str(err)})
        else:raise RuntimeError('NEGATIVE_ACCEPTED:'+kind)
    write(output/'NEGATIVE_ROUTES.json',{'status':'PASS_THREE_REFUSALS','control':'Original accepted','tests':negatives,
      'synthetic_mutations_only':True,'historical_inputs_modified':False});print('PASS_THREE_REFUSALS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
