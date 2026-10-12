from __future__ import annotations
"""C5 final external route closure."""
from typing import Any
from .v05_origin_guard import CONTROLLER_ROOT_ROLE, WORKER_ROLE
REJECT_DIRECT_EXECUTION_ROUTE='REJECT_DIRECT_EXECUTION_ROUTE'; PASSIVE_SUBMIT='PASSIVE_SUBMIT'; READ_ONLY='READ_ONLY'; REJECT='REJECT'
_READ_ONLY={('status',None),('verify','artifact'),('verify','run'),('verify','release'),('verify','graph-export'),('protocol','list'),('protocol','show'),('claim','list'),('claim','show'),('graduation','status'),('uplift','show'),('uplift','registry'),('graph','show'),('graph','roots'),('graph','closure'),('graph','validate'),('resolve',None),('depends',None),('why',None),('roots',None),('contracts','registry'),('contracts','validate'),('state','verify'),('state','show'),('test','list'),('test','show'),('test','coverage')}
_SUB={'verify':'verify_cmd','protocol':'protocol_cmd','claim':'claim_cmd','graduation':'graduation_cmd','uplift':'uplift_cmd','graph':'graph_cmd','contracts':'contracts_cmd','state':'state_cmd','test':'test_cmd','request':'request_cmd'}
def classify_external_cli(args:Any)->str:
    cmd=getattr(args,'cmd',None); sub=getattr(args,_SUB.get(cmd,''),None) if cmd in _SUB else None
    if (cmd,sub)==('request','submit'):return PASSIVE_SUBMIT
    if (cmd,sub) in _READ_ONLY or (cmd,None) in _READ_ONLY:return READ_ONLY
    return REJECT
def rejection_record(route:str)->dict[str,Any]:return {'schema_id':'IG_DECODER_C5_ROUTE_REJECTION_V1','status':'PAUSED','reason':REJECT_DIRECT_EXECUTION_ROUTE,'route':str(route),'authoritative':False}
def route_inventory()->dict[str,Any]:return {'schema_id':'IG_DECODER_C5_ROUTE_INVENTORY_V1','normal_external_request_surfaces':['passive_request.submit','passive_cancel.submit'],'read_only_external_surfaces':[f'{a}:{b}' if b else a for a,b in sorted(_READ_ONLY)],'default_policy':'DENY','worker_entry_requires_role':WORKER_ROLE,'authoritative_mutation_requires_role':CONTROLLER_ROOT_ROLE,'temporary_c5_exceptions':[],'final_origin_exclusivity':True}
