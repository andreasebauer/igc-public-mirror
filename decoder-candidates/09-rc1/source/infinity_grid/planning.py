from __future__ import annotations
from .canon import canonical_sha256
from .protocols import ProtocolRegistry
from .safety import validate_identifier

def create_plan(*,protocol_id:str,protocol_label:str,subject:dict,input_dataset_sha256:str,input_role:str,runner:str,evidence:dict,input_flags:dict|None=None,requested_claims:list|None=None,stage_id:str='execute',stage_params:dict|None=None,run_id:str|None=None):
    reg=ProtocolRegistry(); d=reg.get(protocol_id)
    validate_identifier(stage_id,field='stage_id')
    question={'protocol_id':protocol_id,'subject':subject,'protocol_label':protocol_label,'input_dataset_sha256':input_dataset_sha256,'stopping_rules':d['stopping_rules']}
    qsha=canonical_sha256(question)
    base={'schema_id':'IG_EXECUTION_PLAN_V0_17','protocol_id':protocol_id,'protocol_version':d['version'],'descriptor_sha256':d['descriptor_sha256'],'protocol_label':protocol_label,'subject':subject,'question':question,'question_sha256':qsha,'input_datasets':[{'dataset_sha256':input_dataset_sha256,'logical_role':input_role}],'input_flags':input_flags or {},'requested_claims':requested_claims or [],'evidence':evidence,'stages':[{'stage_id':stage_id,'depends_on':[],'runner':runner,'params':dict(stage_params or {},fixture_dataset_sha256=input_dataset_sha256)}]}
    if run_id is None: run_id=f"run-{protocol_id.lower().replace('_','-')}-{canonical_sha256(base)[:16]}"
    validate_identifier(run_id,field='run_id'); base['run_id']=run_id; base['plan_sha256']=canonical_sha256(base)
    return base
