"""Rooted pair extension. Identity and coverage only; no acceptance authority."""
from .prefix_contract import digest
PARENT_ROOT='e1e2327070b2f9fc3bb548fc58a6961b70a0b885eb857d040ce0dcd5bfe4a351'
PARENT_ARCHIVE='74f5f09dc73f6edc4f660b981f47df9cf7ef4fcb5baf58318bfa48f38d47b4e7'
DATASET='IG_MASTER_CASE5_PAIRS_V1'
RECIPES={
 'AB':{'roles':['A','B'],'classes':[3943,3963],'bridge_slots':[0,0],
       'external_ports':[['A',1],['A',2],['B',1],['B',2]],'destination_order':['A','B']},
 'BC':{'roles':['B','C'],'classes':[3963,3973],'bridge_slots':[1,0],
       'external_ports':[['B',0],['B',2],['C',1],['C',2]],'destination_order':['B','C']}}
CONTRACT={'schema_id':'IG_MASTER_ROOTED_PAIR_CONTRACT_V1','dataset_id':DATASET,
 'record':['L','ordered_P_M_ports','weakest_link_score','ordered_destination_classes','stay'],
 'equality':'EXACT_ORDERED_BOUNDARY_WITHIN_DECLARED_PAIR_RECIPE',
 'formation':'ONE_OCCURRENCE_PER_LAWFUL_ORDERED_PAIR_OF_STORED_J3_BOUNDARY_EVENTS',
 'microscopic_choices':'REFER_TO_PARENT_J3_EVENT_REALIZATIONS; DO_NOT_FLATTEN_OR_DISCARD',
 'parent_scientific_root_sha256':PARENT_ROOT,'recipes':RECIPES,
 'not_claimed':['anonymous_port_quotient','new_J3_coverage','G5_or_G8','universal_algebra_proof'],
 'limits':{'rows_per_family':729,'content_bytes':4194304,'total_bytes':8388608,'files':16}}
SCOPE={'schema_id':'IG_MASTER_CASE5_PAIR_STAGE_V1','parent_root_sha256':PARENT_ROOT,
 'parent_archive_sha256':PARENT_ARCHIVE,'recipes':RECIPES,'max_tasks':54,
 'max_formations':1458,'max_task_formations':27}

def pair_object_id(family,record):
 return digest({'schema':'IG_MASTER_ROOTED_PAIR_OBJECT_V1','parent_root_sha256':PARENT_ROOT,'family':family,'recipe':RECIPES[family],'record':record})

def pair_formation_id(family,oid,components):
 return digest({'schema':'IG_MASTER_ROOTED_PAIR_FORMATION_V1','family':family,'object_id':oid,'components':components})
