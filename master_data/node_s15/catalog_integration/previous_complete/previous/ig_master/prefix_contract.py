"""Scientific export format; data identities, not execution or acceptance state."""
from .support import canonical_bytes, strict_loads
import hashlib

FORMAT='IG_MASTER_FRESH_PREFIX_DATA_V1'
CONTRACT={
 'schema_id':'IG_MASTER_FRESH_PREFIX_DATA_CONTRACT_V1',
 'encoding':'UTF8_CANONICAL_SORTED_KEY_JSON; JSONL_ROWS_TERMINATED_BY_LF',
 'integer_encoding':'JSON_INTEGER; BOOLEAN_IS_NOT_INTEGER',
 'identity':'SCOPED_NATIVE_EXACT_ORDERED_BOUNDARY; NOT_ANONYMOUS_INTERFACE_QUOTIENT',
 'object_id':'sha256(canonical_json({schema:IG_ROOTED_L2_ABC_BOUNDARY_V1,catalog_sha256,classes,record}))',
 'j3_id':'sha256(canonical_json({schema:IG_MASTER_ROOTED_J3_CARRIER_V1,catalog_sha256,representative_tid}))',
 'event_id':'sha256(canonical_json({schema:IG_ROOTED_J3_EVENT_V1,event:record}))',
 'formation_id':'sha256(canonical_json({schema:IG_MASTER_ROOTED_ABC_FORMATION_V1,object_id,components}))',
 'formation_semantics':'ONE_OCCURRENCE_PER_LAWFUL_ORDERED_TRIPLE_OF_DISTINCT_J3_BOUNDARY_ALTERNATIVES; microscopic option witnesses retained inside each J3 carrier',
 'boundary_record':['L','ordered_port_pairs_P_M','weakest_link_score','destination_class_or_ordered_classes','stay_0_or_1'],
 'foundation':'IG_FRESH_FOUNDATION_V1: exact memory states and H-internal options; tri/cid/oc/ot/os/om/on/rk arrays',
 'option_0':'STAY: target=self, supply=missing=need=0, rank=source_rank',
 'positive_option_i':'primitive_states[sid].moves[i-1]; order is retained',
 'j3_fields':['j3_id','class_id','representative_tid','source_sids','events'],
 'j3_event_fields':['event_id','record','realizations'],
 'realization_fields':['option_indices','target_sids','target_tid'],
 'abc_object_fields':['object_id','record'],
 'abc_formation_fields':['formation_id','object_id','components'],
 'component_fields':['role','j3_id','event_id'],
 'limits':{'content_bytes':4194304,'rows_per_shard':1000,'rows_per_table':100000,'total_bytes':67108864,'files':4096},
 'scope_rule':'Only listed J3 representatives and declared ABC wiring are complete; no G5/G8 or raw radius-two graph export is implied',
 'publication_rule':'Root is a data identity only; production provenance and acceptance are external',
}
RECIPE={
 'schema_id':'IG_MASTER_ROOTED_CASE5_RECIPE_V1','classes':[3943,3963,3973],
 'component_roles':['A','B','C'], 'selected_source_roots':[0,1,2],
 'internal_bridges':[[['A',0],['B',0]],[['B',1],['C',0]]],
 'external_ports':[['A',1],['A',2],['B',2],['C',1],['C',2]],
 'destination_order':['A','B','C'], 'boundary_relation':'COMPLETE_UNSELECTED',
 'boundary_equality':'ORDERED_RECORD_EQUALITY',
 'foundation_scope':'H_CORE_ONLY; RAW_15625_STATE_GRAPH_NOT_EXPORTED',
 'l1_scope':'ALL_UNORDERED_TRIPLES_WITH_REPLACEMENT_OF_H; CANDIDATE_CLASSES_ARE_NOT_EXACT_CARRIERS',
 'j3_scope':'ONLY_THE_THREE_DECLARED_EXACT_REPRESENTATIVE_TRIPLES',
 'abc_scope':'ALL_LAWFUL_ORDERED_EVENT_TRIPLES_UNDER_THE_TWO_DECLARED_BRIDGES',
}

def digest(value):
 return hashlib.sha256(canonical_bytes(value)).hexdigest()

def parse(raw,max_bytes=4194304):
 value=strict_loads(raw,max_bytes=max_bytes,max_nodes=500000)
 if canonical_bytes(value)!=raw:raise ValueError('NONCANONICAL_SCIENTIFIC_JSON')
 return value

def normalize(value):return parse(canonical_bytes(value))
def event_id(record):return digest({'schema':'IG_ROOTED_J3_EVENT_V1','event':record})
def object_id(catalog,record):
 return digest({'schema':'IG_ROOTED_L2_ABC_BOUNDARY_V1','catalog_sha256':catalog,'classes':RECIPE['classes'],'record':record})
def j3_id(catalog,tid):
 return digest({'schema':'IG_MASTER_ROOTED_J3_CARRIER_V1','catalog_sha256':catalog,'representative_tid':tid})
def formation_id(oid,components):
 return digest({'schema':'IG_MASTER_ROOTED_ABC_FORMATION_V1','object_id':oid,'components':components})
