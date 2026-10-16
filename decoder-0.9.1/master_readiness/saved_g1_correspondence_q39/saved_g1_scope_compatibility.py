"""Explicit finite saved G1 domain admission and ordered-DAG comparison.

Admission grants byte/scope access, never theorem, generation or global identity.
The comparison is an audit projection preserving source/reservations, ordered
children and complete typed edge witnesses. It is not anonymous isomorphism.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

HISTORICAL='HISTORICAL_V1_R100'
CURRENT='CURRENT_NATIVE_V3_R100'
FIELDS={'scope','artifact_sha256','producer_ref','production_authority'}
class ScopeCompatibilityError(ValueError):pass

def require(ok,code):
    if not ok:raise ScopeCompatibilityError(code)

def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')

@dataclass(frozen=True)
class Admission:
    path:Path
    scope:str
    artifact_sha256:str
    producer_ref:str
    def read(self):
        raw=self.path.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==self.artifact_sha256,'SAVED_SCOPE_BYTES_CHANGED')
        return raw

def admit(path,binding,*,consumer_scope):
    require(type(binding) is dict and set(binding)==FIELDS,'SAVED_SCOPE_BINDING_FIELDS')
    require(binding['scope'] in (HISTORICAL,CURRENT),'SAVED_SCOPE_DOMAIN')
    require(type(consumer_scope) is str and consumer_scope==binding['scope'],'HISTORICAL_CURRENT_SCOPE_CORRESPONDENCE_REQUIRED')
    require(binding['production_authority']=='NONE','SAVED_SCOPE_NO_PRODUCTION_AUTHORITY')
    for key in ('artifact_sha256','producer_ref'):
        value=binding[key]
        require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'SAVED_SCOPE_BINDING_DIGEST')
    result=Admission(Path(path),binding['scope'],binding['artifact_sha256'],binding['producer_ref']);result.read();return result

def compare_ordered_dags(left,right):
    """Full-key interning; token equality requires exact bytes, not digest equality.

    This intentionally omits construction aliases and policy/order metadata.
    Lane and motif remain in the audit profile. No conversion is returned.
    """
    table={};reports=[];roots=[]
    for dag in (left,right):
        nodes=dag['nodes'];memo={};active=set()
        def visit(key):
            if key in memo:return memo[key]
            require(key not in active,'DAG_CYCLE');require(key in nodes,'DAG_MISSING_CHILD');active.add(key)
            row=nodes[key];require(row['construction_digest']==key,'DAG_ALIAS_BINDING')
            if row['kind']=='O7':
                rc=row['reserve_counts'];require(type(rc) is list and len(rc)==7 and all(type(x) is int and x>=0 for x in rc),'DAG_EXACT_RESERVATION_TYPES')
                value={k:row[k] for k in ('kind','source_id','reserve_counts','lane')}
            else:
                require(row['kind']=='LIFT' and type(row['level']) is int and row['level']>=8,'DAG_LIFT_KIND_LEVEL')
                require(type(row['children']) is list and all(type(x) is str for x in row['children']),'DAG_ORDERED_CHILDREN')
                value={k:row[k] for k in ('kind','level','lane','motif_id','top_edges_full')}
                value['children']=[visit(k) for k in row['children']]
            raw=encoded(value)
            token=table.setdefault(raw,len(table));memo[key]=token;active.remove(key);return token
        root_tokens=[visit(k) for k in dag['roots']]
        require(len(memo)==len(nodes),'DAG_UNREACHABLE_NODES')
        roots.append(root_tokens);reports.append({'nodes':len(nodes),'roots':len(root_tokens),'root_aliases':list(dag['roots']),'root_tokens':root_tokens})
    aset,bset=set(roots[0]),set(roots[1]);equal=len(roots[0])==len(roots[1]) and roots[0]==roots[1]
    return {'schema_id':'IG_ORDERED_SAVED_G1_DAG_COMPARISON_V1','profile':'EXACT_ORDERED_SOURCE_RESERVATION_CHILDREN_TYPED_WITNESSES_LANE_MOTIF',
            'hash_equality_used_for_structural_comparison':False,'ordered_root_sequences_equal':equal,
            'common_exact_root_values':len(aset&bset),'left_only_exact_root_values':len(aset-bset),'right_only_exact_root_values':len(bset-aset),
            'domains':reports,'unique_exact_interned_values':len(table),'policy_equivalence_granted':False,'global_identity_granted':False,'production_authority':'NONE'}
