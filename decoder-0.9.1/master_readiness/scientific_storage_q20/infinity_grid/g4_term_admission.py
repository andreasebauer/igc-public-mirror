"""Strict, explicit CAPS7-atomic term read; legacy constructors stay unchanged."""
import re
from .g4_term_state import G3TermState,G4TermError
from .master_full_export_services import check_binding
from .canon import canonical_sha256


def strict_term_read(wire,binding):
    check_binding(binding,canonical_sha256(wire))
    if binding['recipe_ref']!='STRICT_G4_ATOMIC_TERM_READ_V1':raise G4TermError('STRICT_TERM_RECIPE_SCOPE')
    fields={'schema_id','node_caps','typed_edges','total_caps','term_ref','topology_canon_sha256'}
    if type(wire) is not dict or set(wire)!=fields or wire['schema_id']!='IG_CERTIFIED_G3_TERM_STATE_V1':
        raise G4TermError('STRICT_TERM_FIELDS')
    caps,edges,total=wire['node_caps'],wire['typed_edges'],wire['total_caps']
    if type(caps) is not list or not caps or type(edges) is not list or type(total) is not list:
        raise G4TermError('STRICT_TERM_ARRAYS')
    if any(type(r) is not list or len(r)!=7 or any(type(x) is not int or x<0 for x in r) for r in caps+[total]):
        raise G4TermError('STRICT_TERM_NONNEGATIVE_INTEGER_CAPACITIES')
    for e in edges:
        if (type(e) is not list or len(e)!=4 or any(type(x) is not int for x in e)
            or e[0]==e[1] or not 0<=e[0]<len(caps) or not 0<=e[1]<len(caps)
            or not 0<=e[2]<7 or not 0<=e[3]<7):raise G4TermError('STRICT_TERM_EDGE')
    if any(type(wire[k]) is not str or re.fullmatch('[0-9a-f]{64}',wire[k]) is None
           for k in ('term_ref','topology_canon_sha256')):raise G4TermError('STRICT_TERM_HASH_FIELDS')
    if total!=[sum(row[t] for row in caps) for t in range(7)]:raise G4TermError('STRICT_TERM_TOTAL')
    state=G3TermState.from_wire(wire)
    if state.topology_canon_sha256!=wire['topology_canon_sha256']:raise G4TermError('STRICT_TERM_TOPOLOGY')
    return {'schema_id':'IG_STRICT_G4_ATOMIC_TERM_READ_V1','term':state.to_wire(),
        'binding':binding,'scope':'CAPS7_ATOMIC_TERM_ONLY','microscopic_reconstruction':False,
        'production_authority':'NONE','generation_allowed':False}
