"""Complete site-occurrence resource reads, separate from construction sampling.

The internal key is an interned lossless tagged multiset, never a resource hash.
Native digests are output pushforwards, where collisions accumulate. This does
not certify exact-H incidence writes, topology transport or production admission.
"""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import json

from . import exact_carrier_unblinding as ec
from . import direct_v2_observation as direct
from .canon import canonical_sha256

PROTOCOL = 'IG_COMPLETE_RESOURCE_COUNTER_READ@1.0.0'
UNIT = 'ONE_POSITIVE_CAPACITY_SITE_TYPE_OCCURRENCE_NOT_CAPACITY_SLOT'


class ResourceReadError(RuntimeError):
    pass


def require(value, reason):
    if not value:
        raise ResourceReadError(reason)


def endpoint_type(t):
    require(type(t) is int and 0 <= t < 7, 'ENDPOINT_TYPE_REQUIRED')
    return t


def make_contract():
    c = {'schema_id':'IG_COMPLETE_RESOURCE_READ_CONTRACT_V1', 'protocol':PROTOCOL,
        'package_payload_sha256':direct.source_snapshot(),
        'kernel_source_sha256':direct.KERNEL_PINS,
        'runtime_manifest_sha256':direct.make_contract()['runtime_manifest_sha256'],
        'unit':UNIT, 'leaf_domain':'NONNEGATIVE_INTEGRAL_COMPACT_O6_RESOURCE_SITES',
        'read_set':['all_compact_O6_site_ports_and_free_vectors','tagged_child_occurrences',
                    'native_Bridge','selected_endpoint_type','current_level'],
        'labels':'(R_level,min(type_a,type_b),max(type_a,type_b))',
        'owner_domain':'DISTINCT_CURRENT_LEVEL_OWNER_OCCURRENCES',
        'Bridge':'NATIVE_UNIQUE_REVERSAL_CLOSED',
        'internal_identity':'LOSSLESS_INTERNED_TAGGED_MULTISET_PAYLOAD_NOT_DIGEST',
        'input_lifetime':'CAPTURED_IMMUTABLE_MAPPED_STATE_SNAPSHOT',
        'hash_output_policy':'PUSHFORWARD_WITH_COLLISION_ADDITION',
        'construction_sampler':'NOT_CONSUMED', 'topology_transport':'NOT_PROVIDED',
        'exact_incidence_action_witness':'NOT_PROVIDED', 'matched_backbone':'NOT_PROVIDED',
        'production_authority':'NONE', 'generation_allowed':False}
    c['science_sha256']=canonical_sha256(c)
    return c


class CompleteResourceRead:
    def __init__(self, engine, bridge_pairs, contract):
        require(contract==make_contract(), 'RESOURCE_READ_SOURCE_OR_SCOPE_CHANGED')
        require(hashlib.sha256(open(engine.__file__,'rb').read()).hexdigest()==direct.KERNEL_PINS['O7'] and
                hashlib.sha256(open(engine.O6.__file__,'rb').read()).hexdigest()==direct.KERNEL_PINS['O6'],
                'NATIVE_KERNEL_SOURCE_CHANGED')
        require(type(bridge_pairs) in (list,tuple) and bridge_pairs, 'BRIDGE_REQUIRED')
        for pair in bridge_pairs:
            require(type(pair) in (tuple,list) and len(pair)==2, 'BRIDGE_PAIR_REQUIRED')
            for t in pair:endpoint_type(t)
        pairs=tuple(sorted(tuple(p) for p in bridge_pairs))
        native=tuple(sorted(tuple(p) for p in engine.O6.load_rules()[1]))
        require(pairs==native and len(set(pairs))==len(pairs) and
                set(pairs)=={(b,a) for a,b in pairs}, 'NATIVE_BRIDGE_REQUIRED')
        self.engine=engine;self.pairs=pairs;self.contract=deepcopy(contract)
        self.nodes=[];self.index={};self.closed=False
        self._state_refs={};self._state_nodes={}

    def _open(self):
        require(not self.closed, 'RESOURCE_READ_CLOSED')

    def _intern(self, payload):
        if payload not in self.index:
            self.index[payload]=len(self.nodes);self.nodes.append(payload)
        return self.index[payload]

    def site(self, ports, free):
        self._open()
        require(type(ports) is tuple and all(type(x) is int for x in ports), 'SITE_PORTS_REQUIRED')
        require(type(free) is tuple and len(free)==7 and
                all(type(x) is int and x>=0 for x in free), 'SITE_FREE_VECTOR_REQUIRED')
        return self._intern(('SITE',ports,free))

    def container(self, tag, children):
        self._open()
        require(type(tag) is str and tag in {'O2','O3','O4','O5','R6'}|{'R'+str(i) for i in range(7,101)},
                'RESOURCE_TAG_REQUIRED')
        require(all(type(c) is int and 0<=c<len(self.nodes) for c in children), 'RESOURCE_CHILD_REQUIRED')
        return self._intern((tag,tuple(sorted(children))))

    def from_h6(self, h):
        return self.container('R6',[self.container('O5',[self.container('O4',[
            self.container('O3',[self.container('O2',[self.site(p,f) for p,f in block])
            for block in o3]) for o3 in o4]) for o4 in o5]) for o5 in h])

    def state(self, value):
        self._open()
        require(type(value) in (ec.ExactO7State,ec.ExactLiftState), 'MAPPED_EXACT_STATE_REQUIRED')
        require(value.engine is self.engine, 'MIXED_NATIVE_ENGINE')
        key=id(value)
        if key not in self._state_nodes:
            if type(value) is ec.ExactO7State:
                children=[self.from_h6(h) for h in value.owner_h6s]
            else:
                require(8<=value.level<=100 and all(c.level==value.level-1 for c in value.children),
                        'ADJACENT_LIFT_LEVELS_REQUIRED')
                children=[self.state(c) for c in value.children]
            node=self.container('R'+str(value.level),children)
            require(self.digest(node)==value.skin, 'NATIVE_RESOURCE_PROJECTION_MISMATCH')
            self._state_refs[key]=value;self._state_nodes[key]=node
        return self._state_nodes[key]

    @lru_cache(maxsize=None)
    def digest(self,node):
        payload=self.nodes[node]
        if payload[0]=='SITE':
            raw=b'SITE|'+json.dumps([list(payload[1]),list(payload[2])],separators=(',',':')).encode()
        else:raw=(payload[0]+'|'+'|'.join(sorted(self.digest(c) for c in payload[1]))).encode()
        return hashlib.sha256(raw).hexdigest()

    @lru_cache(maxsize=None)
    def _reserve(self,node,t):
        payload=self.nodes[node];out=Counter()
        if payload[0]=='SITE':
            p,f=payload[1:]
            if f[t]>0:
                nf=list(f);nf[t]-=1;out[self.site(p,tuple(nf))]+=1
        else:
            tag,children=payload
            for i,c in enumerate(children):
                for post,count in self._reserve(c,t):
                    updated=list(children);updated[i]=post
                    out[self.container(tag,updated)]+=count
        return tuple(sorted(out.items()))

    def reserve(self,node,t):
        self._open();endpoint_type(t)
        require(type(node) is int and 0<=node<len(self.nodes), 'RESOURCE_NODE_REQUIRED')
        return Counter(dict(self._reserve(node,t)))

    def profile(self,node):
        self._open()
        require(type(node) is int and 0<=node<len(self.nodes), 'RESOURCE_NODE_REQUIRED')
        payload=self.nodes[node]
        require(len(payload)==2, 'CURRENT_LEVEL_RESOURCE_REQUIRED')
        tag,children=payload
        require(tag in {'R'+str(i) for i in range(7,101)}, 'CURRENT_LEVEL_RESOURCE_REQUIRED')
        out=Counter()
        for i in range(len(children)):
            for j in range(i+1,len(children)):
                for a,b in self.pairs:
                    for left,ml in self._reserve(children[i],a):
                        for right,mr in self._reserve(children[j],b):
                            updated=list(children);updated[i]=left;updated[j]=right
                            out[((tag,min(a,b),max(a,b)),self.container(tag,updated))]+=ml*mr
        return out

    def pushed_reservation(self,node,t):
        out=Counter()
        for post,count in self.reserve(node,t).items():out[self.digest(post)]+=count
        return out

    def pushed_profile(self,node):
        out=Counter()
        for (label,post),count in self.profile(node).items():out[label,self.digest(post)]+=count
        return out

    def export_nodes(self):
        self._open()
        return [{'node':i,'payload':payload,'native_digest':self.digest(i)} for i,payload in enumerate(self.nodes)]

    def close(self):
        self.closed=True;self._state_refs.clear();self._state_nodes.clear()
        self._reserve.cache_clear();self.digest.cache_clear()


def witnessed_reservation(scanner, exact, endpoint, witness):
    """Resolve ONE actual scanner reservation, never pretend it is a census.

    Return the exact resource decrement and full owner/site occurrence address.
    This proves constructive membership in the declared choice domain; it does
    not enumerate the entire Counter or validate historical edge-witness tokens.
    """
    from . import regime_scanner as rs
    t=endpoint_type(endpoint)
    require(exact.engine is scanner.engine and exact.level==scanner.level, 'WITNESS_STATE_MISMATCH')
    if type(scanner) is rs.O7State and type(exact) is ec.ExactO7State:
        require(tuple(witness)==(t,scanner.reserve_counts[t]), 'O7_RESERVATION_WITNESS_MISMATCH')
        for owner,(_root,byp) in enumerate(exact.owner_orbit_data):
            for path,(_p,f,_leaf,_ps) in sorted(byp.items()):
                if f[t]>0:return exact._with_decrement(owner,tuple(path),t), (owner,tuple(path),t)
        raise ResourceReadError('WITNESS_NO_POSITIVE_SITE')
    require(type(scanner) is rs.LiftState and type(exact) is ec.ExactLiftState,
            'WITNESS_NATIVE_STATE_REQUIRED')
    require(type(witness) in (tuple,list) and len(witness)==3, 'LIFT_RESERVATION_WITNESS_REQUIRED')
    i,local,tt=witness
    require(type(i) is int and 0<=i<len(scanner.children) and type(tt) is int and tt==t,
            'LIFT_RESERVATION_WITNESS_MISMATCH')
    candidates=[(rs.execution_order_key(c),j) for j,c in enumerate(scanner.children) if c.total_caps[t]>0]
    require(candidates and min(candidates)[1]==i, 'LIFT_SELECTED_OCCURRENCE_MISMATCH')
    post,path=witnessed_reservation(scanner.children[i],exact.children[i],t,local)
    children=list(exact.children);children[i]=post
    return ec.ExactLiftState(exact.engine,exact.level,tuple(children),exact.top_edges_full,
                             exact.lane,exact.motif_id), (i,path,t)
