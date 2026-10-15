"""Independent finite labeled-tree and full-DAG export checks; no kernel imports."""
def freeze(v):
    if type(v) is str:return ('str',v)
    if type(v) is int:return ('int',v)
    return ('seq',tuple(freeze(x) for x in v))

def tuples(v):
    if type(v) in (tuple,list):return tuple(tuples(x) for x in v)
    if type(v) is dict:return {k:tuples(x) for k,x in v.items()}
    return v

def canon(tree,classes):
    adj=[[] for _ in range(tree['n'])]
    for (u,v),(a,b) in zip(tree['edges'],tree['edge_operators']):
        adj[u].append((v,(a,b)));adj[v].append((u,(b,a)))
    def rooted(v,parent):
        return ('V',freeze(classes[tree['H_classes'][v]][0]),
                tuple(sorted((freeze(op),rooted(w,v)) for w,op in adj[v] if w!=parent)))
    return min(rooted(i,-1) for i in range(tree['n']))

def remaining(tree,classes):
    c=[list(classes[h][1]) for h in tree['H_classes']]
    for (u,v),(a,b) in zip(tree['edges'],tree['edge_operators']):c[u][a]-=1;c[v][b]-=1
    assert all(x>=0 for r in c for x in r)
    return c

def graft(left,right,u,v,op):
    n=left['n']
    return {'n':n+right['n'],'edges':left['edges']+[[a+n,b+n] for a,b in right['edges']]+[[u,v+n]],
        'H_classes':left['H_classes']+right['H_classes'],
        'edge_operators':left['edge_operators']+right['edge_operators']+[list(op)]}

def marker_q(tree,classes):
    leaf={'n':1,'edges':[],'H_classes':['D'],'edge_operators':[]}
    assert classes['D'][1][0]>0
    return tuple(sorted({canon(graft(tree,leaf,u,0,(0,0)),classes)
        for u,c in enumerate(remaining(tree,classes)) if c[0]>0},key=repr))

def verify_owner(out):
    classes={name:(key,caps) for name,key,caps in out['authority']['classes']}
    left,right=out['parents']['left'],out['parents']['right'];a,b=out['operator']
    lc,rc=remaining(left,classes),remaining(right,classes)
    lo=[i for i,c in enumerate(lc) if c[a]>0];ro=[i for i,c in enumerate(rc) if c[b]>0]
    assert out['eligible_left_owners']==lo and out['eligible_right_owners']==ro
    rows=out['formation_occurrences'];assert len(rows)==len(lo)*len(ro)==out['formation_count']
    identities=set()
    for row,(u,v) in zip(rows,((u,v) for u in lo for v in ro)):
        expected=graft(left,right,u,v,(a,b));identity=canon(expected,classes)
        assert row['ordinal']==lo.index(u)*len(ro)+ro.index(v)
        assert row['left_owner']==u and row['right_owner']==v
        assert row['operator']==[a,b] and row['full_raw_child']==expected
        assert row['left_owner_remaining']==lc[u] and row['right_owner_remaining']==rc[v]
        assert row['left_component_map']==list(range(left['n']))
        assert row['right_component_map']==list(range(left['n'],expected['n']))
        assert tuples(row['native_identity'])==identity
        retained=out['child_SET'][row['retained_child_ordinal']]
        assert tuples(retained['native_identity'])==identity==canon(retained['full_tree'],classes)
        identities.add(identity)
    assert identities=={tuples(x['native_identity']) for x in out['child_SET']}
    assert len(identities)==out['child_SET_count'] and out['semantic_multiplicity']=='SET'
    return classes

def verify_marker(out,owner):
    classes={name:(key,caps) for name,key,caps in owner['authority']['classes']}
    for side in ('left','right'):
        assert out['parents'][side]==owner['parents'][side]
        assert tuples(out['parent_marker_states'][side])==marker_q(out['parents'][side],classes)
    rows=out['children'];assert len(rows)==owner['child_SET_count']
    assert {tuples(r['native_identity']) for r in rows}=={tuples(r['native_identity']) for r in owner['child_SET']}
    expected=set()
    for row in rows:
        identity=canon(row['raw_tree'],classes);q=marker_q(row['raw_tree'],classes)
        assert tuples(row['native_identity'])==identity and tuples(row['marker_state'])==q
        d=row['decoder'];assert d['candidate_count']==1 and d['status']=='PASS'
        assert tuples(d['candidates'])==(identity,) and tuples(d['unrooted_canon'])==identity
        assert canon(d['tree'],classes)==identity
        expected.add(q)
    assert tuples(out['write_result']['states'])==tuple(sorted(expected,key=repr))
    diag=out['write_result']['diagnostics'];assert diag['raw_child_count']==len(rows)
    assert tuples(diag['projected_marker_states'])==tuples(out['write_result']['states'])
    assert diag['fused_raw_projection'] is True

def verify_dag_export(out,original):
    assert out['full_native_DAG']==original
    nodes=original['nodes'];roots=original['roots']
    assert out['root_occurrences']==[{'root_ordinal':i,'node_alias':r} for i,r in enumerate(roots)]
    slots=[{'parent_node_alias':p,'ordered_child_slot':i,'child_node_alias':ch}
        for p,row in nodes.items() if row['kind']=='LIFT' for i,ch in enumerate(row['children'])]
    assert out['ordered_child_slot_table']==slots
    # Dynamic programming counts logical occurrences without expanding R100.
    sizes={}
    def size(k):
        if k not in sizes:sizes[k]=1+sum(size(c) for c in nodes[k].get('children',[]))
        return sizes[k]
    total=sum(size(r) for r in roots)
    assert set(sizes)==set(nodes)
    def resolve(k,w,t):
        while nodes[k]['kind']=='LIFT':
            i,w,tt=w;assert tt==t;k=nodes[k]['children'][i]
        assert w[0]==t and type(w[1]) is int and 0<=w[1]<nodes[k]['reserve_counts'][t]
    edges=0
    for row in nodes.values():
        if row['kind']!='LIFT':continue
        for u,v,a,b,wu,wv in row['top_edges_full']:
            resolve(row['children'][u],wu,a);resolve(row['children'][v],wv,b);edges+=1
    return {'roots':len(roots),'stored_nodes':len(nodes),'ordered_slots':len(slots),
        'full_typed_edges':edges,'logical_node_occurrences':total}
