"""Pure carrier public projection from the previously earned lower-resource reference."""
def expected_rows(dag, reference):
    rows=[]
    for ordinal, alias in enumerate(dag['roots']):
        node=dag['nodes'][alias];r=reference['node_materializations'][alias]
        owner=r['owner_caps'];total=r['total_caps']
        assert len(total)==7 and all(len(c)==7 for c in owner)
        assert all(type(v) is int and v>=0 for c in owner for v in c)
        assert all(type(v) is int and v>=0 for v in total)
        assert total==[sum(c[t] for c in owner) for t in range(7)]
        public={'schema_id':'IG_MASTER_BOUNDED_CARRIER_PUBLIC_INTERFACE_V1',
            'construction_digest':alias,'execution_policy':dag['execution_policy'],
            'level':7 if node['kind']=='O7' else node['level'],
            'resource_skin_sha256':r['skin'],'owner_caps':owner,'total_caps':total,
            'formal_G1_G3_authority':False}
        rows.append({'carrier_id':f'ROOT.{ordinal:03d}','public_state':public})
    return {'schema_id':'IG_MASTER_CURRENT_CARRIER_PUBLIC_CORRESPONDENCE_V1',
        'carrier_count':len(rows),'ordered_rows':rows,'global_scientific_authority':False}
