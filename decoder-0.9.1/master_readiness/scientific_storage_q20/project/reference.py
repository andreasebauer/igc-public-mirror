"""Independent complete-field storage/witness and small-term reference checks."""
from collections import Counter
from itertools import permutations
import gzip,hashlib,json

def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(x):return hashlib.sha256(encoded(x)).hexdigest()

def reference_term(caps,edges):
    n=len(caps);full=[];topologies=[]
    for p in permutations(range(n)):
        colors=[None]*n
        for old,new in enumerate(p):colors[new]=tuple(caps[old])
        typed=[];pairs=[]
        for u,v,a,b in edges:
            u,v=p[u],p[v]
            typed.append((u,v,a,b) if u<v else (v,u,b,a));pairs.append((min(u,v),max(u,v)))
        full.append((tuple(colors),tuple(sorted(typed))));topologies.append(tuple(sorted(pairs)))
    return {'schema_id':'IG_CERTIFIED_G3_TERM_STATE_V1','node_caps':caps,'typed_edges':edges,
        'total_caps':[sum(r[t] for r in caps) for t in range(7)],
        'term_ref':sha({'schema_id':'IG_CERTIFIED_G3_TERM_CANON_V1','canon':min(full)}),
        'topology_canon_sha256':sha({'schema_id':'IG_G3_TERM_UNCOLORED_TOPOLOGY_CANON_V1','canon':min(topologies)})}

def verify_vertex_map(raw,retained,mapping,classes):
    assert len(mapping)==raw['n']==retained['n']
    assert all(type(x) is int for x in mapping) and sorted(mapping)==list(range(raw['n']))
    assert [classes[h] for h in raw['H_classes']]==[classes[retained['H_classes'][v]] for v in mapping]
    def normalize(u,v,a,b):return (u,v,a,b) if u<v else (v,u,b,a)
    left=Counter(normalize(mapping[u],mapping[v],a,b) for (u,v),(a,b) in zip(raw['edges'],raw['edge_operators']))
    right=Counter(normalize(u,v,a,b) for (u,v),(a,b) in zip(retained['edges'],retained['edge_operators']))
    assert left==right

def verify_all_storage(manifest,groups,read_blob,read_collection,owners,markers):
    def text(ref):return encoded(ref).decode()
    def load(ref):return json.loads(read_blob(ref))
    def payload(ref):
        p=load(ref);assert p['schema_id']=='IG_STORAGE_PAYLOAD_V1' and p['representation_kind']=='WHOLE'
        codec=load(p['whole_codec_ref']);assert codec['schema']=='EXACT_NATIVE_JSON_GZIP_V1'
        raw=gzip.decompress(read_blob(p['whole_storage_ref']))
        assert hashlib.sha256(raw).hexdigest()==p['decoded_content']['sha256'] and len(raw)==int(p['decoded_content']['size_bytes'])
        return json.loads(raw)
    objects={text(r['object_ref']):r for r in groups['objects']}
    occurrences={text(r['occurrence_ref']):r for r in groups['occurrences']}
    relations={r['relation_ref']['native_id']:r for r in groups['relations']}
    incidences={text(r['incidence_ref']):r for r in groups['incidences']}
    assert len(objects)==len(groups['objects']) and len(occurrences)==682 and len(relations)==664 and len(incidences)==2269
    for obj in objects.values():assert encoded(payload(obj['canonical_payload_ref'])).decode()==obj['object_ref']['native_id']
    for kind in groups:
        assert all(r['purpose']=='SCIENCE' for r in groups[kind])
    def related(rel):
        rows=read_collection(rel['incidences'],'relation_incidences')
        assert all(encoded(r)==encoded(incidences[text(r['incidence_ref'])]) for r in rows)
        assert all(r['relation_ref']==rel['relation_ref'] and r['multiplicity']=='1' for r in rows)
        ordered=sorted(rows,key=lambda r:int(r['slot']));assert [r['slot'] for r in ordered]==list(map(str,range(len(rows))))
        assert rel['incidence_count']==str(len(rows))
        return ordered
    witnessed=0
    for case,(owner,marker) in enumerate(zip(owners,markers)):
        descriptor=manifest['cases'][case];task=owner['binding']['task_ref'];assert descriptor['task']==task
        assert encoded(payload(descriptor['case_payload_ref']))==encoded({'owner':owner,'marker':marker})
        assert descriptor['original_binding']==owner['binding']
        rel=relations[task];assert rel['decorations_ref']['content_ref']==descriptor['case_payload_ref']
        main_rows=related(rel);assert len(main_rows)==2+len(owner['child_SET'])
        for i,side in enumerate(('left','right')):
            ir=main_rows[i];assert ir['role']==side+'_parent' and ir['direction']=='INPUT' and ir['endpoint']['kind']=='OCCURRENCE'
            occ=occurrences[text(ir['endpoint']['ref'])]
            assert occ['occurrence_ref']['native_id']==owner['binding'][side+'_parent_occurrence_ref']
            obj=objects[text(occ['object_ref'])]
            assert encoded(payload(obj['canonical_payload_ref']))==encoded(marker['write_result']['diagnostics'][side+'_decoded_unrooted_canon'])
            assert encoded(load(obj['structure_ref']['content_ref']))==encoded(owner['parents'][side])
        for ir,child in zip(main_rows[2:],owner['child_SET']):
            assert ir['endpoint']['kind']=='OBJECT' and ir['role']=='result' and ir['direction']=='OUTPUT'
            obj=objects[text(ir['endpoint']['ref'])];assert encoded(payload(obj['canonical_payload_ref']))==encoded(child['native_identity'])
        witnesses=read_collection(rel['witnesses'],'formation_witnesses');assert len(witnesses)==owner['formation_count']
        assert all(encoded(r)==encoded(occurrences[text(r['occurrence_ref'])]) for r in witnesses)
        classes={h:key for h,key,c in owner['authority']['classes']}
        for row in owner['formation_occurrences']:
            name=task+':FORMATION:'+str(row['ordinal']);occ=next(r for r in witnesses if r['occurrence_ref']['native_id']==name)
            assert occ['position_ref']['state']=='PRESENT' and load(occ['position_ref']['content_ref'])=={'formation_ordinal':row['ordinal']}
            witness=load(occ['formation_ref']);assert encoded(witness['original_formation'])==encoded(row)
            retained=owner['child_SET'][row['retained_child_ordinal']]
            verify_vertex_map(row['full_raw_child'],retained['full_tree'],witness['raw_to_retained_vertex_map'],classes)
            obj=objects[text(occ['object_ref'])];assert encoded(payload(obj['canonical_payload_ref']))==encoded(retained['native_identity'])
            fr=relations[task+':FORMATION_RELATION:'+str(row['ordinal'])];irs=related(fr)
            assert [r['direction'] for r in irs]==['INPUT','INPUT','OUTPUT'] and all(r['endpoint']['kind']=='OCCURRENCE' for r in irs)
            assert [r['endpoint']['ref'] for r in irs]==[main_rows[0]['endpoint']['ref'],main_rows[1]['endpoint']['ref'],occ['occurrence_ref']]
            assert all(r['port_map_ref']['content_ref']==occ['formation_ref'] for r in irs)
            assert encoded(load(fr['operator_ref']['content_ref']))==encoded(owner['operator'])
            assert read_collection(fr['witnesses'],'formation_witnesses')==[occ]
            witnessed+=1
    assert witnessed==646
    return {'formation_vertex_maps':witnessed,'full_cases':18,'counts':{k:len(v) for k,v in groups.items()},
        'all_original_owner_marker_payloads_exact':True,'all_original_bindings_retained':True}
