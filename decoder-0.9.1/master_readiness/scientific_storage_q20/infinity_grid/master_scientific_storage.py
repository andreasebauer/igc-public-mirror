"""Full saved G5/G6 exports into existing SCIENCE storage records and index.

No fresh relation, marker computation or master release admission is performed.
Native structural values and original producing bindings remain authoritative.
"""
from copy import deepcopy
import gzip,hashlib,json,zipfile
from pathlib import Path
from .canon import canonical_sha256
from .storage_schema import canonical_bytes,validate_record_bytes
from .storage_index_descriptor import content_ref
from .storage_collections import ContentDirectory,collection_records
from .storage_catalog import IndexRecord,inventory,build_index,ReadIndex
from .master_full_export_services import check_binding
from .full_marker_export import require,ExportContractError

Q19_CAPTURE='62a59fe276851113e92d222cf57c985e98189fb9fe88aa9f534cb67ee5b9c6d1'
Q19_COMPLETION='35a09fe64c7c96470f5ec70b8373065430db53b6c58a46ee4f19aed2a4c25ddd'
PREFIX={'contract_version':'1.0.0','purpose':'SCIENCE','extensions':[]}


def read_accepted(path):
    with zipfile.ZipFile(path) as z:
        require(len(z.namelist())==len(set(z.namelist())),'ACCEPTED_ARCHIVE_DUPLICATES')
        raw={n:z.read(n) for n in z.namelist()}
    cap=json.loads(raw['CAPTURE.json']);done=json.loads(raw['COMPLETION.json'])
    for row,key,expected in [(cap,'capture_id',Q19_CAPTURE),(done,'completion_sha256',Q19_COMPLETION)]:
        require(row[key]==expected==canonical_sha256({k:v for k,v in row.items() if k!=key}),'ACCEPTED_NATIVE_SEAL')
    require(done['status']=='COMPLETED' and done['evidence_status']=='VERIFIED' and
        done['scientific_outcome']=='REGISTERED_FULL_OUTPUT_EXPORT_ROUTES_FINITE_QUALIFIED','ACCEPTED_OUTCOME')
    require(done['source_sha256']==cap['workspace']['source_sha256'] and
        done['registration_sha256']==cap['job']['registration_sha256'] and
        done['result_sha256']==canonical_sha256(done['result']),'ACCEPTED_NATIVE_BINDING')
    evidence={r['path']:r for r in done['evidence']}
    for n,b in raw.items():
        if n in ('CAPTURE.json','COMPLETION.json','PRESERVED.json'):continue
        r=evidence['script/work/'+n];require(hashlib.sha256(b).hexdigest()==r['sha256'] and len(b)==r['size_bytes'],'ACCEPTED_OUTPUT_BYTES')
    preserved=json.loads(raw['PRESERVED.json'])
    require(preserved['capture_id']==Q19_CAPTURE and preserved['outbox']['status']=='PRESERVED'
        and preserved['outbox']['pending_bytes']==0,'ACCEPTED_PRESERVATION')
    owners=json.loads(gzip.decompress(raw['COMPLETE_OWNER_EXPORTS.json.gz']))['cases']
    markers=json.loads(gzip.decompress(raw['COMPLETE_MARKER_EXPORTS.json.gz']))['cases']
    require(len(owners)==len(markers)==18,'ACCEPTED_CASE_DOMAIN')
    return owners,markers,raw


def vertex_map(raw,retained,classes):
    """Construct one exact labeled-tree isomorphism, retaining every raw vertex."""
    def prepare(tree):
        adj=[[] for _ in range(tree['n'])]
        for (u,v),(a,b) in zip(tree['edges'],tree['edge_operators']):
            adj[u].append((v,(a,b)));adj[v].append((u,(b,a)))
        def rooted(v,parent):
            return (classes[tree['H_classes'][v]],tuple(sorted((op,rooted(w,v)) for w,op in adj[v] if w!=parent)))
        roots=[rooted(v,-1) for v in range(tree['n'])]
        return adj,rooted,roots
    require(raw['n']==retained['n'],'VERTEX_MAP_SIZE')
    aa,ac,ar=prepare(raw);ba,bc,br=prepare(retained)
    key=min(ar);require(key==min(br),'VERTEX_MAP_IDENTITY')
    mapping=[None]*raw['n']
    def walk(a,ap,b,bp):
        mapping[a]=b
        left=sorted((op,ac(v,a),v) for v,op in aa[a] if v!=ap)
        right=sorted((op,bc(v,b),v) for v,op in ba[b] if v!=bp)
        require([(op,c) for op,c,v in left]==[(op,c) for op,c,v in right],'VERTEX_MAP_BRANCHES')
        for (_,_,v),(_,_,w) in zip(left,right):walk(v,a,w,b)
    walk(ar.index(key),-1,br.index(key),-1)
    require(sorted(mapping)==list(range(raw['n'])),'VERTEX_MAP_BIJECTION')
    return mapping


def build_export_storage(evidence_path,destination,binding):
    archive=Path(evidence_path).read_bytes();check_binding(binding,hashlib.sha256(archive).hexdigest())
    require(binding['recipe_ref']=='Q20_SAVED_G5_G6_STORAGE_ONLY','STORAGE_RECIPE_SCOPE')
    owners,markers,raw_inputs=read_accepted(evidence_path)
    root=Path(destination);require(not root.exists(),'STORAGE_DESTINATION_EXISTS');content=root/'content';content.mkdir(parents=True)
    blobs={}
    def put(raw):
        ref=content_ref(raw);p=content/(ref['sha256']+'.blob')
        if not p.exists():p.write_bytes(raw);p.chmod(0o444)
        require(p.read_bytes()==raw,'STORAGE_WRITE_READBACK');blobs[ref['sha256']]=ref
        return ref
    def value(v):return put(canonical_bytes(v))
    def present(ref,text):return {'state':'PRESENT','content_ref':ref,'explanation':text}
    def none(text):return {'state':'NOT_APPLICABLE','content_ref':None,'explanation':text}
    source=value({'capture_id':Q19_CAPTURE,'completion_sha256':Q19_COMPLETION,
        'original_producer_bindings':[o['binding'] for o in owners],
        'authority':owners[0]['authority'],'candidate_storage_binding':binding})
    archive_ref=put(archive)
    identity_definition=value({'identity':'Exact native canon as canonical JSON text; no digest replaces equality',
        'occurrence':'Original task and parent occurrence IDs; each formation ordinal is distinct',
        'scope':'Q19 finite accepted18 cases, original Q18 producing pins retained',
        'layer':'G5_G6_EXACT_DECORATED_G4_TREE','production_authority':'NONE'})
    profile=value(dict(PREFIX,schema_id='IG_STORAGE_IDENTITYPROFILE_V1',profile_name='Q19_NATIVE_TREE_AND_FORMATION_V1',
        native_id_field='native_identity canonical JSON text / original task and occurrence refs',
        native_scope_fields=['source_ref','authority','task_ref'],source_member='COMPLETE_OWNER_EXPORTS.json.gz',
        source_member_sha256=hashlib.sha256(raw_inputs['COMPLETE_OWNER_EXPORTS.json.gz']).hexdigest(),
        source_archive_ref=archive_ref,interpretation_ref=identity_definition,
        identity_policy='PRESERVE_NATIVE_VALUE_AND_DECLARED_SCOPE'))
    def native(n):return {'native_id':n,'scope_ref':source,'profile_ref':profile}
    def collection(kind,sid,rows,field,semantics='SET'):
        definition={'schema_id':'IG_STORAGE_COLLECTION_KEY_PROFILE_V1','format':'CANONICAL_STORAGE_JSONL_V1',
            'key_encoding':'CANONICAL_STRING_ARRAY_UTF8_HEX','order':'ASCII_BYTEWISE','bounds':'INCLUSIVE_DISJOINT','key_fields':[field]}
        pref=value(definition)
        def key(row):
            v=row
            for k in field:v=v[k]
            return canonical_bytes([v]).hex()
        ordered=sorted(rows,key=key);entries=[]
        require(len({key(r) for r in ordered})==len(ordered),'STORAGE_DUPLICATE_COLLECTION_KEY')
        for i in range(0,len(ordered),16):
            group=ordered[i:i+16];bs=[canonical_bytes(r) for r in group]
            for b in bs:validate_record_bytes(b,max_bytes=65536)
            ref=put(b''.join(b+b'\n' for b in bs))
            entries.append({'entry_type':'SHARD','content_ref':ref,'first_key':key(group[0]),'last_key':key(group[-1]),'record_count':str(len(group))})
        page=dict(PREFIX,schema_id='IG_STORAGE_COLLECTIONPAGE_V1',collection_kind=kind,record_schema_id=sid,
            key_definition_ref=pref,depth=0,first_key=entries[0]['first_key'] if entries else None,
            last_key=entries[-1]['last_key'] if entries else None,record_count=str(len(ordered)),entries=entries)
        validate_record_bytes(canonical_bytes(page))
        return {'root':value(page),'record_schema_id':sid,'row_count':str(len(ordered)),
            'key_definition_ref':pref,'cardinality_semantics':semantics}
    empty=collection('evidence','IG_STORAGE_OCCURRENCE_V1',[],['occurrence_ref','native_id'])
    canon_contract=value({'schema':'NATIVE_EXACT_ENDPOINT_TYPED_DECORATED_TREE_CANON','equality':'FULL_STRUCTURAL_VALUE','semantic_multiplicity':'SET'})
    codec=value({'schema':'EXACT_NATIVE_JSON_GZIP_V1','integer_policy':'ARBITRARY_PRECISION_JSON_INTEGER','encoding':'UTF8_JSON_GZIP_MT0'})
    def payload(v):
        b=canonical_bytes(v);decoded=content_ref(b);compressed=put(gzip.compress(b,mtime=0))
        # Full decoded bytes are reconstructed from the declared exact codec.
        return value(dict(PREFIX,schema_id='IG_STORAGE_PAYLOAD_V1',representation_kind='WHOLE',
            decoded_content=decoded,whole_storage_ref=compressed,whole_codec_ref=codec,blocks=None,canonicalization_ref=canon_contract))
    objects={};occurrences=[];relations=[];incidences=[];witnesses=[];cases=[]
    def object_record(can,tree):
        name=canonical_bytes(can).decode();ref=native(name)
        if name not in objects:
            objects[name]=dict(PREFIX,schema_id='IG_STORAGE_OBJECT_V1',object_ref=ref,reference_record_ref=None,
                layer='G5_G6_EXACT_DECORATED_G4_TREE',carrier_schema_ref=canon_contract,
                canonicalization_ref=canon_contract,equality_contract_ref=canon_contract,
                canonical_payload_ref=payload(can),structure_ref=present(value(tree),'Complete retained labeled representative'),evidence=empty)
        return ref
    def occurrence(name,obj,formation,position):
        r=dict(PREFIX,schema_id='IG_STORAGE_OCCURRENCE_V1',occurrence_ref=native(name),object_ref=obj,context_ref=source,
            position_ref=present(value(position),'Exact labeled component/formation position'),parent_occurrence_ref=None,formation_ref=formation)
        occurrences.append(r);return r
    role_registry=value({'roles':['left_parent','right_parent','result'],'slots':'left0/right1/result2plus'})
    types={}
    for name,arity in [('EXACT_SET_RELATION',None),('LABELED_OWNER_FORMATION','3')]:
        types[name]=value(dict(PREFIX,schema_id='IG_STORAGE_RELATIONTYPE_V1',type_name=name,domain='SCIENTIFIC_STRUCTURE',
            definition_ref=identity_definition,allowed_arity_min='2' if arity is None else '3',allowed_arity_max=arity,
            cardinality_semantics='ORDERED',cycle_policy='ACYCLIC_REQUIRED',cycle_scope_ref=source,
            role_registry_ref=role_registry,multiplicity_policy='UNIT_INCIDENCES',scientific_authority_ref=None))
    def relation(name,endpoints,operator,decorations,witness_rows,kind):
        ref=native(name);rows=[]
        for i,(role,direction,endpoint_kind,endpoint_ref,mapref) in enumerate(endpoints):
            rows.append(dict(PREFIX,schema_id='IG_STORAGE_INCIDENCE_V1',incidence_ref=native(name+':INC:'+str(i).zfill(6)),relation_ref=ref,
                endpoint={'kind':endpoint_kind,'ref':endpoint_ref},slot=str(i),role=role,direction=direction,
                multiplicity='1',port_map_ref=present(mapref,'Full typed endpoint and vertex correspondence')))
        incidences.extend(rows)
        ir=collection('relation_incidences','IG_STORAGE_INCIDENCE_V1',rows,['incidence_ref','native_id'],'ORDERED')
        wr=collection('formation_witnesses','IG_STORAGE_OCCURRENCE_V1',witness_rows,['occurrence_ref','native_id'])
        relations.append(dict(PREFIX,schema_id='IG_STORAGE_RELATION_V1',relation_ref=ref,type_ref=types[kind],
            scope={'layer':'G5_G6_EXACT_DECORATED_G4_TREE','lane':'R','definition_ref':source},incidences=ir,incidence_count=str(len(rows)),
            operator_ref=present(value(operator),'Frozen directed endpoint operator'),
            decorations_ref=present(decorations,'Complete original export/formation scientific data'),witnesses=wr))
        return ref
    for case,(owner,marker) in enumerate(zip(owners,markers)):
        require(owner['binding']['task_ref']==marker['binding']['task_ref'] and owner['operator']==marker['operator'],'STORAGE_CASE_CORRESPONDENCE')
        require(owner['authority']==owners[0]['authority'],'STORAGE_AUTHORITY_SCOPE')
        task=owner['binding']['task_ref'];classes={h:k for h,k,c in owner['authority']['classes']}
        decoded=marker['write_result']['diagnostics']
        parent_refs=[]
        for side in ('left','right'):
            tree=owner['parents'][side];obj=object_record(decoded[side+'_decoded_unrooted_canon'],tree)
            occ=occurrence(owner['binding'][side+'_parent_occurrence_ref'],obj,value({'original_binding':owner['binding'],'side':side}),{'side':side,'vertices':list(range(tree['n']))})
            parent_refs.append(occ['occurrence_ref'])
        child_refs=[object_record(r['native_identity'],r['full_tree']) for r in owner['child_SET']]
        local=[]
        for row in owner['formation_occurrences']:
            retained=owner['child_SET'][row['retained_child_ordinal']]['full_tree']
            witness={'original_formation':row,'raw_to_retained_vertex_map':vertex_map(row['full_raw_child'],retained,classes)}
            witness_ref=value(witness);witnesses.append({'task':task,'ordinal':row['ordinal'],'witness_ref':witness_ref})
            child=occurrence(task+':FORMATION:'+str(row['ordinal']),child_refs[row['retained_child_ordinal']],witness_ref,{'formation_ordinal':row['ordinal']})
            local.append(child)
            relation(task+':FORMATION_RELATION:'+str(row['ordinal']),[
                ('left_parent','INPUT','OCCURRENCE',parent_refs[0],witness_ref),
                ('right_parent','INPUT','OCCURRENCE',parent_refs[1],witness_ref),
                ('result','OUTPUT','OCCURRENCE',child['occurrence_ref'],witness_ref)],owner['operator'],witness_ref,[child],'LABELED_OWNER_FORMATION')
        case_payload=payload({'owner':owner,'marker':marker})
        endpoints=[('left_parent','INPUT','OCCURRENCE',parent_refs[0],source),('right_parent','INPUT','OCCURRENCE',parent_refs[1],source)]
        endpoints += [('result','OUTPUT','OBJECT',r,source) for r in child_refs]
        relation(task,endpoints,owner['operator'],case_payload,local,'EXACT_SET_RELATION')
        cases.append({'task':task,'case':case,'original_binding':owner['binding'],'case_payload_ref':case_payload,'formation_count':len(local)})
    groups={'objects':list(objects.values()),'occurrences':occurrences,'relations':relations,'incidences':incidences}
    field={'objects':['canonical_payload_ref','sha256'],'occurrences':['occurrence_ref','native_id'],
        'relations':['relation_ref','native_id'],'incidences':['incidence_ref','native_id']}
    sid={'objects':'OBJECT','occurrences':'OCCURRENCE','relations':'RELATION','incidences':'INCIDENCE'}
    collections={k:collection(k,'IG_STORAGE_'+sid[k]+'_V1',v,field[k],'ORDERED' if k=='incidences' else 'SET') for k,v in groups.items()}
    manifest={'schema_id':'IG_Q19_FULL_EXPORT_STORAGE_CANDIDATE_V1','binding':binding,'scope_ref':source,
        'identity_profile_ref':profile,'accepted_archive_ref':archive_ref,'collections':collections,'cases':cases,
        'formation_witnesses':witnesses,'counts':{k:len(v) for k,v in groups.items()},
        'content_refs':list(blobs.values()),'production_authority':'NONE','release_acceptance':'NOT_GRANTED'}
    raw=canonical_bytes(manifest);(root/'CANDIDATE.json').write_bytes(raw);release=content_ref(raw)
    store=ContentDirectory(content);records=[]
    for kind,c in collections.items():
        for r in collection_records(store,c,collection_kind=kind,purpose='SCIENCE'):
            records.append(IndexRecord(kind+'/'+r.key,r.raw))
    records.sort(key=lambda r:r.key);inv=inventory(records)
    index=build_index(records,root/'index',release_root=release,source_collections=[r['root'] for r in collections.values()],expected_inventory=inv)
    receipt={'schema_id':'IG_Q19_STORAGE_INDEX_RECEIPT_V1','candidate_ref':release,'index':index,'inventory':inv,
        'counts':manifest['counts'],'production_authority':'NONE','scientific_acceptance':'NOT_GRANTED'}
    (root/'INDEX_RECEIPT.json').write_bytes(canonical_bytes(receipt))
    return receipt


def verify_export_storage(directory):
    root=Path(directory);raw=(root/'CANDIDATE.json').read_bytes();m=json.loads(raw);receipt=json.loads((root/'INDEX_RECEIPT.json').read_bytes())
    require(content_ref(raw)==receipt['candidate_ref'],'STORAGE_CANDIDATE_BINDING')
    store=ContentDirectory(root/'content')
    for ref in m['content_refs']:store.read(ref,int(ref['size_bytes']))
    records=[];counts={}
    for kind,c in m['collections'].items():
        rows=list(collection_records(store,c,collection_kind=kind,purpose='SCIENCE'));counts[kind]=len(rows)
        records.extend(IndexRecord(kind+'/'+r.key,r.raw) for r in rows)
    records.sort(key=lambda r:r.key);require(inventory(records)==receipt['inventory'],'STORAGE_INDEX_INPUT_INVENTORY')
    require(counts==m['counts']==receipt['counts'],'STORAGE_COLLECTION_COUNTS')
    with ReadIndex(root/'index',expected_manifest_ref=receipt['index']['manifest_ref'],release_root=receipt['candidate_ref']) as index:
        for sid in ('OBJECT','OCCURRENCE','RELATION','INCIDENCE'):
            rows=[];cursor=None
            while True:
                page=index.records(schema_id='IG_STORAGE_'+sid+'_V1',limit=17,after=cursor);rows.extend(page['rows']);cursor=page['next_cursor']
                if cursor is None:break
            expected=[r.raw for r in records if json.loads(r.raw)['schema_id']=='IG_STORAGE_'+sid+'_V1']
            require([canonical_bytes(r['record']) for r in rows]==expected,'STORAGE_COLD_INDEX_BYTES')
    return {'status':'PASS','counts':counts,'content_objects':len(m['content_refs']),
        'inventory':receipt['inventory'],'candidate_ref':receipt['candidate_ref'],'production_authority':'NONE'}
