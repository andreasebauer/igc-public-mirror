"""Explicit byte bindings for supported original IGRD record types; no authority grant."""
import hashlib
from .storage_schema import strict_loads,canonical_bytes
from .storage_catalog import _ref
from .replay_reference_data import verify_reference_record,ID_RE,HASH_RE


class LegacyBindingError(ValueError):pass


def reference(ref):
    if (not isinstance(ref,dict) or set(ref)!={'record_id','record_sha256','content_ref'}
        or not isinstance(ref['record_id'],str) or ID_RE.fullmatch(ref['record_id']) is None
        or not isinstance(ref['record_sha256'],str) or HASH_RE.fullmatch(ref['record_sha256']) is None):
        raise LegacyBindingError('INVALID_LEGACY_RECORD_REF')
    _ref(ref['content_ref']);return ref


def legacy_dependencies(raw,binding_raw):
    """Original semantic seal and separately bound original file bytes are checked.

    Binding profile describes symbolic schema/formation refs explicitly. Its
    interpretation is not a scientific certificate for those external bytes.
    """
    record=verify_reference_record(strict_loads(raw))
    kind=record['record_type']
    if kind not in {'CANONICAL_OBJECT','MECHANISM','DEPENDENCY_LINK','GRADUATION','EARNED_ALGEBRA','SRCF_EVIDENCE','COMPARISON','AUDIT_AUTHORIZATION','NEGATIVE_RESULT','GENERATION_RECIPE'}:raise LegacyBindingError('UNSUPPORTED_LEGACY_DEPENDENCY_TYPE')
    if kind=='SRCF_EVIDENCE':
        eq=record['payload']['equality_contract']
        if eq['mode']!='CANONICAL_JSON' or eq['compression']!='EXACT' or eq['certificate_sha256'] is not None:
            raise LegacyBindingError('UNSUPPORTED_LEGACY_EVIDENCE_PROFILE')
    if kind=='COMPARISON':
        eq=record['payload']['equality_contract']
        if eq['mode']!='CANONICAL_JSON' or eq['compression']!='EXACT' or eq['certificate_sha256'] is not None:
            raise LegacyBindingError('UNSUPPORTED_LEGACY_COMPARISON_PROFILE')
        if record['payload']['qualification_ids']:
            raise LegacyBindingError('LEGACY_QUALIFICATION_CATALOGUE_REQUIRED')
    if kind=='GENERATION_RECIPE':
        params=record['payload']['parameters'];keys=set(params)
        valid=keys in (set(),{'assertion_id','anchors'},{'assertion_id','anchors','execution_class'},{'assertion_ids','execution_class'})
        for key,value in params.items():
            if key in {'anchors','assertion_ids'}:
                valid=valid and isinstance(value,list) and all(isinstance(x,str) and x for x in value)
            else:valid=valid and isinstance(value,str) and bool(value)
        if not valid:raise LegacyBindingError('UNSUPPORTED_LEGACY_RECIPE_PARAMETERS')
    binding=strict_loads(binding_raw,max_bytes=262144)
    fields={'schema_id','record_ref','record_links','digest_bindings','symbol_bindings'}
    version=binding.get('schema_id') if isinstance(binding,dict) else None
    if version in {'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V2','IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V3'}:fields.add('semantic_bindings')
    if version=='IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V3':fields.add('oracle_witness_binding')
    if (not isinstance(binding,dict) or set(binding)!=fields
        or version not in {'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1','IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V2','IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V3'}
        or canonical_bytes(binding)!=binding_raw):
        raise LegacyBindingError('INVALID_LEGACY_BINDING')
    if kind=='SRCF_EVIDENCE' and version!='IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V2':
        raise LegacyBindingError('LEGACY_SEMANTIC_BINDING_REQUIRED')
    rr=reference(binding['record_ref'])
    if (rr['record_id']!=record['record_id'] or rr['record_sha256']!=record['record_sha256']
        or rr['content_ref']!={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}):
        raise LegacyBindingError('LEGACY_BINDING_IDENTITY_MISMATCH')
    for field in ['record_links','digest_bindings','symbol_bindings']:
        if not isinstance(binding[field],list) or len(binding[field])>4096:raise LegacyBindingError('BINDING_LIST_BUDGET_OR_TYPE')
    payload=record['payload'];needed=set(record['dependencies'])
    oracle_check=None
    if version=='IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V3':
        if kind!='NEGATIVE_RESULT':raise LegacyBindingError('ORACLE_WITNESS_NEGATIVE_REQUIRED')
        oracle_check=binding['oracle_witness_binding']
        if not isinstance(oracle_check,dict) or set(oracle_check)!={'oracle_ref','files'} or not isinstance(oracle_check['files'],list) or len(oracle_check['files'])>256:raise LegacyBindingError('INVALID_ORACLE_WITNESS_BINDING')
        _ref(oracle_check['oracle_ref']);labels=set()
        for row in oracle_check['files']:
            if not isinstance(row,dict) or set(row)!={'file','content_ref'} or type(row['file']) is not str or row['file'] in labels:raise LegacyBindingError('INVALID_ORACLE_WITNESS_FILE_BINDING')
            labels.add(row['file']);_ref(row['content_ref'])
        if any(not isinstance(w,dict) or set(w)!={'file','pointer','equals'} or type(w['file']) is not str for w in payload['witnesses']) or labels!={w['file'] for w in payload['witnesses']}:raise LegacyBindingError('ORACLE_WITNESS_FILE_INVENTORY')
    if kind=='MECHANISM':needed.update(payload['output_record_ids'])
    links={}
    for row in binding['record_links']:
        reference(row)
        if row['record_id'] in links:raise LegacyBindingError('DUPLICATE_LEGACY_RECORD_BINDING')
        links[row['record_id']]=row
    if set(links)!=needed:raise LegacyBindingError('LEGACY_RECORD_LINK_CLOSURE_MISMATCH')
    digests={f'/provenance/source_hashes/{i}/sha256':r['sha256'] for i,r in enumerate(record['provenance']['source_hashes'])}
    symbols={};symbol_digests={}
    if kind=='CANONICAL_OBJECT':
        digests['/payload/canonical_bytes_sha256']=payload['canonical_bytes_sha256']
        symbols={f'/payload/{k}':payload[k] for k in ['carrier_schema','formation_provenance']}
    elif kind=='MECHANISM':
        digests.update({f'/payload/implementation_hashes/{i}/sha256':r['sha256'] for i,r in enumerate(payload['implementation_hashes'])})
    elif kind=='GENERATION_RECIPE':
        digests['/payload/implementation_sha256']=payload['implementation_sha256']
    elif kind=='AUDIT_AUTHORIZATION':
        digests.update({f'/payload/evidence_hashes/{i}/sha256':r['sha256'] for i,r in enumerate(payload['evidence_hashes'])})
    elif kind=='NEGATIVE_RESULT' and oracle_check is None:
        sources={row['ref']:row['sha256'] for row in record['provenance']['source_hashes']}
        for i,witness in enumerate(payload['witnesses']):
            if isinstance(witness,str) and witness:continue
            if not isinstance(witness,dict):raise LegacyBindingError('UNSUPPORTED_NEGATIVE_WITNESS_PROFILE')
            if set(witness)=={'source','anchors'}:
                if (not isinstance(witness['anchors'],list) or not witness['anchors']
                    or not all(isinstance(a,str) and a for a in witness['anchors'])):
                    raise LegacyBindingError('UNSUPPORTED_NEGATIVE_WITNESS_PROFILE')
            elif set(witness)=={'source','field','value'}:
                if not all(isinstance(witness[k],str) and witness[k] for k in ['field','value']):
                    raise LegacyBindingError('UNSUPPORTED_NEGATIVE_WITNESS_PROFILE')
            else:raise LegacyBindingError('UNSUPPORTED_NEGATIVE_WITNESS_PROFILE')
            source=witness['source']
            if not isinstance(source,str) or source not in sources:raise LegacyBindingError('WITNESS_SOURCE_NOT_PINNED')
            path=f'/payload/witnesses/{i}/source';symbols[path]=source;symbol_digests[path]=sources[source]
    contents=[];seen=set()
    for row in binding['digest_bindings']:
        if not isinstance(row,dict) or set(row)!={'path','content_ref'}:raise LegacyBindingError('INVALID_DIGEST_BINDING')
        path=row['path'];_ref(row['content_ref'])
        if not isinstance(path,str) or path in seen or path not in digests or row['content_ref']['sha256']!=digests[path]:
            raise LegacyBindingError('DIGEST_BINDING_MISMATCH')
        seen.add(path);contents.append({'path':path,'ref':row['content_ref'],'role':'LEGACY_RAW_DIGEST'})
    if seen!=set(digests):raise LegacyBindingError('MISSING_DIGEST_BINDING')
    seen=set()
    for row in binding['symbol_bindings']:
        if not isinstance(row,dict) or set(row)!={'path','value','content_ref'}:raise LegacyBindingError('INVALID_SYMBOL_BINDING')
        path=row['path'];_ref(row['content_ref'])
        if not isinstance(path,str) or path in seen or path not in symbols or row['value']!=symbols[path]:
            raise LegacyBindingError('SYMBOL_BINDING_MISMATCH')
        if path in symbol_digests and row['content_ref']['sha256']!=symbol_digests[path]:
            raise LegacyBindingError('WITNESS_SOURCE_DIGEST_MISMATCH')
        seen.add(path);contents.append({'path':path,'ref':row['content_ref'],'role':'LEGACY_SYMBOL_DEFINITION'})
    if seen!=set(symbols):raise LegacyBindingError('MISSING_SYMBOL_BINDING')
    semantics=binding.get('semantic_bindings',[])
    if not isinstance(semantics,list) or len(semantics)>4096:raise LegacyBindingError('BINDING_LIST_BUDGET_OR_TYPE')
    expected={'/payload/result_identity':payload['result_identity']} if kind=='SRCF_EVIDENCE' else {}
    semantic_checks=[];seen=set()
    for row in semantics:
        if not isinstance(row,dict) or set(row)!={'path','profile','content_ref'}:raise LegacyBindingError('INVALID_SEMANTIC_BINDING')
        path=row['path'];_ref(row['content_ref'])
        if (not isinstance(path,str) or path in seen or path not in expected
            or row['profile']!='IG_CANONICAL_JSON_V1'):
            raise LegacyBindingError('SEMANTIC_BINDING_MISMATCH')
        seen.add(path);semantic_checks.append({'path':path,'profile':row['profile'],'ref':row['content_ref'],'sha256':expected[path]})
        contents.append({'path':path,'ref':row['content_ref'],'role':'LEGACY_CANONICAL_JSON_RESULT'})
    if seen!=set(expected):raise LegacyBindingError('MISSING_SEMANTIC_BINDING')
    if oracle_check is not None:
        contents.append({'path':'/oracle_witness_binding/oracle_ref','ref':oracle_check['oracle_ref'],'role':'LEGACY_ORACLE_MANIFEST'})
        contents.extend({'path':'/oracle_witness_binding/files/'+str(i),'ref':row['content_ref'],'role':'LEGACY_ORACLE_WITNESS_FILE'} for i,row in enumerate(oracle_check['files']))
    return {'oracle_witness_check':oracle_check,'semantic_checks':semantic_checks,'record_ref':rr,'record_links':list(links.values()),'stored_content':contents,
        'record_type':kind,'epistemic_status':record['epistemic_status'],'science_execution':record['science_execution'],
        'authority_effect':record['authority_effect'],'scientific_acceptance':'NOT_GRANTED'}
