"""Schema-directed direct references, not full closure or release acceptance.

Only the pinned storage schema drives traversal. Native identity and decoded
commitments are not mistaken for stored-content edges. External interpretation
and codec dependencies require their own registered profiles in a closure walk.
"""
from .storage_schema import validate_record_bytes, _compiled, SCHEMA_SHA256


class DependencyError(ValueError):
    pass


_IDENTITIES = {
    'Object': 'object_ref', 'Occurrence': 'occurrence_ref',
    'Relation': 'relation_ref', 'Incidence': 'incidence_ref',
}


def record_dependencies(raw):
    record=validate_record_bytes(raw)
    defs=_compiled().schema['$defs']
    matches=[name for name,schema in defs.items()
             if schema.get('properties',{}).get('schema_id',{}).get('const')==record['schema_id']]
    if len(matches)!=1:raise DependencyError('UNSUPPORTED_REFERENCE_SCHEMA')
    name=matches[0]
    result={'schema_id':'IG_STORAGE_DIRECT_DEPENDENCIES_V1','source_schema_id':record['schema_id'],
            'contract_schema_sha256':SCHEMA_SHA256,'verification_scope':'DIRECT_SCHEMA_REFERENCES',
            'scientific_acceptance':'NOT_GRANTED','stored_content':[], 'collections':[],
            'native_references':[], 'reference_records':[], 'decoded_checks':[],
            'lineage':[], 'unresolved_fields':[], 'skipped_optional_extensions':[]}

    def pointer(path):return '/'+ '/'.join(str(x).replace('~','~0').replace('/','~1') for x in path)
    def content(value,path,role):result['stored_content'].append({'path':pointer(path),'ref':value,'role':role})
    def walk(value,schema,path):
        if value is None:return
        if '$ref' in schema:
            tag=schema['$ref'].removeprefix('#/$defs/')
            if tag not in defs:raise DependencyError('UNSUPPORTED_SCHEMA_REFERENCE')
            if tag=='ContentRef':
                if name in {'Payload','Block'} and path==['decoded_content']:
                    result['decoded_checks'].append({'path':pointer(path),'ref':value})
                elif name=='ReleaseRoot' and path==['previous_release_root']:
                    result['lineage'].append({'path':pointer(path),'ref':value})
                else:content(value,path,'REQUIRES_INTERPRETATION_PROFILE')
                return
            if tag=='NativeRef':
                role='IDENTITY' if path==[_IDENTITIES.get(name)] else 'REFERENCE'
                result['native_references'].append({'path':pointer(path),'ref':value,'role':role})
                content(value['scope_ref'],path+['scope_ref'],'NATIVE_SCOPE')
                content(value['profile_ref'],path+['profile_ref'],'IDENTITY_PROFILE')
                return
            if tag=='CollectionRef':
                result['collections'].append({'path':pointer(path),'ref':value})
                content(value['root'],path+['root'],'COLLECTION_PAGE')
                content(value['key_definition_ref'],path+['key_definition_ref'],'COLLECTION_KEY_PROFILE')
                return
            if tag=='ReferenceRecordRef':
                result['reference_records'].append({'path':pointer(path),'ref':value,'requires_legacy_seal_check':True})
                content(value['content_ref'],path+['content_ref'],'NATIVE_REFERENCE_RECORD')
                return
            if tag=='OptionalFieldRef':
                if value['state']=='UNKNOWN':result['unresolved_fields'].append({'path':pointer(path),'explanation':value['explanation']})
                # PRESENT alone creates a content dependency; schema enforces null otherwise.
                if value['content_ref'] is not None:content(value['content_ref'],path+['content_ref'],'OPTIONAL_FIELD_CONTENT')
                return
            if tag=='Extension':
                # Required extensions have already been refused by admission.
                if value['required_to_interpret']:raise DependencyError('UNSUPPORTED_REQUIRED_EXTENSION')
                result['skipped_optional_extensions'].append({'path':pointer(path),'extension_id':value['extension_id'],'version':value['version']})
                content(value['definition_ref'],path+['definition_ref'],'OPTIONAL_EXTENSION_DEFINITION')
                content(value['payload_ref'],path+['payload_ref'],'OPTIONAL_EXTENSION_PAYLOAD')
                return
            if tag=='PageEntry':
                content(value['content_ref'],path+['content_ref'],'COLLECTION_PAGE' if value['entry_type']=='PAGE' else 'COLLECTION_SHARD')
                return
            walk(value,defs[tag],path);return
        if 'anyOf' in schema:
            branches=[s for s in schema['anyOf'] if s.get('type')!='null']
            if len(branches)!=1:raise DependencyError('AMBIGUOUS_REFERENCE_BRANCH')
            walk(value,branches[0],path);return
        if isinstance(value,dict):
            for field,child in sorted(schema.get('properties',{}).items()):
                if field in value:walk(value[field],child,path+[field])
        elif isinstance(value,list):
            for i,item in enumerate(value):walk(item,schema['items'],path+[i])
    walk(record,defs[name],[])
    return result
