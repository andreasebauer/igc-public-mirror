"""Bounded manifest/catalogue integrity and explicit raw pin verification.

References are labels in a supplied byte map, never paths to open or execute.
This checks transport integrity, not evidence meaning or transitive closure.
"""
import hashlib
from .storage_schema import strict_loads
from .canon import canonical_sha256
from .replay_obligation_compiler import compile_obligations, verify_compiled_manifest
from .replay_reference_data import HASH_RE


class ManifestSourceError(ValueError):
    pass


def verify_manifest_sources(manifest_raw, catalogue_raw, sources, *,
                            max_total_bytes=67108864, max_records=4096):
    if type(max_total_bytes) is not int or not 1 <= max_total_bytes <= 67108864 or type(max_records) is not int or not 2 <= max_records <= 4096:
        raise ManifestSourceError('INVALID_MANIFEST_SOURCE_BUDGET')
    if type(sources) is not dict or 2+len(sources) > max_records:
        raise ManifestSourceError('SOURCE_RECORD_BUDGET_OR_TYPE')
    if any(type(k) is not str or not k or type(v) is not bytes for k,v in sources.items()):
        raise ManifestSourceError('SOURCE_LABEL_AND_BYTES_REQUIRED')
    raws=[manifest_raw,catalogue_raw,*sources.values()]
    if any(type(raw) is not bytes for raw in raws):
        raise ManifestSourceError('MANIFEST_SOURCE_BYTES_REQUIRED')
    total=sum(len(raw) for raw in raws)
    if total > max_total_bytes or any(len(raw)>4194304 for raw in raws):
        raise ManifestSourceError('MANIFEST_SOURCE_BYTE_BUDGET')
    manifest=strict_loads(manifest_raw,max_bytes=4194304)
    catalogue=strict_loads(catalogue_raw,max_bytes=4194304)
    if not isinstance(manifest,dict) or not isinstance(catalogue,dict):
        raise ManifestSourceError('MANIFEST_SOURCE_OBJECT_REQUIRED')
    verify_compiled_manifest(manifest)
    if canonical_sha256(catalogue) != manifest['source_catalogue_sha256']:
        raise ManifestSourceError('CATALOGUE_SEMANTIC_HASH_MISMATCH')
    compiled=compile_obligations(catalogue)
    if compiled != manifest:
        raise ManifestSourceError('RECOMPILED_MANIFEST_MISMATCH')
    expected={};occurrences=0
    def pin(row):
        nonlocal occurrences
        occurrences+=1
        if occurrences>32768:
            raise ManifestSourceError('SOURCE_PIN_OCCURRENCE_BUDGET')
        if not isinstance(row,dict) or set(row)!={'ref','sha256'} or type(row['ref']) is not str or not row['ref'] or type(row['sha256']) is not str or HASH_RE.fullmatch(row['sha256']) is None:
            raise ManifestSourceError('INVALID_SOURCE_PIN')
        label=row['ref'];digest=row['sha256']
        if label in expected and expected[label]!=digest:
            raise ManifestSourceError('CONFLICTING_SOURCE_PIN')
        expected[label]=digest
        if 2+len(expected)>max_records:
            raise ManifestSourceError('SOURCE_RECORD_BUDGET_OR_TYPE')
    for group,fields in [
        ('obligations',('source_hashes','evidence_hashes')),
        ('historical_assertion_mappings',('source_hashes',)),
        ('historical_audit_authorizations',('historical_source_hashes','accepted_equivalence_certificates','audit_provenance','audit_evidence')),
        ('known_replay_qualifications',('evidence_pins',))]:
        rows=catalogue.get(group,[])
        if not isinstance(rows,list):raise ManifestSourceError('INVALID_SOURCE_OWNER_LIST')
        for owner in rows:
            for field in fields:
                pins=owner.get(field,[])
                if not isinstance(pins,list):raise ManifestSourceError('INVALID_SOURCE_PIN_LIST')
                for row in pins:pin(row)
    for field in ('audit_provenance_ledger','known_replay_qualification_ledger','execution_class_ledger','cost_budget_ledger'):
        if field in catalogue and catalogue[field] is not None:pin(catalogue[field])
    if not expected:raise ManifestSourceError('SOURCE_PINS_REQUIRED')
    if set(sources)!=set(expected):raise ManifestSourceError('SOURCE_INVENTORY_MISMATCH')
    for label,digest in expected.items():
        if hashlib.sha256(sources[label]).hexdigest()!=digest:
            raise ManifestSourceError('SOURCE_BYTES_MISMATCH')
    def ref(raw):return {'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}
    return {'status':'MANIFEST_SOURCES_VERIFIED','scope':'RECOMPILED_MANIFEST_AND_EXPLICIT_RAW_PINS_ONLY',
        'execution_authorized':False,'scientific_acceptance':'NOT_GRANTED','dependency_closure_verified':False,
        'manifest_sha256':manifest['dag_sha256'],'source_catalogue_sha256':manifest['source_catalogue_sha256'],
        'manifest_content_ref':ref(manifest_raw),'catalogue_content_ref':ref(catalogue_raw),
        'sources':[{'ref':label,'content_ref':ref(sources[label])} for label in sorted(sources)],
        'records_checked':len(sources)+2,'pin_occurrences_checked':occurrences,'bytes_checked':total}
