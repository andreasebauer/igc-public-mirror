"""Verify a bounded, fully committed reference dataset without recovery writes."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from .storage_schema import strict_loads
from .replay_reference_data import (ReplayReferenceDataStore, empty_manifest,
    HASH_RE, RECORD_SCHEMA, TRANSACTION_SCHEMA, RECOVERABLE_TRANSACTION_SCHEMA,
    COMMIT_SCHEMA)


class DatasetSnapshotError(ValueError):
    pass


def verify_dataset_snapshot(manifest_raw,*,records=(),transactions=(),commits=(),
                            max_total_bytes=67108864,max_records=4096):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864 or type(max_records) is not int or not 1<=max_records<=4096:
        raise DatasetSnapshotError('INVALID_DATASET_BUDGET')
    groups=[('records',records,'record_sha256',{RECORD_SCHEMA}),
        ('transactions',transactions,'transaction_sha256',{TRANSACTION_SCHEMA,RECOVERABLE_TRANSACTION_SCHEMA}),
        ('commits',commits,'commit_sha256',{COMMIT_SCHEMA})]
    if any(not isinstance(rows,(list,tuple)) for _,rows,_,_ in groups):raise DatasetSnapshotError('DATASET_LIST_REQUIRED')
    if 1+sum(len(rows) for _,rows,_,_ in groups)>max_records:raise DatasetSnapshotError('DATASET_RECORD_BUDGET')
    raws=[manifest_raw,*records,*transactions,*commits]
    if any(type(raw) is not bytes for raw in raws):raise DatasetSnapshotError('DATASET_BYTES_REQUIRED')
    total=sum(len(raw) for raw in raws)
    if total>max_total_bytes or any(len(raw)>4194304 for raw in raws):raise DatasetSnapshotError('DATASET_BYTE_BUDGET')
    def parse(raw):
        obj=strict_loads(raw,max_bytes=4194304)
        if not isinstance(obj,dict):raise DatasetSnapshotError('DATASET_OBJECT_REQUIRED')
        return obj
    manifest=parse(manifest_raw);objects={};inventory=[]
    def add(raw,role,seal):
        inventory.append({'role':role,'semantic_sha256':seal,'content_ref':{'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}})
    with TemporaryDirectory(prefix='ig-dataset-snapshot-') as temp:
        root=Path(temp);(root/'MANIFEST.json').write_bytes(manifest_raw)
        for directory,rows,field,schemas in groups:
            target=root/directory;target.mkdir();seen={}
            for raw in rows:
                obj=parse(raw);digest=obj.get(field)
                if obj.get('schema_id') not in schemas or type(digest) is not str or HASH_RE.fullmatch(digest) is None:
                    raise DatasetSnapshotError('INVALID_DATASET_MEMBER_IDENTITY')
                if digest in seen:raise DatasetSnapshotError('DUPLICATE_DATASET_MEMBER')
                seen[digest]=obj;(target/(digest+'.json')).write_bytes(raw);add(raw,directory,digest)
            objects[directory]=seen
        # __init__ invokes recovery. Bypass it deliberately and use verify(),
        # which reads and validates the original staged bytes without repairing.
        store=object.__new__(ReplayReferenceDataStore);store.root=root
        store.verify()
        txs=objects['transactions'];commit_rows=objects['commits']
        committed=[c['transaction_sha256'] for c in commit_rows.values()]
        if len(committed)!=len(set(committed)) or set(committed)!=set(txs):
            raise DatasetSnapshotError('DATASET_PENDING_OR_DUPLICATE_COMMIT')
        by_record={}
        for digest in txs:
            tx=store._transaction(root/'transactions'/(digest+'.json'))
            if tx['record_id'] in by_record:raise DatasetSnapshotError('DUPLICATE_DATASET_TRANSACTION')
            by_record[tx['record_id']]=tx
        if set(by_record)!=set(manifest['record_ids']):raise DatasetSnapshotError('DATASET_HISTORY_INVENTORY_MISMATCH')
        store.manifest=empty_manifest()
        for rid in manifest['record_ids']:
            tx=by_record[rid];record=objects['records'][manifest['record_sha256_by_id'][rid]]
            proposed=store._proposed_manifest(record)
            if tx['parent_manifest_sha256']!=store.manifest['manifest_sha256'] or tx['proposed_manifest_sha256']!=proposed['manifest_sha256'] or tx['record_sha256']!=record['record_sha256']:
                raise DatasetSnapshotError('DATASET_TRANSACTION_CHAIN_MISMATCH')
            store.manifest=proposed
        if store.manifest!=manifest:raise DatasetSnapshotError('DATASET_FINAL_MANIFEST_MISMATCH')
        if (root/'MANIFEST.json').read_bytes()!=manifest_raw:raise DatasetSnapshotError('DATASET_VERIFIER_MUTATED_MANIFEST')
    add(manifest_raw,'MANIFEST',manifest['manifest_sha256'])
    return {'status':'DATASET_SNAPSHOT_VERIFIED','scope':'REFERENCE_INVENTORY_AND_COMPLETE_COMMITTED_HISTORY_ONLY',
        'manifest_sha256':manifest['manifest_sha256'],'record_ids':list(manifest['record_ids']),
        'record_sha256_by_id':dict(manifest['record_sha256_by_id']),
        'execution_authorized':False,'scientific_acceptance':'NOT_GRANTED','dependency_closure_verified':False,
        'recovery_performed':False,'records_checked':len(raws),'bytes_checked':total,'inventory':inventory}
