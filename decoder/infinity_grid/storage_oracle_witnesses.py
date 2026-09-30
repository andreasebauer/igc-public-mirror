"""Bounded original oracle witness verification; no execution or closure grant."""
import hashlib,re
from .storage_schema import strict_loads,canonical_bytes
from .replay_reference_data import verify_reference_record


class OracleWitnessError(ValueError):pass


def verify_oracle_negative_witnesses(record_raw,oracle_raw,files,*,max_total_bytes=67108864):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864:raise OracleWitnessError('INVALID_WITNESS_BUDGET')
    if type(files) is not dict or len(files)>256 or type(record_raw) is not bytes or type(oracle_raw) is not bytes or any(type(v) is not bytes for v in files.values()):raise OracleWitnessError('WITNESS_BYTES_REQUIRED')
    raws=[record_raw,oracle_raw,*files.values()];total=sum(map(len,raws))
    if total>max_total_bytes or any(len(b)>4194304 for b in raws):raise OracleWitnessError('WITNESS_BYTE_BUDGET')
    sha=lambda b:hashlib.sha256(b).hexdigest()
    record=verify_reference_record(strict_loads(record_raw,max_bytes=4194304))
    if record['record_type']!='NEGATIVE_RESULT':raise OracleWitnessError('NEGATIVE_RECORD_REQUIRED')
    if record['provenance']['source_hashes']!=[{'ref':'infinity_grid/resources/oracle/V026_SCIENTIFIC_ORACLE_MANIFEST.json','sha256':sha(oracle_raw)}]:raise OracleWitnessError('ORACLE_RAW_BINDING_MISMATCH')
    oracle=strict_loads(oracle_raw,max_bytes=4194304)
    if not isinstance(oracle,dict) or oracle.get('schema_id')!='IG_V026_SCIENTIFIC_ORACLE_MANIFEST_V1' or oracle.get('oracle_sha256')!=sha(canonical_bytes({k:v for k,v in oracle.items() if k!='oracle_sha256'})):raise OracleWitnessError('ORACLE_SEAL_MISMATCH')
    witnesses=record['payload']['witnesses']
    if not 1<=len(witnesses)<=256:raise OracleWitnessError('WITNESS_COUNT_BUDGET')
    for w in witnesses:
        if not isinstance(w,dict) or set(w)!={'file','pointer','equals'} or type(w['file']) is not str or type(w['pointer']) is not str:raise OracleWitnessError('UNSUPPORTED_WITNESS_PROFILE')
        if type(w['equals']) not in (str,int,bool) and w['equals'] is not None:raise OracleWitnessError('UNSUPPORTED_WITNESS_VALUE')
    needed={w['file'] for w in witnesses}
    if set(files)!=needed:raise OracleWitnessError('WITNESS_FILE_INVENTORY_MISMATCH')
    corpus=oracle.get('corpus_files');assertions=oracle.get('assertions')
    if not isinstance(corpus,list) or not isinstance(assertions,list) or len(corpus)>4096 or len(assertions)>4096:raise OracleWitnessError('ORACLE_INVENTORY_BUDGET')
    index={}
    for row in corpus:
        if not isinstance(row,dict) or type(row.get('path')) is not str:raise OracleWitnessError('INVALID_CORPUS_ROW')
        if row['path'] in index:raise OracleWitnessError('DUPLICATE_CORPUS_PATH')
        index[row['path']]=row
    parsed={};inventory=[]
    for label,raw in files.items():
        row=index.get(label)
        if row is None or row.get('sha256')!=sha(raw) or type(row.get('size_bytes')) is not int or row['size_bytes']!=len(raw):raise OracleWitnessError('WITNESS_FILE_PIN_MISMATCH')
        obj=strict_loads(raw,max_bytes=4194304)
        if row.get('canonical_sha256')!=sha(canonical_bytes(obj)):raise OracleWitnessError('WITNESS_CANONICAL_HASH_MISMATCH')
        parsed[label]=obj;inventory.append({'file':label,'sha256':sha(raw),'size_bytes':len(raw)})
    allowed={canonical_bytes(a) for a in assertions}
    for w in witnesses:
        if canonical_bytes(w) not in allowed:raise OracleWitnessError('WITNESS_NOT_IN_ORACLE')
        pointer=w['pointer']
        if not pointer.startswith('/') or len(pointer)>4096 or len(pointer.split('/'))>65 or re.search(r'~(?![01])',pointer):raise OracleWitnessError('UNSUPPORTED_OBJECT_POINTER')
        obj=parsed[w['file']]
        for part in pointer[1:].split('/'):
            key=part.replace('~1','/').replace('~0','~')
            if not isinstance(obj,dict) or key not in obj:raise OracleWitnessError('WITNESS_POINTER_MISSING')
            obj=obj[key]
        if type(obj) is not type(w['equals']) or obj!=w['equals']:raise OracleWitnessError('WITNESS_VALUE_MISMATCH')
    return {'status':'ORACLE_NEGATIVE_WITNESSES_VERIFIED','scope':'PINNED_OBJECT_POINTER_ASSERTIONS_ONLY','witnesses_checked':len(witnesses),'files_checked':len(files),'bytes_checked':total,'inventory':inventory,'record_sha256':record['record_sha256'],'record_content_ref':{'sha256':sha(record_raw),'size_bytes':str(len(record_raw))},'oracle_content_ref':{'sha256':sha(oracle_raw),'size_bytes':str(len(oracle_raw))},'scientific_acceptance':'NOT_GRANTED','execution_authorized':False,'dependency_closure_verified':False}
