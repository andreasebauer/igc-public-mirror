#!/usr/bin/env python3
"""Bounded synthetic metadata scaling; no remote receipt or science authority."""
from pathlib import Path
import argparse
import collections
import json
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infinity_grid import preservation as pr,submission as sub
from infinity_grid.canon import canonical_sha256,write_json_atomic
from infinity_grid.replay_reference_data import ReplayReferenceDataStore
from tests.test_step2_bounded_storage import rows


def metadata_sample(root,count):
    shared=pr._put(root,b'SYNTHETIC_ENGINEERING_BYTES');previous=None
    for i in range(count):
        row={'schema_id':pr.SCHEMA,'previous':previous,'objects':[shared],'created_unix':time.time(),
            'capture_id':'synthetic_step2_metadata_no_remote_authority','source':shared,'state':shared,
            'project':{'base':shared,'delta':shared},'base_objects':[],'reason':'SYNTHETIC_'+str(i)}
        raw=sub._json_bytes(row);saved=pr._put(root,raw);digest=saved['sha256']
        path=pr._root(root)/'commits'/(digest+'.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        for obj in pr._checkpoint_obligations(digest,row,raw):
            receipt=dict(obj,schema_id='IG_DECODER_EXPLICIT_SAVE_RECEIPT_V2',provider='google_drive',
                drive_file_id='synthetic_fixture_not_saved',raw_readback_verified=True)
            receipt['receipt_sha256']=canonical_sha256(receipt)
            write_json_atomic(pr._root(root)/'receipts'/obj['sha256']/(receipt['receipt_sha256']+'.json'),receipt)
        previous=digest
    write_json_atomic(pr._root(root)/'CURRENT.json',{'sha256':previous})
    start=time.monotonic();state=pr._status_from_inventory(root,*pr._inventory(root),verified_count=0)
    assert not state['pending_objects']
    expected=len(list((pr._root(root)/'receipts').rglob('*.json')))
    assert state['receipt_files_read']==expected
    return {'checkpoints':count,'receipt_reads':state['receipt_files_read'],'seconds':time.monotonic()-start}


def ledger_sample(root,count):
    store=ReplayReferenceDataStore.initialize(root);items=rows(count);reads=collections.Counter();original=Path.read_text
    def read(path,*a,**kw):reads[path.parent.name]+=1;return original(path,*a,**kw)
    Path.read_text=read;start=time.monotonic()
    try:store.put_many(items)
    finally:Path.read_text=original
    elapsed=time.monotonic()-start;store.verify()
    return {'records':count,'seconds':elapsed,'read_counts':dict(reads),'manifest_sha256':store.manifest['manifest_sha256']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='step2-bounded-metadata-') as temp:
        root=Path(temp)
        meta=[metadata_sample(root/str(n),n) for n in (128,256)]
        ledger=[ledger_sample(root/('ledger'+str(n)),n) for n in (200,400,800)]
    assert meta[1]['receipt_reads']/meta[0]['receipt_reads']<=2.1
    assert all(b['read_counts']['records']/a['read_counts']['records']<=2.2 for a,b in zip(ledger,ledger[1:]))
    result={'schema_id':'IG_STEP2_METADATA_SCALING_V1','status':'PASS','scope':'SYNTHETIC_METADATA_ONLY; NO_ACTUAL_SAVE_ACKNOWLEDGMENT',
            'receipt_status':meta,'ledger_batches':ledger,'old_single_row_writer_retained':True,
            'bulk_science_storage':'SQLite/shards; provenance ledger batching is bounded metadata'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
