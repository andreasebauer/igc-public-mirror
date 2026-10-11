"""Real saved bytes, injected ID/ACK publication faults, exact cold closure."""
from pathlib import Path
import sys
import traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'candidate/source'))
sys.path.insert(0,str(HERE/'candidate/operational'))
from p0_control import SaveJournal,read,write,require,sha,confirm_checkpoint
from infinity_grid import preservation as pr,preservation_batch as batch,portable_registry as reg
from infinity_grid import submission as sub

def main():
    pending=read(HERE/'results/LIVE_PENDING_OBJECTS.json')
    work=Path(pending['workspace']);root=Path(pending['root'])
    readbacks=read(HERE/'results/PROVIDER_EXACT_READBACKS.json')
    uploads=read(HERE/'results/PROVIDER_UPLOAD_RESPONSES.json')
    by_id={x['drive_file_id']:x for x in readbacks}
    by_name={x['provider_name']:x['id'] for x in uploads}
    matrix=read(HERE/'results/NATIVE_MATRIX.json');rows=matrix['rows']
    journal=HERE/'results/LIVE_SAVE_JOURNAL.json'
    def forbidden_upload(*args):raise RuntimeError('RESTART_DUPLICATE_UPLOAD_REFUSED')
    def reconcile(name):return [by_name[name]]
    calls=0
    def fault_after_id(provider_id):
        nonlocal calls
        calls+=1
        if calls==2:raise InterruptedError('AFTER_DURABLE_ID_BEFORE_FINAL_READBACK_VERIFICATION')
        return Path(by_id[provider_id]['path']).read_bytes()
    first=pending['native_plan']['physical_objects'][0]['path']
    try:SaveJournal(journal,forbidden_upload,fault_after_id,reconcile).save(first)
    except InterruptedError:pass
    else:raise RuntimeError('EXPECTED_ID_BOUNDARY_FAULT_MISSING')
    require(any(x['phase']=='ID_RECORDED' for x in read(journal)['objects'].values()),'ID_NOT_DURABLE')
    def download(provider_id):return Path(by_id[provider_id]['path']).read_bytes()
    service=SaveJournal(journal,forbidden_upload,download,reconcile)
    saved=[service.save(o['path']) for o in pending['native_plan']['physical_objects']]
    require(all(x['phase']=='EXACT' for x in read(journal)['objects'].values()),'JOURNAL_NOT_EXACT')
    rows.append({'operation':'REAL_PROVIDER_SAVE_ID_INTERRUPTION_AND_MISSING_ONLY_RETRY','status':'PASS',
                 'physical_objects':len(saved),'actual_uploads':len(uploads),'retry_uploads':0,
                 'raw_verified_bytes':sum(x['bytes'] for x in saved),'fault_after_durable_id':True,
                 'reconciliation_basis':'Canonical IDs from actual successful upload responses and exact raw bytes'})
    write(HERE/'results/LIVE_SAVE_RECEIPTS.json',saved)
    write(HERE/'results/NATIVE_MATRIX.json',dict(matrix,status='RUNNING',rows=rows))
    request={'schema_id':batch.SCHEMA,'capture_id':pending['native_plan']['capture_id'],
             'obligations':pending['native_plan']['obligations'],
             'readbacks':[{'sha256':x['sha256'],'kind':'RAW','path':x['path'],
                           'drive_file_id':x['drive_file_id']} for x in readbacks]}
    ack=HERE/'results/ACK_BATCH.json';write(ack,request)
    original=batch._publish;injected=False
    def publish_then_interrupt(path,row):
        nonlocal injected
        original(path,row)
        if not injected and 'receipts' in Path(path).parts:
            injected=True
            raise InterruptedError('AFTER_FIRST_NATIVE_ROLE_RECEIPT')
    batch._publish=publish_then_interrupt
    try:
        try:pr.confirm_batch(work,ack)
        except InterruptedError:pass
        else:raise RuntimeError('EXPECTED_NATIVE_RECEIPT_FAULT_MISSING')
    finally:batch._publish=original
    receipts=lambda:{str(p.relative_to(pr._root(work))):sha(p)
                     for p in (pr._root(work)/'receipts').rglob('*.json')}
    first_receipts=receipts();require(bool(first_receipts),'FIRST_NATIVE_RECEIPT_NOT_DURABLE')
    pr.confirm_batch(work,ack)
    require(all(receipts()[k]==v for k,v in first_receipts.items()),'NATIVE_RETRY_REWROTE_RECEIPT')
    all_receipts=receipts();pr.confirm_batch(work,ack)
    require(receipts()==all_receipts,'SECOND_ACK_REWROTE_RECEIPTS')
    gate=HERE/'results/CHECKPOINT_PRESERVED.json'
    confirm_checkpoint(work,ack,gate,pr);first_gate=gate.read_bytes()
    confirm_checkpoint(work,ack,gate,pr)
    require(gate.read_bytes()==first_gate,'CONFIRMATION_RETRY_CHANGED_GATE')
    state=pr.status(work)
    require(state['status']=='PRESERVED' and state['pending_bytes']==0
            and not state['pending_objects'] and not state['pending_checkpoints'],'NATIVE_NOT_PRESERVED')
    rows.append({'operation':'INTERRUPTED_NATIVE_ACK_DRAIN_CONFIRMATION_RETRY','status':'PASS',
                 'pending_roles':0,'pending_bytes':0,'native_status':'PRESERVED',
                 'role_receipts':len(all_receipts),'retained_partial_receipts':len(first_receipts),
                 'existing_receipt_bytes_unchanged':True,'confirmation_gate_bytes_unchanged':True})
    write(HERE/'results/NATIVE_MATRIX.json',dict(matrix,status='RUNNING',rows=rows))
    a=root/'exact-export-a.zip';b=root/'exact-export-b.zip'
    pr.export_checkpoint(work,a);pr.export_checkpoint(work,b)
    require(a.read_bytes()==b.read_bytes(),'EXPORT_RETRY_BYTES_CHANGED')
    slim=HERE/'results/NATIVE_CHECKPOINT_SLIM.zip';pr.export_checkpoint(work,slim,slim=True)
    cold=root/'cold-after-ack';restored=pr.restore_checkpoint(a,cold,sha(a))
    require(reg.snapshot(cold/'coordination')==reg.snapshot(work.parent.parent/'coordination'),
            'COLD_PROJECT_BYTES_CHANGED')
    capsule=reg.verify_capsule(cold/'coordination',pending['capsule_sha256'])
    rows.append({'operation':'BYTE_EXACT_EXPORT_RETRY_AND_INDEPENDENT_COLD_CLOSURE','status':'PASS',
                 'export_sha256':sha(a),'export_bytes':a.stat().st_size,
                 'slim_export_sha256':sha(slim),'slim_export_bytes':slim.stat().st_size,
                 'checkpoint_sha256':restored['checkpoint_sha256'],
                 'historical_completion_sha256':capsule['completion_sha256'],
                 'scope':'Independent cold directory on same host; not independent-host RC'})
    write(HERE/'results/NATIVE_MATRIX.json',dict(matrix,status='BOUNDED_NATIVE_CLOSURE_GATE_PASS',rows=rows,
              scientific_dispatches=0,full_native_fault_matrix_qualified=False,
              remaining=['Incomplete paused scientific execution','Identity impact/migration',
                         'Independent-host full RC','Provider-wide outage/unknown GitHub and Library outcomes']))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        old=read(HERE/'results/NATIVE_MATRIX.json')
        write(HERE/'results/NATIVE_MATRIX.json',dict(old,status='STOPPED_AT_FIRST_NATIVE_MISMATCH',
              error=str(exc),traceback=traceback.format_exc(),full_native_fault_matrix_qualified=False))
        raise
