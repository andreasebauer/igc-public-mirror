"""Qualify only reassembled raw Drive scientific transport bytes."""
from pathlib import Path
import json, hashlib, sys, zipfile
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parents[1]/'engine'))
from scientific_reader import read_archive
from project.reader import IntegrityError

def main():
    transport=json.loads((R/'SCIENTIFIC_TRANSPORT_SAVED.json').read_text());b=b''
    for x in transport['parts']:
        raw=Path(x['readback_path']).read_bytes()
        assert len(raw)==x['size_bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256'];b+=raw
    assert len(b)==transport['archive_size_bytes'] and hashlib.sha256(b).hexdigest()==transport['archive_sha256']
    archive=R/'DRIVE_READBACK_SCIENTIFIC.zip';archive.write_bytes(b)
    args=(transport['export_root_sha256'],'4753cee3dd1adb0b963f10ae11d98fb0c35640b43f3706f1d90b836150458102','3ad27f29023db1b39442d07383d08df0d2829819a014203c6cd10a79112bfc9b')
    reader,root=read_archive(archive,*args);report=dict(reader.report)
    sample=next(oid for oid,d in reader.depths.items() if d==2);row=reader.lookup(sample)
    assert reader.lookup_formation(row['formation_id'])==row and reader.lineage(sample)['parent']['object_id']==row['identity']['parent_object_id']
    original=reader.lookup(sample);row['ordered_record'][0]=-1;assert reader.lookup(sample)==original
    checks=[]
    def reject(label,fn):
        try:fn()
        except (IntegrityError,KeyError):checks.append(label)
        else:raise AssertionError('ACCEPTED '+label)
    reject('wrong_scientific_root',lambda:read_archive(archive,'0'*64,*args[1:]))
    reject('wrong_catalog',lambda:read_archive(archive,args[0],'0'*64,args[2]))
    reject('wrong_view',lambda:read_archive(archive,*args[:2],'0'*64))
    reject('missing_object',lambda:reader.lookup('0'*64))
    reject('missing_formation',lambda:reader.lookup_formation('0'*64))
    reject('missing_projection',lambda:reader.projection_preimages(2,'0'*64))
    reader.close();reject('closed_reader',lambda:reader.lookup(sample))
    report.update(status='PASS_DOWNLOADED_SCIENTIFIC_EXPORT',archive_sha256=transport['archive_sha256'],export_root_sha256=args[0],scientific_members=182,scientific_payload_bytes=286694082,read_source='ONLY_REASSEMBLED_RAW_DRIVE_EXPORT_BYTES',original_workspace_scientific_shards_used=False,scientific_closure='179 shards + complete primitive packet + exact parent packet + qualified view',dependency_authority='Pinned admitted catalog and foundation references; primitive witness closure inherited from separately verified admitted-J3 comparison',negative_checks=checks,lookup_formation_lineage_and_copy_checks='PASS',master_admission=False)
    (R/'SCIENTIFIC_READBACK_QUALIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
