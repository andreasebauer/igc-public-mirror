"""Transport-only packaging of the registered exact scientific selection."""
from pathlib import Path
import hashlib, json, zipfile, io

R = Path(__file__).resolve().parent
def sha(raw): return hashlib.sha256(raw).hexdigest()
def write(name, value): (R/name).write_text(json.dumps(value, indent=2)+'\n')
def main():
    result=json.loads((R/'NATIVE_RESULT.json').read_text())
    assert result['evidence_status']=='VERIFIED' and result['result']['outcome']=='PASS'
    mapping=json.loads((R/'CHECKPOINT_READBACK_MAPPING.json').read_text())
    capture=json.loads((R/'CAPTURE_READBACK_MAPPING.json').read_text())
    spec=json.loads((R/'CAPTURE_SPEC.json').read_text())
    ref=result['result']['export_root'];root=Path(mapping[ref['sha256']]['path']).read_bytes()
    assert sha(root)==ref['sha256'] and len(root)==ref['size_bytes']
    scientific=json.loads(root);raws={}
    inputs={x['logical_name']:x for x in spec['inputs']}
    pieces=[]
    for i in range(spec['execution']['parameters']['saved_contents_part_count']):
        h=inputs['saved_contents_part'+str(i).zfill(2)]['sha256']
        b=Path(capture[h]['path']).read_bytes();assert sha(b)==h;pieces.append(b)
    bundle=b''.join(pieces)
    assert sha(bundle)==spec['execution']['parameters']['saved_contents_bundle_sha256']
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        assert len(z.namelist())==len(set(z.namelist()))==181
        for name in z.namelist():
            assert len(name)==68 and name.endswith('.bin')
            b=z.read(name);assert sha(b)==name[:-4];raws[name[:-4]]=b
    h=inputs['qualified_view']['sha256'];b=Path(capture[h]['path']).read_bytes();assert sha(b)==h;raws[h]=b
    members=scientific['scientific_members']
    assert len(members)==len(raws)==182 and {x['sha256'] for x in members}==set(raws)
    for x in members:assert len(raws[x['sha256']])==x['size_bytes']
    assert sum(map(len,raws.values()))==result['result']['scientific_payload_bytes']
    archive=R/'NODE_NATIVE_ROOTED_DEPTH1_DEPTH2_V1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,b in [('ROOT.json',root)]+[('content/'+h+'.blob',raws[h]) for h in sorted(raws)]:
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,b)
    raw=archive.read_bytes();parts=[]
    for i,offset in enumerate(range(0,len(raw),24*1024*1024)):
        p=R/('NODE_NATIVE_ROOTED_DEPTH1_DEPTH2_V1.part'+str(i).zfill(2));b=raw[offset:offset+24*1024*1024];p.write_bytes(b)
        parts.append({'index':i,'path':str(p),'sha256':sha(b),'size_bytes':len(b)})
    report={'status':'PASS_EXACT_REGISTERED_SCIENTIFIC_PACKAGING','export_root_sha256':ref['sha256'],'archive_sha256':sha(raw),'archive_size_bytes':len(raw),'scientific_members':182,'scientific_payload_bytes':sum(map(len,raws.values())),'all_members_equal_saved_captured_bytes':True,'generator_calls':0,'master_admission':False,'parts':parts}
    write('SCIENTIFIC_PACKAGING.json',report);print(json.dumps(report))
if __name__=='__main__':main()
