"""Freeze depths 87–92 from verified bootstrap 86; unchanged historical science."""
from pathlib import Path
import json,hashlib,gzip
B=Path(__file__).resolve().parent;P=B.parent/'continuation0226'
assert json.loads((P/'AUDIT.json').read_text())['status']=='PASS_COLD_HISTORICAL_DEPTH86_CONTINUATION'
r=json.loads((P/'SAVE_RECEIPT.json').read_text());assert r['raw_drive_readback_verified'] and r['library_save_completed']
meta=json.loads((P/'BOOTSTRAP86_META.json').read_text());raw=Path(meta['path']).read_bytes();assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256']
craw=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(craw)==raw;gz=B/'BOOTSTRAP86_HISTORICAL.json.gz';gz.write_bytes(craw)
for f in (B/'project/historical').glob('*.py'):assert f.read_bytes()==(P/'project/historical'/f.name).read_bytes()
p=B/'project/handler.py';s=p.read_text().replace('depths81–86','depths87–92').replace('bootstrap80','bootstrap86').replace('range(81,87)','range(87,93)').replace('depth86','depth92').replace('DEPTH86','DEPTH92').replace("'completed_depth':86","'completed_depth':92");p.write_text(s)
p=B/'project/worker.py';s=p.read_text().replace('level==86','level==92').replace('depth86','depth92').replace('DEPTH86','DEPTH92');p.write_text(s)
spec=json.loads((P/'SPEC.json').read_text());spec['job_id']='MASTER.G1.HISTORICAL.CONTINUATION.0227';spec['project_source']=str(B/'project');spec['question']['stage_id']=spec['job_id'];spec['question']['description']='Historical bounded87–92 continuation with archived92 exact root skin capacity gate';spec['question']['outcomes']=['PASS_HISTORICAL_DEPTH92_CONTINUATION'];c=spec['execution']['parameters'];c.update(start_depth=87,stop_depth=92,bootstrap_raw_sha256=meta['sha256'])
spec['output_contract']['preservation'].update(max_state_bytes=4294967296,max_pending_commits=16)
for x in spec['inputs']:
 if x['logical_name']=='bootstrap':x.update(path=str(gz),sha256=hashlib.sha256(craw).hexdigest())
 if x['logical_name']=='anchor':x.update(path=str(B/'HISTORICAL_DEPTH92_SOURCE_INPUT.json'),sha256='93abe894168978b43bba0d1ebf7befde4b1cc756b9c6500f9e8a1a0423c8619f')
 c['bindings'][x['logical_name']]=x['sha256'];assert hashlib.file_digest(Path(x['path']).open('rb'),'sha256').hexdigest()==x['sha256']
for x in spec['output_contract']['result_checks']:
 if x['pointer']=='/outcome':x['equals']='PASS_HISTORICAL_DEPTH92_CONTINUATION'
 if x['pointer']=='/completed_depth':x['equals']=92
 if x['pointer']=='/historical_depth86_anchor':x['pointer']='/historical_depth92_anchor'
(B/'SPEC.json').write_text(json.dumps(spec,indent=2))
a=json.loads((B/'SOURCE_ATTESTATION.json').read_text());a['continuation_scope']={'start':87,'stop':92,'raw_bootstrap_sha256':meta['sha256'],'transport_sha256':hashlib.sha256(craw).hexdigest(),'transport_bytes':len(craw),'historical_helpers_byte_identical_to0226':True,'candidate_generation_before_capture':0};(B/'SOURCE_ATTESTATION.json').write_text(json.dumps(a,indent=2));(B/'BOOTSTRAP86_TRANSPORT.json').write_text(json.dumps({'raw':meta,'compressed_path':str(gz),'compressed_bytes':len(craw),'compressed_sha256':hashlib.sha256(craw).hexdigest(),'lossless_exact_roundtrip':True},indent=2));print(json.dumps(a['continuation_scope']))
