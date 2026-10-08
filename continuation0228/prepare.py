"""Freeze depths 93–98 from verified bootstrap 92; unchanged historical science."""
from pathlib import Path
import json,hashlib,gzip
B=Path(__file__).resolve().parent;P=B.parent/'continuation0227'
assert json.loads((P/'AUDIT.json').read_text())['status']=='PASS_COLD_HISTORICAL_DEPTH92_CONTINUATION'
r=json.loads((P/'SAVE_RECEIPT.json').read_text());assert r['raw_drive_readback_verified'] and r['library_save_completed']
meta=json.loads((P/'BOOTSTRAP92_META.json').read_text());raw=Path(meta['path']).read_bytes();assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256']
craw=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(craw)==raw;gz=B/'BOOTSTRAP92_HISTORICAL.json.gz';gz.write_bytes(craw)
for f in (B/'project/historical').glob('*.py'):assert f.read_bytes()==(P/'project/historical'/f.name).read_bytes()
spec=json.loads((P/'SPEC.json').read_text());spec['job_id']='MASTER.G1.HISTORICAL.CONTINUATION.0228';spec['project_source']=str(B/'project');spec['question']['stage_id']=spec['job_id'];spec['question']['description']='Historical bounded93–98 continuation with archived98 exact root skin capacity gate';spec['question']['outcomes']=['PASS_HISTORICAL_DEPTH98_CONTINUATION'];c=spec['execution']['parameters'];c.update(start_depth=93,stop_depth=98,bootstrap_raw_sha256=meta['sha256'])
spec['output_contract']['preservation'].update(max_state_bytes=4294967296,max_pending_commits=16)
for x in spec['inputs']:
 if x['logical_name']=='bootstrap':x.update(path=str(gz),sha256=hashlib.sha256(craw).hexdigest())
 if x['logical_name']=='anchor':x.update(path=str(B/'HISTORICAL_DEPTH98_SOURCE_INPUT.json'),sha256='0c6f8b13472443250d8fcd9f663ac29e230284ec99d3ff45453f6494dab3ea5d')
 if x['logical_name'] not in ('bootstrap','anchor'):x['path']=str(Path('/tmp/ig_verified0228')/(x['sha256']+'.bin'))
 c['bindings'][x['logical_name']]=x['sha256'];assert hashlib.file_digest(Path(x['path']).open('rb'),'sha256').hexdigest()==x['sha256']
for x in spec['output_contract']['result_checks']:
 if x['pointer']=='/outcome':x['equals']='PASS_HISTORICAL_DEPTH98_CONTINUATION'
 if x['pointer']=='/completed_depth':x['equals']=98
 if x['pointer']=='/historical_depth92_anchor':x['pointer']='/historical_depth98_anchor'
for x in spec['environment']['artifacts']:x['path']=str(Path('/tmp/ig_verified0228')/(x['sha256']+'.bin'))
(B/'SPEC.json').write_text(json.dumps(spec,indent=2))
a=json.loads((B/'SOURCE_ATTESTATION.json').read_text());a['continuation_scope']={'start':93,'stop':98,'raw_bootstrap_sha256':meta['sha256'],'transport_sha256':hashlib.sha256(craw).hexdigest(),'transport_bytes':len(craw),'historical_helpers_byte_identical_to0227':True,'candidate_generation_before_capture':0};(B/'SOURCE_ATTESTATION.json').write_text(json.dumps(a,indent=2));(B/'BOOTSTRAP92_TRANSPORT.json').write_text(json.dumps({'raw':meta,'compressed_path':str(gz),'compressed_bytes':len(craw),'compressed_sha256':hashlib.sha256(craw).hexdigest(),'lossless_exact_roundtrip':True},indent=2));print(json.dumps(a['continuation_scope']))
