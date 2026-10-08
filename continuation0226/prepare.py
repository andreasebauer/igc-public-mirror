"""Freeze depths81–86 with lossless bootstrap transport and unchanged science helpers."""
from pathlib import Path
import json,hashlib,gzip
B=Path(__file__).resolve().parent;P=B.parent/'continuation0225';Q=B.parent/'continuation0224'
assert json.loads((P/'AUDIT.json').read_text())['status']=='PASS_COLD_NATIVE_DEPTH80_COMPLETION_RECOVERY'
meta=json.loads((P/'BOOTSTRAP80_META.json').read_text());raw=Path(meta['path']).read_bytes();assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256']
compressed=gzip.compress(raw,compresslevel=6,mtime=0);assert gzip.decompress(compressed)==raw;gz=B/'BOOTSTRAP80_HISTORICAL.json.gz';gz.write_bytes(compressed)
for f in (B/'project/historical').glob('*.py'):assert f.read_bytes()==(Q/'project/historical'/f.name).read_bytes()
p=B/'project/handler.py';s=p.read_text().replace('import hashlib','import hashlib,gzip').replace('range(75,81)','range(81,87)').replace('depth80','depth86').replace('DEPTH80','DEPTH86').replace("'completed_depth':80", "'completed_depth':86")
s=s.replace(" base={'path':i['bootstrap']", " if hashlib.sha256(gzip.decompress(Path(i['bootstrap']).read_bytes())).hexdigest()!=c['bootstrap_raw_sha256']:raise ValueError('BOOTSTRAP_RAW_IDENTITY')\n base={'path':i['bootstrap']")
s=s.replace('Native bounded historical seed/8–14qualification; no prior replay state reuse.', 'Native bounded depths81–86 from exact verified historical bootstrap80.')
p.write_text(s)
p=B/'project/worker.py';s=p.read_text().replace('import hashlib,json,statistics','import hashlib,json,statistics,gzip').replace('level==80','level==86').replace('depth80','depth86').replace('DEPTH80','DEPTH86').replace(' return json.loads(raw)',' return json.loads(gzip.decompress(raw) if raw[:2]==b"\\x1f\\x8b" else raw)');p.write_text(s)
p=B/'project/partitions.py';s=p.read_text().replace('import json,hashlib','import json,hashlib,gzip').replace(' return json.loads(raw)',' return json.loads(gzip.decompress(raw) if raw[:2]==b"\\x1f\\x8b" else raw)');p.write_text(s)
spec=json.loads((Q/'SPEC.json').read_text());spec['job_id']='MASTER.G1.HISTORICAL.CONTINUATION.0226';spec['project_source']=str(B/'project');spec['question']['stage_id']=spec['job_id'];spec['question']['description']='Historical bounded81–86 continuation with archived86 exact root skin capacity gate';spec['question']['outcomes']=['PASS_HISTORICAL_DEPTH86_CONTINUATION'];c=spec['execution']['parameters'];c.update(start_depth=81,stop_depth=86,task_work_budget_seconds=1800,bootstrap_raw_sha256=meta['sha256']);spec['resources']['workspace_budget_bytes']=4294967296
for x in spec['inputs']:
 if x['logical_name']=='bootstrap':x.update(path=str(gz),sha256=hashlib.sha256(compressed).hexdigest())
 if x['logical_name']=='anchor':x.update(path=str(B/'HISTORICAL_DEPTH86_SOURCE_INPUT.json'),sha256='efee14669805e2116a26bb900e7f259d674da423f10e26d59567eee48db91a14')
 c['bindings'][x['logical_name']]=x['sha256'];assert hashlib.file_digest(Path(x['path']).open('rb'),'sha256').hexdigest()==x['sha256']
for x in spec['output_contract']['result_checks']:
 if x['pointer']=='/outcome':x['equals']='PASS_HISTORICAL_DEPTH86_CONTINUATION'
 if x['pointer']=='/completed_depth':x['equals']=86
 if x['pointer']=='/historical_depth80_anchor':x['pointer']='/historical_depth86_anchor'
(B/'SPEC.json').write_text(json.dumps(spec,indent=2))
for script in ('preflight.py','transfer.py'):
 p=B/script;p.write_text(p.read_text().replace('0224','0226'))
att=json.loads((B/'SOURCE_ATTESTATION.json').read_text());att['continuation_scope']={'start':81,'stop':86,'raw_bootstrap_sha256':meta['sha256'],'transport_sha256':hashlib.sha256(compressed).hexdigest(),'transport_bytes':len(compressed),'historical_helpers_byte_identical_to0224':True,'candidate_generation_before_capture':0};(B/'SOURCE_ATTESTATION.json').write_text(json.dumps(att,indent=2));(B/'BOOTSTRAP80_TRANSPORT.json').write_text(json.dumps({'raw':meta,'compressed_path':str(gz),'compressed_bytes':len(compressed),'compressed_sha256':hashlib.sha256(compressed).hexdigest(),'lossless_exact_roundtrip':True},indent=2));print(json.dumps(att['continuation_scope']))
