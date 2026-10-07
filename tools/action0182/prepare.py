"""Pin saved stream evidence and admitted G1 input projections; no full replay."""
from pathlib import Path
import gzip,hashlib,json,shutil,zipfile
B=Path(__file__).resolve().parent;R=B.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
(B/'inputs').mkdir(exist_ok=True)
paths={'SOURCE_PINS.json':R/'g_repair0177/SOURCE_PINS.json','TRANSPORT.json':R/'g_repair0177/TRANSPORT.json','REPAIR_VALIDATION.json':R/'g_repair0177/VALIDATION.json','REPAIRED_STREAM_SUMMARY.json':R/'g_repair0177/inputs/reaudit/evidence/REPAIRED_STREAM_SUMMARY.json','REAUDIT_RESULT.json':R/'g_repair0177/inputs/reaudit/RESULT.json','S2_UNLOCK_CERTIFICATE.json':R/'g_repair0177/inputs/reaudit/S2_UNLOCK_CERTIFICATE.json','G1_SCOPED_ADMISSION.json':R/'g_public_integrate0181/SCOPED_ADMISSION.json','CATALOG_0150.json':R/'g_public_integrate0181/CATALOG_0150.json','G1_INTEGRATION_RESULT.json':R/'g_public_integrate0181/NATIVE_RESULT.json','G1_COLD_REUSE.json':R/'g_public_integrate0181/COLD_REUSE_RESULT.json','G2_S0_INTERFACE_POPULATION.json':R/'g_public_register0179/inputs/G2_S0_INTERFACE_POPULATION.json','POST_RESERVATION_CONTINUATION_HASHES.json':R/'g_public_register0179/inputs/POST_RESERVATION_CONTINUATION_HASHES.json'}
for n,p in paths.items():shutil.copy2(p,B/'inputs'/n)
pin=json.loads((B/'inputs/SOURCE_PINS.json').read_text())['reaudit'];p=R/pin['source_path'];assert sha(p.read_bytes())==pin['sha256'] and p.stat().st_size==pin['bytes']
preview={};members={}
with zipfile.ZipFile(p) as z:
 for name,n in [('records','evidence/G2_S1_REPAIRED_D4_Q2_PAIR_CONNECTION_RECORDS.jsonl.gz'),('audits','evidence/G2_S1_REPAIRED_D4_Q2_AUDIT_SIGNATURES.jsonl.gz')]:
  with gzip.GzipFile(fileobj=z.open(n)) as f:row=f.readline()
  preview[name]={'source_ordinal':0,'raw_line_sha256':sha(row),'raw_line_bytes':len(row),'payload':json.loads(row)}
  members[name]={'archive_member':n,'gzip_bytes':z.getinfo(n).file_size}
dump(B/'FIRST_ROW_BINDING_PREVIEW.json',preview)
dump(B/'SOURCE_BINDINGS.json',{'archive_sha256':pin['sha256'],'archive_bytes':pin['bytes'],'archive_hash_freshly_verified':True,'members':members,'prior_full_stream_verification':'inputs/REPAIR_VALIDATION.json: action0177;580351 records+580351 paired signatures;not rerun here','whole_archive_library_file_id':'libfile_31daf44b95d88191a02cebcc4aa03078','Drive_transport':'inputs/TRANSPORT.json'})
dump(B/'INPUT_PINS.json',{n:{'sha256':sha((B/'inputs'/n).read_bytes()),'bytes':(B/'inputs'/n).stat().st_size} for n in paths})
