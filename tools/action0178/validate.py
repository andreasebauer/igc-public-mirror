"""Saved byte validation and scope gate. No scientific producer imports."""
from pathlib import Path
import ast,hashlib,io,json,zipfile

B=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
EXPECTED={
 'V0301':'0a8edd002a967750d996b0424937d0426b55d0445c865034b8eb471c2ce9ec84',
 'V0302':'b2522a07fa6a92210211d1af3b833d4efdbe64a4f1b768822d69c8887d8aef57'}
STREAM='9023c8d4a53ff6bba87a5b6954bac8d37008257aaf5214c67c0c6ba8ddf28049'

def scan(z,label,summary):
 summary['zip_count']+=1
 for n in z.namelist():
  if n.endswith('.zip'):
   scan(zipfile.ZipFile(io.BytesIO(z.read(n))),label+'!'+n,summary)
  elif n.endswith('.json'):
   summary['json_count']+=1
   try:d=json.loads(z.read(n))
   except (json.JSONDecodeError,UnicodeDecodeError):
    summary['unparseable_json_members'].append(label+'!'+n);continue
   if isinstance(d,dict) and 'nodes' in d and 'roots' in d:
    summary['dag_candidates'].append(label+'!'+n)
   if isinstance(d,dict) and 'interfaces' in d:
    summary['interface_population_members'].append(label+'!'+n)

def validate():
 for n,d in json.loads((B/'INPUT_PINS.json').read_text()).items():
  b=(B/'inputs'/n).read_bytes();assert sha(b)==d['sha256'] and len(b)==d['bytes']
 sources={};summ={'zip_count':0,'json_count':0,'dag_candidates':[],'interface_population_members':[],'unparseable_json_members':[]}
 for tag,want in EXPECTED.items():
  p=next((B/'sources').glob('*'+tag+'*.zip'))
  assert sha(p.read_bytes())==want
  assert Path(str(p)+'.sha256.txt').read_text().split()[0]==want
  with zipfile.ZipFile(p) as z:
   root='' if tag=='V0301' else 'g2_s4_v0302_closeout_2026-09-03/'
   mn='final_v0301/MANIFEST.json' if tag=='V0301' else root+'MANIFEST.json'
   m=json.loads(z.read(mn));checked=0;missing=[]
   for x in m['files']:
    if root+x['path'] not in z.namelist():
     missing.append(x);continue
    b=z.read(root+x['path']);assert sha(b)==x['sha256'] and len(b)==x['bytes'];checked+=1
   if tag=='V0301':
    assert {x['path'] for x in missing}=={'prerequisites/s1_reconstructed/G2_S1_REPAIRED_D4_Q2_PAIR_CONNECTION_RECORDS.jsonl.gz','prerequisites/s1_reconstructed/G2_S1_REPAIRED_D4_Q2_AUDIT_SIGNATURES.jsonl.gz'}
   else:assert not missing
   scan(z,p.name,summ)
   script=z.read(root+'scripts/fast_reconstruct_s1_v0301.py').decode()
   calls=sorted({n.func.attr for n in ast.walk(ast.parse(script)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)})
   assert 'ensure_g1_r100_population' in calls and 'reserve_external' in calls
   sources[tag]={'sha256':want,'bytes':p.stat().st_size,'manifest_files_verified':checked,'manifest_references_absent_from_capsule':missing,'script_is_stored_read':False,'unsafe_for_no_regeneration_read':['ensure_g1_r100_population','reserve_external']}
   if tag=='V0301':
    reconstruction=json.loads(z.read('prerequisites/s1_reconstructed/RECONSTRUCTION_RESULT.json'))
    assert reconstruction['status']=='FAIL_CLOSED_IDENTITY_MISMATCH'
    assert reconstruction['repaired_record_stream_sha256']==STREAM
    assert reconstruction['repaired_file_sha256']!=reconstruction['expected_repaired_file_sha256']
    eq=json.loads(z.read('prerequisites/s1_reconstructed/RECONSTRUCTION_SCIENCE_EQUIVALENCE.json'))
   else:
    eq=json.loads(z.read(root+'science/prerequisites/RECONSTRUCTION_SCIENCE_EQUIVALENCE.json'))
    reval=json.loads(z.read(root+'science/prerequisites/S2_REVALIDATION.json'))
    assert reval['old']==reval['new'] and all(reval['field_matches'].values())
    assert reval['new']['record_count']==580351 and reval['new']['s2_quotient_class_count']==576735
    assert reval['new']['input_repaired_record_stream_sha256']==STREAM
    sources[tag]['S2_saved_revalidation_science_sha256']=reval['new_function_science_sha256']
   assert eq['physical_gzip_container_match'] is False
   assert eq['repaired_decompressed_content_exact'] is True and eq['repaired_record_stream_sha256']==STREAM
   assert eq['authorizes']=='G2:S4_RERUN_INPUT_USE_ONLY' and eq['g2_graduated'] is False
   pop=z.read(root+'prerequisites/G2_S0_INTERFACE_POPULATION.json' if tag=='V0301' else root+'science/prerequisites/G2_S0_INTERFACE_POPULATION.json')
   assert pop==(B/'inputs/G2_S0_INTERFACE_POPULATION.json').read_bytes()
   sources[tag]['same_saved_S0_population']=True
 assert not summ['dag_candidates']
 prior=json.loads((B/'inputs/REPAIR_VALIDATION.json').read_text())
 assert prior['stream_validation']['record_rows_checked']==580351
 assert prior['stream_validation']['record_stream_sha256']==STREAM
 gate=json.loads((B/'LOCAL_SUFFICIENCY_GATE.json').read_text())
 pop=json.loads((B/'inputs/G2_S0_INTERFACE_POPULATION.json').read_text())
 rows=pop['interfaces'];assert len(rows)==193 and len({r['interface_sha256'] for r in rows})==192
 assert sum(len(r['one_endpoint_reservations']) for r in rows)==1351
 cont=json.loads((B/'inputs/POST_RESERVATION_CONTINUATION_HASHES.json').read_text())['rows']
 assert set(cont)=={r['carrier_ref'] for r in rows} and sum(map(len,cont.values()))==1351
 assert gate['exact_carrier_admission']['status']=='BLOCKED_MISSING_SAVED_PARENT_DAG_AND_WITNESSES'
 assert gate['saved_projection_export']['status']=='PASS_FOR_REGISTERED_BYTE_ONLY_EXPORT_PREPARATION'
 assert gate['new_scientific_admission_authorized'] is False
 return {'status':'PASS_SOURCE_BINDINGS_AND_SCOPE_GATE','sources':sources,'bounded_source_scan':summ,'historical_reconstruction_container_failure_retained':True,'later_content_equivalence_retained':True,'saved_S2_quotient_classes':576735,'S2_revalidation_is_fresh_replay':False,'master_release':'MASTER_DATA_V1_0149','scientific_slices':149,'generation_calls':0,'new_admissions':0,'exact_G1_DAG_recovered':False,'Q2_payload_recovered':False,'next_scope':'WP6_REGISTERED_SAVED_G1_PUBLIC_PROJECTION_EXPORT'}

if __name__=='__main__':print(json.dumps(validate(),indent=2))
