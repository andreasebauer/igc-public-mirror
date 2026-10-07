"""Verify immutable release and handoff pins without invoking science."""
from pathlib import Path
import json,hashlib,zipfile
B=Path(__file__).resolve().parent
def read(n):return json.loads((B/n).read_text())
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify():
 m=read('MANIFEST.json')
 for x in m['files']:
  p=B/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],x['path']
 cat,old,ad=read('CATALOG_0151.json'),read('CATALOG_0150.json'),read('SCOPED_ADMISSION.json')
 assert cat['release_id']=='MASTER_DATA_V1_0151' and len(cat['slices'])==151
 assert cat['slices'][:-1]==old['slices'] and cat['previous_catalog_sha256']==sha(B/'CATALOG_0150.json')
 assert cat['slices'][-1]['scoped_admission_sha256']==sha(B/'SCOPED_ADMISSION.json')
 assert cat['slices'][-1]['scope']==ad['scope'] and cat['slices'][-1]['files']==ad['files']
 assert ad['generation_calls']==0 and ad['scope']['Q2_payload_available'] is False
 assert ad['scope']['exact_G1_parent_DAG_available'] is False and ad['scope']['fresh_realization'] is False
 native,cold=read('NATIVE_RESULT.json'),read('COLD_REUSE_RESULT.json')
 assert native['status']==cold['status']=='COMPLETED' and native['evidence_status']==cold['evidence_status']=='VERIFIED'
 assert cold['reused'] and native['result']==cold['result']
 for k in ['completion_sha256','result_sha256','source_sha256']:assert native[k]==cold[k]
 result=native['result']
 for k,v in {'outcome':'PASS','master_release':'MASTER_DATA_V1_0151','scientific_slices':151,
             'record_rows_checked':580351,'outcome_classes_checked':576785,'collision_classes_checked':3554,
             'stored_observer_split_classes':0,'generation_calls':0,'prior_slices_unchanged':150,
             'prior_reader_bytes_unchanged':True,'saved_reader_bytes_unchanged':True,
             'published_reader_qualified':True,'scoped_admission_verified':True}.items():assert result[k]==v,k
 p=read('FINAL_PRESERVATION.json');assert p['status']=='PRESERVED' and p['pending_bytes']==0 and not p['pending_objects']
 assert sha(B/'FINAL_CHECKPOINT_SLIM.zip')==read('EXPORT.json')['sha256']
 for p in B.glob('*.zip'):
  with zipfile.ZipFile(p) as z:assert z.testzip() is None,p.name
 reg=read('PROJECT_REGISTRATION.json')
 for n,h in reg['project_source'].items():assert sha(B/'project'/n)==h,n
 assert sum(n.startswith('verified_candidate/') for n in reg['project_source'])==107
 print('PASS_MASTER151_RELEASE_AND_HANDOFF: native qualification, cold reuse, preservation, prior150 unchanged')
if __name__=='__main__':verify()
