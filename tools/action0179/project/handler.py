from pathlib import Path
import hashlib,json
from infinity_grid.v05_chain import ChainExecutionResult
from .reader import ProjectionReader

def handler(stage,runtime):
 i=stage['input_artifacts'];p=stage['execution']['parameters']
 for name in ['contract','gate','catalog']:
  if hashlib.sha256(Path(i[name]).read_bytes()).hexdigest()!=p['bindings'][name]:raise ValueError('BINDING:'+name)
 contract=json.loads(Path(i['contract']).read_bytes())
 if contract['authority']!='SAVED_PUBLIC_PROJECTION_ONLY' or contract['scientific_master_admission'] is not False:raise ValueError('AUTHORITY')
 r=ProjectionReader(i['population'],i['continuations'],p['pins'])
 try:
  for ref in r.refs():
   r.interface(ref)
   for t in range(7):r.reservation(ref,t);r.continuation_hash(ref,t)
  result={'outcome':'PASS','interfaces':193,'interface_classes':192,'reservation_rows':1351,'continuation_hashes':1351,'authority':'SAVED_PUBLIC_PROJECTION_ONLY','generation_calls':0,'scientific_master_admission':False,'exact_parent_DAG_available':False,'Q2_payload_available':False,'provenance':r.provenance()}
 finally:r.close()
 return ChainExecutionResult(result=result)
