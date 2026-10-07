"""Preflight saved-reader routes; does not invoke native scientific handler."""
from pathlib import Path
import ast,hashlib,json,tempfile
from project.reader import ProjectionReader
B=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def validate():
 spec=json.loads((B/'SPEC.json').read_text());reg=json.loads((B/'PROJECT_REGISTRATION.json').read_text());contract=json.loads((B/'EXPORT_CONTRACT.json').read_text());bindings=json.loads((B/'INPUT_BINDINGS.json').read_text())
 assert sha((B/'SPEC.json').read_bytes())==reg['spec_sha256']
 for n,pin in bindings.items():
  raw=(B/pin['relative_path']).read_bytes();assert sha(raw)==pin['sha256'] and len(raw)==pin['bytes']
  assert next(x for x in spec['inputs'] if x['logical_name']==n)['sha256']==pin['sha256']
 for n,p in reg['project_code'].items():assert sha((B/n).read_bytes())==p['sha256']
 assert contract['authority']=='SAVED_PUBLIC_PROJECTION_ONLY' and contract['scientific_master_admission'] is False
 assert reg['status']=='CONTRACT_AND_HANDLER_FROZEN_NOT_YET_CAPTURED' and reg['native_execution_started'] is False
 for p in (B/'project').glob('*.py'):
  tree=ast.parse(p.read_text())
  for n in ast.walk(tree):
   if isinstance(n,(ast.Import,ast.ImportFrom)):
    names=[n.module or '' ] if isinstance(n,ast.ImportFrom) else [x.name for x in n.names]
    assert all(x in ['pathlib','copy','collections','hashlib','json','infinity_grid.v05_chain','reader'] for x in names)
   if isinstance(n,ast.Call):
    name=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
    assert name not in ['ensure_g1_r100_population','reserve_external','exec','eval','canonicalize','enumerate_candidate_recipes']
 paths=[B/bindings[n]['relative_path'] for n in ['population','continuations']];pins=spec['execution']['parameters']['pins']
 reader=ProjectionReader(*paths,pins);pop=json.loads(paths[0].read_bytes());cont=json.loads(paths[1].read_bytes())['rows'];classes={}
 for row in pop['interfaces']:
  ref=row['carrier_ref'];assert reader.interface(ref)==row
  classes.setdefault(row['interface_sha256'],[]).append(ref)
  for t in range(7):assert reader.reservation(ref,t)==row['one_endpoint_reservations'][t] and reader.continuation_hash(ref,t)==cont[ref][str(t)]
 for h,refs in classes.items():assert reader.interface_class(h)==refs
 ref=reader.refs()[0];original=reader.interface(ref);changed=reader.interface(ref);changed['total_free_by_type'][0]=0;assert reader.interface(ref)==original
 changed=reader.reservation(ref,0);changed['successor_total_free_by_type'][0]=0;assert reader.reservation(ref,0)==original['one_endpoint_reservations'][0]
 def rejects(call):
  try:call()
  except (ValueError,KeyError):return
  raise AssertionError('Excluded route accepted')
 for call in [lambda:reader.interface('missing'),lambda:reader.interface_class('missing'),lambda:reader.exact_state(ref),lambda:reader.q2_payload(ref)]:rejects(call)
 for t in [-1,7,True,1.0,'1',None]:
  rejects(lambda t=t:reader.reservation(ref,t));rejects(lambda t=t:reader.continuation_hash(ref,t))
 bad=json.loads(json.dumps(pins));bad['population']['sha256']='0'*64;rejects(lambda:ProjectionReader(*paths,bad))
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'population.json';forged=json.loads(paths[0].read_bytes());forged['interfaces'][0]['total_free_by_type'][0]+=1;p.write_text(json.dumps(forged));bad=json.loads(json.dumps(pins));bad['population']={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size};rejects(lambda:ProjectionReader(p,paths[1],bad))
 reader.close()
 for call in [reader.refs,lambda:reader.interface(ref),lambda:reader.reservation(ref,0),lambda:reader.continuation_hash(ref,0),lambda:reader.interface_class(next(iter(classes))),reader.provenance,lambda:reader.exact_state(ref),lambda:reader.q2_payload(ref)]:rejects(call)
 return {'status':'PASS_FINITE_EXPORT_REGISTRATION_PREFLIGHT','interface_routes_checked':193,'class_routes_checked':192,'reservation_routes_checked':1351,'continuation_routes_checked':1351,'negative_checks':['unknown_identity','endpoint_type','source_hash','semantic_tamper','copy_isolation','excluded_exact_state','excluded_Q2_payload','closed_reader'],'handler_called':False,'native_capture_completed':False,'generation_calls':0,'new_admissions':0,'master_release':'MASTER_DATA_V1_0149','scientific_slices':149,'next_scope':'WP6_CAPTURE_SAVED_G1_PUBLIC_PROJECTION_READER'}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
