"""Finite saved-authority consumption audit. Never construct scientific states."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import hashlib,json,os,zipfile
from infinity_grid.canon import canonical_sha256
from infinity_grid.uplift_g3_s0 import verify_authority, G3S0Error
from infinity_grid.uplift_g3_s1 import verify_s1_authority, G3S1Error

EXPECTED={
 'g2':'2d1b2c870c8367884271de59c0859ce4f2961130bdf0b2b0579d3b1863f9cb46',
 's0':'985f92f110b4cab744becc36cba108a41bb053dabec8f9c99803ed719d5a566a',
 's1':'0a2a0644a31f6f9d3378df65fd6e8a4a970a4a609f280c7c4e454495c8aa922c'}
SCIENCE={'g2':'64d503d6c9680df893618b84c16f45af7592ae124ed6bd758b21e1f83110ff2f','s0':'42832d682c1e0175f6f7c02eff6047f9ba8bf2a165c6017d8f4776166e766199','s1':'0d4ab3493930ac1b0b3b29deaca52496ffacf30484d3742e8b4e54c7d4937e7c'}
def digest(b):return hashlib.sha256(b).hexdigest()
def validate_payload(r,expected):
 p={k:v for k,v in r.items() if k not in ('science_sha256','source_sha256','source_version','registry_sha256','execution_metadata')}
 if canonical_sha256(p)!=r['science_sha256'] or r['science_sha256']!=expected:raise ValueError('saved science content mismatch')
def main():
 inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text()); records={}; bindings=[]
 out=Path('work');out.mkdir(exist_ok=True);sources=out/'saved_records';sources.mkdir()
 for key in ('g2','s0','s1'):
  raw=Path(inputs[key]).read_bytes();assert digest(raw)==EXPECTED[key]
  z=zipfile.ZipFile(inputs[key]);assert z.testzip() is None
  assert len(z.namelist())==len(set(z.namelist())), 'ambiguous duplicate paths'
  pre='closeout/' if key=='g2' else ''
  manifest=json.loads(z.read(pre+'MANIFEST_SHA256.json')); checked=0;missing=[];mismatches=[]
  for n,h in manifest.items():
   # Historical manifest path is relative to closeout directory.
   actual=n if n in z.namelist() else pre+n
   if actual not in z.namelist():missing.append(n);continue
   if digest(z.read(actual))!=h:mismatches.append(n)
   checked+=1
  root=pre+'decoder_source/src/infinity_grid/'
  sm={n[len(root):]:{'sha256':digest(z.read(n)),'size_bytes':len(z.read(n))} for n in sorted(z.namelist()) if n.startswith(root) and not n.endswith('/') and Path(n).name!='_build_meta.json' and '__pycache__' not in Path(n).parts and not n.endswith('.pyc')}
  r=json.loads(z.read(pre+'science/PRIMARY_RESULT.json'));cold=json.loads(z.read(pre+'science/COLD_RESULT.json'))
  validate_payload(r,SCIENCE[key]);validate_payload(cold,SCIENCE[key])
  assert canonical_sha256(sm)==r['source_sha256']==cold['source_sha256']
  (sources/(key+'_PRIMARY_RESULT.json')).write_bytes(z.read(pre+'science/PRIMARY_RESULT.json'))
  (sources/(key+'_COLD_RESULT.json')).write_bytes(z.read(pre+'science/COLD_RESULT.json'))
  records[key]=r
  for n in z.namelist():
   if n.startswith(pre+'frozen/') and n.endswith('.json'):
    (sources/(key+'_'+Path(n).name)).write_bytes(z.read(n))
  bindings.append({'role':key,'archive_sha256':EXPECTED[key],'archive_bytes':len(raw),'source_files':len(sm),'computed_original_source_sha256':canonical_sha256(sm),'source_version':r['source_version'],'science_sha256':r['science_sha256'],'primary_cold_science_match':True,'manifest_entries_checked':checked,'manifest_missing':missing,'manifest_mismatches':mismatches,'manifest_policy':'retain discrepancies; original native source is independently recomputed from archived bytes'})
  if key=='g2':
   cert=json.loads(z.read(pre+'authority/G2_GRADUATION_CERTIFICATE.json'));(sources/'G2_GRADUATION_CERTIFICATE.json').write_bytes(z.read(pre+'authority/G2_GRADUATION_CERTIFICATE.json'))
 assert canonical_sha256({k:v for k,v in cert.items() if k!='science_sha256'})==cert['science_sha256']=='025713aab599dacbcffe397721d781e5854bc4216245a21bfd70f92c34a8a209'
 auth0=verify_authority(graduation_certificate=cert,r0_result=records['g2']);auth1=verify_s1_authority(records['s0'])
 assert auth0==records['s0']['authority'];assert auth1==records['s1']['authority']
 rows=records['s0']['challenge_corpus']['records'];assert len(rows)==6
 refs={r['g2_carrier_ref'] for r in rows};assert len(refs)==6
 assert Counter(r['hidden_challenge_diagnostic']['topology_canon'] for r in rows)=={'((())())':4,'(()()())':2}
 caps=[r['public_interface']['coordinates'] for r in rows];assert all(c==caps[0] for c in caps)
 w=records['g2']['caps7_topology_separation_witness'];pair=records['s0']['certified_collision_pair']
 for side,old in [('A','topology_A'),('B','topology_B')]:
  assert pair[side] in rows;assert pair[side]['g2_carrier_ref']==w[old]['example_construction_digest'];assert pair[side]['public_interface']['coordinates']==w['shared_caps7']
 summaries=records['s1']['target_behavior_summaries'];assert {r['target_ref'] for r in summaries}==refs
 assert all(r['context_row_count']==372 for r in summaries);assert records['s1']['context_basis']['task_count']==2232
 assert records['s1']['challenge_reproduction']['authority']==auth1
 refusals=[]
 def refuse(name,fn):
  try:fn()
  except (ValueError,G3S0Error,G3S1Error):refusals.append(name)
  else:raise AssertionError('not refused:'+name)
 for key in ('g2','s0','s1'):
  bad=deepcopy(records[key]);bad['status']='FAIL';refuse(key+'_stale_science_label',lambda bad=bad,key=key:validate_payload(bad,SCIENCE[key]))
 for field,value in [('status','FAIL'),('classification','OTHER'),('science_sha256','0'*64),('g3_s1_unlocked',False),('g3_graduated',True)]:
  bad=deepcopy(records['s0']);bad[field]=value;refuse('s0_'+field,lambda bad=bad:verify_s1_authority(bad))
 for field,value in [('science_sha256','0'*64),('promotion',True),('g2_graduated_before_r0',False)]:
  bad=deepcopy(records['g2']);bad[field]=value;refuse('g2_'+field,lambda bad=bad:verify_authority(graduation_certificate=cert,r0_result=bad))
 report={'status':'PASS','scientific_outcome':'FINITE_ORIGINAL_G2_G3_SAVED_AUTHORITY_CHAIN_BOUND','bindings':bindings,'authority_edges_verified':3,'challenge_records_bound':6,'reported_context_tasks':2232,'context_task_payloads_reexecuted':0,'refusals':refusals,'generation_calls':0,'production_authority':'NONE','original_plan_complete':2,'revised_plan_complete':1,'package2_complete':False,'payload_scope':'S0 six saved public/hidden-diagnostic records; S1 six target summaries, not 2232 raw per-context result payloads','execution_reuse_authorized':False,'next_requirement':'source/cohort-bound saved-carrier read path; full witness closure before dependent new execution','no_new_graduation_or_topology_promotion':True}
 (out/'QUALIFICATION_RESULT.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ('status','scientific_outcome','authority_edges_verified','challenge_records_bound','generation_calls')}))
if __name__=='__main__':main()
