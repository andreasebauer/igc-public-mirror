"""Complete saved-record reads of the recorded G4 rebase chain; no new science."""
from pathlib import Path
from io import BytesIO
from copy import deepcopy
import hashlib,json,re,zipfile
from .canon import canonical_sha256

SCHEMA='IG_G4_REBASE_SAVED_PIPELINE_PLAN_V1'
ROUTE='IG_G4_REBASE_SAVED_PIPELINE@1.0.0'
MANIFEST=Path(__file__).with_name('g4_rebase_saved_pins.json')
class G4SavedPipelineError(RuntimeError):pass
def need(ok,why):
 if not ok:raise G4SavedPipelineError(why)
def compile_plan(plan):
 expected={'schema_id':SCHEMA,'route':ROUTE,'scope':'ORIGINAL_SAVED_RECORDS_AND_AUTHORITY_LINKS','r6_authority':'DRIVE_BRANCH','generation_allowed':False,'production_authority':'NONE'}
 need(type(plan) is dict and set(plan)==set(expected),'PLAN_FIELDS')
 need(all(type(plan[k]) is type(v) and plan[k]==v for k,v in expected.items()),'PLAN_SCOPE')
 return dict(expected,required_theorems=[])
def _snapshot(path,pin):
 raw=Path(path).read_bytes();need(hashlib.sha256(raw).hexdigest()==pin,'ARCHIVE_HASH');return raw
def _seal(obj):
 need(obj['science_sha256']==canonical_sha256({k:v for k,v in obj.items() if k!='science_sha256'}),'SCIENCE_SEAL')
def _stable(obj):return {k:v for k,v in obj.items() if k not in {'source_sha256','source_version','registry_sha256','execution_metadata','science_sha256'}}
def _stage(raw,pin):
 with zipfile.ZipFile(BytesIO(raw)) as z:
  need(len(z.namelist())==len(set(z.namelist())),'DUPLICATE_PATH')
  selected={}
  for n,h in pin['members'].items():
   b=z.read(n);need(hashlib.sha256(b).hexdigest()==h,'MEMBER_HASH');selected[n]=json.loads(b)
 main=selected[pin['primary']];cold=selected[pin['cold']];cert=selected[pin['certificate']];replay=selected[pin['replay']]
 need(main['status']==cold['status']=='PASS','STAGE_STATUS')
 need(main['science_sha256']==cold['science_sha256']==pin['science_sha256'],'PRIMARY_COLD_SCIENCE')
 need(main['source_sha256']==cold['source_sha256']==pin['source_sha256'],'SOURCE_BINDING')
 need(_stable(main)==_stable(cold) and canonical_sha256(_stable(main))==pin['science_sha256'],'FULL_STABLE_PAYLOAD')
 _seal(cert);_seal(replay)
 need(cert['status']=='CERTIFIED_PASS' and replay['status']=='PASS' and replay['certification']=='CERTIFIED_PASS','CERTIFICATION')
 need(cert['primary_science_sha256']==cert['cold_science_sha256']==replay['primary_science_sha256']==replay['cold_science_sha256']==pin['science_sha256'],'CERTIFICATE_BINDING')
 return {'main':main,'certificate':cert,'replay':replay,'records':selected}
def _links(stages):
 links=[]
 for name,stage in stages.items():
  for field,value in stage['main']['authority'].items():
   m=re.fullmatch(r'([sr][0-7])_(?:(closeout|replay)_)?science_sha256',field)
   if not m:continue
   target=m[1].upper();kind={'closeout':'certificate','replay':'replay',None:'main'}[m[2]]
   need(value==stages[target][kind]['science_sha256'],'AUTHORITY_LINK:'+name+':'+field)
   links.append({'from':name,'field':field,'to':target,'kind':kind,'science_sha256':value})
 # Additional term/candidate-interface links are not anonymous carrier equality.
 corpus=stages['S0']['main']['certified_term_corpus'];_seal(corpus)
 need(stages['S1']['main']['authority']['s0_term_corpus_sha256']==corpus['science_sha256'],'S0_CORPUS_LINK')
 candidate=stages['S3']['main']['candidate_interface'];_seal(candidate)
 for name,stage in stages.items():
  a=stage['main']['authority']
  if 's3_candidate_interface_science_sha256' in a:need(a['s3_candidate_interface_science_sha256']==candidate['science_sha256'],'S3_INTERFACE_LINK')
 return links
def _branch(stages):
 r7=stages['R7'];records=r7['records'];rec=records['reconciliation/G4_R6_BRANCH_RECONCILIATION_V1.json'];corr=records['reconciliation/G4_R4_R5_INTERPRETATION_CORRECTION_V1.json']
 _seal(rec);_seal(corr)
 need(rec['status']==corr['status']=='PASS' and rec['reconciliation_decision']['frontier_authority']=='DRIVE_BRANCH','BRANCH_AUTHORITY')
 a=r7['main']['authority']
 need(a['branch_reconciliation_science_sha256']==rec['science_sha256'] and a['r4_r5_correction_science_sha256']==corr['science_sha256'],'CORRECTION_BINDING')
 for name,key in [('R6','drive_authoritative_branch'),('R6_PARALLEL','parallel_local_branch')]:
  for k,field in [('main','primary_science_sha256'),('certificate','closeout_science_sha256'),('replay','replay_science_sha256')]:need(stages[name][k]['science_sha256']==rec[key][field],'R6_BRANCH_BINDING')
 need(stages['R6']['main']['science_sha256']!=stages['R6_PARALLEL']['main']['science_sha256'],'BRANCHES_DISTINCT')
 # Resource-realizability support refers to the exact saved rebased C/D terms.
 corpus=stages['S0']['main']['certified_term_corpus']['terms'];bridge=r7['main']['actual_g4_resource_realizability'];_seal(bridge)
 seed=bridge['actual_certified_g3_seed'];partner=bridge['same_caps_distinct_H_partner']
 need(seed['term_key']=='C' and seed['term_ref']==corpus['C']['term']['term_ref'] and seed['caps7']==corpus['C']['term']['total_caps'],'R7_C_TERM_LINK')
 need(partner['term_key']=='D' and partner['term_ref']==corpus['D']['term']['term_ref'],'R7_D_TERM_LINK')
 return {'authority':'DRIVE_BRANCH','parallel_scope':'SUPPORTING_RECONSTRUCTION_ONLY','reconciliation':rec,'interpretation_correction':corr,'saved_C_D_terms_bound':True,'fresh_generation_or_theorem_reproof':False}
def execute(plan,archives):
 compiled=compile_plan(plan);pins=json.loads(MANIFEST.read_text());need(type(archives) is dict and set(archives)==set(pins),'INPUT_SET')
 # Hold immutable original byte snapshots until the entire route succeeds.
 snapshots={name:_snapshot(archives[name],pin['archive_sha256']) for name,pin in pins.items()}
 stages={name:_stage(snapshots[name],pin) for name,pin in pins.items()}
 links=_links(stages);branch=_branch(stages)
 counts={name:len(stage['records']) for name,stage in stages.items()}
 return {'schema_id':'IG_G4_REBASE_SAVED_PIPELINE_RESULT_V1','compiled':compiled,'stages':stages,'record_counts':counts,'saved_records':sum(counts.values()),'rebase_stages':15,'parallel_supporting_branches':1,'authority_links':links,'branch_resolution':branch,'generation_calls':0,'production_authority':'NONE','complete_g4_carrier_population':False,'consumed_theorem_transport_authority':False,'scope':'READ_EXACT_SAVED_JSON_RECORDS; NO SCIENTIFIC REGENERATION_OR_NEW_ADOPTION'}
