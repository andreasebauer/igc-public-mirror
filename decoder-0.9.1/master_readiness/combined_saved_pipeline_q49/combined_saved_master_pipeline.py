"""One explicit current-source entry for the five qualified finite saved readers."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,tempfile,zipfile
from . import saved_read_run_plan as saved
from . import saved_g2_g3_read_plan as authority
from . import historical_g4_saved_terms as terms
from . import g4_rebase_saved_pipeline as chain
from . import master_existing_export_pipeline as stores
from .canon import canonical_sha256

SCHEMA='IG_COMBINED_SAVED_MASTER_PIPELINE_PLAN_V1'
ROUTE='IG_COMBINED_SAVED_MASTER_PIPELINE@1.0.0'
COMPONENTS=['O7_R100_SAVED_READ','ORIGINAL_G2_G3_READ','HISTORICAL_G4_TERMS','G4_REBASE_CHAIN','EXISTING_NATIVE_STORES']
class CombinedSavedPipelineError(RuntimeError):pass
def need(ok,why):
 if not ok:raise CombinedSavedPipelineError(why)
def compile_plan(plan):
 expected={'schema_id':SCHEMA,'route':ROUTE,'components':COMPONENTS,'scope':'FINITE_QUALIFIED_SAVED_READS_ONLY','generation_allowed':False,'production_authority':'NONE'}
 need(type(plan) is dict and set(plan)==set(expected),'COMBINED_FIELDS')
 need(all(type(plan[k]) is type(v) and plan[k]==v for k,v in expected.items()),'COMBINED_SCOPE')
 return dict(deepcopy(expected),required_theorems=[])
def plans(calibration):
 requests=[{'kind':'saved_packet','domain':d,'level':n} for d,n in calibration['saved_packet_cases']]
 requests += [{'kind':'complete_read','domain':a,'level':7} for a in calibration['counter_anchor_ids']]
 requests += [{'kind':'matched_findings','domain':'FROZEN_MOTIFS','level':n} for n in calibration['matched_levels']]
 requests += [{'kind':'temporal_window','domain':'FROZEN_MOTIFS','level':n} for n in calibration['closed_temporal_depths']]
 requests += [{'kind':'r100_descriptors','domain':'CURRENT_Q07','level':100}]
 p0={'schema_id':saved.SCHEMA,'route':saved.ROUTE,'calibration_sha256':calibration['science_sha256'],'requests':requests,'stop_policy':calibration['stop_policy'],'production_authority':'NONE','generation_allowed':False}
 p1={'schema_id':authority.SCHEMA,'route':authority.ROUTE,'requests':[{'kind':'saved_result','source_scope':k} for k in ['g2','s0','s1']]+[{'kind':'s0_challenge_records','source_scope':'s0'},{'kind':'s1_target_summaries','source_scope':'s1'}],'production_authority':'NONE','generation_allowed':False}
 p2={'schema_id':terms.SCHEMA,'route':terms.ROUTE,'cohorts':['original','rebase'],'scope':'SAVED_CAPS7_ATOMIC_TERMS_ONLY','production_authority':'NONE','generation_allowed':False}
 p3={'schema_id':chain.SCHEMA,'route':chain.ROUTE,'scope':'ORIGINAL_SAVED_RECORDS_AND_AUTHORITY_LINKS','r6_authority':'DRIVE_BRANCH','generation_allowed':False,'production_authority':'NONE'}
 p4={'schema_id':stores.SCHEMA,'route':stores.ROUTE,'components':stores.COMPONENTS,'g4_fixture_policy':'EVIDENCE_ONLY','generation_allowed':False,'production_authority':'NONE'}
 return [p0,p1,p2,p3,p4]
def execute(plan,inputs,*,sink=None,progress=None):
 compiled=compile_plan(plan)
 pin_roles=json.loads(chain.MANIFEST.read_text())
 expected={'saved_inputs','saved_binding','q11_evidence','master_contract'}|{'authority_'+k for k in authority.PINS}|{'term_'+k for k in terms.PINS}|{'chain_'+k for k in pin_roles}|{'store_'+k for k in stores.DATA_PINS}
 need(type(inputs) is dict and set(inputs)==expected,'COMBINED_INPUT_SET')
 binding=json.loads(Path(inputs['saved_binding']).read_text());master=json.loads(Path(inputs['master_contract']).read_text())
 calibration=saved.make_calibration(binding,inputs['q11_evidence'],master)
 subplans=plans(calibration)
 from . import run_plan_v2 as rp
 for p in subplans:rp.compile_run_plan_v2(p,**({'saved_read_calibration':calibration} if p['schema_id']==saved.SCHEMA else {}))
 # A finite saved-input archive, never a scanner source or corpus constructor.
 raw=Path(inputs['saved_inputs']).read_bytes()
 need(hashlib.sha256(raw).hexdigest()=='51cc2df89254cb94a22307ace944e4edaa29d5aa117febdd6353ba44bb3e8fb2','SAVED_INPUT_ARCHIVE')
 with tempfile.TemporaryDirectory(prefix='ig_combined_saved_') as folder:
  from io import BytesIO
  with zipfile.ZipFile(BytesIO(raw)) as z:
   need(len(z.namelist())==len(set(z.namelist())) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist()),'SAVED_ARCHIVE_PATHS');z.extractall(folder)
  r0=rp.execute_saved_read_run_plan(subplans[0],root=folder,binding=binding,calibration=calibration,evidence_path=inputs['q11_evidence'],master_contract=master)
 if progress is not None:progress('O7_R100_SAVED_READ')
 r1=rp.execute_original_g2_g3_saved_read_run_plan(subplans[1],archives={k:inputs['authority_'+k] for k in authority.PINS})
 if progress is not None:progress('ORIGINAL_G2_G3_READ')
 r2=rp.execute_historical_g4_saved_term_run_plan(subplans[2],archives={k:inputs['term_'+k] for k in terms.PINS})
 if progress is not None:progress('HISTORICAL_G4_TERMS')
 r3=rp.execute_g4_rebase_saved_pipeline_run_plan(subplans[3],archives={k:inputs['chain_'+k] for k in pin_roles})
 if progress is not None:progress('G4_REBASE_CHAIN')
 r4=rp.execute_existing_export_attach_run_plan(subplans[4],archives={k:inputs['store_'+k] for k in stores.DATA_PINS},sink=sink)
 if progress is not None:progress('EXISTING_NATIVE_STORES')
 return {'schema_id':'IG_COMBINED_SAVED_MASTER_PIPELINE_RESULT_V1','compiled':compiled,'results':dict(zip(COMPONENTS,[r0,r1,r2,r3,r4])),'component_count':5,'saved_requests':len(r0['observations']),'original_authority_reads':len(r1['observations']),'source_scoped_g4_terms':r2['source_scoped_terms'],'g4_saved_documents':r3['saved_records'],'g4_authority_links':len(r3['authority_links']),'store_records':r4['total_records'],'generation_allowed':False,'production_authority':'NONE','full_master_producer_compatible':False,'commit_scope':'ALL_FIVE_COMPONENTS_COMPLETE; SINK_ROWS_PROVISIONAL_UNTIL_SUCCESS; ORIGINAL_SCOPES_AND_IDENTITIES_RETAINED'}
