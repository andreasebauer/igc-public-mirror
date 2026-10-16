"""Integrate preserved exports through installed RunPlan dispatch, no science rerun."""
from pathlib import Path
from io import BytesIO
from copy import deepcopy
from unittest.mock import patch
import hashlib,json,os,time,zipfile
import reference
from infinity_grid import master_existing_export_pipeline as pipeline
from infinity_grid.run_plan_v2 import compile_run_plan_v2,execute_existing_export_attach_run_plan,RunPlanV2Error
from infinity_grid.canon import write_json_atomic
from infinity_grid import semantic_sentinel,theorem_registry

def main():
    started=time.monotonic();inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text());paths={k:inputs[k] for k in pipeline.DATA_PINS}
    plan={'schema_id':pipeline.SCHEMA,'route':pipeline.ROUTE,'components':pipeline.COMPONENTS,'g4_fixture_policy':'EVIDENCE_ONLY','generation_allowed':False,'production_authority':'NONE'}
    old_semantics=semantic_sentinel.verify_native_semantics(raise_on_change=False);old_theorems=theorem_registry.verify_all_theorems(raise_on_stale=False)
    assert old_semantics['status']=='REOPEN_REQUIRED' and len(old_theorems['stale_theorems'])==4
    refs={};z3=zipfile.ZipFile(inputs['g3']);z6=zipfile.ZipFile(inputs['g6']);z22=zipfile.ZipFile(inputs['q22']);zf=zipfile.ZipFile(BytesIO(z22.read('accepted_outputs/SCIENTIFIC_STORAGE_FULL.zip')))
    for name,z,prefix in [(pipeline.COMPONENTS[0],z3,''),(pipeline.COMPONENTS[1],zf,''),(pipeline.COMPONENTS[2],z6,'partial_store/')]:
        c=json.loads(z.read(prefix+'CANDIDATE.json'));counts,digest=reference.read(z,prefix,c);refs[name]={'counts':counts,'stream_sha256':digest}
    for z in (z3,z6,z22,zf):z.close()
    seen={};sink_calls=0
    def sink(component,kind,key,raw):
        nonlocal sink_calls
        h=seen.setdefault(component,hashlib.sha256());h.update(len(raw).to_bytes(8,'big'));h.update(raw);sink_calls+=1
    # These producer/rebuild functions must not run during the installed read route.
    from infinity_grid import master_g3_saved_storage,master_g6_initial_storage,master_scientific_storage,storage_catalog,full_owner_export,full_marker_export
    from contextlib import ExitStack
    forbidden=[]
    def no(*a,**kw):forbidden.append('FORBIDDEN');raise AssertionError('scientific/rebuild call')
    with ExitStack() as stack:
        for obj,name in [(master_g3_saved_storage,'saved_dag_storage'),(master_g6_initial_storage,'initial_storage'),(master_scientific_storage,'build_export_storage'),(storage_catalog,'build_index')]:stack.enter_context(patch.object(obj,name,no))
        compiled=compile_run_plan_v2(plan);result=execute_existing_export_attach_run_plan(plan,archives=paths,sink=sink)
    assert result['compiled']==compiled and sink_calls==result['total_records']==267406 and not forbidden
    for c in result['components']:
        assert c['counts']==refs[c['component']]['counts']
        assert c['full_record_stream_sha256']==refs[c['component']]['stream_sha256']==seen[c['component']].hexdigest()
        assert c['native_references_rewritten'] is False and c['cross_store_identity_merge'] is False
    assert result['g4_fixture_terms']==12 and result['g4_fixtures_added_to_science']==0 and result['g4_historical_corpus_coverage'] is False
    refusals=[]
    def refuse(name,fn):
        try:fn()
        except (pipeline.ExistingExportError,RunPlanV2Error):refusals.append(name)
        else:raise AssertionError('missing refusal:'+name)
    with patch.object(pipeline,'_snapshot',side_effect=AssertionError('input read before scope refusal')):
        for field,value in [('generation_allowed',True),('generation_allowed',0),('production_authority','MASTER'),('g4_fixture_policy','MASTER_SCIENCE'),('components',pipeline.COMPONENTS+['HISTORICAL_G4_CORPUS']),('components',['GLOBAL_LOWER_CLOSURE']),('route','OTHER')]:
            bad=deepcopy(plan);bad[field]=value;refuse(field+'_'+str(value),lambda bad=bad:execute_existing_export_attach_run_plan(bad,archives={}))
        bad=deepcopy(plan);bad['allow_historical_current_equivalence']=True;refuse('historical_equivalence_override',lambda:execute_existing_export_attach_run_plan(bad,archives={}))
    refuse('frontier_adoption',lambda:compile_run_plan_v2(plan,frontier={}))
    refuse('missing_input',lambda:execute_existing_export_attach_run_plan(plan,archives={}))
    assert semantic_sentinel.verify_native_semantics(raise_on_change=False)==old_semantics and theorem_registry.verify_all_theorems(raise_on_stale=False)==old_theorems
    write_json_atomic('MASTER_EXISTING_EXPORT_PIPELINE.json',result)
    write_json_atomic('INDEPENDENT_STREAM_COMPARISON.json',{'status':'PASS','components':refs,'exact_typed_record_streams':True,'total_records':sink_calls})
    write_json_atomic('REFUSALS.json',{'status':'PASS','cases':refusals,'count':len(refusals),'unsupported_scopes_refused_before_input':True})
    write_json_atomic('QUALIFICATION_RESULT.json',{'status':'PASS','scientific_outcome':'EXISTING_FULL_EXPORT_MASTER_PIPELINE_ATTACH_QUALIFIED','installed_runplan_dispatch':True,'components':3,'scientific_records':sink_calls,'g3_records':158507,'finite18_records':3806,'g6_records':105093,'g4_fixture_terms_excluded':12,'g4_historical_corpus_coverage':False,'generation_calls':0,'storage_rebuilds':0,'index_rebuilds':0,'refusal_count':len(refusals),'full_master_producer_compatible':False,'production_authority':'NONE','package2_complete':False,'body_seconds':time.monotonic()-started})
if __name__=='__main__':main()
