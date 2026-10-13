"""Source-bound rolling launch, with a distinct registered engineering fixture route."""
from pathlib import Path
import argparse,hashlib,json,sys
from . import submission as sub,portable_registry as registry,preservation as pr
from .canon import canonical_sha256,canonical_bytes,write_json_atomic
from .bulk_submission import _coordinate_registered_windows,execute_adopted_windows,window_workspace,_prepared_json
from .operator_transport import OperatorQueueTransport

ENGINEERING_SCHEMA='IG_DECODER_ENGINEERING_ROLLING_REGISTRATION_V1'


def run_engineering(root,engine,project,artifacts,registration,transport):
    """Two frozen sparse qualification windows; never a data-owner adoption route."""
    from . import v05_controller_event_loop as loop
    from .isolated_runtime import verified_runtime,BUNDLED_MANIFEST_SHA256
    from .runtime_capacity import host_memory_ceiling_bytes
    from .sqlite_attestation import admit_capture_runtime
    from .bulk_descriptor_plan import admit_frozen_plan
    from .bulk_mapping_store import read_mapping_rows
    verified_runtime();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    fields={'schema_id','purpose','scientific_adoption','source_sha256','package_sha256','runtime_manifest_sha256',
            'windows','workers','batch_size','artifacts'}
    if set(registration)!=fields or registration['schema_id']!=ENGINEERING_SCHEMA or registration['purpose']!='QUALIFY_NATIVE_TRANSPORT_AND_RECOVERY' or registration['scientific_adoption'] is not False:
        raise ValueError('ENGINEERING_REGISTRATION_REQUIRED')
    source_sha,package_sha=loop._source_ids(Path(__file__).resolve().parents[1])
    if registration['source_sha256']!=source_sha or registration['package_sha256']!=package_sha or registration['runtime_manifest_sha256']!=BUNDLED_MANIFEST_SHA256:
        raise ValueError('ENGINEERING_SOURCE_PIN')
    if len(registration['windows'])!=2 or type(registration['workers']) is not int or type(registration['batch_size']) is not int or registration['workers']<=0 or registration['batch_size']<=0:
        raise ValueError('ENGINEERING_WINDOW_SCOPE')
    for name,path in artifacts.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=registration['artifacts'].get(name):raise ValueError('ENGINEERING_ARTIFACT_PIN')
    if set(artifacts)!=set(registration['artifacts']):raise ValueError('ENGINEERING_ARTIFACT_SCOPE')
    admit_frozen_plan(artifacts['raw_descriptor_archive'],artifacts['preserved_prefix_archive'])
    sample=json.loads(Path(artifacts['descriptor_sample']).read_bytes())
    reference=json.loads(Path(artifacts['reference_full_witnesses']).read_bytes())
    by_ordinal={r['ordinal']:r for r in sample};expected={r['ordinal']:r for r in reference}
    selected=[]
    for i,window in enumerate(registration['windows']):
        if set(window)!= {'index','ordinals'} or window['index']!=i or type(window['ordinals']) is not list or not window['ordinals'] or window['ordinals']!=sorted(set(window['ordinals'])):
            raise ValueError('ENGINEERING_WINDOW_SCOPE')
        if any(type(o) is not int or o not in by_ordinal or o not in expected for o in window['ordinals']):raise ValueError('ENGINEERING_DESCRIPTOR_SCOPE')
        selected.extend(window['ordinals'])
    if len(set(selected))!=len(selected):raise ValueError('ENGINEERING_WINDOW_OVERLAP')
    plan={'schema_id':ENGINEERING_SCHEMA,'windows':registration['windows'],'plan_sha256':canonical_sha256(registration)}
    def prepare(index,directory):
        directory.mkdir(parents=True,exist_ok=True);window=plan['windows'][index]
        rows=[by_ordinal[o] for o in window['ordinals']];rowpath=directory/'descriptor-window.json'
        _prepared_json(rowpath,rows)
        spec={'schema_id':sub.SPEC_SCHEMA,'job_id':f'ENG.DECODER091.ROLLING.{index:02d}',
             'engine_source':str(Path(engine).resolve()),'project_source':str(Path(project).resolve()),
             'question':{'stage_id':f'ENG:DECODER091-ROLLING-{index}','description':'Registered native sparse engineering fixture; no census or scientific adoption',
                         'outcomes':['PASS'],'stopping_rule':'Exactly this declared sparse window, saved and independently compared'},
             'execution':{'kind':'STAGE','handler_ref':'project.bulk_adapter:handler','evaluator_refs':['project.bulk_adapter:evaluate'],
                          'parameters':{'registered_rows':len(rows),'batch_size':registration['batch_size'],'workers':registration['workers'],
                                        'window_index':None,'window_rows':16384}},
             'resources':{'workers':registration['workers'],'start_method':'spawn','memory_budget_bytes':host_memory_ceiling_bytes(),
                          'workspace_budget_bytes':None,'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'},
             'inputs':[],'environment':{'python':'3.13','requirements':[],
                 'artifacts':[{'logical_name':name,'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                              for name,path in dict(artifacts,descriptor_sample=rowpath).items()]},
             'output_contract':{'schema_id':'IG_DECODER_RESULT_CONTRACT_V1','claim':'EXECUTION_ONLY','required_artifacts':[],
                  'result_checks':[{'pointer':'/outcome','equals':'PASS'}],'prerequisites':[],
                  'preservation':{'project_evidence_externalization':1,'max_project_raw_bytes':None,'completion_project_closure':1,'rolling_unbounded_execution':1}}}
        _prepared_json(directory/'SPECIFICATION.json',spec)
        registry.initialize(directory/'store','Registered Decoder091 engineering rolling fixture',engine,resume=True)
        captured=sub.capture(directory/'store',spec)
        _prepared_json(directory/'WORKSPACE.json',{'path':captured['workspace']})
        return {'capture':captured}
    def verify(index,workspace):
        rec=sub.capture_record(workspace)
        if rec['job']['job_id']!=f'ENG.DECODER091.ROLLING.{index:02d}' or rec['workspace']['source_sha256']!=source_sha or rec['workspace']['package_sha256']!=package_sha:
            raise ValueError('ENGINEERING_CAPTURE_SCOPE')
        if sub.save_status(workspace)['status']!='SAVED' or pr.status(workspace)['status']!='PRESERVED':raise ValueError('ENGINEERING_SAVE_REQUIRED')
        admission=loop.validate_workspace_job(workspace,rec['job']['job_id'],check_loaded=False);admit_capture_runtime(workspace,rec)
        done=loop.verified_completion(admission)
        if done is None or done['status']!='COMPLETED' or done['result'].get('outcome')!='PASS':raise ValueError('ENGINEERING_COMPLETION_REQUIRED')
        sealed=Path(workspace)/done['evidence_root'];databases=list(sealed.rglob('mapping.sqlite3'))
        if len(databases)!=1:raise ValueError('ENGINEERING_MAPPING_DATABASE_REQUIRED')
        rows=list(read_mapping_rows(databases[0],expected_ordinals=plan['windows'][index]['ordinals']))
        wanted=[expected[o] for o in plan['windows'][index]['ordinals']]
        if canonical_bytes(rows)!=canonical_bytes(wanted):raise ValueError('ENGINEERING_INDEPENDENT_REFERENCE_DIFFERENCE')
        return {'index':index,'capture_id':rec['capture_id'],'source_sha256':source_sha,'package_sha256':package_sha,
                'rows':len(rows),'full_rows_stream_sha256':hashlib.sha256(b''.join(canonical_bytes(r)+b'\n' for r in rows)).hexdigest(),
                'completion_sha256':done['completion_sha256'],'status':'VERIFIED_ENGINEERING_NATIVE_SAVED','scientific_adoption':False}
    def validate(progress,verify_saved):
        if not progress:return
        if set(progress)!= {'schema_id','plan_sha256','completed_windows','scientific_adoption'} or progress['schema_id']!='IG_ENGINEERING_ROLLING_PROGRESS_V1' or progress['plan_sha256']!=plan['plan_sha256'] or progress['scientific_adoption'] is not False:
            raise ValueError('ENGINEERING_PROGRESS_BINDING')
        completed=progress['completed_windows']
        if type(completed) is not list or len(completed)>2:raise ValueError('ENGINEERING_PROGRESS_COVERAGE')
        for i,receipt in enumerate(completed):
            if receipt['index']!=i:raise ValueError('ENGINEERING_PROGRESS_ORDER')
            verify_saved(i,receipt)
    def advance(progress,receipt):
        completed=list(progress.get('completed_windows',[]))
        if receipt['index']!=len(completed):raise ValueError('ENGINEERING_PROGRESS_ORDER')
        return {'schema_id':'IG_ENGINEERING_ROLLING_PROGRESS_V1','plan_sha256':plan['plan_sha256'],
                'completed_windows':completed+[receipt],'scientific_adoption':False}
    return _coordinate_registered_windows(root,plan,source_sha,package_sha,transport,prepare,verify,validate,advance)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',type=Path,required=True);parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--engineering-registration',type=Path);group.add_argument('--data-owner-adoption',type=Path)
    parser.add_argument('--workers',type=int)
    args=parser.parse_args(argv);artifacts=json.loads(args.artifacts.read_bytes());transport=OperatorQueueTransport(args.queue)
    if args.engineering_registration:
        result=run_engineering(args.root,args.engine,args.project,artifacts,json.loads(args.engineering_registration.read_bytes()),transport)
    else:
        result=execute_adopted_windows(args.root,args.engine,args.project,artifacts['raw_descriptor_archive'],artifacts['preserved_prefix_archive'],
              artifacts['sqlite_runtime_binding'],json.loads(args.data_owner_adoption.read_bytes()),transport,workers=args.workers)
    print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
