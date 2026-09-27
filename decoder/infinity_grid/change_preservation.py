"""Versioned administrative recovery for a preserved, pre-test change capture.

This module does not execute tests, impersonate a controller, alter frozen
records, or relax source/origin/receipt gates. It registers a NEW execution
through submission.capture, preserving the original failed admissions. Execute
that capture using its own retained infinity_grid.controller and then record
its native completion. Only max_pending_bytes may be increased. The exact
candidate, nodes, run resources, inputs and semantic result checks are fixed.

Production mutators require a native PASS whose tested candidate source equals
this complete recovery-source tree. This is task-local administrative tooling,
not permission to promote this source as a common Decoder release.
"""
from __future__ import annotations
from pathlib import Path
import json
from . import submission as sub, portable_registry as pr, result_contracts as rc
from . import change_sessions as cs
from .canon import canonical_sha256

SCHEMA='IG_CHANGE_PRESERVATION_AMENDMENT_V1'
HANDLER='infinity_grid.change_validation:validate_revision'
NODE='tests/test_change_preservation_rebind.py'
QUALIFICATION_NODES=[NODE]


def semantics(record):
    """Fields a preservation-only administrative revision must not change."""
    job=record['job'];contract=record['result_contract']
    return {'workspace':record['workspace'],'engine':next(o['sha256'] for o in record['objects'] if o['role']=='engine_source'),
        'question':job['question'],'execution':job['execution'],'resources':job['resources'],
        'inputs':[o for o in job['input_artifacts'] if o['logical_name']!='submission_contract'],
        'environment':record['environment'],
        'result_rules':{k:contract[k] for k in ('schema_id','claim','required_artifacts','result_checks','prerequisites')},
        'outcome':contract.get('outcome')}


def revised_contract(record, pending_bytes):
    if (record['job']['execution'].get('handler_ref')!=HANDLER or
        any(o['role']=='project_source' for o in record['objects'])):
        raise sub.SubmissionError('PRESERVATION_ENGINE_CHANGE_ONLY')
    resources=record['job']['resources']
    if resources.get('execution_policy')!='NO_AUTOMATIC_RUNTIME_DEADLINE_V1' or resources.get('wall_seconds_max') is not None:
        raise sub.SubmissionError('PRESERVATION_NO_DEADLINE_REQUIRED')
    prior=rc.policy(record['result_contract'])
    if type(pending_bytes) is not int or pending_bytes<=prior['max_pending_bytes']:
        raise sub.SubmissionError('PRESERVATION_INCREASE_REQUIRED')
    if pending_bytes>resources['workspace_budget_bytes']:
        raise sub.SubmissionError('PRESERVATION_EXCEEDS_WORKSPACE')
    contract={k:record['result_contract'][k] for k in ('schema_id','claim','required_artifacts','result_checks','prerequisites')}
    if 'outcome' in record['result_contract']:contract['outcome']=record['result_contract']['outcome']
    contract['preservation']=dict(prior,max_pending_bytes=pending_bytes)
    return rc.normalize(contract,record['job']['execution'],record['job']['question'])


def anchor(session):
    row=cs._sealed({'schema_id':'IG_DECODER_LOCAL_DEVELOPMENT_POINTER_V1','epoch':0,
        'target':session['parent'],'previous_pointer_sha256':None,
        'scope':'ISOLATED_LOCAL_DEVELOPMENT_NOT_SHARED_AUTHORITY'})
    if row['record_sha256']!=session['parent_pointer_sha256']:
        raise sub.SubmissionError('RECOVERY_ORIGINAL_POINTER_REQUIRED')
    return row


def _qualified(workspace):
    done=cs._completion(Path(workspace))
    rec=sub.capture_record(workspace)
    here=Path(__file__).resolve().parents[1]
    sid,_=cs.loop._source_ids(here)
    result=done['result']
    if (rec['job']['execution'].get('handler_ref')!=HANDLER or
        result.get('candidate_source_sha256')!=sid or result.get('outcome')!='PASS'):
        raise sub.SubmissionError('RECOVERY_BOOTSTRAP_QUALIFICATION_REQUIRED')
    names=result.get('validation',{}).get('nodes')
    # The canonical validation result carries exact node selectors in its groups.
    revision_path=_input_path(workspace,'change_revision')
    rev=cs._verified(revision_path)
    if rev['requirements']['nodes']!=QUALIFICATION_NODES:
        raise sub.SubmissionError('RECOVERY_BOOTSTRAP_TEST_SCOPE')
    return {'source_sha256':sid,'capture_id':rec['capture_id'],
            'completion_sha256':done['completion_sha256'],'requirements':rev['requirements']}


def _input_path(workspace,name):
    rec=sub.capture_record(workspace)
    rows=[r for r in rec['job']['input_artifacts'] if r['logical_name']==name]
    if len(rows)!=1:raise sub.SubmissionError('RECOVERY_INPUT_REQUIRED',name)
    p=Path(workspace)/'runtime/intake/artifacts'/(rows[0]['sha256']+'.bin')
    if sub._sha(p.read_bytes())!=rows[0]['sha256']:raise sub.SubmissionError('RECOVERY_INPUT_HASH',name)
    return p


def pretest_only(workspace):
    ws=Path(workspace);rec=sub.capture_record(ws)
    cs.loop.validate_workspace_job(ws,rec['job']['job_id'],check_loaded=False)
    sub.require_saved(ws,rec['job']['job_id'])
    if any((ws/'runtime/intake/completed').glob('*.json')) or any((ws/'runtime/runs').glob('*/candidate_validation')):
        raise sub.SubmissionError('PRESERVATION_TEST_ALREADY_STARTED')
    paths=sorted((ws/'runtime/attempts').glob('*/*.json'))
    if not paths:raise sub.SubmissionError('PRESERVATION_PAUSED_ADMISSION_REQUIRED')
    for p in paths:
        row=sub._read(p)
        if row.get('status')!='PAUSED' or 'SAVE_BACKLOG_PAUSE:' not in str(row.get('reason','')):
            raise sub.SubmissionError('PRESERVATION_OTHER_ATTEMPT',p.name)
    return {p.relative_to(ws).as_posix():sub._sha(p.read_bytes()) for p in paths}


def recover_store(store_root,originals,qualification_workspace):
    proof=_qualified(qualification_workspace)
    store=Path(store_root).resolve(strict=True);originals=Path(originals).resolve(strict=True)
    session=cs._verified(originals/'SESSION.json');cid=session['change_id']
    latest=cs._verified(originals/'LATEST.json');rid=latest['revision_id']
    rev=cs._verified(originals/'revisions'/(rid+'.json'))
    binding=cs._verified(originals/'bindings'/(rid+'.json'))
    ws=store/'execution/captures'/binding['capture_id'];rec=sub.capture_record(ws)
    if (cs._verified(_input_path(ws,'change_session'))!=session or
        cs._verified(_input_path(ws,'change_revision'))!=rev or
        rev['session_sha256']!=session['record_sha256'] or binding['revision_id']!=rid or
        rec['job']['job_id']!=binding['job_id']):
        raise sub.SubmissionError('RECOVERY_CONTROL_BINDING')
    proot,pbind=pr.locate(ws);head,release=pr.current(proot,'release')
    if head!=session['project_parent_head'] or release['engine']!=session['parent']['source_object']['sha256']:
        raise sub.SubmissionError('RECOVERY_RELEASE_MOVED')
    pointer=anchor(session)
    folder=store/'changes'/cid
    with cs.loop._workspace_lock(store):
        if (store/'CURRENT_LOCAL.json').exists() and cs._verified(store/'CURRENT_LOCAL.json')!=pointer:
            raise sub.SubmissionError('RECOVERY_LOCAL_POINTER_CONFLICT')
        raw=pr.snapshot(proot);p=store/'RECOVERED_PROJECT_ANCHOR.zip'
        if p.exists() and p.read_bytes()!=raw:raise sub.SubmissionError('RECOVERY_HISTORY_CHANGED')
        if not p.exists():p.write_bytes(raw)
        pr.import_history(store,p,sub._sha(raw))
        for rel in ['SESSION.json','LATEST.json','revisions/'+rid+'.json','bindings/'+rid+'.json']:
            src=originals/rel;row=cs._verified(src);cs._immutable(folder/rel,row)
            if (folder/rel).read_bytes()!=src.read_bytes():raise sub.SubmissionError('RECOVERY_RAW_CONTROL_BYTES')
        cs._immutable(store/'CURRENT_LOCAL.json',pointer)
        cs._immutable(store/'pointers'/(pointer['record_sha256']+'.json'),pointer)
        for obj in rec['objects']:
            if obj['role']=='engine_source':raw=sub._archive(sub._tree(ws/'source'))
            elif obj['role'].startswith('input:'):raw=_input_path(ws,obj['role'][6:]).read_bytes()
            else:continue
            if sub._sha(raw)!=obj['sha256']:raise sub.SubmissionError('RECOVERY_OBJECT_BINDING')
            cs._immutable_bytes(store/'objects'/obj['object_name'],raw)
        marker=cs._sealed({'schema_id':'IG_CHANGE_STORE_RECOVERY_V1','capture_id':rec['capture_id'],
            'project_id':pbind['project_id'],'session_sha256':session['record_sha256'],
            'revision_id':rid,'pointer_sha256':pointer['record_sha256'],'qualification':proof,
            'gap':'Original enclosing change-event tail was not exported. No historical event or missing receipt was invented.',
            'scope':'Exact sealed control records and hash-matched initial pointer; explicit fresh recovery anchor.'})
        cs._immutable(store/'RECOVERY_BOOTSTRAP.json',marker)
    return {'status':'CHANGE_STORE_RECOVERED','recovery':marker}


def make_spec(workspace,pending_bytes):
    ws=Path(workspace).resolve(strict=True);rec=sub.capture_record(ws)
    contract=revised_contract(rec,pending_bytes)
    # normalize adds derived declared_outcomes, which is not a submission field.
    contract={k:v for k,v in contract.items() if k!='declared_outcomes'}
    env=dict(rec['environment']);envnames={r['logical_name'] for r in env['artifacts']}
    env['artifacts']=[dict(r,path=str(_input_path(ws,r['logical_name']))) for r in env['artifacts']]
    inputs=[dict(r,path=str(_input_path(ws,r['logical_name']))) for r in rec['job']['input_artifacts']
            if r['logical_name'] not in envnames|{'submission_contract'}]
    identity=canonical_sha256({'original':rec['capture_id'],'preservation':contract['preservation']})
    job=rec['job']
    return {'schema_id':sub.SPEC_SCHEMA,'job_id':job['job_id']+'.PRESERVE.'+identity[:12],
        'engine_source':str(ws/'source'),'project_source':None,'question':job['question'],
        'execution':job['execution'],'resources':job['resources'],'environment':env,
        'inputs':inputs,'output_contract':contract}


def selected_binding(store,folder,rid,original,original_ws):
    selector=folder/'execution_amendments'/(rid+'.json')
    if not selector.exists():return original,original_ws
    amended=cs._verified(selector)
    if amended.get('schema_id')!=SCHEMA or amended.get('original_binding')!=original or amended.get('revision_id')!=rid:
        raise sub.SubmissionError('PRESERVATION_AMENDMENT_BINDING')
    old=sub.capture_record(original_ws);new_binding=amended['binding']
    ws=store/'execution/captures'/new_binding['capture_id'];new=sub.capture_record(ws)
    if (new_binding['job_id']!=new['job']['job_id'] or new_binding['revision_id']!=rid or
        semantics(old)!=semantics(new) or canonical_sha256(semantics(old))!=amended['semantics_sha256'] or
        rc.policy(new['result_contract'])!=amended['new_policy'] or rc.policy(old['result_contract'])!=amended['old_policy']):
        raise sub.SubmissionError('PRESERVATION_SEMANTICS_CHANGED')
    revised_contract(old,amended['new_policy']['max_pending_bytes'])
    expected=dict(amended['old_policy'],max_pending_bytes=amended['new_policy']['max_pending_bytes'])
    if expected!=amended['new_policy']:raise sub.SubmissionError('PRESERVATION_NONALLOWANCE_CHANGE')
    for rel,digest in amended['prior_attempts'].items():
        if sub._sha((original_ws/sub._relative(rel)).read_bytes())!=digest:raise sub.SubmissionError('PRESERVATION_OLD_ATTEMPT_CHANGED')
    return new_binding,ws


def rebind(store_root,change_id,revision_id,pending_bytes,reason,qualification_workspace):
    if not isinstance(reason,str) or not reason.strip():raise sub.SubmissionError('PRESERVATION_REASON_REQUIRED')
    proof=_qualified(qualification_workspace)
    store=Path(store_root).resolve(strict=True)
    with cs.loop._workspace_lock(store):
        folder,session=cs._session(store,change_id);rev=cs._revision(folder,revision_id)
        original,ws=cs._original_binding(store,folder,revision_id)
        selector=folder/'execution_amendments'/(revision_id+'.json')
        if selector.exists():
            b,w=selected_binding(store,folder,revision_id,original,ws)
            if rc.policy(sub.capture_record(w)['result_contract'])['max_pending_bytes']!=pending_bytes:
                raise sub.SubmissionError('PRESERVATION_AMENDMENT_ALREADY_BOUND')
            return {'status':sub.save_status(w)['status'],'workspace':str(w),'capture':sub.save_status(w),'reused_amendment':True}
        attempts=pretest_only(ws);old=sub.capture_record(ws)
        if cs._verified(_input_path(ws,'change_revision'))!=rev or cs._verified(_input_path(ws,'change_session'))!=session:
            raise sub.SubmissionError('PRESERVATION_CHANGE_INPUT_BINDING')
        spec=make_spec(ws,pending_bytes)
        state=sub.capture(store/'execution',spec);state=pr.adopt(store,state['workspace'])
        new=sub.capture_record(state['workspace'])
        if semantics(old)!=semantics(new):raise sub.SubmissionError('PRESERVATION_SEMANTICS_CHANGED')
        new_binding=cs._sealed({'revision_id':revision_id,'capture_id':state['capture_id'],'job_id':state['job_id']})
        amendment=cs._sealed({'schema_id':SCHEMA,'change_id':change_id,'revision_id':revision_id,
            'original_binding':original,'binding':new_binding,'prior_attempts':attempts,
            'semantics_sha256':canonical_sha256(semantics(old)),'old_policy':rc.policy(old['result_contract']),
            'new_policy':rc.policy(new['result_contract']),'reason':reason,'qualification':proof})
        cs._immutable(selector,amendment)
        cs._immutable(folder/'amendment_history'/(amendment['record_sha256']+'.json'),amendment)
        cs._event(store,'PRESERVATION_REBOUND',{'amendment_sha256':amendment['record_sha256'],
            'original_capture_id':old['capture_id'],'new_capture_id':new['capture_id']})
        return {'status':state['status'],'workspace':state['workspace'],'capture':state,'amendment':amendment}


def activation_amendment_files(folder,rid):
    p=folder/'execution_amendments'/(rid+'.json')
    return {'PRESERVATION_AMENDMENT.json':p.read_bytes()} if p.exists() else {}


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description='Qualified preservation-only recovery bootstrap; no execution command.')
    cmds=parser.add_subparsers(dest='command',required=True)
    p=cmds.add_parser('recover');p.add_argument('store_root');p.add_argument('originals');p.add_argument('qualification_workspace')
    p=cmds.add_parser('rebind');p.add_argument('store_root');p.add_argument('change_id');p.add_argument('revision_id');p.add_argument('pending_bytes',type=int);p.add_argument('reason');p.add_argument('qualification_workspace')
    args=vars(parser.parse_args(argv));command=args.pop('command')
    result=(recover_store if command=='recover' else rebind)(**args)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
