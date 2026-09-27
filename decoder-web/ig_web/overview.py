"""Read-only Jobs projection. Transport state never grants scientific authority."""
from contextlib import closing
import json
from pathlib import Path

from .native import AdapterError, canonical_hash, contained, read_json
from .results import results_observation
from .tracking import catalogue


def job_overview(settings, queue, job):
    item={'id':job.id,'name':job.name,'native_job_id':job.native_job_id,
          'question':None,'execution':'Unknown','category':'attention',
          'backup':'Unknown','last_activity':None,'state_source':'Unavailable',
          'note':'No current execution observation is available.','reason':None}
    try:
        root=contained(Path(settings.workspace_root),job.workspace,directory=True)
        reg=read_json(contained(root,'registry/'+job.native_job_id+'.json',directory=False))
        if (reg.get('schema_id')!='IG_DECODER_WORKSPACE_JOB_V1' or reg.get('job_id')!=job.native_job_id or reg.get('registration_sha256')!=canonical_hash({k:v for k,v in reg.items() if k!='registration_sha256'})):
            raise AdapterError('REGISTRATION_MISMATCH')
        question=reg.get('question',{})
        if isinstance(question,dict) and isinstance(question.get('description'),str):
            item['question']=question['description'][:1000]
        result=results_observation(settings,job)
        if result['record_status']=='PUBLISHED_RECORD':
            reported=result['reported']
            item.update(execution='Finished',category='finished',state_source='Native record',
                        note='A published completion record exists. Evidence and backup need separate checks.')
            if reported.get('evidence_status')=='REJECTED' or reported.get('status') in ('RESULT_REJECTED','VALIDATION_FAILED'):
                item.update(category='attention',execution='Finished · needs review',note='The native record reports rejected evidence or failed validation.')
        elif result['record_status']=='PENDING_CHECKPOINT_RECORD':
            item.update(execution='Checkpoint pending',category='attention',state_source='Native record',note='A completion is prepared but not published.')
        else:
            item.update(execution='Prepared',category='all',state_source='Registered job',note='Registered job found. Readiness and backup have not been checked.')
    except (AdapterError,KeyError,TypeError,ValueError) as exc:
        item['reason']=exc.code if isinstance(exc,AdapterError) else 'RECORD_UNAVAILABLE'
        return item
    if queue:
        with closing(queue.connect()) as db:
            # Match native identity too, so aliases see the same transport state.
            rows=db.execute("SELECT * FROM requests WHERE (operation='run' OR (operation='pause' AND status!='finished')) AND (target=? OR run_key=?) ORDER BY created DESC LIMIT 1",(job.id,canonical_hash([str(root),job.native_job_id]))).fetchall()
        for row in rows:
            cmd=json.loads(row['command']);argv=cmd['argv']
            match=row['target']==job.id or (row['operation']=='run' and len(argv)>=2 and argv[-2:]==[str(root),job.native_job_id])
            if not match: continue
            item['last_activity']=row['updated']
            state=row['status']
            if state in ('queued','dispatching','running'):
                item.update(execution={'queued':'Queued','dispatching':'Dispatching','running':'Run requested'}[state] if row['operation']=='run' else 'Pause requested',
                            category='active',state_source='Web request',note='Request state only. Open the job to check native execution.')
            elif state=='finished' and result['record_status']=='ABSENT':
                item.update(execution='Request finished · check result',category='attention',state_source='Web request',note='The request ended, but no local published result was found.')
            elif state in ('needs_reconciliation','interrupted','refused'):
                item.update(execution='Needs reconciliation' if state!='refused' else 'Request refused',category='attention',state_source='Web request',note='Inspect the retained request before taking further action.',reason=row['error'])
            break
    return item


def overview(settings,queue,query='',filter_by='all',offset=0,limit=20):
    items=[job_overview(settings,queue,j) for j in catalogue(settings,queue).jobs]
    counts={key:sum(x['category']==key for x in items) for key in ('active','attention','finished')}
    counts['all']=len(items)
    needle=query.casefold().strip()
    selected=[x for x in items if (filter_by=='all' or x['category']==filter_by) and
              (not needle or needle in (x['name']+' '+x['native_job_id']+' '+x['id']).casefold())]
    priority={'active':0,'attention':1,'all':2,'finished':2}
    selected.sort(key=lambda x:(priority[x['category']],-(x['last_activity'] or 0),x['id']))
    return {'items':selected[offset:offset+limit],'total':len(selected),'offset':offset,'limit':limit,'counts':counts}
