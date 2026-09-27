import {mountPrepare} from './prepare.mjs';
export function validateOverview(value) {
  if (!value || !Array.isArray(value.items) || !Number.isInteger(value.total) || value.total < 0 || !value.counts) throw new Error('Invalid job list');
  for (const row of value.items) {
    if (!row || !['id','name','native_job_id','execution','category','backup','state_source','note'].every(k => typeof row[k] === 'string') || !['all','active','attention','finished'].includes(row.category)) throw new Error('Invalid job record');
  }
  return value;
}
export function nativeExecution(observation, jobId) {
  if (!observation || observation.classification !== 'RESPONSE' || !observation.native) return 'Unknown';
  const native = observation.native, record = native.last_record;
  if (!record || record.job_id !== jobId) return native.workspace_locked_now ? 'Workspace busy · job unconfirmed' : 'No current job observation';
  if (record.status === 'RUNNING') return native.workspace_locked_now ? 'Running reported · workspace busy' : 'Needs reconciliation';
  if (native.workspace_locked_now) return 'Workspace busy · job unconfirmed';
  return ({PAUSED:'Paused',COMPLETED:'Finished',RESULT_REJECTED:'Finished · needs review',VALIDATION_FAILED:'Finished · needs review',COMPLETION_PENDING_CHECKPOINT:'Checkpoint pending'})[record.status] || 'Unknown';
}
export function backupState(value) {
  if (!value || value.capture?.classification !== 'RESPONSE' || value.outbox?.classification !== 'RESPONSE') return 'Unknown';
  const capture=value.capture.native, outbox=value.outbox.native;
  if (!capture || !outbox || !Array.isArray(capture.pending_objects) || !Array.isArray(outbox.pending_objects) || !Array.isArray(outbox.pending_checkpoints)) return 'Unknown';
  if (capture.pending_objects.length || outbox.pending_objects.length || outbox.pending_checkpoints.length) return 'Backup pending';
  return outbox.status === 'PRESERVED' ? 'Native reports preserved' : 'Unknown';
}
export function timeLabel(seconds) {
  return typeof seconds === 'number' && Number.isFinite(seconds) ? new Date(seconds*1000).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : 'Activity time unknown';
}

if (typeof document !== 'undefined') {
  const $=id=>document.getElementById(id);
  const state={token:'',filter:'all',query:'',offset:0,limit:20,cached:null,updated:null,epoch:0,detailEpoch:0,controller:null,opener:null};
  const text=(tag,value,className)=>{const el=document.createElement(tag);el.textContent=value;if(className)el.className=className;return el;};
  function message(value){$('message').textContent=value;$('message').hidden=!value;}
  function connection(ok){$('connection').textContent=ok?'Connected':'Connection unavailable · cached values may be stale';$('connection').classList.toggle('stale',!ok);}
  async function api(path, signal) {
    const response=await fetch('/api/v1/'+path,{headers:{Authorization:'Bearer '+state.token},cache:'no-store',credentials:'omit',signal,redirect:'error'});
    if (response.status===401) throw new Error('Access token was not accepted. Disconnect and reconnect with the correct token.');
    if (!response.ok) throw new Error('The server could not provide this view. Retained jobs have not been changed.');
    return response.json();
  }
  function empty(title,description){const box=text('div','', 'empty');box.append(text('h2',title),text('p',description));$('jobs-list').replaceChildren(box);}
  function render(value) {
    const list=$('jobs-list');list.replaceChildren();
    for(const job of value.items) {
      const button=text('button','', 'job-row '+job.category);button.type='button';button.setAttribute('aria-label','Open '+job.name);
      const identity=text('span','', 'job-identity');identity.append(text('span',job.name,'job-name'),text('span',job.question || 'Question unavailable','job-question'),text('span',job.native_job_id,'job-id'));
      const status=text('span','', 'job-status');status.append(text('span',job.execution,'job-state'),text('span',job.state_source,'job-meta'),text('span',timeLabel(job.last_activity),'job-meta'));
      const backup=text('span','', 'job-backup');backup.append(text('span','Backup','backup-label'),text('span',job.backup,'backup-value'));
      const chevron=text('span','›','chevron');chevron.setAttribute('aria-hidden','true');button.append(identity,status,backup,chevron);
      button.addEventListener('click',()=>openJob(job,button));list.append(button);
    }
    if (!value.items.length) empty(value.counts.all===0?'No jobs yet':'No matching jobs',value.counts.all===0?'Choose New job to review and prepare an available task.':'Try another filter or search by name or job ID.');
    for(const key of ['all','active','attention','finished']) $('count-'+key).textContent=String(value.counts[key] ?? '—');
    $('page-label').textContent=value.total?`${state.offset+1}–${state.offset+value.items.length} of ${value.total} jobs`:'0 jobs';
    $('previous').disabled=state.offset===0;$('next').disabled=state.offset+state.limit>=value.total;
  }
  async function refresh() {
    if (!state.token) return;
    const epoch=++state.epoch;state.controller?.abort();state.controller=new AbortController();const controller=state.controller;const timer=setTimeout(()=>controller.abort(),25000);
    $('jobs-list').setAttribute('aria-busy','true');$('refresh').disabled=true;
    if (!state.cached) empty('Loading jobs…','Reading the registered job list.');
    try {
      const params=new URLSearchParams({query:state.query,filter_by:state.filter,offset:String(state.offset),limit:String(state.limit)});
      const value=validateOverview(await api('job-overview?'+params,controller.signal));
      if(epoch!==state.epoch || !state.token)return;
      state.cached=value;state.updated=new Date();render(value);connection(true);message('');
      $('updated').textContent='Updated '+state.updated.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit',second:'2-digit'});
      $('view-description').textContent='Active jobs and items needing attention appear first.';
    } catch(error) {
      if(epoch!==state.epoch || !state.token)return;
      connection(false);message((error.name==='AbortError'?'The server did not respond in time.':error.message)+' Execution may still continue.');
      if(!state.cached)empty('Unable to load jobs','This is a connection or read error, not an empty job list.');
      else $('view-description').textContent='Showing the last successful list. Search, filter and page changes have not been confirmed.';
      $('previous').disabled=true;$('next').disabled=true;
      if(!$('job-detail').hidden)$('detail-note').textContent='This observation may be stale. Refresh the job when the connection returns.';
    } finally {
      clearTimeout(timer);if(epoch===state.epoch){$('jobs-list').setAttribute('aria-busy','false');$('refresh').disabled=false;}
    }
  }
  async function openJob(job,opener) {
    state.opener=opener;const epoch=++state.detailEpoch;const token=state.token;
    $('job-detail').hidden=false;$('detail-heading').textContent=job.name;$('detail-question').textContent=job.question || 'Question unavailable';
    $('detail-note').textContent='Reading native execution and backup observations…';$('detail-facts').replaceChildren();$('detail-raw').textContent='';$('job-detail').focus();$('job-detail').scrollIntoView({behavior:'smooth',block:'start'});
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),25000);
    try {
      const id=encodeURIComponent(job.id);const [status,preservation]=await Promise.all([api('jobs/'+id+'/status',controller.signal),api('jobs/'+id+'/preservation',controller.signal)]);
      if(epoch!==state.detailEpoch || token!==state.token)return;
      const facts=[['Execution',nativeExecution(status,job.native_job_id)],['Backup',backupState(preservation)],['Job ID',job.native_job_id],['Observed',status.observed_at ? new Date(status.observed_at).toLocaleString() : 'Unknown']];
      for(const [label,value] of facts){const div=document.createElement('div');div.append(text('dt',label),text('dd',value));$('detail-facts').append(div);}
      $('detail-note').textContent=status.classification==='RESPONSE'?'Native observations are separate from scientific outcome and evidence verification.':'Native observation was refused or could not be parsed. See Technical details.';
      $('detail-raw').textContent=JSON.stringify({status,preservation},null,2);
    } catch(error) {
      if(epoch!==state.detailEpoch || token!==state.token)return;
      $('detail-note').textContent='Unable to read this job. Its execution may still continue. Retry by reopening it.';
    } finally {clearTimeout(timer);}
  }
  const prepare=mountPrepare(()=>state.token,refresh);
  $('connect-form').addEventListener('submit',event=>{event.preventDefault();state.token=$('token').value;$('token').value='';$('connect-panel').hidden=true;$('jobs-panel').hidden=false;$('disconnect').hidden=false;refresh();});
  $('disconnect').addEventListener('click',()=>{prepare.reset();state.token='';state.epoch++;state.detailEpoch++;state.controller?.abort();state.cached=null;state.updated=null;state.offset=0;$('connect-panel').hidden=false;$('jobs-panel').hidden=true;$('job-detail').hidden=true;$('disconnect').hidden=true;$('jobs-list').replaceChildren();$('detail-raw').textContent='';message('');$('token').focus();});
  $('refresh').addEventListener('click',refresh);
  document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{state.filter=button.dataset.filter;state.offset=0;document.querySelectorAll('[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));refresh();}));
  let searchTimer; $('search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.query=$('search').value;state.offset=0;refresh();},250);});
  $('previous').addEventListener('click',()=>{state.offset=Math.max(0,state.offset-state.limit);refresh();});$('next').addEventListener('click',()=>{state.offset+=state.limit;refresh();});
  $('close-detail').addEventListener('click',()=>{state.detailEpoch++;$('job-detail').hidden=true;state.opener?.focus();});
  window.addEventListener('online',refresh);window.addEventListener('offline',()=>{connection(false);message('Connection lost. Displayed values may be stale; execution may still continue.');});
  setInterval(()=>{if(!document.hidden && state.token)refresh();},30000);
}
