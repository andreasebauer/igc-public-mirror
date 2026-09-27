import {mountPreserve} from './preserve.mjs';
export function presentResults(value){
  if(!value || !value.facts || !value.save || !value.job || typeof value.observed_at!=='number')throw new Error('INVALID_RESULTS_VIEW');
  const f=value.facts,outcome=f.scientific_outcome?.value;
  const scientific=outcome===null || outcome===undefined?'Not reported':typeof outcome==='string'?outcome:JSON.stringify(outcome);
  const evidence=({VERIFIED:'Native reports verified',REJECTED:'Native reports rejected',PENDING:'Native verification pending',UNKNOWN:'Unknown'})[f.evidence?.reported_status] || 'Unknown';
  const save=({NATIVE_REPORTS_PRESERVED:'Native reports preserved',PENDING:'Save pending',NO_CHECKPOINT:'No checkpoint recorded',UNKNOWN:'Unknown'})[value.save.status] || 'Unknown';
  const publication=({PUBLISHED_RECORD:'Published completion record',PENDING_CHECKPOINT_RECORD:'Partial · checkpoint publication pending',ABSENT:'No completion record yet',UNAVAILABLE:'Result record unavailable or invalid'})[f.record_status] || 'Result record unknown';
  return {scientific,evidence,save,publication};
}
export function mountResults(token){
  const preserve=mountPreserve(token);
  const $=id=>document.getElementById(id),s={job:null,epoch:0,controller:null,busy:false};
  const node=(tag,value)=>{const el=document.createElement(tag);el.textContent=value;return el;};
  function clear(){for(const id of ['results-outcome','results-evidence','results-save'])$(id).textContent='Unknown';$('results-publication').textContent='Loading result observations…';$('results-summary').replaceChildren();$('results-raw').textContent='';$('results-observed').textContent='';$('results-error').hidden=true;}
  async function refresh(){if(!s.job || !token() || s.busy)return;const epoch=s.epoch;s.busy=true;$('results-refresh').disabled=true;s.controller=new AbortController();const controller=s.controller,timer=setTimeout(()=>controller.abort(),55000);
    try{const response=await fetch('/api/v1/jobs/'+encodeURIComponent(s.job.id)+'/results-view',{headers:{Authorization:'Bearer '+token()},cache:'no-store',credentials:'omit',redirect:'error',signal:controller.signal});if(!response.ok)throw new Error('SERVER_READ_FAILED');const value=await response.json(),display=presentResults(value);if(epoch!==s.epoch)return;if(value.job.id!==s.job.id)throw new Error('RESULT_JOB_MISMATCH');
      $('results-outcome').textContent=display.scientific;$('results-evidence').textContent=display.evidence;$('results-save').textContent=display.save;$('results-publication').textContent=display.publication;$('results-observed').textContent='Observed '+new Date(value.observed_at*1000).toLocaleString();$('results-error').hidden=true;if(value.job.question)$('run-question').textContent=value.job.question;
      const summary=$('results-summary');summary.replaceChildren();const rows=[['Job',value.job.native_job_id],['Execution reported',value.facts.execution_reported ?? 'Unknown'],['Completion reported',value.facts.completion_reported ?? 'Unknown'],['Capture obligations pending',value.save.capture_pending ?? 'Unknown'],['Checkpoint objects pending',value.save.checkpoint_objects_pending ?? 'Unknown'],['Checkpoints pending',value.save.checkpoints_pending ?? 'Unknown']];for(const [label,v] of rows){const d=node('div','');d.append(node('dt',label),node('dd',String(v)));summary.append(d);}$('results-raw').textContent=JSON.stringify(value,null,2);
    }catch(error){if(epoch!==s.epoch)return;$('results-error').hidden=false;$('results-error').textContent='Unable to refresh results. Displayed observations may be stale; no new outcome, evidence or save status is confirmed. '+error.message;}finally{clearTimeout(timer);if(epoch===s.epoch){s.busy=false;$('results-refresh').disabled=false;}}
  }
  $('results-refresh').addEventListener('click',refresh);
  window.addEventListener('offline',()=>{if(s.job){$('results-error').hidden=false;$('results-error').textContent='Offline. Displayed observations may be stale.';}});window.addEventListener('online',refresh);
  setInterval(()=>{if(s.job && !document.hidden)refresh();},15000);
  function hide(){preserve.hide();s.epoch++;s.job=null;s.busy=false;s.controller?.abort();$('results-content').hidden=true;}
  return {hide,reset(){preserve.reset();hide();clear();},open(job){hide();clear();s.job=job;preserve.open(job);$('results-content').hidden=false;$('results-content').focus();refresh();}};
}
