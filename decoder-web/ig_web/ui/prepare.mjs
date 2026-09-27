// Tokens stay in the parent tab's memory; all durable reviews live on the server.
export function mountPrepare(token,refreshJobs,onCaptured) {
  const $=id=>document.getElementById(id), state={epoch:0,draft:null,key:null,busy:false};
  const node=(tag,value)=>{const el=document.createElement(tag);el.textContent=value;return el;};
  const status=value=>{$('prepare-status').textContent=value;};
  async function request(path,method='GET',body,key) {
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),25000);
    try {
      const response=await fetch('/api/v1/'+path,{method,headers:{Authorization:'Bearer '+token(),...(body?{'Content-Type':'application/json'}:{}),...(key?{'Idempotency-Key':key}:{})},body:body?JSON.stringify(body):undefined,cache:'no-store',credentials:'omit',redirect:'error',signal:controller.signal});
      const value=await response.json();
      if(!response.ok)throw new Error(value.error?.code || 'REQUEST_UNAVAILABLE');
      return value;
    } finally {clearTimeout(timer);}
  }
  function controls(){const disabled=state.busy || !navigator.onLine;$('prepare-task').disabled=disabled;$('prepare-review').disabled=disabled || !$('prepare-task').value;$('prepare-submit').disabled=disabled || !state.draft?.review.can_prepare || !!state.draft?.request;$('prepare-check').disabled=disabled;}
  function show(draft){
    state.draft=draft;const r=draft.review,summary=$('prepare-summary');summary.replaceChildren();
    const section=(title,lines)=>{const s=node('section','');s.append(node('h2',title));for(const line of lines)s.append(node('p',line));summary.append(s);};
    section(r.name,[r.question.description,'Outcomes: '+r.question.outcomes.join(', '),'Stopping rule: '+r.question.stopping_rule]);
    section('Inputs',r.inputs.length?r.inputs.map(i=>i.role+' — '+i.name+' · '+(i.verified?'Bytes verified at review':'Unavailable or unverified')):['This task declares no file inputs.']);
    section('Resources',Object.entries(r.resources).map(([k,v])=>k.replaceAll('_',' ')+': '+v));
    section('Expected outputs',[r.expected_outputs.length?JSON.stringify(r.expected_outputs):'No required artifact list declared.',r.configuration_note]);
    const details=node('details','');details.append(node('summary','Review identity'),node('pre',JSON.stringify({draft_id:draft.id,review_sha256:draft.review_sha256,specification_sha256:r.specification_sha256,inputs:r.inputs},null,2)));summary.append(details);
    $('prepare-submit').hidden=!!draft.request;$('prepare-check').hidden=!draft.request;
    status(draft.request?'Capture request saved · '+draft.request.status:r.can_prepare?'Review saved. Prepare creates a capture; it does not start execution.':'Review saved. Required inputs are unavailable or unverified; preparation is blocked.');controls();
  }
  async function perform(action){if(state.busy)return;const epoch=state.epoch;state.busy=true;controls();try{await action(epoch);}catch(error){if(epoch===state.epoch)status('Unable to confirm: '+error.message+'. Reopen the saved review or retry the same action to reconcile.');}finally{if(epoch===state.epoch){state.busy=false;controls();}}}
  async function saved(epoch){const value=await request('drafts');if(epoch!==state.epoch)return;$('prepare-drafts').replaceChildren();for(const draft of value.items){const b=node('button',draft.review.name+' · '+(draft.request?.status || 'Review saved'));b.addEventListener('click',()=>perform(async e=>{const current=await request('drafts/'+draft.id);if(e===state.epoch)show(current);}));$('prepare-drafts').append(b);}}
  $('new-job').addEventListener('click',()=>{state.epoch++;state.busy=false;state.key=null;state.draft=null;$('prepare-summary').replaceChildren();$('prepare-submit').hidden=true;$('prepare-check').hidden=true;$('jobs-panel').hidden=true;$('job-detail').hidden=true;$('prepare-panel').hidden=false;$('prepare-panel').focus();perform(async epoch=>{status('Loading available task versions…');const value=await request('tasks');if(epoch!==state.epoch)return;$('prepare-task').replaceChildren();for(const task of value.items){const option=node('option',task.name+' · '+task.specification_sha256.slice(0,12));option.value=task.id;$('prepare-task').append(option);}status(value.items.length?'Choose a task version to save and review.':'No task versions are available. A server task definition is required.');await saved(epoch);});});
  $('prepare-task').addEventListener('change',()=>{state.key=null;state.draft=null;$('prepare-summary').replaceChildren();$('prepare-submit').hidden=true;$('prepare-check').hidden=true;controls();});
  $('prepare-review').addEventListener('click',()=>perform(async epoch=>{state.key ||= 'review-'+crypto.randomUUID();const draft=await request('drafts','POST',{task_id:$('prepare-task').value},state.key);if(epoch!==state.epoch)return;show(draft);await saved(epoch);}));
  $('prepare-submit').addEventListener('click',()=>perform(async epoch=>{const draft=state.draft;const accepted=await request('drafts/'+draft.id+'/capture','POST',{review_sha256:draft.review_sha256});if(epoch!==state.epoch)return;show({...draft,request:accepted});await saved(epoch);}));
  $('prepare-check').addEventListener('click',()=>perform(async epoch=>{const draft=await request('drafts/'+state.draft.id);const result=draft.request?await request('requests/'+draft.request.request_id):null;if(epoch!==state.epoch)return;show(draft);if(result?.status==='finished'){const job=await request('requests/'+draft.request.request_id+'/job');if(epoch===state.epoch){state.epoch++;state.busy=false;onCaptured(job);}return;}if(result){status('Capture request: '+result.status+'. '+(result.status==='finished'?'Capture completed. Backup readiness has not been verified; return to Jobs to inspect it.':'No execution has been requested by Prepare.'));const details=node('details','');details.append(node('summary','Capture response'),node('pre',JSON.stringify(result,null,2)));$('prepare-summary').append(details);}await refreshJobs();}));
  $('prepare-close').addEventListener('click',()=>{state.epoch++;state.busy=false;$('prepare-panel').hidden=true;$('jobs-panel').hidden=false;refreshJobs();});
  window.addEventListener('offline',controls);window.addEventListener('online',controls);
  return {reset(){state.epoch++;state.busy=false;state.draft=null;state.key=null;$('prepare-panel').hidden=true;$('prepare-summary').replaceChildren();$('prepare-drafts').replaceChildren();$('prepare-task').replaceChildren();$('prepare-submit').hidden=true;$('prepare-check').hidden=true;}};
}
