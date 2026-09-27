export function mountSaves(token){
  const $=id=>document.getElementById(id),s={job:null,epoch:0,busy:false,reading:false,ready:false,active:false,pending:new Map()};
  const node=(tag,text)=>{const n=document.createElement(tag);n.textContent=text;return n;};
  async function api(path,options={}){const c=new AbortController(),timer=setTimeout(()=>c.abort(),55000);
    try{const r=await fetch(path,{...options,headers:{Authorization:'Bearer '+token(),'Content-Type':'application/json',...options.headers},credentials:'omit',redirect:'error',cache:'no-store',signal:c.signal});if(!r.ok){const e=new Error('Request refused ('+r.status+')');e.definite=r.status>=400&&r.status<500;throw e;}return await r.json();}finally{clearTimeout(timer);}}
  function controls(){const pending=s.job&&s.pending.has(s.job.id);$('save-submit').disabled=!s.job||!s.ready||s.busy||s.active||pending||!navigator.onLine;$('save-reconcile').hidden=!pending;$('save-reconcile').disabled=s.busy;}
  async function submit(){if(!s.job||s.busy)return;const job=s.job,epoch=s.epoch;let key=s.pending.get(job.id);if(!key){key='drive-'+crypto.randomUUID();s.pending.set(job.id,key);}s.busy=true;controls();
    try{const r=await api('/api/v1/jobs/'+encodeURIComponent(job.id)+'/save-requests',{method:'POST',headers:{'Idempotency-Key':key},body:'{}'});s.pending.delete(job.id);if(epoch===s.epoch)$('save-message').textContent='Save request recorded: '+r.request_id;}
    catch(e){if(e.definite)s.pending.delete(job.id);if(epoch===s.epoch)$('save-message').textContent=e.definite?e.message:'Response uncertain. Reconcile last request to recover the same request.';}
    finally{if(epoch===s.epoch){s.busy=false;controls();refresh();}}}
  async function retry(rid){if(s.busy)return;const epoch=s.epoch;s.busy=true;controls();try{await api('/api/v1/save-requests/'+encodeURIComponent(rid)+'/retry',{method:'POST',body:'{}'});if(epoch===s.epoch)$('save-message').textContent='The same save request is queued for reconciliation.';}catch(e){if(epoch===s.epoch)$('save-message').textContent=e.message;}finally{if(epoch===s.epoch){s.busy=false;controls();refresh();}}}
  async function refresh(){if(!s.job||s.reading||!token())return;s.reading=true;const epoch=s.epoch;
    try{const v=await api('/api/v1/jobs/'+encodeURIComponent(s.job.id)+'/save-requests');if(epoch!==s.epoch)return;s.ready=v.transport!=='UNCONFIGURED';s.active=v.items.some(r=>r.status!=='finished');$('save-transport').textContent=s.ready?'Host Drive transport configured. A save is confirmed only after native verification.':'Host Drive authorization is not configured. Required saves remain pending.';const root=$('save-requests');root.replaceChildren();
      for(const r of v.items){const row=node('article','');row.append(node('h3','Drive save · '+r.status),node('p',r.id));if(r.error)row.append(node('p',r.error));row.append(node('p','Native obligations confirmed by this request: '+r.confirmed_obligations));
        for(const o of r.objects)row.append(node('p',o.sha256.slice(0,12)+'… · '+o.size_bytes+' bytes · '+o.phase));
        if(r.native_remaining)row.append(node('p','Native obligations still pending: capture '+r.native_remaining.capture+', checkpoint '+r.native_remaining.outbox));
        if(['failed','needs_reconciliation'].includes(r.status)){const b=node('button','Retry / reconcile this save');b.disabled=s.busy;b.addEventListener('click',()=>retry(r.id));row.append(b);}root.append(row);}
    }catch(e){if(epoch===s.epoch){s.ready=false;$('save-message').textContent='Transfer status unavailable; displayed information may be stale.';}}
    finally{if(epoch===s.epoch){s.reading=false;controls();}}}
  $('save-submit').addEventListener('click',submit);$('save-reconcile').addEventListener('click',submit);$('save-refresh').addEventListener('click',refresh);
  window.addEventListener('online',refresh);window.addEventListener('offline',controls);setInterval(()=>{if(s.job&&!document.hidden)refresh();},15000);
  function hide(){s.epoch++;s.job=null;s.busy=false;s.reading=false;s.ready=false;s.active=false;controls();$('save-requests').replaceChildren();$('save-message').textContent='';$('save-transport').textContent='';}
  return {hide,reset(){s.pending.clear();hide();},open(job){hide();s.job=job;refresh();}};
}
