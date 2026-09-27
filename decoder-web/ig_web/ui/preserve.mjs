export function mountPreserve(token){
  const $=id=>document.getElementById(id),s={job:null,epoch:0,pending:new Map(),busy:false,reading:false};
  const node=(tag,text)=>{const n=document.createElement(tag);n.textContent=text;return n;};
  async function api(path,options={}){
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),55000);
    try{const r=await fetch(path,{...options,headers:{Authorization:'Bearer '+token(),'Content-Type':'application/json',...options.headers},credentials:'omit',cache:'no-store',redirect:'error',signal:controller.signal});
      if(!r.ok){const error=new Error('Request refused ('+r.status+')');error.refused=r.status>=400&&r.status<500;throw error;}return await r.json();
    }finally{clearTimeout(timer);}
  }
  function controls(){const pending=s.job&&s.pending.has(s.job.id);for(const b of document.querySelectorAll('[data-preserve]'))b.disabled=!s.job||s.busy||pending||!navigator.onLine;$('preserve-retry').hidden=!pending;$('preserve-retry').disabled=s.busy;}
  async function submit(operation){if(!s.job||s.busy)return;const job=s.job,epoch=s.epoch;let pending=s.pending.get(job.id);
    if(!pending){pending={operation,key:'preserve-'+crypto.randomUUID()};s.pending.set(job.id,pending);}s.busy=true;controls();
    try{const r=await api('/api/v1/jobs/'+encodeURIComponent(job.id)+'/preserve/'+pending.operation,{method:'POST',headers:{'Idempotency-Key':pending.key},body:'{}'});s.pending.delete(job.id);if(epoch===s.epoch)$('preserve-message').textContent='Request recorded: '+r.request_id+'. '+r.status+'.';}
    catch(error){if(error.refused)s.pending.delete(job.id);if(epoch===s.epoch)$('preserve-message').textContent=error.refused?error.message:'Response uncertain. Reconcile last request to retrieve the same durable request.';}
    finally{if(epoch===s.epoch){s.busy=false;controls();refresh();}}
  }
  async function download(rid,container){const epoch=s.epoch;try{const v=await api('/api/v1/requests/'+encodeURIComponent(rid)+'/download-ticket',{method:'POST',body:'{}'});if(epoch!==s.epoch)return;
      if(!/^\/downloads\/[A-Za-z0-9_-]{43}$/.test(v.url))throw new Error('Invalid download link');
      const a=node('a','Download verified archive ('+v.size_bytes+' bytes)');a.href=v.url;a.download='decoder-'+rid+'.zip';container.replaceChildren(a,node('p','SHA-256: '+v.sha256+' · Link expires in two minutes and can be used once.'));
    }catch(error){if(epoch===s.epoch)container.textContent='Download unavailable: '+error.message;}}
  async function refresh(){if(!s.job||s.reading||!token())return;const epoch=s.epoch;s.reading=true;
    try{const v=await api('/api/v1/jobs/'+encodeURIComponent(s.job.id)+'/preserve-requests');if(epoch!==s.epoch)return;const root=$('preserve-requests');root.replaceChildren();
      for(const r of v.items){const row=node('article','');row.append(node('h3',r.operation+' · '+r.status),node('p',r.id));if(r.error)row.append(node('p',r.error));
        if(r.native){const details=node('details','');details.append(node('summary','Native response'),node('pre',JSON.stringify(r.native,null,2)));row.append(details);}
        if(r.status==='finished'&&['export-full','export-slim'].includes(r.operation)){const b=node('button','Prepare verified download'),link=node('div','');b.addEventListener('click',()=>download(r.id,link));row.append(b,link);}root.append(row);}
      if(!v.items.length)root.textContent='No checkpoint or export requests recorded yet.';
    }catch(error){if(epoch===s.epoch)$('preserve-message').textContent='Unable to refresh requests. Existing information may be stale.';}
    finally{if(epoch===s.epoch)s.reading=false;}
  }
  for(const b of document.querySelectorAll('[data-preserve]'))b.addEventListener('click',()=>submit(b.dataset.preserve));
  $('preserve-retry').addEventListener('click',()=>submit());$('preserve-refresh').addEventListener('click',refresh);
  window.addEventListener('online',()=>{controls();refresh();});window.addEventListener('offline',controls);
  setInterval(()=>{if(s.job&&!document.hidden)refresh();},15000);
  function hide(){s.epoch++;s.job=null;s.busy=false;s.reading=false;controls();$('preserve-requests').replaceChildren();$('preserve-message').textContent='';}
  return {hide,reset(){s.pending.clear();hide();},open(job){hide();s.job=job;controls();refresh();}};
}
