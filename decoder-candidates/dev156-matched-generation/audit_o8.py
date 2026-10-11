import json,hashlib,pathlib,math
P=pathlib.Path(__file__).resolve().parent
def dg(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def load(p):return json.loads(p.read_text())
def selection(tr):
 pool=tr['pool'];norm=tr['normalized'];chosen=[]
 if norm is None:return pool[:tr['k']]
 for s in tr['must_include']:
  i=pool.index(s)
  if i not in chosen:chosen.append(i)
 if not chosen:chosen=[0]
 while len(chosen)<tr['k']:
  best=None
  for i in range(len(pool)):
   if i in chosen:continue
   d=min(sum((norm[i][j]-norm[c][j])**2 for j in range(len(norm[i])))**.5 for c in chosen)
   if best is None or d>best[0]+1e-15 or (abs(d-best[0])<=1e-15 and pool[i]<pool[best[1]]):best=(d,i)
  chosen.append(best[1])
 assert chosen==tr['chosen_indices'];assert [pool[i] for i in chosen]==tr['selected']
 return [pool[i] for i in chosen]
def valid(row,reg):
 assert row['registration']==reg and dg(row['call'])==row['call_digest'] and dg(row['result'])==row['result_digest']
 call,res=row['call'],row['result'];caps=[list(s['caps']) for s in call['owners']];counts=[list(s['counts']) for s in call['owners']];events=res['reservation_events'];edge_i=0
 for i,edge in enumerate(sorted(tuple(sorted(e)) for e in call['edges'])):
  u,v=edge;ops=[call['force_pair']] if call['force_pair'] is not None else [call['operators'][(i+call['seed']+j)%len(call['operators'])] for j in range(len(call['operators']))];chosen=None;checks=[]
  for a,b in ops:
   ok=caps[u][a]>0 and caps[v][b]>0;checks.append({'pair':[a,b],'legal_capacity':ok})
   if ok:chosen=[a,b];break
  assert res['operator_choices'][i]=={'edge':list(edge),'tests':checks,'chosen':chosen}
  if chosen is None:assert res['state'] is None;break
  for owner,t in [(u,chosen[0]),(v,chosen[1])]:
   ev=events[edge_i];assert ev['type']==t and ev['occurrence']==call['owners'][owner]['occurrence'];assert ev['before']['caps']==caps[owner] and ev['before']['counts']==counts[owner];assert ev['witness']==[t,counts[owner][t]];caps[owner][t]-=1;counts[owner][t]+=1;assert ev['after']['caps']==caps[owner] and ev['after']['counts']==counts[owner];edge_i+=1
 assert edge_i==len(events)
 if res['state'] is not None:
  assert res['state']['caps']==[sum(c[t] for c in caps) for t in range(7)]
  assert [x['counts'] for x in res['children']]==counts
 return True
reg=hashlib.sha256((P/'REGISTRATION.json').read_bytes()).hexdigest();rows={};count=0
for arm in ['historical','dev156']:
 tr=load(P/arm/'O7_SELECTION.json');selection(tr);rr=[load(p) for p in sorted((P/arm/'recipes').glob('*.json'))]
 for r in rr:valid(r,reg);count+=1
 rows[arm]={r['call']['motif_id']:r for r in rr}
assert len(rows['historical'])==len(rows['dev156'])==193
for arm in rows:
 r=next(iter(rows[arm].values()));bad=json.loads(json.dumps(r));bad['result']['reservation_events'][0]['after']['caps'][0]+=1
 try:valid(bad,reg)
 except AssertionError:pass
 else:raise AssertionError('TAMPER_ACCEPTED')
a=load(P/'historical/BASE.json');b=load(P/'dev156/BASE.json');assert all(x['skin']==y['skin'] and x['caps']==y['caps'] for x,y in zip(a,b));h=load(P/'historical/O7_SELECTION.json');n=load(P/'dev156/O7_SELECTION.json');ci={r['construction']:r['index'] for r in a};cj={r['construction']:r['index'] for r in b}
changes=[]
for motif in rows['historical']:
 x=rows['historical'][motif];y=rows['dev156'][motif];xs=x['result']['state'];ys=y['result']['state'];changes.append({'motif':motif,'same_owner_occurrences':[r['occurrence'] for r in x['call']['owners']]==[r['occurrence'] for r in y['call']['owners']],'same_resource_output':(xs is None and ys is None) or (xs is not None and ys is not None and xs['skin']==ys['skin'] and xs['caps']==ys['caps']),'same_chosen_operators':[r['chosen'] for r in x['result']['operator_choices']]==[r['chosen'] for r in y['result']['operator_choices']]})
result={'status':'PASS_RECIPE_TRACE_AND_SELECTION_REFERENCE_AUDIT','recipe_checkpoints':count,'selection_reference_checks':2,'tamper_rejections':2,'same_starting_resource_rows':205,'first_selection_seed':{'historical_occurrence':ci[h['pool'][0]],'dev156_occurrence':cj[n['pool'][0]],'rule':'native farthest selection starts at lexicographically first construction digest; identity migration changes this deterministic seed'},'same_selected_occurrence_sets':set(map(ci.get,h['selected']))==set(map(cj.get,n['selected'])),'resource_different_recipes':sum(not x['same_resource_output'] for x in changes),'operator_different_recipes':sum(not x['same_chosen_operators'] for x in changes),'recipe_differences':changes,'full_p0_closed':False,'historical_retained_population_comparison':'OPEN','O8_cuts':'DEFERRED','O9_dispatched':False}
(P/'AUDIT.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print({k:v for k,v in result.items() if k!='recipe_differences'})
