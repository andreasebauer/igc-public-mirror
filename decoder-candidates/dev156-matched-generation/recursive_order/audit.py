"""Cold audit of bound O8 recursive reservation outputs; no dispatch."""
import copy,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'recursive_order'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def bind(row,regsha):
 if row['registration']!=regsha or digest(row['result'])!=row['result_digest']:raise ValueError('binding')
def main():
 reg=json.loads((OUT/'REGISTRATION.json').read_text());regsha=hashlib.sha256((OUT/'REGISTRATION.json').read_bytes()).hexdigest()
 for p,h in reg['pins'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 inputs={}
 for p in (ROOT/'matched_run2/historical/recipes').glob('*.json'):
  r=json.loads(p.read_text());inputs[r['call']['motif_id']]={'caps':r['result']['state']['caps'],'counts':[c['counts'] for c in r['result']['children']],'edges':r['result']['edges']}
 for p in (ROOT/'panel_o8').glob('historical_BACKBONE_*.json'):
  r=json.loads(p.read_text());inputs['BACKBONE:'+r['binding']['name']]={'caps':r['result']['state']['caps'],'counts':[c['counts'] for c in r['result']['children']],'edges':r['result']['edges']}
 seen=set();differences=0;rows=[]
 for p in sorted((OUT/'cases').glob('*.json')):
  row=json.loads(p.read_text());bind(row,regsha);assert digest(row['call'])==p.stem
  c,r=row['call'],row['result'];key=(c['motif'],c['type']);assert key not in seen;seen.add(key);before=inputs[c['motif']];t=c['type']
  assert r['historical_output']==r['compat_output'] and r['historical_witness']==r['compat_witness'] and r['historical_output_key']==r['compat_order_key']
  expected=list(before['caps']);expected[t]-=1;assert r['compat_output']['caps']==expected and r['compat_output']['edges']==before['edges']
  i,local,typ=r['compat_witness'];assert typ==t and local==[t,before['counts'][i][t]]
  counts=copy.deepcopy(before['counts']);counts[i][t]+=1;assert [c['counts'] for c in r['compat_output']['children']]==counts
  assert r['default_v3_route_differs']==(r['default_v3_witness']!=r['compat_witness'])
  assert r['compat_v3_output_key']!=r['historical_output_key'];differences+=r['default_v3_route_differs'];rows.append(row)
 assert seen=={(m,t) for m in inputs for t in range(7)} and len(seen)==1400
 bad=copy.deepcopy(rows[0]);bad['result']['compat_witness'][0]+=1;rejected=False
 try:bind(bad,regsha)
 except ValueError:rejected=True
 assert rejected
 result={'status':'PASS_COLD_RECURSIVE_RESERVATION_OUTPUT_AUDIT','cases':1400,'states':200,'default_v3_route_differences':differences,'unit_capacity_and_counter_changes_checked':True,'nested_ordinal_witnesses_checked':True,'tamper_rejections':1,'native_calls':0,'deeper_than_one_lift_qualified':False,'independent_host':False,'full_p0_closed':False}
 (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
