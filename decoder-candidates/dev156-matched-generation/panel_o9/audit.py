"""Cold checkpoint audit: hashes, disposition coverage and cross-arm resources."""
import pathlib,json,hashlib,copy
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'panel_o9'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(row):assert digest(row['result'])==row['result_digest']
def main():
 reg=json.loads((OUT/'REGISTRATION.json').read_text())
 for p,h in reg['pins'].items():assert sha(ROOT/p)==h,p
 h,n=[json.loads((OUT/(a+'_SELECTION.json')).read_text()) for a in ['historical','compat']]
 assert len(h['dispositions'])==len(n['dispositions'])==193
 assert h['selected']==n['selected'] and h['selected_order_keys']==n['selected_order_keys'] and len(n['selected'])==24
 assert all(m in n['selected'] for m in ['TWIN:A','TWIN:B'])
 for x,y in zip(h['dispositions'],n['dispositions']):
  assert {k:v for k,v in x.items() if k!='native_id'}=={k:v for k,v in y.items() if k!='native_id'}
  assert x['native_id']!=y['native_id']
 names=[]
 for p in sorted(OUT.glob('historical_BACKBONE_*.json')):
  a=json.loads(p.read_text());b=json.loads((OUT/p.name.replace('historical_','compat_')).read_text());check(a);check(b)
  assert a['binding']['registration']==b['binding']['registration']==sha(OUT/'REGISTRATION.json')
  x,y=a['result'],b['result'];assert x['physical']==y['physical'] and x['wire']==y['wire'] and x['order_key']==y['order_key'];assert x['state']['construction']!=y['state']['construction'];names.append(a['binding']['name'])
 assert set(names)=={'P4','S4','C4','K4','C5','TWIN_A','TWIN_B'}
 bad=copy.deepcopy(a);bad['result']['wire']['children'][0]['children'][0]['counts'][0]+=1
 try:check(bad)
 except AssertionError:pass
 else:raise AssertionError('tamper accepted')
 result={'status':'PASS','dispositions':386,'backbone_checkpoints':14,'tamper_rejected':1,'scope':'Cold same-host saved-checkpoint digest and cross-arm audit; no independent-host or independent skin materialization claim.'}
 (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
