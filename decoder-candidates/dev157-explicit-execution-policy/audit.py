"""Cold source/policy/checkpoint audit for the candidate-native chain recovery."""
import pathlib,json,hashlib,copy
R=pathlib.Path(__file__).resolve().parents[1];P=R/'integration157'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def check(row):assert digest(row['result'])==row['result_digest']
def main():
 reg=json.loads((P/'REGISTRATION.json').read_text())
 for p,h in reg['pins'].items():assert sha(R/p)==h,p
 cps=sorted((P/'checkpoints').glob('*.json'));assert len(cps)==193;events=0
 for cp in cps:
  row=json.loads(cp.read_text());check(row);assert row['binding']['registration']==sha(P/'REGISTRATION.json') and row['binding']['policy']=='HISTORICAL_COORDINATE_V1';assert row['binding']['candidate_scanner']==sha(R/'decoder157/infinity_grid/regime_scanner.py')
  expected=json.loads((R/'restored/o9_qualification/chains'/cp.name).read_text());assert len(row['result']['steps'])==14
  for got,want in zip(row['result']['steps'],expected['steps']):assert got=={'type':want['type'],'witness':want['compat_witness'],'construction':want['compat_v3_key'],'execution_order_key':want['compat_order_key']};events+=1
  assert row['result']['final_wire_digest']==digest(expected['final_compat_wire']) and row['result']['native_policy_preserved']
 bad=copy.deepcopy(row);bad['result']['steps'][0]['witness'][0]+=1
 try:check(bad)
 except AssertionError:pass
 else:raise AssertionError('tamper accepted')
 result={'status':'PASS','checkpoints':193,'native_steps':events,'tamper_rejected':1,'scope':'same-host cold source/policy/result binding and expected-trace audit; no independent skin algorithm'};(P/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
