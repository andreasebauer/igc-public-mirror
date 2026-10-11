"""Read-only checkpoint and disposition audit; not independent-host qualification."""
import copy,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'panel_o8'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def binding(row,regsha):
 if row['binding']['registration']!=regsha or digest(row['result'])!=row['result_digest']:raise ValueError('binding')
def main():
 regsha=hashlib.sha256((OUT/'REGISTRATION.json').read_bytes()).hexdigest();reg=json.loads((OUT/'REGISTRATION.json').read_text())
 for p,h in reg['pins'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 selections={a:json.loads((OUT/(a+'_SELECTION.json')).read_text()) for a in ['historical','compat']}
 for s in selections.values():
  assert s['raw']==193 and s['after_cap']==193 and s['after_dedup']==193 and not s['cap_active']
  assert len(s['selected'])==24 and len(set(s['selected']))==24
  assert set(s['twins'])<=set(s['selected'])
  assert len(s['dispositions'])==193 and all(r['kept_by_cap'] and r['dedup_representative'] for r in s['dispositions'])
  assert {r['motif'] for r in s['dispositions'] if r['selected']}==set(s['selected'])
  by_motif={r['motif']:r for r in s['dispositions']};assert len(by_motif)==193
  assert [by_motif[m]['order_key'] for m in s['selected']]==s['selected_order_keys']
  assert [by_motif[m]['native_id'] for m in s['selected']]==s['selected_native_ids']
 h,n=selections['historical'],selections['compat'];assert h['selected']==n['selected'] and h['selected_order_keys']==n['selected_order_keys']
 rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('*_BACKBONE_*.json'))];assert len(rows)==14
 for row in rows:binding(row,regsha)
 groups={}
 for row in rows:groups.setdefault(row['binding']['name'],{})[row['binding']['arm']]=row['result']
 assert len(groups)==7
 for pair in groups.values():
  h,n=pair['historical'],pair['compat'];assert h['order_key']==n['order_key'] and h['children']==n['children'] and h['edges']==n['edges'] and h['state']['skin']==n['state']['skin'] and h['state']['caps']==n['state']['caps']
 bad=copy.deepcopy(rows[0]);bad['result']['state']['caps'][0]+=1
 rejected=False
 try:binding(bad,regsha)
 except ValueError:rejected=True
 assert rejected
 result={'status':'PASS_COLD_O8_DISPOSITION_AND_BACKBONE_COMPARISON','dispositions_checked':386,'backbone_checkpoints':14,'backbone_pairs':7,'tamper_rejections':1,'beam_order_and_mandatory_twins_match':True,'native_generation_calls':0,'cap_branch_exercised':False,'independent_host':False,'full_p0_closed':False}
 (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
