"""Independent saved-manifest class completeness and proof-scope audit."""
from pathlib import Path
import json,gzip,hashlib,collections
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pre=json.loads((B/'PREREGISTRATION.json').read_text());r=json.loads((B/'REVIEW.json').read_text());m=json.loads(gzip.decompress((B/'COLLISION_MEMBERS.json.gz').read_bytes()));p=json.loads((B/'PILOT_CASES.json').read_text())
 assert sha(B/'review_coverage.py')==pre['registered_script_sha256']
 assert sha(B/'COLLISION_MEMBERS.json.gz')==r['collision_members_sha256'] and sha(B/'PILOT_CASES.json')==r['pilot_cases_sha256']
 by={x['ordinal']:x for x in m['rows']};assert len(by)==7120
 grouped=collections.defaultdict(list)
 for x in m['rows']:grouped[x['record']['outcome_science_sha256']].append(x['ordinal'])
 assert len(grouped)==3554 and dict(collections.Counter(map(len,grouped.values())))=={2:3542,3:12}
 for c in m['classes']:assert c['member_ordinals']==grouped[c['outcome_science_sha256']] and all(by[i]['stored_realized_public_sha256']==c['stored_realized_public_sha256'] for i in c['member_ordinals'])
 keys=sorted(grouped);chosen=[];n=0
 for k in keys:
  if n+len(grouped[k])>62:break
  chosen.append(k);n+=len(grouped[k])
 assert p['class_keys']==chosen and len(p['cases'])==n==62
 expected=sorted(i for k in chosen for i in grouped[k]);assert [x['ordinal'] for x in p['cases']]==expected
 assert all(x==by[x['ordinal']] for x in p['cases'])
 assert r['existing_native_cold']['G1_carriers']==2 and r['existing_complete_collision_classes']==0 and r['pilot_prior_Q2_cases_reusable']==[]
 assert r['generation_calls']==0 and r['pilot_cases_executed']==0 and r['FULL_REALIZED_PUBLIC_observer_gate']=='BLOCKED_UNTIL_EXACT_PRODUCER_BOUND'
 out=dict(status='PASS_INDEPENDENT_COLLISION_MANIFEST_AND_SCOPE_AUDIT',class_members_complete=3554,member_rows=7120,pilot_classes_complete=31,pilot_cases=62,pilot_executed=0,scope='Saved membership audit; no repeated full input scan or fresh realization',review_sha256=sha(B/'REVIEW.json'),master_slices=152,new_admissions=0,G2_promotion=False)
 (B/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
