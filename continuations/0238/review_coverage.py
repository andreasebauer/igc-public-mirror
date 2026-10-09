"""Read-only coverage and exact saved collision membership; no realization."""
from pathlib import Path
import json,gzip,hashlib,collections,sys
B=Path(__file__).resolve().parent;W=B.parent;Q=W/'continuation0237';R=W/'continuation0234'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pre=json.loads((B/'PREREGISTRATION.json').read_text())
 for n,h in pre['inputs_sha256'].items():assert sha(W/n)==h,n
 assert sys.version_info[:3]==(3,13,5) and sys.flags.optimize==0
 fixed=json.loads((W/'continuation0235/CASES.json').read_text())['cases'];proof=json.loads((Q/'Q2_PROBE.json').read_text());cold=json.loads((Q/'COLD_AUDIT.json').read_text());assert cold['cases_exact']==62
 proved={x['ordinal']:x for x in proof['cases']};assert set(proved)==set(range(62))
 counts=collections.Counter();observers={};rh=hashlib.sha256();ah=hashlib.sha256();n=0
 with gzip.open(R/'data/records.jsonl.gz','rb') as rf,gzip.open(R/'data/audits.jsonl.gz','rb') as af:
  for i,(line,aline) in enumerate(zip(rf,af,strict=True)):
   rh.update(line);ah.update(aline);r=json.loads(line);a=json.loads(aline);o=r['outcome_science_sha256'];obs=a['realized_public_sha256'];assert a['outcome_science_sha256']==o
   if o in observers:assert observers[o]==obs
   observers[o]=obs;counts[o]+=1;n+=1
   if i<62:
    assert fixed[i]=={'ordinal':i,'record':r};p=proved[i];assert p['public_projection']['science_sha256']==r['q2_one_reservation_successor_skins_sha256'] and p['repaired_outcome_sha256']==o
   if n%100000==0:print('coverage rows',n,flush=True)
 assert n==580351 and rh.hexdigest()==pre['record_uncompressed_sha256'] and ah.hexdigest()==pre['audit_uncompressed_sha256']
 collision_keys=sorted(k for k,v in counts.items() if v>1);assert len(collision_keys)==3554 and max(counts.values())==3
 selected=[];size=0
 for k in collision_keys:
  if size+counts[k]>pre['next_pilot_case_budget']:break
  selected.append(k);size+=counts[k]
 selected_set=set(selected);collision_members=[];pilot=[];classes={k:[] for k in collision_keys};h2=hashlib.sha256()
 with gzip.open(R/'data/records.jsonl.gz','rb') as f:
  for i,line in enumerate(f):
   h2.update(line);r=json.loads(line);o=r['outcome_science_sha256']
   if o in classes:
    item={'ordinal':i,'record':r,'stored_realized_public_sha256':observers[o]};collision_members.append(item);classes[o].append(i)
    if o in selected_set:pilot.append(item)
 assert h2.hexdigest()==rh.hexdigest() and len(pilot)==size and sum(counts[k] for k in collision_keys)==len(collision_members)
 for k,ordinals in classes.items():assert len(ordinals)==counts[k]
 def coverage(items):
  rows=[x['record'] for x in items];return dict(cases=len(rows),parent_pairs=len({(r['left_carrier_ref'],r['right_carrier_ref']) for r in rows}),G1_carriers=len({r[z] for r in rows for z in ('left_carrier_ref','right_carrier_ref')}),operators=len({r['connection_operator_ref'] for r in rows}),outcome_classes=len({r['outcome_science_sha256'] for r in rows}))
 all_members=dict(schema_id='IG_SAVED_S1_COLLISION_MEMBERS_V1',scope='Saved index membership only, not fresh observer certification',rows=collision_members,classes=[dict(outcome_science_sha256=k,stored_realized_public_sha256=observers[k],member_ordinals=classes[k]) for k in collision_keys])
 (B/'COLLISION_MEMBERS.json.gz').write_bytes(gzip.compress(json.dumps(all_members,sort_keys=True,separators=(',',':')).encode(),mtime=0))
 pilot_obj=dict(schema_id='IG_COMPLETE_COLLISION_CLASS_PILOT_CASES_V1',selection_rule=pre['selection_rule'],cases=sorted(pilot,key=lambda x:x['ordinal']),class_keys=selected,bridge_pairs=json.loads((W/'continuation0235/CASES.json').read_text())['bridge_pairs'])
 (B/'PILOT_CASES.json').write_text(json.dumps(pilot_obj,indent=2)+'\n')
 done_outcomes={x['record']['outcome_science_sha256'] for x in fixed};complete=sum(all(i in proved for i in classes[k]) for k in collision_keys);touched=sum(k in done_outcomes for k in collision_keys);reuse=[x['ordinal'] for x in pilot if x['ordinal'] in proved]
 result=dict(status='PASS_SAVED_COVERAGE_AND_COLLISION_REGISTRATION_REVIEW',master_slices=152,new_admissions=0,G2_promotion=False,generation_calls=0,new_DAG_decodes=0,record_rows_reviewed=n,existing_native_cold=coverage(fixed),existing_collision_classes_touched=touched,existing_complete_collision_classes=complete,existing_FULL_REALIZED_PUBLIC_observer_comparisons=0,remaining_fresh_record_realizations=n-62,collision_class_size_histogram=dict(collections.Counter(counts[k] for k in collision_keys)),all_collision_members=coverage(collision_members),pilot=coverage(pilot),pilot_complete_classes=len(selected),pilot_prior_Q2_cases_reusable=reuse,pilot_cases_executed=0,observer_producer_binding='EXACT_HISTORICAL_REALIZED_PUBLIC_ASSEMBLY_V1_SERIALIZER_NOT_YET_PINNED',FULL_REALIZED_PUBLIC_observer_gate='BLOCKED_UNTIL_EXACT_PRODUCER_BOUND',next_scope='BOUND_COMPLETE_COLLISION_CLASS_Q2_RESOURCE_PREFLIGHT_AND_RECOVER_HISTORICAL_FULL_PUBLIC_OBSERVER_SERIALIZER',collision_members_sha256=sha(B/'COLLISION_MEMBERS.json.gz'),pilot_cases_sha256=sha(B/'PILOT_CASES.json'))
 (B/'REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
