"""Full saved-stream reference review; no carrier construction or authority rewrite."""
from pathlib import Path
import json,gzip,hashlib,collections,traceback,zipfile,sys
B=Path(__file__).resolve().parent;W=B.parent
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    pre=json.loads((B/'PREREGISTRATION.json').read_text());manifest=json.loads((B/'INDEX_MANIFEST.json').read_text())
    assert digest(B/'INDEX_MANIFEST.json')==pre['source_index_manifest_sha256']
    assert digest(W/'continuation0233/CATALOG_0152.json')==pre['catalog152_sha256']
    assert json.loads((W/'continuation0233/AUDIT.json').read_text())['status']=='PASS_COLD_SCOPED_EXACT_PARENT_ADMISSION'
    assert sys.version_info[:3]==(3,13,5) and sys.flags.optimize==0
    cat=json.loads((W/'continuation0233/CATALOG_0152.json').read_text());seal=W/'continuation0232/G1_HISTORICAL_V1_EXACT_PARENT_SEAL0232.zip'
    assert digest(seal)==cat['slices'][-1]['archive']['sha256']
    with zipfile.ZipFile(seal) as z:
        sealed=json.loads(z.read('SEAL_MANIFEST.json'))
        assert sealed['scientific_root_sha256']==pre['admitted_G1_scientific_root_sha256']
        assert digest(W/'continuation0231/HISTORICAL_V1_INTERFACE_POPULATION.json')==sealed['payload']['files']['HISTORICAL_V1_INTERFACE_POPULATION.json']['sha256']
    for n in ['records.jsonl.gz','audits.jsonl.gz']:assert digest(B/'data'/n)==manifest['files'][n]['sha256']
    pop=json.loads((W/'continuation0231/HISTORICAL_V1_INTERFACE_POPULATION.json').read_text());rows={x['carrier_ref']:x for x in pop['interfaces']};assert len(rows)==193
    assert digest(B/'HISTORICAL_CONTINUATIONS.json')=='35bb6899ab4d1fae4da11f12393658a233f70f3ad7a3dfaad95e617424499d85'
    continuations=json.loads((B/'HISTORICAL_CONTINUATIONS.json').read_text())['rows'];assert set(continuations)==set(rows) and sum(map(len,continuations.values()))==1351
    ops={};pairs={};refs=set();sources=collections.Counter();statuses=collections.Counter();outcomes={};h=hashlib.sha256();ah=hashlib.sha256();n=0
    with gzip.open(B/'data/records.jsonl.gz','rb') as rf,gzip.open(B/'data/audits.jsonl.gz','rb') as af:
        for ordinal,(line,auditline) in enumerate(zip(rf,af,strict=True)):
            h.update(line);ah.update(auditline);x=json.loads(line);audit=json.loads(auditline)
            l,r,op=x['left_carrier_ref'],x['right_carrier_ref'],x['connection_operator_ref'];assert l in rows and r in rows and l<=r,('CARRIER_REF',ordinal)
            assert x['left_interface_sha256']==rows[l]['interface_sha256'] and x['right_interface_sha256']==rows[r]['interface_sha256'],('INTERFACE',ordinal)
            assert op.startswith('G1_PUBLIC_BRIDGE_RELATION_V1:');a,b=op.rsplit(':',1)[1].split('>')
            assert x['d4_left_post_reservation_public_continuation_sha256']==continuations[l][a] and x['d4_right_post_reservation_public_continuation_sha256']==continuations[r][b],('D4',ordinal)
            assert x['q2_projection_serialized'] is False and x['schema_id']=='IG_G_UPLIFT_PAIR_CONNECTION_RECORD_V2'
            assert x['source_authority_sha256']=='0cf1a49659af124d59aa587a5a7b901c37c1270fde3edd22edaaa3f21cf2559a'
            assert audit['outcome_science_sha256']==x['outcome_science_sha256'],('AUDIT_JOIN',ordinal)
            outcome=x['outcome_science_sha256'];observer=audit['realized_public_sha256']
            if outcome in outcomes:
                val,c=outcomes[outcome];assert val==observer,('STORED_OBSERVER_SPLIT',ordinal);outcomes[outcome]=(val,c+1)
            else:outcomes[outcome]=(observer,1)
            if op not in ops:ops[op]=len(ops)
            bit=1<<ops[op];key=(l,r);old=pairs.get(key,0);assert not old&bit,('DUPLICATE_PAIR_OPERATOR',ordinal);pairs[key]=old|bit
            refs.update([l,r]);sources[x['source_authority_sha256']]+=1;statuses[x['strict_public_repair_status']]+=1;n+=1
            if n%100000==0:print('checked',n,flush=True)
    assert h.hexdigest()==pre['record_uncompressed_sha256'] and ah.hexdigest()==pre['audit_uncompressed_sha256']
    assert n==580351 and refs==set(rows) and len(ops)==31 and len(pairs)==18721 and set(pairs.values())=={(1<<31)-1}
    assert len(outcomes)==576785 and sum(c>1 for _,c in outcomes.values())==3554 and max(c for _,c in outcomes.values())==3
    result=dict(status='PASS_FULL_SAVED_G2_SOURCE_BINDING_REVIEW',scope=pre['scope'],record_rows_checked=n,audit_rows_checked=n,G1_refs_joined=193,interface_hash_bindings_checked=2*n,stored_D4_bindings_checked=2*n,continuation_rows_bound=1351,bridge_operators=31,pairs=18721,outcome_classes=576785,collision_classes=3554,maximum_class_size=3,stored_observer_split_classes=0,record_uncompressed_sha256=h.hexdigest(),audit_uncompressed_sha256=ah.hexdigest(),stored_source_authorities=dict(sources),stored_row_statuses=dict(statuses),admitted_G1_scientific_root_sha256=pre['admitted_G1_scientific_root_sha256'],historical_G1_spec_sha256=pop['implementation_spec_sha256'],historical_G1_source_authority_sha256=pop['source_authority_sha256'],source_authorities_rewritten=False,master_slices=152,new_admissions=0,generation_calls=0,new_DAG_decodes=0,Q2_payload_available=False,fresh_realization=False,G2_promotion=False,index_restore_required=False)
    (B/'REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (B/'STOPPED.txt').write_text(traceback.format_exc());raise
