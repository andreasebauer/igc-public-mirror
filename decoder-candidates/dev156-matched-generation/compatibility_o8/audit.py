"""Cold checkpoint binding and physical trace comparison; no generator calls."""
import copy,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'compatibility_o8'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def physical(x):
    if isinstance(x,dict):return {k:physical(v) for k,v in x.items() if k!='construction'}
    if isinstance(x,list):return [physical(v) for v in x]
    return x
def binding(row,regsha):
    if row['registration']!=regsha or digest(row['call'])!=row['call_digest'] or digest(row['result'])!=row['result_digest']:
        raise ValueError('checkpoint binding')
def main():
    regsha=hashlib.sha256((OUT/'REGISTRATION.json').read_bytes()).hexdigest()
    reg=json.loads((OUT/'REGISTRATION.json').read_text())
    for p,h in reg['pins'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    old={json.loads(p.read_text())['call']['motif_id']:json.loads(p.read_text()) for p in (ROOT/'matched_run2/historical/recipes').glob('*.json')}
    seen=set();changed=0;count=0;events=0
    rows=[]
    for p in sorted((OUT/'compat/recipes').glob('*.json')):
        row=json.loads(p.read_text());binding(row,regsha);assert p.stem==row['call_digest']
        motif=row['call']['motif_id'];assert motif not in seen;seen.add(motif);reference=old[motif]
        assert physical(row['call'])==physical(reference['call']),motif
        assert physical(row['result'])==physical(reference['result']),motif
        assert row['result']['state'] is not None
        changed+=row['result']['state']['construction']!=reference['result']['state']['construction']
        count+=1;events+=len(row['result']['reservation_events']);rows.append(row)
    assert count==193 and seen==set(old) and changed==193
    rejected=0
    for field in ('call','result'):
        bad=copy.deepcopy(rows[0]);bad[field]['tampered']=True
        try:binding(bad,regsha)
        except ValueError:rejected+=1
    assert rejected==2
    h=json.loads((ROOT/'matched_run2/historical/CENTER.json').read_text());n=json.loads((OUT/'compat/CENTER.json').read_text())
    assert h['ordered_parent_occurrences']==n['ordered_parent_occurrences'] and h['median']==n['median']
    assert physical(h['center'])==physical(n['center'])
    result={'status':'PASS_COLD_CHECKPOINT_TRACE_COMPARISON','recipes':count,'reservation_events_compared':events,'distinct_v3_ids_preserved':changed,'tamper_rejections':rejected,'exact_calls_resources_children_edges_operator_tests_reservation_events':True,'comparison_excludes_only_construction_namespace':True,'generator_calls':0,'independent_host':False,'O8_cuts_dispatched':False,'O9_dispatched':False,'full_p0_closed':False}
    (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
