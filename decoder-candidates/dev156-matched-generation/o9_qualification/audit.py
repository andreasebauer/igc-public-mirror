"""Cold wire-level repeated witness audit, without generation or materialization."""
import copy,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'o9_qualification'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def apply_witness(w,path,t):
    if w['level']==7:
        assert path==[t,w['counts'][t]];w['counts'][t]+=1;return
    i,local,typ=path;assert typ==t and 0<=i<len(w['children'])
    apply_witness(w['children'][i],local,t)
def main():
    reg=json.loads((OUT/'REGISTRATION.json').read_text());regsha=hashlib.sha256((OUT/'REGISTRATION.json').read_bytes()).hexdigest()
    for p,h in reg['pins'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    seen=set();count=0;steps_count=0
    for p in sorted((OUT/'recipes').glob('*.json')):
        row=json.loads(p.read_text());assert row['registration']==regsha and digest(row['recipe'])==p.stem and digest(row['result'])==row['result_digest'];seen.add(row['recipe']['motif_id'])
        h,n=row['result']['historical'],row['result']['compat'];assert h['physical']==n['physical'] and h['wire']==n['wire'] and h['construction']==n['order_key'] and h['construction']!=n['construction']
        chain=json.loads((OUT/'chains'/p.name).read_text());assert chain['registration']==regsha and chain['recipe_digest']==p.stem and digest(chain['steps'])==chain['steps_digest']
        assert [s['type'] for s in chain['steps']]==reg['repeated_sequence']
        wire=copy.deepcopy(h['wire']);caps=h['physical']['caps'][:]
        for step in chain['steps']:
            t=step['type'];assert step['before_caps']==caps;caps[t]-=1;assert step['after_caps']==caps
            assert step['historical_witness']==step['compat_witness'] and step['historical_physical_digest']==step['compat_physical_digest'] and step['historical_output_key']==step['compat_order_key']
            apply_witness(wire,step['historical_witness'],t);steps_count+=1
        assert wire==chain['final_historical_wire']==chain['final_compat_wire'];count+=1
    frozen=json.loads((ROOT/'matched_protocol/ORDERED_RECIPES.json').read_text())['with_twins'];assert count==193 and seen=={r['motif_id'] for r in frozen} and steps_count==2702
    bad=copy.deepcopy(chain);bad['steps'][0]['compat_witness'][0]+=1;assert digest(bad['steps'])!=bad['steps_digest']
    result={'status':'PASS_COLD_O9_RAW_AND_CHAIN_WITNESS_AUDIT','raw_recipe_pairs':193,'chains':193,'steps':2702,'final_wires_reproduced_from_nested_witnesses':True,'tamper_rejections':1,'generation_calls':0,'independent_materialized_skin_calculation':False,'independent_host':False,'full_p0_closed':False}
    (OUT/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
