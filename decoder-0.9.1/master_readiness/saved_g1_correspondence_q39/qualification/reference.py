"""Independent literal mismatch path; no primary DAG-interning calls."""
import json

def enc(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def first_difference(a,b,left,right):
    stack=[(a,b,[])];seen=set()
    while stack:
        ka,kb,path=stack.pop()
        if (ka,kb) in seen:continue
        seen.add((ka,kb));x=left[ka];y=right[kb]
        fields=('kind','lane') if x['kind']!=y['kind'] else (('kind','lane','source_id','reserve_counts') if x['kind']=='O7' else ('kind','lane','level','motif_id','top_edges_full'))
        for f in fields:
            if enc(x.get(f))!=enc(y.get(f)):
                return {'ordered_child_path':path,'left_alias':ka,'right_alias':kb,'field':f,'left_value':x.get(f),'right_value':y.get(f),'reference':'DIRECT_LITERAL_TYPED_JSON_FIELD_COMPARISON'}
        ca=x.get('children',[]);cb=y.get('children',[])
        if len(ca)!=len(cb):return {'ordered_child_path':path,'left_alias':ka,'right_alias':kb,'field':'child_count','left_value':len(ca),'right_value':len(cb),'reference':'DIRECT_LITERAL_CHILD_ARITY'}
        stack.extend((aa,bb,path+[i]) for i,(aa,bb) in reversed(list(enumerate(zip(ca,cb)))))
    return None
