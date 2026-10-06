"""Verify recovered bootstrap evidence and dry-replay its pure constructor.

No historical top-level script or pickle object deserialization is executed.
"""
from pathlib import Path
import ast,json,hashlib,pickletools
from collections import Counter
B=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
R=D/'original/Infinity_Grid_L14A0_BOOTSTRAP_2026-08-24'
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((R/'docs/SHA256_MANIFEST.json').read_text())
for name,h in manifest.items():assert sha((R/name).read_bytes())==h

def literal_pickle(raw):
    # Exact data-only opcode whitelist for this 778-byte protocol-4 packet.
    # Reject GLOBAL, STACK_GLOBAL, REDUCE, NEWOBJ and every other unknown opcode.
    stack=[];memo={};mark=object();stopped=False
    for opcode,arg,pos in pickletools.genops(raw):
        n=opcode.name
        if n in ('PROTO','FRAME'):continue
        if n=='EMPTY_LIST':stack.append([])
        elif n=='EMPTY_DICT':stack.append({})
        elif n=='MARK':stack.append(mark)
        elif n in ('BININT1','BININT2','SHORT_BINUNICODE'):stack.append(arg)
        elif n=='MEMOIZE':memo[len(memo)]=stack[-1]
        elif n=='BINGET':stack.append(memo[arg])
        elif n in ('TUPLE1','TUPLE2','TUPLE3'):
            k=int(n[-1]);values=stack[-k:];del stack[-k:];stack.append(tuple(values))
        elif n in ('TUPLE','APPENDS','SETITEMS'):
            idx=max(i for i,v in enumerate(stack) if v is mark);values=stack[idx+1:];del stack[idx:]
            if n=='TUPLE':stack.append(tuple(values))
            elif n=='APPENDS':assert type(stack[-1]) is list;stack[-1].extend(values)
            else:
                assert type(stack[-1]) is dict and len(values)%2==0
                for i in range(0,len(values),2):stack[-1][values[i]]=values[i+1]
        elif n=='STOP':assert len(stack)==1 and pos==len(raw)-1;stopped=True;break
        else:raise ValueError('UNSAFE_OR_UNSUPPORTED_PICKLE_OPCODE:'+n)
    assert stopped;return stack[0]

primitive_raw=(R/'inputs/primitive.pkl').read_bytes();original=literal_pickle(primitive_raw)
frozen_path=B/'engine/infinity_grid/resources/replay/NODE_IN_ORIGINAL_FROZEN_PRIMITIVES.json'
if not frozen_path.exists():frozen_path=D/'bindings/NODE_IN_ORIGINAL_FROZEN_PRIMITIVES.json'
frozen=json.loads(frozen_path.read_text())['primitive']
assert len(original)==len(frozen)==9
for a,b in zip(original,frozen):
    assert set(a)=={'tid','rank','record','canon'} and a['tid']==b['tid'] and a['rank']==b['rank']
    r=a['record'];expected=ast.literal_eval(b['record'])
    actual=(r[0],tuple(sorted(r[1])),r[2],r[3] if isinstance(r[3],tuple) else (r[3],),r[4])
    assert actual==a['canon']==expected and sha(repr(expected).encode())==b['sha256']
# Recovered source's pure functions only; no script imports, file I/O or CLI.
src=(R/'scripts/01_l14a0_official.py').read_bytes();tree=ast.parse(src.decode());functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'dtup','canon','sha','bridge','compose'}],type_ignores=[]);ns={'hashlib':hashlib};exec(compile(functions,'original_bootstrap_pure_functions','exec'),ns)
panel=json.loads((R/'inputs/FROZEN_L13_SOURCE_PANEL.json').read_text());assert len(panel)==12
global_rows={};total_attempts=total_lawful=0;results=[]
for x in sorted(panel,key=lambda x:x['panel_pos']):
    a=ns['canon'](ast.literal_eval(x['selected_L13_boundary']));assert ns['sha'](a)==x['selected_L13_sha256']
    assert len(a[1])==16 and len(a[3])==14
    saved=json.loads((R/f"chunks/panel_{x['panel_pos']:02d}.json").read_text());children={};witnesses=[];attempts=lawful=0
    for p in original:
        for sa in range(len(a[1])):
            for sp in range(3):
                attempts+=1;r=ns['compose'](a,sa,p['record'],sp)
                if r is None:continue
                lawful+=1;h=ns['sha'](r);children[h]=repr(r)
                witnesses.append({'child_sha256':h,'primitive_tid':p['tid'],'primitive_rank':p['rank'],'source_port':sa,'primitive_port':sp})
    rows=sorted([{'sha256':h,'record':r} for h,r in children.items()],key=lambda x:(x['sha256'],x['record']))
    witnesses.sort(key=lambda x:(x['child_sha256'],x['primitive_tid'],x['primitive_rank'],x['source_port'],x['primitive_port']))
    assert saved['L13_source_sha256']==x['selected_L13_sha256'] and attempts==saved['attempts'] and lawful==saved['lawful_attachment_types']
    assert rows==saved['children'] and witnesses==saved['attachment_witnesses']
    global_rows.update(children);total_attempts+=attempts;total_lawful+=lawful
    results.append({'panel_pos':x['panel_pos'],'attempts':attempts,'lawful':lawful,'children':len(children),'all_endpoint_witnesses_match':True})
saved_global=json.loads((R/'evidence/GLOBAL_L14_PANEL_CHILDREN.json').read_text());assert {x['sha256']:x['record'] for x in saved_global}==global_rows and len(global_rows)==499
scout_panel=next((B/'node_frontier_binding').rglob('GLOBAL_L14_PANEL_CHILDREN.json'),D/'bindings/SCOUT_GLOBAL_L14_PANEL_CHILDREN.json');assert scout_panel.read_bytes()==(R/'evidence/GLOBAL_L14_PANEL_CHILDREN.json').read_bytes()
selected=[];used=set()
for x in sorted(panel,key=lambda x:x['panel_pos']):
    chunk=json.loads((R/f"chunks/panel_{x['panel_pos']:02d}.json").read_text());pick=next(c for c in chunk['children'] if c['sha256'] not in used);used.add(pick['sha256'])
    selected.append({'panel_pos':x['panel_pos'],'L13_source_sha256':x['selected_L13_sha256'],'selected_L14_sha256':pick['sha256'],'selected_L14_boundary':pick['record'],'exposed_ports':17})
assert selected==json.loads((R/'evidence/FROZEN_L14_CHILD_PANEL.json').read_text())
terminal=json.loads((R/'evidence/TERMINAL_REPLAY.json').read_text())
assert terminal=={'status':'PASS','panel_pos':12,'attempts':attempts,'lawful_attachment_types':lawful,'unique_canonical_L14_children':len(rows),'child_rows_digest':sha(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()),'byte_semantic_match':True}
report={'status':'PASS_BOOTSTRAP_SOURCE_INPUT_AND_WITNESS_BINDING','execution':'Dry full semantic comparison of recovered bootstrap evidence; no official population job/admission','original_manifest_members_verified':len(manifest),'source_sha256':sha(src),'frozen_l13_input_sha256':sha((R/'inputs/FROZEN_L13_SOURCE_PANEL.json').read_bytes()),'primitive_original_sha256':sha(primitive_raw),'primitive_mapping':'Original nine primitive records exactly equal the frozen nine selectors already bound to admitted J3','input_sources':12,'attempts':total_attempts,'lawful_endpoint_witnesses':total_lawful,'global_l14_boundaries':499,'all_12_chunk_child_sets_and_endpoint_witness_lists_match':True,'scout_499_panel_byte_identical':True,'deterministic_12_selected_panel_matches':True,'per_source':results,'resolved_blockers':['Original L14A0 producer','Exact frozen L13 boundary input packet','All bounded L14 bootstrap attachment endpoint witnesses'],'remaining':'Fresh exact construction of the L13 inputs and their recursive native lineage still needs producer/admission binding. The recovered L13 boundary snapshot is historical; it is not newly admitted master data.'}
(D/'BOOTSTRAP_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='per_source'}))
# JSON replacement packet for future adapters; original bytes remain pinned.
(D/'PRIMITIVE_JSON_REPLACEMENT.json').write_text(json.dumps({'original_sha256':sha(primitive_raw),'primitives':original,'native_mapping_ref':'node_input_audit/PRIMITIVE_MAPPING.json','interpretation':'Historical projected primitive domain, not native-event expansion'},indent=2)+'\n')
