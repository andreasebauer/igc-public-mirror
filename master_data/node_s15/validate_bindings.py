import pathlib,json,hashlib,importlib.util,sys,itertools,ast
B=pathlib.Path(__file__).resolve().parent; S=B/'source_bundle';root=B.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
checks=[]
for line in (S/'SHA256_MANIFEST.txt').read_text().splitlines():
 if not line.strip():continue
 h,n=line.split(None,1);p=S/n.strip();assert p.is_file(),n;assert sha(p.read_bytes())==h,n;checks.append(n)
sp=importlib.util.spec_from_file_location('historical',S/'code/node_graduation_audit.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
records,prim,hfail,pfail=m.load_inputs();assert not hfail and not pfail
original=json.loads((S/'evidence/NODE_GRADUATION_RESULT.json').read_text());f=json.loads((B/'replay_forward/NODE_GRADUATION_RESULT.json').read_text());r=json.loads((B/'replay_reverse/NODE_GRADUATION_RESULT.json').read_text());assert f==r==original
sys.path.insert(0,str(root/'takeover/master_reader'));from ig_master.prefix_reader import PrefixReader
P=PrefixReader('/tmp/ig_takeover_20261006/DATA_SLICE','e1e2327070b2f9fc3bb548fc58a6961b70a0b885eb857d040ce0dcd5bfe4a351');P.verify()
bind=[];a=P.foundation['arrays']
for tid,rank,rec in prim:
 sids=tuple(a['tri'][3*tid:3*tid+3]);hits=[];partial=[]
 for c in itertools.product(*(range(a['oc'][s]) for s in sids)):
  v=P._choice_record(sids,c)
  if v is None:continue
  raw=v[0];normalized=m.canon((raw[0],raw[1],raw[2],(raw[3],),raw[4]))
  if normalized==rec:hits.append({'option_indices':list(c),'target_tid':v[2],'boundary':raw})
  if normalized[:3]+normalized[4:]==rec[:3]+rec[4:] and normalized!=rec:partial.append(raw[3])
 bind.append({'historical_tid':tid,'historical_rank':rank,'historical_record_sha256':m.sha(rec),'foundation_source_sids':list(sids),'exact_choice_witnesses':hits,'T_only_different_values':sorted(set(partial)),'status':'EXACT_FOUNDATION_SEMANTIC_MATCH_NOT_YET_ADMITTED_EVENT_ROUTE_BOUND' if hits else 'NO_EXACT_FOUNDATION_MATCH'})
res={'schema':'IG_NODE_S15_INPUT_BINDING_VALIDATION_V1','source_manifest_files_checked':len(checks),'source_manifest':'PASS','records_checked':len(records),'record_hash_failures':0,'distinct_record_hashes':len(set(h for h,_ in records)),'primitives_checked':len(prim),'primitive_hash_failures':0,'historical_forward_reverse_full_result_equality':True,'historical_science_sha256':f['science_sha256'],'historical_result':f['overall_status'],'historical_regression_exact_cases':f['NG3_exact_mature_macro_closure']['regression_exact_cases'],'foundation_root':P.expected_root if hasattr(P,'expected_root') else 'e1e2327070b2f9fc3bb548fc58a6961b70a0b885eb857d040ce0dcd5bfe4a351','primitive_bindings':bind,'master_admission':False,'master_release_unchanged':'MASTER_DATA_V1_0141','new_population_generation':False,'scope':'Recovered historical finite observed boundary panel; whole microscopic derivation/witness lineage is not present in observed input. Foundation choices checked only as validation, no new science exported.','next':'Bind matching primitive boundaries to admitted J3 event routes; audit recovered observed records for derivation/lineage witnesses before choosing admission contract.'}
(B/'VALIDATION.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='primitive_bindings'},indent=2));print([(x['historical_tid'],x['historical_rank'],len(x['exact_choice_witnesses'])) for x in bind])
