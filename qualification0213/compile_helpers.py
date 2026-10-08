"""Extract historical pure helper closure with exact retained-definition AST attestation."""
from pathlib import Path
import ast,json,hashlib
B=Path(__file__).resolve().parent;H=B.parent/'diagnosis0210/source_evidence/historical';OUT=B/'project/historical';OUT.mkdir(parents=True,exist_ok=True);(B/'project/__init__.py').write_text('');(OUT/'__init__.py').write_text('')
roots={'regime_scanner.py':['O7State','LiftState','_farthest_select','_state_pre_features','_build_lift','load_regime_scanner_spec','load_motif_library','_load_module'],'maturation_parallel.py':['state_dag_wire','_states_from_dag','install_candidate_context','clear_candidate_context','_candidate_worker','enumerate_candidate_recipes','_farthest_select_descriptors'],'materialized_discovery.py':['_get_process_o7_runtime']};allrows=[]
# Cross-module dependencies are part of the closure, including sibling attribute calls.
module_symbols={};module_trees={}
for name in roots:
 tree=ast.parse((H/name).read_text());module_trees[name]=tree;symbols={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for n in tree.body:
  if isinstance(n,(ast.Assign,ast.AnnAssign)):
   for t in (n.targets if isinstance(n,ast.Assign) else [n.target]):
    if isinstance(t,ast.Name):symbols[t.id]=n
 module_symbols[name]=symbols
selected_by_module={name:set() for name in roots};pending=[(name,key) for name,keys in roots.items() for key in keys]
while pending:
 name,key=pending.pop()
 if key in selected_by_module[name]:continue
 selected_by_module[name].add(key);node=module_symbols[name][key]
 aliases={}
 for imp in module_trees[name].body:
  if isinstance(imp,ast.ImportFrom) and imp.level and imp.module is None:
   for alias in imp.names:
    sibling=alias.name+'.py'
    if sibling in roots:aliases[alias.asname or alias.name]=sibling
 for v in ast.walk(node):
  if isinstance(v,ast.Name) and isinstance(v.ctx,ast.Load) and v.id in module_symbols[name]:pending.append((name,v.id))
  if isinstance(v,ast.Attribute) and isinstance(v.value,ast.Name) and v.value.id in aliases:
   sibling=aliases[v.value.id]
   if v.attr in module_symbols[sibling]:pending.append((sibling,v.attr))
roots={name:sorted(keys) for name,keys in selected_by_module.items()}
for name,wanted in roots.items():
 source=(H/name).read_text();tree=ast.parse(source);defs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};assigns={}
 for n in tree.body:
  if isinstance(n,(ast.Assign,ast.AnnAssign)):
   targets=n.targets if isinstance(n,ast.Assign) else [n.target]
   for t in targets:
    if isinstance(t,ast.Name):assigns[t.id]=n
 symbols=dict(assigns,**defs);selected=set();pending=list(wanted)
 while pending:
  key=pending.pop()
  if key in selected:continue
  selected.add(key);node=symbols[key]
  pending.extend(n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id in symbols and n.id not in selected)
 if name=='regime_scanner.py':selected.discard('_load_module')
 imports=[]
 # Append shared loader after all historical imports, preserving future-first rule.
 for node in tree.body:
  if isinstance(node,(ast.Import,ast.ImportFrom)):
   if isinstance(node,ast.ImportFrom) and node.level and node.module=='execution':continue
   if isinstance(node,ast.ImportFrom) and node.level and node.module=='o7_science_compat':continue
   text=ast.get_source_segment(source,node)
   if isinstance(node,ast.ImportFrom) and node.level and node.module in ('frontier','canon'):text=text.replace('from .'+node.module,'from infinity_grid.'+node.module)
   imports.append(text)
 if name=='regime_scanner.py':imports.append('from infinity_grid.regime_scanner import _load_module')
 retained=[n for n in tree.body if any(n is symbols[k] for k in selected)];result='\n'.join(imports)+'\n\n'+'\n\n'.join('\n'.join(source.splitlines()[min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1:n.end_lineno]) for n in retained)+'\n';compile(result,str(OUT/name),'exec');(OUT/name).write_text(result);new=ast.parse(result);nd={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 rows=[]
 for key in sorted(selected):
  if key not in defs:continue
  original=ast.dump(defs[key],include_attributes=False);output=ast.dump(nd[key],include_attributes=False);assert original==output;rows.append({'definition':key,'historical_AST_sha256':hashlib.sha256(original.encode()).hexdigest(),'output_AST_equal':True})
 allrows.append({'module':name,'historical_module_sha256':hashlib.file_digest((H/name).open('rb'),'sha256').hexdigest(),'extracted_module_sha256':hashlib.file_digest((OUT/name).open('rb'),'sha256').hexdigest(),'retained_definitions':rows,'excluded_execution_wrappers':sorted(set(defs)-selected),'import_changes':'Only canon/frontier routes to AST/byte-qualified current common helpers; historical scientific sibling imports preserved'})
# The historical seed kernel uses the frozen packaged ZIP and extracted historical
# sibling scanner, not the current scanner class namespace.
oldroot=Path('/workspace/scratch/c89172f01c5f/binding0190/recovered/Infinity_Grid_Decoder_PHASE3_EMERGENCY_RESCUE_CHECKPOINT_V2_2026-09-02/source/Infinity_Grid_Algebra_Decoder_v0.28.8_TRUST_REPAIR_COMPLETE_2026-09-02/src/infinity_grid');current=Path('/tmp/ig_engine0204/infinity_grid');assert (oldroot/'core/canonical.py').read_bytes()==(current/'core/canonical.py').read_bytes()
a=next(n for n in ast.parse((oldroot/'frontier.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_apply_o7_overlay_skin');b=next(n for n in ast.parse((current/'frontier.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name==a.name);assert ast.dump(a)==ast.dump(b)
oa=next(n for n in ast.parse((H/'regime_scanner.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_load_module');na=next(n for n in ast.parse((current/'regime_scanner.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_load_module');assert ast.dump(oa)==ast.dump(na)
(B/'SOURCE_ATTESTATION.json').write_text(json.dumps({'status':'PASS_HISTORICAL_PURE_HELPER_AST_CLOSURE','modules':allrows,'canonicalizer_byte_equal':True,'overlay_helper_AST_equal':True,'common_loader_route':'infinity_grid.regime_scanner._load_module (same historical definition; native capture prohibits project-owned dynamic loaders)','generator_calls':0,'state_construction_calls':0,'native_capture_created':False},indent=2));print('PASS historical helper extraction/AST closure')
