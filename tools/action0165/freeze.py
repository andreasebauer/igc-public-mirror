import ast,hashlib,json
from pathlib import Path
def ast_bytes(node):
 def normalized(value):
  if isinstance(value,ast.AST):return {'node':type(value).__name__,'fields':{k:normalized(v) for k,v in ast.iter_fields(value) if v is not None and v!=[]}}
  if isinstance(value,list):return [normalized(v) for v in value]
  if isinstance(value,bytes):return {"bytes_hex":value.hex()}
  return value
 return json.dumps(normalized(node),sort_keys=True,separators=(',',':')).encode()
FUNCTIONS=['shaj','dist','nested_o5','canon_block','canon_o3','canon_o4','canon_o5','usage','base_free','edge_bytes','state_payload','state_digest','graph_components','materialize','component_invariants','validate_exact_state','normalize_component','verify_prototype_selection']
def freeze(source,target):
 raw=source.read_bytes();text=raw.decode();tree=ast.parse(text);nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)};chunks=[];mapping={}
 for name in FUNCTIONS:
  chunk=ast.get_source_segment(text,nodes[name]);chunks.append(chunk);mapping[name]=hashlib.sha256(ast_bytes(nodes[name])).hexdigest()
 out='from __future__ import annotations\nimport json,hashlib,math,collections,struct\nfrom typing import Iterable\nPN=7\nSTATE_HASH_PREFIX=b"IG-E6-STATE-v1|"\n\n'+'\n\n'.join(chunks)+'\n';target.write_text(out);new={n.name:n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef)}
 for name,h in mapping.items():assert hashlib.sha256(ast_bytes(new[name])).hexdigest()==h
 return {'schema':'IG_O6_PURE_SOURCE_EXTRACTION_V1','original_sha256':hashlib.sha256(raw).hexdigest(),'pure_sha256':hashlib.sha256(out.encode()).hexdigest(),'functions_ast_sha256':mapping,'excluded':'main, generation workers/reservoirs, checkpoint IO, private multiprocessing and adversarial execution','generation_calls':0}
