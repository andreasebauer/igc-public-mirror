"""Existing saved level14 root comparison under historical digest formula."""
from pathlib import Path
import json,hashlib,zipfile
B=Path(__file__).resolve().parent;M=json.loads((B/'EARLY_REPLAY_READBACK.json').read_text());A=json.loads(Path('/workspace/scratch/c89172f01c5f/execution0192/AUDIT.json').read_text());pins={x['level']:x['snapshot_sha256'] for x in A['rows']}
with zipfile.ZipFile(M['path']) as z:
 for level in (7,8,13,14):
  name=next(n for n in z.namelist() if n.endswith('/artifacts/g1_exact_depth_'+str(level)+'.json'));raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==pins[level]
  if level==14:(B/'SAVED_DEPTH14.json').write_bytes(raw)
with zipfile.ZipFile(B/'PINNED_O7_SEED_AUTHORITY.zip') as z:records=json.loads(z.read(next(n for n in z.namelist() if n.endswith('O7_IMMUTABLE_SURVIVORS.json'))))['records']
lookup={x['state_digest']:x for x in records};D=json.loads((B/'SAVED_DEPTH14.json').read_text())['dag'];nodes=D['nodes'];cache={}
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def old(k):
 if k in cache:return cache[k]
 v=nodes[k]
 if v['kind']=='O7':
  r=lookup[v['source_id']];payload={'level':7,'parents':r['parent_ids'],'edges':r['edges'],'reserve_counts':v['reserve_counts'],'source':v['source_id']}
 else:payload={'level':v['level'],'children':[old(c) for c in v['children']],'edges':v['top_edges_full'],'lane':v['lane'],'motif':v['motif_id']};assert sha(dict(payload,children=v['children']))==k
 cache[k]=sha(payload);return cache[k]
p=Path('/workspace/scratch/c89172f01c5f/binding0190/recovered/Infinity_Grid_Decoder_PHASE4_REPAIRED_G1_R14_R20_COMPLETE_2026-09-02/run/checkpoints/O00014/SOURCE_INPUT.json');h=json.loads(p.read_text())['materialized_discovery_evidence']['state_probes'];expected={x['construction_digest'] for x in h};new=set(D['roots']);legacy={old(k) for k in D['roots']}
out={'status':'PASS_READ_ONLY_DIRECT_SAVED_DEPTH14_ANCHOR','replay_roots':len(new),'historical_selected_probes':len(expected),'current_root_overlap':len(new&expected),'historical_formula_rehashed_root_overlap':len(legacy&expected),'saved_depth14_sha256':pins[14],'historical_source_input_sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'generator_calls':0,'states_constructed':0,'candidate_build_calls':0,'historical_refs':sorted(expected),'replay_refs':sorted(new),'historical_formula_rehashed_replay_refs':sorted(legacy)};(B/'EARLY_DEPTH14_ANCHOR.json').write_text(json.dumps(out,indent=2));(B/'HISTORICAL_DEPTH14_SOURCE_INPUT.json').write_bytes(p.read_bytes());print(json.dumps({k:v for k,v in out.items() if not k.endswith('refs')}))
