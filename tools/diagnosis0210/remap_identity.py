"""Administrative historical digest audit of existing saved DAG; no state construction."""
from pathlib import Path
import json,hashlib,sqlite3,zipfile
B=Path(__file__).resolve().parent;C=Path('/tmp/ig_native0209/forensic_cold');R=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'));prior=Path('/tmp/ig_verified0208/BOOTSTRAP99.json');base=json.loads(prior.read_text());nodes=base['dag']['nodes'];conn=sqlite3.connect((R/'phases/g1_partition_depth_100/state_store.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);raw=conn.execute('SELECT state_json FROM states').fetchone()[0];conn.close();part=json.loads(raw)
for k,v in part['nodes'].items():
 assert k not in nodes or nodes[k]==v;nodes[k]=v
seed=Path('/workspace/scratch/c89172f01c5f/binding0190/inputs/O7_MATERIAL_ROOT_cb6f48641eb9.zip')
with zipfile.ZipFile(seed) as z:records=json.loads(z.read(next(n for n in z.namelist() if n.endswith('O7_IMMUTABLE_SURVIVORS.json'))))['records']
lookup={r['state_digest']:r for r in records}
def sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
# Verify the unchanged LIFT digest formula against every observed saved node.
for k,v in nodes.items():
 if v['kind']=='LIFT':assert sha({'level':v['level'],'children':v['children'],'edges':v['top_edges_full'],'lane':v['lane'],'motif':v['motif_id']})==k
mapped={}
def old(k):
 if k in mapped:return mapped[k]
 v=nodes[k]
 if v['kind']=='O7':
  r=lookup[v['source_id']];payload={'level':7,'parents':r['parent_ids'],'edges':r['edges'],'reserve_counts':v['reserve_counts'],'source':v['source_id']}
 else:payload={'level':v['level'],'children':[old(c) for c in v['children']],'edges':v['top_edges_full'],'lane':v['lane'],'motif':v['motif_id']}
 mapped[k]=sha(payload);return mapped[k]
for k in nodes:old(k)
refsha=json.loads((B.parent/'terminal0209/SPEC.json').read_text())['execution']['parameters']['bindings']['reference'];reference=json.loads((C/'runtime/intake/artifacts'/(refsha+'.bin')).read_text());expected={x['carrier_ref'] for x in reference['interfaces']};observed=set(part['roots']);legacy={old(x) for x in part['roots']}
rows=[]
roots=Path('/workspace/scratch/c89172f01c5f/binding0190/recovered/Infinity_Grid_Decoder_PHASE4_REPAIRED_G1_R14_R20_COMPLETE_2026-09-02/run/checkpoints');phase5=Path('/workspace/scratch/c89172f01c5f/binding0189/recovered/Infinity_Grid_Decoder_PHASE5_REPAIRED_G1_R21_R100_COMPLETE_2026-09-02/run/checkpoints')
for n in (13,14,20,21,100):
 f=(roots if n<=20 else phase5)/f'O{n:05d}'/'SOURCE_INPUT.json'
 if not f.exists():continue
 h=json.loads(f.read_text()).get('materialized_discovery_evidence',{});probes=h.get('state_probes',[]);hr={x['construction_digest'] for x in probes};pool={k for k,v in nodes.items() if v['kind']=='LIFT' and v['level']==n};rows.append({'level':n,'historical_selected_probes':len(hr),'saved_reachable_node_variants':len(pool),'current_ref_overlap':len(pool&hr),'historical_identity_remapped_overlap':len({mapped[k] for k in pool}&hr),'historical_source_input_sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()})
out={'status':'PASS_ADMINISTRATIVE_EXISTING_DAG_IDENTITY_REMAP_DIAGNOSIS','generator_calls':0,'states_constructed':0,'candidate_build_calls':0,'saved_nodes_examined':len(nodes),'O7_nodes':sum(v['kind']=='O7' for v in nodes.values()),'O7_sources':len({v['source_id'] for v in nodes.values() if v['kind']=='O7'}),'O7_current_legacy_digests_equal':sum(k==mapped[k] for k,v in nodes.items() if v['kind']=='O7'),'terminal_current_ref_overlap':len(observed&expected),'terminal_legacy_identity_remapped_overlap':len(legacy&expected),'terminal_roots':193,'source_seed_sha256':hashlib.file_digest(seed.open('rb'),'sha256').hexdigest(),'bootstrap99_sha256':hashlib.file_digest(prior.open('rb'),'sha256').hexdigest(),'terminal_partition_state_json_sha256':hashlib.sha256(raw.encode()).hexdigest(),'historical_selected_probe_comparison':rows,'nonclaim':'Rehashing existing witness structures is diagnostic only; not a historical state replay, matching physical interface proof or new admission'};(B/'IDENTITY_REMAP.json').write_text(json.dumps(out,indent=2));(B/'NODE_IDENTITY_MAP.json').write_text(json.dumps(mapped,sort_keys=True));print(json.dumps(out))
