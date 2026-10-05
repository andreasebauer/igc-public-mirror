"""Read-only provenance/coverage audit; no scientific population generation."""
from pathlib import Path
import zipfile,io,json,hashlib,ast,gzip,re
B=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
outer=B/'project_sources/09-Nodes.zip';nested_name='Infinity_Grid_WHOLE_NODE_GRAMMAR_CLOSURE_AUDIT_REPRO_BUNDLE_2026-08-26.zip'
with zipfile.ZipFile(outer) as z:nb=z.read(nested_name)
with zipfile.ZipFile(io.BytesIO(nb)) as q:
 prefix='Infinity_Grid_WHOLE_NODE_GRAMMAR_CLOSURE_AUDIT_2026-08-26/'
 manifest_raw=q.read(prefix+'SHA256_MANIFEST.txt');manifest={}
 for line in manifest_raw.decode().splitlines():
  h,path=line.split(None,1);manifest[prefix+path.strip().removeprefix('./')]=h
 verified=0
 for member,h in manifest.items():assert sha(q.read(member))==h;verified+=1
 def get(suffix):
  member=prefix+suffix;raw=q.read(member);assert sha(raw)==manifest[member];return raw,json.loads(raw)
 def rows_checked(rows):
  out={}
  for row in rows:
   r=ast.literal_eval(row['record']);assert isinstance(r,tuple) and len(r)==5
   canonical=(int(r[0]),tuple(sorted(tuple(map(int,p)) for p in r[1])),int(r[2]),tuple(sorted(map(int,r[3]))),int(r[4]))
   assert r==canonical and sha(repr(r).encode())==row['sha256'] and row['sha256'] not in out
   assert len(r[1])==len(r[3])+2;out[row['sha256']]=r
  return out
 raw,l44=get('evidence/stress/inputs/L44_SELECTED_RELATION.json');previous=rows_checked(l44['selected']);start_count=len(previous)
 _,maturation=get('evidence/stress/inputs/L39_L44_MATURATION_STATE.json');assert maturation['through_level']==44
 levels=[];union={};total_edges=0;missing_parent=0
 for level in range(45,65):
  member=f'evidence/stress/levels/L{level}/evidence/SELECTED_RELATION.json';rbytes,rel=get(member);_,result=get(f'evidence/stress/levels/L{level}/results/L{level}_RESULT.json')
  states=rows_checked(rel['selected']);edges=[tuple(e) for e in rel['edges']];assert len(set(edges))==len(edges)
  assert all(parent in previous and child in states for parent,child in edges)
  assert {c for p,c in edges}==set(states)
  parents={c:set() for c in states}
  for parent,child in edges:parents[child].add(parent)
  assert all(row['bounded_parent_count']==len(parents[row['sha256']]) and row['witness_count']>=len(parents[row['sha256']]) for row in rel['selected'])
  assert len(states)==result['selection']['union_selected']==result['bounded_relation']['selected_union']
  assert len(edges)==result['bounded_relation']['relation_edges'] and len(previous)==result['generation']['sources']
  attempts=sum(len(r[1])*9*3 for r in previous.values());assert attempts==result['generation']['attempts']
  for h,r in states.items():
   if (r[0],r[2],r[4])==(15,1,0):
    if h in union:assert union[h]==r
    union[h]=r
  levels.append({'level':level,'tier':result['tier'],'lane_caps':result['selection']['lane_caps'],'sources':len(previous),'selected':len(states),'edges':len(edges),'attempts':attempts,'reported_lawful':result['generation']['lawful'],'reported_unique_candidates':result['generation']['global_unique_candidates'],'selected_member':prefix+member,'selected_sha256':sha(rbytes),'source_and_target_references_close':True})
  previous=states;total_edges+=len(edges)
 # Exact equality to the retained replay observation set, not just count equality.
 obs_path=B/'engine/infinity_grid/resources/replay/NODE_IN_ORIGINAL_OBSERVED_S15_RECORDS.json.gz';obs_raw=obs_path.read_bytes();assert sha(obs_raw)=='ab397b2d40891a6f3cec742edbfd25c5eb1b5d627a98641a9666f8a265862f5f'
 observed=json.loads(gzip.decompress(obs_raw));observed_rows=rows_checked(observed['records']);assert observed_rows==union and observed['count']==len(union)==22885
 source_members=['evidence/stress/code/controller.sh','evidence/stress/spec/NODE_GRAMMAR_STRESS_SPEC_V1.txt']
 for member in source_members:
  full=prefix+member;raw=q.read(full);assert sha(raw)==manifest[full];dest=D/'pinned_stress_controls'/Path(member).name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(raw)
 report={'status':'PASS_HISTORICAL_FRONTIER_BINDING','scientific_generation_calls':0,'outer_archive_sha256':sha(outer.read_bytes()),'nested_archive':nested_name,'nested_archive_sha256':sha(nb),'original_manifest_verified_members':verified,'l44_selected_boundaries':start_count,'l44_selected_sha256':manifest[prefix+'evidence/stress/inputs/L44_SELECTED_RELATION.json'],'maturation_through_level':44,'historical_levels':levels,'checked_parent_edges':total_edges,'s15_observed_union':len(union),'retained_observation_set_exactly_equal':True,'replay_observed_sha256':sha(obs_raw),'missing_endpoint_witnesses':'Selected records store boundary, parent count, witness count and parent-to-child edges. Individual primitive/event/endpoint choices and exact internal carriers are absent from this representation.','fresh_population_readiness':'BLOCKED_ON_BOOTSTRAP_PRODUCER_AND_FULL_LINEAGE_SCOPE','historical_replay_readiness':'L44 inputs, L45-L64 constructor/controller and observed set are now bound. Stress sidecar update_stress.py also requires CORE_ROLE_ENVELOPE.json, absent from this nested bundle and requiring separate recovery. Fresh generation/admission is distinct; no retained snapshot copied into master.'}
 (D/'HISTORICAL_FRONTIER_BINDING.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='historical_levels'}))
