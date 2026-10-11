"""Bind matched O7-to-O8 generation inputs without scientific dispatch."""
import hashlib, importlib.util, json, pathlib, shutil, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'decoder156'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(n,x):(ROOT/'matched_protocol'/n).write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def require(x,label):
 if not x:raise RuntimeError(label)
def main():
 require(shutil.disk_usage(ROOT).free>=10*1024**3,'SPACE_FLOOR')
 manifest=json.loads((ROOT/'candidate156/dev156/SOURCE_MANIFEST.json').read_text())
 actual={p.relative_to(ROOT/'decoder156').as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (ROOT/'decoder156').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 require(actual==manifest['candidate_files'],'EXACT_SOURCE')
 require(sha(ROOT/'candidate156/dev156/SOURCE_MANIFEST.json')=='a2ba9e3fb9728db79f0dcd713473b3e0dc57ddf986eb03de708adc8efc84409c','SOURCE_MANIFEST_PIN')
 from infinity_grid.isolated_runtime import verified_runtime,BUNDLED_MANIFEST_SHA256
 runtime=verified_runtime()
 from infinity_grid.frontier import _load_engine
 from infinity_grid.regime_scanner import load_motif_library,load_regime_scanner_spec,BACKBONE_MOTIFS
 engine=_load_engine(next((ROOT/'kernel').iterdir()))
 _,ops=engine.O6.load_rules();pairs=sorted(tuple(map(int,x)) for x in ops)
 require(len(pairs)==31 and len(set(pairs))==31,'OPERATOR_CENSUS')
 motifs=load_motif_library();spec=load_regime_scanner_spec();old=json.loads((ROOT/'recovery/FROZEN_RECIPE_REGISTER.json').read_text())
 require(len(motifs)==139 and spec['panel']==old['panel'],'FROZEN_SPEC')
 recipes=[]
 for mi,m in enumerate(motifs):
  n=int(m['n']);edges=m['edges']
  recipes.append(dict(n=n,edges=edges,lane='HOM',motif_id=f'HOM:{n}:{mi}',schedule_seed=mi%len(pairs),force_pair=None))
  if n in (4,5) or (n==6 and mi%6==0):recipes.append(dict(n=n,edges=edges,lane='HET',motif_id=f'HET:{n}:{mi}',schedule_seed=(mi*3+1)%len(pairs),force_pair=None))
  if n==4:recipes.append(dict(n=n,edges=edges,lane='MIX',motif_id=f'MIX:4:{mi}',schedule_seed=(mi*5+2)%len(pairs),force_pair=None))
 require(recipes==old['recipes']['False'],'RECIPE_ORDER')
 twin=old['recipes']['True'][-2:];require(old['recipes']['True']==recipes+twin,'TWINS')
 seed=next((ROOT/'material').iterdir());records_path=seed/'graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json';records=json.loads(records_path.read_text())['records']
 require(len(records)==205,'O7_COHORT_CENSUS')
 require(records==json.loads((seed/'evidence/O7_IMMUTABLE_SURVIVORS.json').read_text())['records'],'O7_DUAL_COPY')
 retained=[json.loads(x) for x in (ROOT/'recovery/identity/results/IDENTITY_MAPPING.jsonl').read_text().splitlines() if json.loads(x)['role']=='retained'];by={r['record']:r for r in retained}
 cohort=[]
 for i,r in enumerate(records):
  pin=by[f'O7_SURVIVOR:{i:03d}']
  require(r['edges']==pin['raw_edges'] and r['parent_ids']==pin['parent_ids'],'COHORT_COORDINATES')
  cohort.append({'index':i,'historical_source_id':r['state_digest'],'record_sha256':digest(r),'historical_context_identity':pin['old_context_identity'],'v3_identity':pin['new_global_identity']})
 require(len({x['index'] for x in cohort})==205,'UNIQUE_OCCURRENCES')
 write('HISTORICAL_ALIAS_GROUPS.json',{'distinct_aliases':len({x['historical_source_id'] for x in cohort}),'groups':{k:[x['index'] for x in cohort if x['historical_source_id']==k] for k in sorted({x['historical_source_id'] for x in cohort})},'binding':'occurrence index plus exact record hash; historical alias alone is not a unique occurrence key'})
 write('O7_COHORT_BINDING.json',cohort);write('ORDERED_OPERATORS.json',pairs);write('ORDERED_RECIPES.json',{'without_twins':recipes,'with_twins':recipes+twin,'backbone':BACKBONE_MOTIFS})
 kernel_files={p.relative_to(ROOT/'kernel').as_posix():sha(p) for p in (ROOT/'kernel').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 write('KERNEL_FILE_BINDING.json',kernel_files)
 pins={str(p.relative_to(ROOT)):sha(p) for p in [records_path,ROOT/'recovery/FROZEN_RECIPE_REGISTER.json',ROOT/'recovery/g1/decoder/regime_scanner.py',ROOT/'recovery/g1/BOOTSTRAP100_HISTORICAL.json.gz',ROOT/'recovery/overlay/results/MAPPING.json',ROOT/'candidate156/dev156/SOURCE_MANIFEST.json',ROOT/'baseline/runtime/IG_RUNTIME_CPY3135_SQLITE3513.tar.gz',pathlib.Path(__file__)]}
 for n in ['O7_COHORT_BINDING.json','ORDERED_OPERATORS.json','ORDERED_RECIPES.json','HISTORICAL_ALIAS_GROUPS.json','KERNEL_FILE_BINDING.json']:pins['matched_protocol/'+n]=sha(ROOT/'matched_protocol'/n)
 protocol={'schema':'IG_DEV156_MATCHED_GENERATION_PROTOCOL_V1','status':'FROZEN_INPUT_PROTOCOL_NOT_EXECUTED','inputs':pins,'source_files':len(actual),'runtime_manifest_sha256':BUNDLED_MANIFEST_SHA256,'runtime':str(runtime),'scope':{'first_gate':'O7 selection and complete O8 recipe generation','o7_authority_records':205,'historical_aliases':163,'record_binding':'ordered occurrence index and exact record hash; preserve repeated aliases','o7_beam':24,'reserved_o7_mapping_states':193,'reserved_mapping_role':'ancestry evidence only, never substitute for authority starting cohort','operators':31,'motifs':139,'recipes_without_twins':191,'recipes_with_twins':193,'candidate_cap':220,'subsequent_levels':'O9-O100 only after O8 comparison is explained and recorded'},'arms':{'historical':'sealed historical scanner with original construction identity and reservation ordering','candidate':'exact dev156 scanner with V3 construction identity and native reservation operation'},'procedure':['Load the identical 205 immutable O7 records with exact coordinates in both arms; validate base skins and capacity vectors.','Record each original-to-V3 base construction identity and every farthest-point feature, normalized vector, tie and selection decision. Select each arm natively; retain both ordered 24-state beams.','Record capacity totals, median, center and ordered heterogeneous/mixed owner choices separately for each arm.','Generate all frozen recipes in order, without early candidate cap, deduplication or beam cut. Keep every failed attempt with the first impossible edge/type.','For every attempted edge record sorted edge order, rotated ordered operator search, capacity predicates, chosen type pair, before/after construction keys, skins and capacities, and exact native reservation witness paths.','Compare complete raw candidate sequences to the sealed historical implementation and retained historical DAG where evidence exists. A retained DAG omits raw failed/discarded candidates and does not prove raw closure.','Only after recording raw comparison perform cap, deduplication, mandatory twins, farthest selection and fixed backbone; record every disposition and selection tie.','Stop at the first unexplained discrepancy. Identity/order-driven differences require an explicit diagnosis and may not be silently normalized or accepted.'], 'execution':{'workers':1,'memory_bytes':4294967296,'minimum_free_bytes':10737418240,'overall_deadline':None,'background':True,'checkpoint':'per arm and recipe; preregistration/source/runtime/input/result hashes; atomic durable write; reuse verified completed recipes only'},'acceptance':{'full_p0_closed':False,'native_generation_executed':False,'independent_host_qualified':False,'raw_pre_dedup_closed':False,'historical_identity_rewrites':0,'controller_transition':False,'promotion':False},'remaining_historical_limitations':['Earlier O7 action-stream cap remains unresolved.','Q2 historical operational/projection witnesses remain incomplete.','Recomputed generation may differ due to construction ordering; retained-witness replay alone is insufficient.']}
 write('PROTOCOL.json',protocol)
 write('PREFLIGHT_RESULT.json',{'status':'PASS_INPUT_BINDING','source_files':1558,'runtime_files':4729,'python':'3.13.5','sqlite':'3.51.3','o7_cohort':len(cohort),'ordered_operators':len(pairs),'ordered_motifs':len(motifs),'recipes':[len(recipes),len(recipes+twin)],'generation_calls':0,'full_p0_closed':False})
 print(json.dumps({'status':'PASS_INPUT_BINDING','o7_cohort':len(cohort),'operators':len(pairs),'recipes':[len(recipes),len(recipes+twin)]}),flush=True)
if __name__=='__main__':main()
