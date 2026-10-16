"""Read the exact saved original and rebased G4 input corpora; no forward authority."""
from pathlib import Path
from io import BytesIO
import hashlib,json,zipfile
from .canon import canonical_sha256
from .g4_term_admission import strict_term_read
from .direct_v2_observation import source_snapshot
from .isolated_runtime import BUNDLED_MANIFEST_SHA256

SCHEMA='IG_HISTORICAL_G4_SAVED_TERM_PLAN_V1'
ROUTE='IG_HISTORICAL_G4_SAVED_TERM_READ@1.0.0'
PINS={
 'original':('8698b9a97f38b3b1a92ce0b6371385767ed10f5fb3826acb5f433bd1e283cd2f','ec0162c79642253f5c91d0dce8a293bdea865b208fa005e9d54dd133e7596141','4dd3d39128861aa23b1278bf18c6efe9181a88114b0ef81cb4128ad4992b8ace','b19fb1edccba7e7768e8034f9b5d323460704ced33774cb3e1435fab3ed6eb3a',2),
 'rebase':('298366d39f1132eff4dbf32964a2791b512080cbb174560ac866a9727fd6a0d9','41fd656d86fb81440c85e1d4314b226021fea6a6a49d9efaed01d2ef9ce06abe','3c463d91898e781915863270195950b4025f72dda33fca71059051d97cd7d1a9','36443cdb52f2810120b3bf167c9747cc6f92d18149b7fb29f867aba170f9c7a6',4)}
class HistoricalG4Error(RuntimeError):pass
def require(ok,why):
 if not ok:raise HistoricalG4Error(why)
def compile_plan(plan):
 expected={'schema_id':SCHEMA,'route':ROUTE,'cohorts':['original','rebase'],'scope':'SAVED_CAPS7_ATOMIC_TERMS_ONLY','generation_allowed':False,'production_authority':'NONE'}
 require(type(plan) is dict and set(plan)==set(expected),'PLAN_FIELDS')
 require(all(type(plan[k]) is type(v) and plan[k]==v for k,v in expected.items()),'PLAN_SCOPE')
 return dict(expected)
def _snapshot(path,pin):
 raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==pin,'ARCHIVE_PIN');return raw
def _one(z,predicate):
 names=[n for n in z.namelist() if predicate(n)];require(len(names)==1,'AMBIGUOUS_ARCHIVE_ROLE');return json.loads(z.read(names[0]))
def _seal(v,field='science_sha256'):
 require(v.get(field)==canonical_sha256({k:x for k,x in v.items() if k!=field}),'SELF_SEAL')
def execute(plan,archives):
 compiled=compile_plan(plan);require(type(archives) is dict and set(archives)==set(PINS),'INPUT_SET')
 # Snapshot all bytes before processing; a later replacement cannot mix sources.
 raws={c:_snapshot(archives[c],p[0]) for c,p in PINS.items()};result=[]
 current_source=source_snapshot()
 for cohort,pin in PINS.items():
  with zipfile.ZipFile(BytesIO(raws[cohort])) as z:
   require(len(z.namelist())==len(set(z.namelist())),'DUPLICATE_ZIP_PATH')
   suffix='G4_S0_RESULT_V2.json' if cohort=='original' else 'RESULT.json'
   primary=_one(z,lambda n:'/primary/' in n and n.endswith(suffix))
   cold=_one(z,lambda n:'/cold/' in n and n.endswith(suffix))
   cert=_one(z,lambda n:'/certification/' in n and n.endswith('G4_S0_CERTIFIED_CLOSEOUT_V2.json' if cohort=='original' else 'G4_S0_REBASE_CERTIFIED_CLOSEOUT.json'))
   replay=_one(z,lambda n:'/certification/' in n and n.endswith('G4_S0_COLD_REPLAY_COMPARISON_V2.json' if cohort=='original' else 'G4_S0_REBASE_REPLAY_COMPARISON.json'))
  corpus=primary['certified_term_corpus']
  # Primary science seals intentionally exclude execution metadata in the old source.
  # Immutable archive + the original explicit science/source/corpus seals bind it.
  require(primary['status']==cold['status']=='PASS' and primary['science_sha256']==cold['science_sha256']==pin[1],'PRIMARY_COLD_SCIENCE')
  require(primary['source_sha256']==cold['source_sha256']==cert['source_sha256']==pin[2],'SOURCE_SEAL')
  require(corpus==cold['certified_term_corpus'] and corpus['science_sha256']==cert['certified_term_corpus_sha256']==pin[3],'CORPUS_BINDING')
  _seal(corpus)
  require(cert['status']=='CERTIFIED_PASS' and replay['status']=='PASS' and replay['certification']=='CERTIFIED_PASS','CERTIFICATION')
  require(replay['primary_science_sha256']==replay['cold_science_sha256']==pin[1],'REPLAY_BINDING')
  require(corpus['lower_layer_rematerialization'] is False and corpus['status']=='PASS','ATOMIC_BOUNDARY')
  require(set(corpus['terms'])==set('AB' if cohort=='original' else 'ABCD'),'TERM_KEYS')
  rows=[]
  for key,record in sorted(corpus['terms'].items()):
   wire=record['term'];binding={'recipe_ref':'STRICT_G4_ATOMIC_TERM_READ_V1','source_ref':current_source,'runtime_ref':BUNDLED_MANIFEST_SHA256,'input_root_ref':canonical_sha256(wire),'task_ref':cohort+':'+key}
   read=strict_term_read(wire,binding);require(read['term']==wire,'EXACT_WIRE_READ')
   rows.append({'cohort':cohort,'key':key,'original_record':record,'strict_read':read})
  result.append({'cohort':cohort,'archive_sha256':pin[0],'original_science_sha256':pin[1],'original_source_sha256':pin[2],'corpus_sha256':pin[3],'terms':rows,'forward_use':'NOT_GRANTED_BY_THIS_READ'})
 return {'schema_id':'IG_HISTORICAL_G4_SAVED_TERMS_READ_RESULT_V1','compiled':compiled,'cohorts':result,'source_scoped_terms':6,'original_terms':2,'rebase_terms':4,'cohorts_merged':False,'production_authority':'NONE','generation_allowed':False,'lower_layer_rematerialization':False,'complete_historical_g4_outputs':False}
