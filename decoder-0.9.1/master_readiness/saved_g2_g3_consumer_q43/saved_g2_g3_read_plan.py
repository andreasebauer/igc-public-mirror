"""Original saved G2/G3 records: explicit finite reads, never state reconstruction."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import hashlib
import json
import zipfile
from .canon import canonical_sha256

SCHEMA = 'IG_ORIGINAL_G2_G3_SAVED_READ_PLAN_V1'
ROUTE = 'IG_ORIGINAL_G2_G3_SAVED_READ@1.0.0'
PINS = {
 'g2': ('2d1b2c870c8367884271de59c0859ce4f2961130bdf0b2b0579d3b1863f9cb46', 'e7fe5a41ab9aeda3022c7588b19118537467e5f658e9335c7787158941f9a807', '64d503d6c9680df893618b84c16f45af7592ae124ed6bd758b21e1f83110ff2f'),
 's0': ('985f92f110b4cab744becc36cba108a41bb053dabec8f9c99803ed719d5a566a', 'ff1e28a46ee09c2b8d9baedea9c55b67a2424f3ccce367eb7d782aa8c5c2b9ff', '42832d682c1e0175f6f7c02eff6047f9ba8bf2a165c6017d8f4776166e766199'),
 's1': ('0a2a0644a31f6f9d3378df65fd6e8a4a970a4a609f280c7c4e454495c8aa922c', 'f7f0a1753919aaa1380c39191b9a65ee2ea864aa65f1c0f101078f9c2ea91871', '0d4ab3493930ac1b0b3b29deaca52496ffacf30484d3742e8b4e54c7d4937e7c')}
KINDS = {'saved_result': set(PINS), 's0_challenge_records': {'s0'}, 's1_target_summaries': {'s1'}}

class SavedAuthorityReadError(ValueError):
    pass

def require(ok, reason):
    if not ok:
        raise SavedAuthorityReadError(reason)

def compile_plan(plan):
    require(type(plan) is dict and set(plan) == {'schema_id','route','requests','generation_allowed','production_authority'}, 'SAVED_AUTHORITY_PLAN_FIELDS')
    require(plan['schema_id'] == SCHEMA and plan['route'] == ROUTE and plan['generation_allowed'] is False and plan['production_authority'] == 'NONE', 'SAVED_AUTHORITY_PLAN_SCOPE')
    requests = plan['requests']
    require(type(requests) is list and requests and len({canonical_sha256(r) for r in requests}) == len(requests), 'SAVED_AUTHORITY_REQUEST_LIST')
    for r in requests:
        require(type(r) is dict and set(r) == {'kind','source_scope'}, 'SAVED_AUTHORITY_REQUEST_FIELDS')
        require(type(r['kind']) is str and type(r['source_scope']) is str, 'SAVED_AUTHORITY_REQUEST_TYPES')
        require(r['kind'] in KINDS and r['source_scope'] in KINDS[r['kind']], 'FULL_CARRIER_OR_CONTEXT_PAYLOAD_NOT_ADMITTED')
    return {'schema_id':'IG_COMPILED_ORIGINAL_G2_G3_SAVED_READ_V1', 'route':ROUTE, 'plan_sha256':canonical_sha256(plan), 'requests':deepcopy(requests), 'required_theorems':[], 'generation_allowed':False, 'production_authority':'NONE'}

def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

def _reject_duplicate_keys(pairs):
    out = {}
    for k,v in pairs:
        require(k not in out, 'SAVED_AUTHORITY_DUPLICATE_JSON_KEY')
        out[k] = v
    return out

def _json(raw):
    return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)

def _load_archives(paths):
    require(type(paths) is dict and set(paths) == set(PINS), 'SAVED_AUTHORITY_ARCHIVE_SET')
    data = {}; provenance = []
    for scope,(archive_hash,producer,science) in PINS.items():
        # Pin one immutable byte snapshot, then decode that snapshot only.
        raw = Path(paths[scope]).read_bytes()
        require(_sha(raw) == archive_hash, 'SAVED_AUTHORITY_ARCHIVE_BYTES')
        with zipfile.ZipFile(BytesIO(raw)) as z:
            names = z.namelist()
            require(len(names) == len(set(names)), 'SAVED_AUTHORITY_DUPLICATE_MEMBER')
            pre = 'closeout/' if scope == 'g2' else ''
            root = pre+'decoder_source/src/infinity_grid/'
            manifest = {n[len(root):]:{'sha256':_sha(z.read(n)), 'size_bytes':len(z.read(n))}
                for n in sorted(names) if n.startswith(root) and not n.endswith('/') and Path(n).name != '_build_meta.json' and '__pycache__' not in Path(n).parts and not n.endswith('.pyc')}
            require(canonical_sha256(manifest) == producer, 'SAVED_AUTHORITY_ORIGINAL_PRODUCER_BYTES')
            rows = [_json(z.read(pre+'science/'+phase+'_RESULT.json')) for phase in ('PRIMARY','COLD')]
            for r in rows:
                payload = {k:v for k,v in r.items() if k not in ('science_sha256','source_sha256','source_version','registry_sha256','execution_metadata')}
                require(canonical_sha256(payload) == r['science_sha256'] == science and r['source_sha256'] == producer, 'SAVED_AUTHORITY_SCIENCE_OR_PRODUCER')
            data[scope] = rows[0]
            if scope == 'g2':
                cert = _json(z.read(pre+'authority/G2_GRADUATION_CERTIFICATE.json'))
                require(canonical_sha256({k:v for k,v in cert.items() if k != 'science_sha256'}) == cert['science_sha256'] == '025713aab599dacbcffe397721d781e5854bc4216245a21bfd70f92c34a8a209', 'SAVED_AUTHORITY_CERTIFICATE')
            provenance.append({'scope':scope,'archive_sha256':archive_hash,'original_source_sha256':producer,'original_science_sha256':science,'original_source_version':rows[0]['source_version']})
    require(data['s0']['authority']['g2_graduation_science_sha256'] == cert['science_sha256'] and data['s0']['authority']['g2_r0_science_sha256'] == data['g2']['science_sha256'], 'SAVED_AUTHORITY_G2_S0_EDGE')
    require(data['s1']['authority']['g3_s0_science_sha256'] == data['s0']['science_sha256'], 'SAVED_AUTHORITY_S0_S1_EDGE')
    return data,provenance

def execute(plan, paths):
    # Refuse stronger claims before reading even one archive or importing producers.
    compiled = compile_plan(plan)
    data,provenance = _load_archives(paths)
    observations = []
    for request in compiled['requests']:
        scope = request['source_scope'];kind = request['kind'];r = data[scope]
        if kind == 'saved_result':
            payload = r;profile = 'ORIGINAL_SAVED_RESULT_NOT_EXECUTABLE_STATE'
        elif kind == 's0_challenge_records':
            payload = r['challenge_corpus']['records'];profile = 'SAVED_PUBLIC_AND_HIDDEN_DIAGNOSTIC_RECORDS_NOT_FULL_IMPLEMENTATION_CARRIERS'
        else:
            payload = r['target_behavior_summaries'];profile = 'SAVED_TARGET_SUMMARIES_NOT_RAW_CONTEXT_PAYLOADS'
        observations.append({'request':deepcopy(request),'payload':deepcopy(payload),'payload_profile':profile,'provenance':deepcopy(next(p for p in provenance if p['scope'] == scope))})
    return {'schema_id':'IG_ORIGINAL_G2_G3_SAVED_READ_RESULT_V1','compiled':compiled,'observations':observations,'generation_allowed':False,'production_authority':'NONE','historical_current_equivalence':False,'execution_reuse_authorized':False,'required_theorems':[]}
