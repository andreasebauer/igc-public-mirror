"""Explicit finite saved-input integration; no production generation authority."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .canon import canonical_sha256

SCOPE = 'QUALIFIED_SAVED_DISCOVERY_INPUTS_ONLY'
FILES = {'prefix_dag','prefix_index','materialized','materialized_audit','full_audit',
         'prefix_capture','prefix_completion','r100_dag','r100_manifest','r100_binding',
         'r100_capture','r100_completion'}


class SavedInputError(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise SavedInputError(code)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)


def raw_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def unchanged(root, binding):
    for role, pin in binding['artifacts'].items():
        require(type(pin) is dict and set(pin)=={'file_name','sha256','size_bytes'}, 'ARTIFACT_PIN_FIELDS')
        relative = Path(pin['file_name'])
        require(not relative.is_absolute() and len(relative.parts)==1, 'ARTIFACT_PATH')
        path = root/relative
        require(not path.is_symlink() and path.is_file(), 'ARTIFACT_FILE')
        require(type(pin['size_bytes']) is int and path.stat().st_size==pin['size_bytes']
                and raw_sha(path)==pin['sha256'], 'SAVED_INPUT_BYTES_CHANGED:'+role)


def completion(capture, done, outcome):
    require(capture['capture_id']==canonical_sha256({k:v for k,v in capture.items() if k!='capture_id'}), 'ORIGIN_CAPTURE_SEAL')
    require(done['completion_sha256']==canonical_sha256({k:v for k,v in done.items() if k!='completion_sha256'}), 'ORIGIN_COMPLETION_SEAL')
    require(done['status']=='COMPLETED' and done['evidence_status']=='VERIFIED'
            and done['scientific_outcome']==outcome, 'ORIGIN_VERIFIED_OUTCOME')
    require(done['source_sha256']==capture['workspace']['source_sha256']
            and done['registration_sha256']==capture['job']['registration_sha256']
            and done['result_sha256']==canonical_sha256(done['result']), 'ORIGIN_REGISTRATION_BINDING')


@dataclass(frozen=True)
class Admission:
    root: Path
    binding_json: str


def preflight(root, binding):
    """Validate origin metadata and bytes before importing any reconstruction code."""
    root = Path(root).resolve(strict=True)
    require(type(binding) is dict and set(binding)=={'schema_id','scope','production_authority',
            'runtime_manifest_sha256','artifacts','prefix_capture_id','prefix_completion_sha256',
            'r100_capture_id','r100_completion_sha256'}, 'BINDING_FIELDS')
    require(binding['schema_id']=='IG_QUALIFIED_SAVED_DISCOVERY_INPUT_BINDING_V1'
            and binding['scope']==SCOPE and binding['production_authority']=='NONE', 'FINITE_INPUT_SCOPE_REQUIRED')
    require(set(binding['artifacts'])==FILES, 'COMPLETE_INPUT_DOMAIN')
    unchanged(root, binding)
    def item(role):return read(root/binding['artifacts'][role]['file_name'])
    pc, pd = item('prefix_capture'), item('prefix_completion')
    rc, rd = item('r100_capture'), item('r100_completion')
    completion(pc,pd,'EXACT_JOINT_SOURCE_FINITE_CONSUMPTION_QUALIFIED_NO_PRODUCTION_AUTHORITY')
    completion(rc,rd,'Q07_VERIFIED_OUTPUTS_REUSED_NO_SCIENTIFIC_REEXECUTION')
    for prefix,c,d in [('prefix',pc,pd),('r100',rc,rd)]:
        require(c['capture_id']==binding[prefix+'_capture_id']
                and d['completion_sha256']==binding[prefix+'_completion_sha256'], 'DECLARED_ORIGIN_IDENTITY')
    pe = {row['path']:row for row in pd['evidence']}
    re = {row['path']:row for row in rd['evidence']}
    for role, name in [('prefix_dag','QUALIFIED_PANELS_STATE_DAG_V2.json'),
            ('prefix_index','STATE_ROOT_INDEX.json'),('materialized','MATERIALIZED_CALIBRATION_CANDIDATE_V2.json')]:
        row = pe['script/work/candidate_inputs/'+name]
        pin = binding['artifacts'][role]
        require(row['sha256']==pin['sha256'] and row['size_bytes']==pin['size_bytes'], 'PREFIX_EVIDENCE_BINDING')
    for role, original in [('materialized_audit','materialized_audit'),('full_audit','full_audit')]:
        pin = binding['artifacts'][role]
        row = next(row for row in pc['objects'] if row['role']=='input:'+original)
        require(row['sha256']==pin['sha256'] and row['size_bytes']==pin['size_bytes'], 'PREFIX_AUDIT_INPUT_BINDING')
    for role, path in [('r100_dag','current_r100/EXACT_STATE_DAG_V1.json'),
            ('r100_manifest','current_r100/CERTIFIED_CARRIER_SET_V1.json'),('r100_binding','PRODUCER_BINDING.json')]:
        row = re['script/work/q07_evidence/script/work/'+path]
        pin = binding['artifacts'][role]
        require(row['sha256']==pin['sha256'] and row['size_bytes']==pin['size_bytes'], 'R100_EVIDENCE_BINDING')
    calibration = item('materialized')
    require(calibration['status']=='CANDIDATE_NO_CONSUMPTION_AUTHORITY'
            and calibration['diagnostic_schema']=='IG_RESOURCE_ACTION_DIAGNOSTIC_V2'
            and calibration['organizational_schema']=='IG_RESOURCE_ORGANIZATIONAL_SIGNATURE_V2'
            and calibration['calibration_installed'] is False and calibration['theorems_adopted'] is False,
            'V2_FINITE_CALIBRATION_SCOPE')
    require(calibration['science_sha256']==canonical_sha256({k:v for k,v in calibration.items() if k!='science_sha256'}), 'CALIBRATION_SEAL')
    producer = item('r100_binding')
    require(producer['producer_experiment_id']=='MASTER.STEP3.CURRENT.R100.Q07'
            and producer['source_sha256']=='76f3bfee70c13d34ac2eafc91060d96b199ee7d9fa794dd20d938417179ce2a2',
            'R100_PRODUCING_BINDING_RETAINED')
    return Admission(root,json.dumps(binding,sort_keys=True,allow_nan=False))


class SavedDiscoveryInputs:
    def __init__(self, admission):
        require(type(admission) is Admission, 'ADMISSION_REQUIRED')
        self.root = admission.root
        self.binding = json.loads(admission.binding_json)
        self.closed = False
        unchanged(self.root,self.binding)
        from .isolated_runtime import BUNDLED_MANIFEST_SHA256, verified_runtime
        require(self.binding['runtime_manifest_sha256']==BUNDLED_MANIFEST_SHA256, 'RUNTIME_BINDING')
        verified_runtime()
        from .maturation_parallel import _states_from_dag, state_wire
        from . import r100_reuse
        def item(role):return read(self.root/self.binding['artifacts'][role]['file_name'])
        dag, index = item('prefix_dag'), item('prefix_index')
        self.engine, self.states = _states_from_dag(dag)
        require(len(self.states)==len(index['roots'])==747, 'PREFIX_747_ROOT_DOMAIN')
        self.rows = index['roots']
        for state,row in zip(self.states,self.rows):
            require(state.construction_digest==row['construction_digest']
                    and canonical_sha256(state_wire(state))==row['public_state_sha256'], 'PREFIX_PUBLIC_CORRESPONDENCE')
        self.packets = {}
        for role, domain, key in [('materialized_audit','selected','levels'),('full_audit','full_pool','depths')]:
            for row in item(role)[key]:
                packet = row['scan']
                require(packet['diagnostic_schema']=='IG_RESOURCE_ACTION_DIAGNOSTIC_V2'
                        and packet['normalized_signature']['schema_id']=='IG_RESOURCE_ORGANIZATIONAL_SIGNATURE_V2', 'PACKET_V2_SCHEMA')
                self.packets[(domain,row['level'])] = packet
        require(set(self.packets)=={('selected',i) for i in range(7,14)}|{('full_pool',i) for i in (11,12,13)}, 'TEN_FINITE_PACKET_DOMAIN')
        # Preserve the original filenames expected by the qualified R100 reader.
        require(self.binding['artifacts']['r100_dag']['file_name']=='EXACT_STATE_DAG_V1.json'
                and self.binding['artifacts']['r100_manifest']['file_name']=='CERTIFIED_CARRIER_SET_V1.json', 'R100_READER_FILENAMES')
        self.r100 = r100_reuse.load(r100_reuse.preflight(self.root,item('r100_binding')))
        unchanged(self.root,self.binding)

    def check(self):
        require(not self.closed, 'SAVED_SESSION_CLOSED')
        unchanged(self.root,self.binding)

    def selected_level(self, level):
        self.check()
        require(type(level) is int and 7<=level<=13, 'SAVED_SESSION_GENERATION_NOT_AUTHORIZED')
        from .materialized_discovery import MaterializedDiscoveryLevel
        states = [s for s,r in zip(self.states,self.rows) if r['domain']=='selected' and r['level']==level]
        require(len(states)==24, 'SELECTED_24_ROOT_DOMAIN')
        return MaterializedDiscoveryLevel(level,deepcopy(self.packets[('selected',level)]),
            {'scope':SCOPE,'production_authority':'NONE','generated_here':False},list(states))

    def packet(self, domain, level):
        self.check()
        require(type(level) is int and (domain,level) in self.packets, 'UNSUPPORTED_SAVED_PACKET')
        return deepcopy(self.packets[(domain,level)])

    def r100_descriptors(self):
        self.check()
        from .r100_reuse import public_descriptor
        return [public_descriptor(state) for state in self.r100]

    def close(self):
        self.states.clear()
        self.r100.clear()
        self.packets.clear()
        self.engine = None
        self.closed = True
