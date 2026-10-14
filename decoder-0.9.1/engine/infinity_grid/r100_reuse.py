"""Exact saved R100 admission; this transport gate grants no scientific graduation."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import carrier_reuse as cr
from .maturation_dag_contract import validate_state_dag

PUBLIC_SCHEMA = 'IG_G1_R100_EXACT_REUSE_PUBLIC_V1'

class R100ReuseError(RuntimeError):
    pass

def public_descriptor(state: Any) -> dict[str, Any]:
    if type(state.level) is not int or state.level != 100 or state.execution_policy != 'NATIVE_V3':
        raise R100ReuseError('R100_CURRENT_LEVEL_POLICY_REQUIRED')
    caps = list(state.total_caps)
    if len(caps) != 7 or any(type(x) is not int or x < 0 for x in caps):
        raise R100ReuseError('R100_EXACT_CAPS7_REQUIRED')
    cr._digest(state.construction_digest, 'construction_digest')
    cr._digest(state.skin, 'resource_skin_sha256')
    return dict(schema_id=PUBLIC_SCHEMA, source_stage='G1:R100', level=100,
                execution_policy='NATIVE_V3', construction_digest=state.construction_digest,
                resource_skin_sha256=state.skin, total_caps=caps)

@dataclass(frozen=True)
class R100Admission:
    root: Path
    identities: tuple[tuple[str, str], ...]
    manifest_sha256: str
    bundle_sha256: str
    ordered_roots: tuple[str, ...]

def preflight(root: str | Path, binding: Mapping[str, str]) -> R100Admission:
    """Inspect exact frozen metadata and DAG shape before loading any authority."""
    root = Path(root)
    try:
        expected = cr._identity(dict(binding))
        manifest_bytes = (root/'CERTIFIED_CARRIER_SET_V1.json').read_bytes()
        manifest = json.loads(manifest_bytes)
        rows = cr._validated_rows(manifest, expected)
        if len(rows) != 193:
            raise R100ReuseError('R100_COMPLETE_193_COHORT_REQUIRED')
        if manifest['exact_payload_schema'] != 'IG_MATURATION_STATE_DAG_V1':
            raise R100ReuseError('R100_NATIVE_DAG_ONLY')
        roots = [r['construction_digest'] for r in rows]
        if roots != sorted(set(roots)) or any(r['carrier_id'] != r['construction_digest'] for r in rows):
            raise R100ReuseError('R100_UNIQUE_SORTED_CONSTRUCTION_IDS_REQUIRED')
        for row in rows:
            p = row['public_state']
            fields = {'schema_id','source_stage','level','execution_policy','construction_digest','resource_skin_sha256','total_caps'}
            if type(p) is not dict or set(p) != fields:
                raise R100ReuseError('R100_PUBLIC_DESCRIPTOR_FIELDS')
            if p['schema_id'] != PUBLIC_SCHEMA or p['source_stage'] != 'G1:R100':
                raise R100ReuseError('R100_PUBLIC_DESCRIPTOR_SCOPE')
            if type(p['level']) is not int or p['level'] != 100 or p['execution_policy'] != 'NATIVE_V3':
                raise R100ReuseError('R100_CURRENT_LEVEL_POLICY_REQUIRED')
            if p['construction_digest'] != row['construction_digest']:
                raise R100ReuseError('R100_PUBLIC_CONSTRUCTION_MISMATCH')
            cr._digest(p['resource_skin_sha256'],'resource_skin_sha256')
            caps=p['total_caps']
            if type(caps) is not list or len(caps) != 7 or any(type(x) is not int or x < 0 for x in caps):
                raise R100ReuseError('R100_EXACT_CAPS7_REQUIRED')
        bundle_bytes=(root/'EXACT_STATE_DAG_V1.json').read_bytes()
        dag=json.loads(bundle_bytes)
        validate_state_dag(dag)
        if dag['execution_policy'] != 'NATIVE_V3':
            raise R100ReuseError('R100_CURRENT_DAG_POLICY_REQUIRED')
        if dag['roots'] != roots:
            raise R100ReuseError('R100_ORDERED_ROOT_CLOSURE')
        for alias in roots:
            node=dag['nodes'][alias]
            if node['level'] != 100:
                raise R100ReuseError('R100_ROOT_LEVEL_REQUIRED')
        return R100Admission(root,tuple(sorted(expected.items())),hashlib.sha256(manifest_bytes).hexdigest(),
                             hashlib.sha256(bundle_bytes).hexdigest(),tuple(roots))
    except R100ReuseError:
        raise
    except Exception as exc:
        raise R100ReuseError('R100_SAVED_INPUT_REFUSED:'+str(exc)) from exc

def load(admission: R100Admission) -> list[Any]:
    """Load one preflighted cohort, checking exact bytes and public correspondence."""
    try:
        def unchanged():
            for name, digest in [('CERTIFIED_CARRIER_SET_V1.json',admission.manifest_sha256),
                                 ('EXACT_STATE_DAG_V1.json',admission.bundle_sha256)]:
                if hashlib.sha256((admission.root/name).read_bytes()).hexdigest()!=digest:
                    raise R100ReuseError('R100_INPUT_CHANGED_AFTER_PREFLIGHT')
        unchanged()
        binding=dict(admission.identities)
        states,manifest=cr.load_certified_carrier_set(admission.root,**{'expected_'+k:v for k,v in binding.items()})
        unchanged()
        if list(states)!=list(admission.ordered_roots):
            raise R100ReuseError('R100_LOADED_ROOT_CLOSURE')
        carriers=[states[k] for k in admission.ordered_roots]
        for state,row in zip(carriers,manifest['carriers']):
            if public_descriptor(state)!=row['public_state']:
                raise R100ReuseError('R100_LOADED_PUBLIC_CORRESPONDENCE')
        return carriers
    except R100ReuseError:
        raise
    except Exception as exc:
        raise R100ReuseError('R100_SAVED_INPUT_REFUSED:'+str(exc)) from exc
