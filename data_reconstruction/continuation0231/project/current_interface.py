from __future__ import annotations
from importlib.resources import files
from typing import Any, Iterable, Mapping, Sequence
import json
from infinity_grid.canon import canonical_sha256
from infinity_grid.uplift_architecture import uplift_contract
_SPEC_RESOURCE="resources/uplift/G_UPLIFT_S0_S1_IMPLEMENTATION_SPEC_V2.json"
class UpliftStructuralError(RuntimeError):
    pass

def _load_spec() -> dict[str, Any]:
    obj = json.loads(files('infinity_grid').joinpath(_SPEC_RESOURCE).read_text(encoding='utf-8'))
    expected = obj.get('spec_sha256')
    payload = {k: v for k, v in obj.items() if k != 'spec_sha256'}
    observed = canonical_sha256(payload)
    if expected != observed:
        raise UpliftStructuralError(f'S0/S1 implementation spec identity mismatch: expected {expected}, observed {observed}')
    if obj.get('architecture_contract_sha256') != uplift_contract()['contract_sha256']:
        raise UpliftStructuralError('S0/S1 implementation spec is not bound to the frozen uplift architecture')
    return obj

def implementation_spec() -> dict[str, Any]:
    return _load_spec()

def _require_sha256(value: str, *, field: str) -> str:
    value = str(value)
    if len(value) != 64 or any((c not in '0123456789abcdef' for c in value)):
        raise UpliftStructuralError(f'{field} must be a lowercase SHA-256 hex digest')
    return value

def _semantic_interface_payload(*, boundary_skin: str, total_caps: Sequence[int], reservations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {'schema_id': 'IG_G1_PUBLIC_ONE_ENDPOINT_INTERFACE_SEMANTICS_V1', 'boundary_resource_skin_sha256': str(boundary_skin), 'total_free_by_type': [int(x) for x in total_caps], 'one_endpoint_reservations': [{'endpoint_type': int(r['endpoint_type']), 'available': bool(r['available']), 'successor_boundary_resource_skin_sha256': r.get('successor_boundary_resource_skin_sha256'), 'successor_total_free_by_type': r.get('successor_total_free_by_type')} for r in reservations], 'scope': 'ONE_EXTERNAL_ENDPOINT_RESERVATION_FOR_WHOLE_CARRIER_PAIR_CONNECTION'}

def extract_carrier_interface(carrier: Any, *, source_stage: str, source_authority_sha256: str) -> dict[str, Any]:
    """Extract the frozen S0 public interface from one complete G1 carrier.

    The only carrier reads allowed here are the earned public boundary/resource
    surface: ``construction_digest`` (provenance ref), ``skin``, ``total_caps``
    and the public ``reserve_external(type)`` operation.  The reservation witness
    is intentionally discarded so internal owner/path information never enters
    the uplift state.
    """
    if source_stage != 'G1:R100':
        raise UpliftStructuralError(f'v1 S0 implementation is frozen to G1:R100, got {source_stage!r}')
    source_authority_sha256 = _require_sha256(source_authority_sha256, field='source_authority_sha256')
    carrier_ref = str(carrier.construction_digest)
    _require_sha256(carrier_ref, field='carrier construction_digest')
    boundary_skin = str(carrier.skin)
    _require_sha256(boundary_skin, field='carrier skin')
    total_caps = tuple((int(x) for x in carrier.total_caps))
    if len(total_caps) != 7 or any((x < 0 for x in total_caps)):
        raise UpliftStructuralError('G1 public resource vector must contain seven nonnegative counters')
    reservations: list[dict[str, Any]] = []
    for t in range(7):
        if total_caps[t] <= 0:
            reservations.append({'endpoint_type': t, 'available': False, 'successor_boundary_resource_skin_sha256': None, 'successor_total_free_by_type': None})
            continue
        try:
            successor, _hidden_witness_discarded = carrier.reserve_external(t)
        except Exception as exc:
            raise UpliftStructuralError(f'public reserve_external({t}) failed despite positive capacity') from exc
        succ_skin = str(successor.skin)
        _require_sha256(succ_skin, field=f'reserve_external({t}) successor skin')
        succ_caps = [int(x) for x in successor.total_caps]
        if len(succ_caps) != 7 or succ_caps[t] != total_caps[t] - 1:
            raise UpliftStructuralError(f'reserve_external({t}) does not consume exactly one public type-{t} endpoint')
        for j in range(7):
            expected = total_caps[j] - (1 if j == t else 0)
            if succ_caps[j] != expected:
                raise UpliftStructuralError(f'reserve_external({t}) changed public type-{j} counter unexpectedly')
        reservations.append({'endpoint_type': t, 'available': True, 'successor_boundary_resource_skin_sha256': succ_skin, 'successor_total_free_by_type': succ_caps})
    semantic = _semantic_interface_payload(boundary_skin=boundary_skin, total_caps=total_caps, reservations=reservations)
    interface_sha = canonical_sha256(semantic)
    row = {'schema_id': 'IG_G_UPLIFT_CARRIER_INTERFACE_V1', 'schema_version': '1.0.0', 'uplift_layer': 2, 'stage_ref': 'G2:S0', 'source_stage': source_stage, 'carrier_ref': carrier_ref, 'interface_sha256': interface_sha, 'boundary_resource_skin_sha256': boundary_skin, 'total_free_by_type': list(total_caps), 'one_endpoint_reservations': reservations, 'hidden_internal_reads': False, 'public_operation_witnesses_retained': False, 'source_authority_sha256': source_authority_sha256, 'implementation_spec_sha256': implementation_spec()['spec_sha256'], 'relabel_invariance_status': 'PASS_BY_PREVIOUS_LAYER_CANONICAL_PUBLIC_BOUNDARY', 'nonclaims': ['NOT_A_COMPLETE_UNBOUNDED_FUTURE_INTERFACE', 'NOT_A_G2_STATE_DESCRIPTOR', 'NOT_G2_GRADUATION']}
    row['science_sha256'] = canonical_sha256(row)
    return row

def extract_interface_population(carriers: Iterable[Any], *, source_stage: str, source_authority_sha256: str) -> dict[str, Any]:
    rows = [extract_carrier_interface(c, source_stage=source_stage, source_authority_sha256=source_authority_sha256) for c in carriers]
    rows.sort(key=lambda r: r['carrier_ref'])
    refs = [r['carrier_ref'] for r in rows]
    if len(refs) != len(set(refs)):
        raise UpliftStructuralError('S0 population requires unique carrier refs')
    classes: dict[str, int] = {}
    for r in rows:
        classes[r['interface_sha256']] = classes.get(r['interface_sha256'], 0) + 1
    obj = {'schema_id': 'IG_G_UPLIFT_S0_INTERFACE_POPULATION_V1', 'schema_version': '1.0.0', 'status': 'PASS', 'stage_ref': 'G2:S0', 'source_stage': source_stage, 'carrier_count': len(rows), 'interface_class_count': len(classes), 'interface_class_histogram': [{'interface_sha256': k, 'carrier_count': classes[k]} for k in sorted(classes)], 'interfaces': rows, 'carrier_ref_set_sha256': canonical_sha256(refs), 'source_authority_sha256': _require_sha256(source_authority_sha256, field='source_authority_sha256'), 'implementation_spec_sha256': implementation_spec()['spec_sha256'], 'g2_graduated': False}
    obj['science_sha256'] = canonical_sha256(obj)
    return obj
