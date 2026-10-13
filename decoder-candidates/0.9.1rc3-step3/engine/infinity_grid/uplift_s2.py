from __future__ import annotations

"""G2:S2 exact observer-kernel quotient over certified repaired S1 pair outcomes.

This stage is non-promoting.  It does not define a G2 state descriptor.  It computes the
coarsest exact quotient that preserves the frozen REALIZED_PUBLIC_ASSEMBLY_V1 observer
used by the S1 realization/congruence audit.
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping
import gzip
import hashlib
import json

from .canon import canonical_sha256


class UpliftS2Error(RuntimeError):
    pass


def _require_sha256(value: str, *, field: str) -> str:
    value=str(value)
    if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise UpliftS2Error(f"{field} must be lowercase sha256")
    return value


def verify_s2_unlock_certificate(
    certificate: Mapping[str, Any], *, expected_parent_source_sha256: str,
    expected_repaired_record_stream_sha256: str,
) -> dict[str, Any]:
    expected_parent_source_sha256=_require_sha256(expected_parent_source_sha256,field='expected_parent_source_sha256')
    expected_repaired_record_stream_sha256=_require_sha256(expected_repaired_record_stream_sha256,field='expected_repaired_record_stream_sha256')
    failures=[]
    if certificate.get('schema_id')!='IG_G2_S2_UNLOCK_CERTIFICATE_V1': failures.append('SCHEMA')
    if certificate.get('status')!='EARNED': failures.append('STATUS')
    if certificate.get('authorizes')!='G2:S2_PAIR_OUTCOME_QUOTIENT_ONLY': failures.append('AUTHORIZATION')
    if certificate.get('g2_graduated') is not False: failures.append('G2_GRADUATION_FIREWALL')
    if certificate.get('source_sha256')!=expected_parent_source_sha256: failures.append('PARENT_SOURCE')
    if certificate.get('repaired_record_stream_sha256')!=expected_repaired_record_stream_sha256: failures.append('REPAIRED_STREAM')
    payload={
        'schema_id':'IG_G2_S2_UNLOCK_VERIFICATION_V1',
        'status':'PASS' if not failures else 'FAIL',
        'failures':failures,
        'certificate_science_sha256':certificate.get('science_sha256'),
        'expected_parent_source_sha256':expected_parent_source_sha256,
        'expected_repaired_record_stream_sha256':expected_repaired_record_stream_sha256,
        'g2_graduated':False,
    }
    payload['science_sha256']=canonical_sha256(payload)
    if failures: raise UpliftS2Error('S2 unlock certificate verification failed: '+','.join(failures))
    return payload


def s2_class_ref(realized_public_sha256: str) -> str:
    return 'G2S2Q:'+_require_sha256(realized_public_sha256,field='realized_public_sha256')


def exact_observer_kernel_quotient(
    repaired_records_path: str | Path,
    audit_signatures_path: str | Path,
    *,
    unlock_certificate: Mapping[str, Any],
    expected_parent_source_sha256: str,
    expected_repaired_record_stream_sha256: str,
) -> dict[str, Any]:
    """Compute the exact S2 quotient from aligned certified S1/audit streams.

    Equivalence is exactly equality of REALIZED_PUBLIC_ASSEMBLY_V1 observer hashes.
    This is the coarsest quotient preserving that declared observer.  Carrier refs,
    construction identities, ancestry and hidden witnesses do not enter the quotient key.
    """
    cert=verify_s2_unlock_certificate(
        unlock_certificate,
        expected_parent_source_sha256=expected_parent_source_sha256,
        expected_repaired_record_stream_sha256=expected_repaired_record_stream_sha256,
    )
    repaired_records_path=Path(repaired_records_path); audit_signatures_path=Path(audit_signatures_path)
    if not repaired_records_path.is_file() or not audit_signatures_path.is_file():
        raise UpliftS2Error('S2 input stream missing')

    input_stream_hash=hashlib.sha256(); rows=0
    s1_to_obs: dict[str,str]={}
    obs_to_s1: dict[str,set[str]]=defaultdict(set)
    obs_record_counts: Counter[str]=Counter()
    outcome_counts: Counter[str]=Counter()

    with gzip.open(repaired_records_path,'rb') as rf, gzip.open(audit_signatures_path,'rb') as sf:
        while True:
            rb=rf.readline(); sb=sf.readline()
            if not rb and not sb: break
            if not rb or not sb: raise UpliftS2Error('aligned S1/audit streams have different lengths')
            input_stream_hash.update(rb)
            r=json.loads(rb); s=json.loads(sb); rows+=1
            out=_require_sha256(str(r['outcome_science_sha256']),field='S1 outcome')
            if str(s.get('outcome_science_sha256'))!=out: raise UpliftS2Error(f'alignment mismatch at row {rows}')
            obs=_require_sha256(str(s['realized_public_sha256']),field='realized public observer')
            prev=s1_to_obs.get(out)
            if prev is not None and prev!=obs:
                raise UpliftS2Error(f'S1 congruence violation for outcome {out}')
            s1_to_obs[out]=obs
            obs_to_s1[obs].add(out)
            obs_record_counts[obs]+=1
            outcome_counts[out]+=1

    observed_stream=input_stream_hash.hexdigest()
    if observed_stream!=expected_repaired_record_stream_sha256:
        raise UpliftS2Error(f'repaired record stream sha mismatch: {observed_stream}')

    # Canonical mapping stream identity without storing the full 576k-line map.
    map_hash=hashlib.sha256()
    for out in sorted(s1_to_obs):
        item={'s1_outcome_science_sha256':out,'s2_class_ref':s2_class_ref(s1_to_obs[out])}
        map_hash.update((json.dumps(item,sort_keys=True,separators=(',',':'))+'\n').encode())

    merged=[]
    for obs in sorted(obs_to_s1):
        members=sorted(obs_to_s1[obs])
        if len(members)>1:
            merged.append({
                's2_class_ref':s2_class_ref(obs),
                'realized_public_sha256':obs,
                's1_outcome_class_count':len(members),
                'record_count':int(obs_record_counts[obs]),
                's1_outcome_science_sha256s':members,
            })

    s1_classes=len(s1_to_obs); s2_classes=len(obs_to_s1)
    class_size_hist=Counter(len(v) for v in obs_to_s1.values())
    record_size_hist=Counter(obs_record_counts.values())
    result={
        'schema_id':'IG_G2_S2_PAIR_OUTCOME_QUOTIENT_V1',
        'schema_version':'1.0.0',
        'status':'PASS',
        'stage_ref':'G2:S2',
        'source_stage':'G2:S1',
        'observer':'REALIZED_PUBLIC_ASSEMBLY_V1',
        'quotient_definition':'EXACT_KERNEL_OF_DECLARED_REALIZED_PUBLIC_OBSERVER',
        'minimality_basis':'COARSEST_EQUIVALENCE_RELATION_PRESERVING_EXACT_DECLARED_OBSERVER_VALUE',
        'record_count':rows,
        's1_repaired_outcome_class_count':s1_classes,
        's2_quotient_class_count':s2_classes,
        's1_classes_merged_by_s2':s1_classes-s2_classes,
        's2_classes_with_multiple_s1_classes':len(merged),
        's1_classes_per_s2_class_histogram':[{'s1_class_count':k,'s2_class_count':v} for k,v in sorted(class_size_hist.items())],
        'records_per_s2_class_histogram':[{'record_count':k,'s2_class_count':v} for k,v in sorted(record_size_hist.items())],
        'merged_class_details':merged,
        'quotient_mapping_stream_sha256':map_hash.hexdigest(),
        'input_repaired_record_stream_sha256':observed_stream,
        'unlock_verification_sha256':cert['science_sha256'],
        'relabel_invariance_status':'PASS_BY_CANONICAL_PREVIOUS_LAYER_PUBLIC_OBSERVER_HASH',
        'hidden_internal_reads_in_quotient_key':False,
        'observer_is_nonpromoting':True,
        'g2_graduated':False,
        'next_stage_if_independently_verified':'G2:S3',
        'nonclaims':[
            'NOT_G2_STATE_DESCRIPTOR',
            'NOT_FINITE_READ_WRITE_STATE',
            'NOT_TRIPLE_IRREDUCIBILITY_RESULT',
            'NO_G2_GRADUATION',
        ],
    }
    result['science_sha256']=canonical_sha256(result)
    return result
