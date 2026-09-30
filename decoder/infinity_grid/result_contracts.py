"""Frozen, data-only result checks. Scientific negatives can satisfy a contract."""
from pathlib import Path
import hashlib
import json

from .canon import canonical_sha256
from . import submission as sub

SCHEMA = 'IG_DECODER_RESULT_CONTRACT_V1'
SCIENCE_SCHEMA = 'IG_DECODER_RESULT_CONTRACT_V2'
SCIENCE_POLICY = 'EXPLICIT_ARTIFACT_CONTENT_V1'
_PROCESS_OUTCOMES = {'PASS', 'FAIL', 'PROCESS_COMPLETED', 'PROCESS_FAILED', 'COMPLETED', 'SUCCESS'}


def _content_check(check):
    # This is a syntactic admission floor, not an assessment of scientific truth.
    pointer_text = check['pointer']
    if not pointer_text or pointer_text.split('/')[1] in {'status', 'return_code', 'exit_code'}:
        return False
    if pointer_text == '/outcome':
        values = check.get('one_of', [check.get('equals')])
        return ('count' not in check and all(isinstance(v, str) and v not in _PROCESS_OUTCOMES for v in values))
    return True


def _science_policy(contract, question):
    if contract['schema_id'] != SCIENCE_SCHEMA:
        raise sub.SubmissionError('SCIENCE_V2_EXPLICIT_CONTENT_REQUIRED')
    paths = contract.get('scientific_content')
    if (type(paths) is not list or not paths or any(type(p) is not str for p in paths)
            or len(set(paths)) != len(paths)):
        raise sub.SubmissionError('SCIENTIFIC_CONTENT_PATHS')
    rows = {row['path']: row for row in contract['required_artifacts']}
    for path in paths:
        row = rows.get(path)
        if row is None or not ('sha256' in row or any(_content_check(c) for c in row.get('json_checks', []))):
            raise sub.SubmissionError('SCIENTIFIC_CONTENT_CHECK_REQUIRED', path)
    outcomes = question.get('outcomes')
    if (type(outcomes) is not list or not outcomes or any(type(v) is not str or not v for v in outcomes)
            or not any(v not in _PROCESS_OUTCOMES for v in outcomes)
            or len(set(outcomes)) != len(outcomes)):
        raise sub.SubmissionError('SCIENTIFIC_OUTCOMES_REQUIRED')


def _strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value): raise ValueError('nonfinite JSON value')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=nonfinite)
    # Also rejects overflow such as 1e999, which parse_constant does not see.
    json.dumps(value, allow_nan=False)
    return value


def _validate_science(contract):
    raw = {k: v for k, v in contract.items() if k != 'declared_outcomes'}
    normalize(raw, {'kind': 'SCRIPT'}, {'outcomes': contract.get('declared_outcomes')})



def normalize(contract, execution, question):
    if 'schema_id' in contract and contract['schema_id'] not in {SCHEMA, SCIENCE_SCHEMA}:
        raise sub.SubmissionError('RESULT_CONTRACT_SCHEMA')
    if contract.get('schema_id') not in {SCHEMA, SCIENCE_SCHEMA}:
        kind = execution['kind']
        engineering = execution.get('handler_ref') == 'infinity_grid.change_validation:validate_revision'
        return {'schema_id': SCHEMA, 'claim': 'VALIDATION' if kind == 'VALIDATION' or engineering else 'EXECUTION_ONLY',
                'required_artifacts': [], 'result_checks': [{'pointer': '/status' if kind == 'VALIDATION' else '/outcome', 'equals': 'PASS'}] if kind == 'VALIDATION' or engineering else [],
                'prerequisites': [], 'preservation': {}, 'legacy_description': contract}
    allowed = {'schema_id', 'claim', 'required_artifacts', 'result_checks', 'prerequisites', 'preservation'}
    if contract.get('schema_id') == SCIENCE_SCHEMA:
        allowed = allowed | {'scientific_content'}
        if contract.get('claim') != 'SCIENCE': raise sub.SubmissionError('SCIENCE_V2_CLAIM_REQUIRED')
    if set(contract) not in (allowed, allowed|{'outcome'}) or contract['claim'] not in {'SCIENCE', 'VALIDATION', 'EXECUTION_ONLY'}:
        raise sub.SubmissionError('RESULT_CONTRACT_FIELDS')
    if any(type(contract[k]) is not list for k in ('required_artifacts', 'result_checks', 'prerequisites')):
        raise sub.SubmissionError('RESULT_CONTRACT_LIST')
    if contract['claim'] == 'SCIENCE' and not (contract['required_artifacts'] or contract['result_checks']):
        raise sub.SubmissionError('SCIENTIFIC_RESULT_CHECK_REQUIRED')
    paths = []
    for row in contract['required_artifacts']:
        if not isinstance(row, dict) or not {'path'} <= set(row) <= {'path', 'sha256', 'json_checks'}:
            raise sub.SubmissionError('RESULT_ARTIFACT_FIELDS')
        sub._relative(row['path']); paths.append(row['path'])
        if 'sha256' in row and not _digest(row['sha256']): raise sub.SubmissionError('RESULT_ARTIFACT_HASH')
        _checks(row.get('json_checks', []))
    if len(paths) != len(set(paths)): raise sub.SubmissionError('RESULT_ARTIFACT_DUPLICATE')
    if 'outcome' in contract:
        rule=contract['outcome']
        if not isinstance(rule,dict) or set(rule)!={'artifact','pointer'} or rule['artifact'] not in paths:
            raise sub.SubmissionError('SCIENTIFIC_OUTCOME_ARTIFACT_REQUIRED')
        _checks([{'pointer':rule['pointer'],'one_of':question.get('outcomes', [])}])
    _checks(contract['result_checks'])
    for row in contract['prerequisites']:
        if set(row) != {'capsule_sha256', 'completion_sha256'} or not all(_digest(v) for v in row.values()):
            raise sub.SubmissionError('PREREQUISITE_FIELDS')
    if contract['claim'] == 'SCIENCE': _science_policy(contract, question)
    policy(contract)
    canonical_sha256(contract)
    return dict(contract,declared_outcomes=list(question.get('outcomes',[])))


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _checks(rows):
    if type(rows) is not list: raise sub.SubmissionError('RESULT_CHECK_LIST')
    for row in rows:
        if not isinstance(row, dict) or len(row) != 2 or 'pointer' not in row or not ({'equals', 'one_of', 'count'} & set(row)):
            raise sub.SubmissionError('RESULT_CHECK_FIELDS')
        p = row['pointer']
        if not isinstance(p, str) or (p and not p.startswith('/')): raise sub.SubmissionError('RESULT_JSON_POINTER')
        if 'one_of' in row and (type(row['one_of']) is not list or not row['one_of']): raise sub.SubmissionError('RESULT_CHECK_CHOICES')
        if 'count' in row and (type(row['count']) is not int or row['count'] < 0): raise sub.SubmissionError('RESULT_CHECK_COUNT')


def policy(contract):
    defaults = {'interval_seconds': 30, 'max_pending_bytes': 536870912,
                'max_commit_bytes': 201326592, 'max_pending_commits': 8,
                'max_pending_age_seconds': 3600}
    custom = contract.get('preservation', {})
    if type(custom) is not dict or set(custom) - set(defaults): raise sub.SubmissionError('PRESERVATION_POLICY_FIELDS')
    defaults.update(custom)
    if any(type(v) is not int or v <= 0 for v in defaults.values()): raise sub.SubmissionError('PRESERVATION_POLICY_VALUE')
    if defaults['max_commit_bytes'] > defaults['max_pending_bytes']: raise sub.SubmissionError('PRESERVATION_RESERVE_REQUIRED')
    return defaults


def pointer(obj, text):
    if not text: return obj
    for part in text[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        if isinstance(obj, list):
            if not part.isdigit() or (len(part) > 1 and part.startswith('0')): raise KeyError(text)
            obj = obj[int(part)]
        else: obj = obj[part]
    return obj


def evaluate_checks(obj, checks):
    errors = []
    for check in checks:
        try:
            value = pointer(obj, check['pointer'])
            if 'equals' in check: passed = canonical_sha256(value) == canonical_sha256(check['equals'])
            elif 'one_of' in check: passed = any(canonical_sha256(value) == canonical_sha256(v) for v in check['one_of'])
            else: passed = isinstance(value, (list, dict, str)) and len(value) == check['count']
            if not passed: errors.append({'check': check, 'reason': 'VALUE_MISMATCH'})
        except (KeyError, IndexError, TypeError, ValueError): errors.append({'check': check, 'reason': 'VALUE_MISSING'})
    return errors


def contract_for(workspace):
    rec = sub.capture_record(workspace)
    contract = rec.get('result_contract') or normalize(rec['output_contract'], rec['job']['execution'], rec['job']['question'])
    if contract.get('claim') == 'SCIENCE':
        _validate_science(contract)
        if contract['declared_outcomes'] != rec['job']['question']['outcomes']:
            raise sub.SubmissionError('SCIENTIFIC_OUTCOME_BINDING')
    return contract


def prerequisites(admission):
    contract = contract_for(admission['workspace']); rows = contract['prerequisites']
    if not rows: return []
    from .portable_registry import locate, verify_capsule
    root, _ = locate(admission['workspace']); verified = []
    for row in rows:
        try: done = verify_capsule(root, row['capsule_sha256'])
        except Exception as exc: raise sub.SubmissionError('PREREQUISITE_EVIDENCE_UNRESOLVED', row['capsule_sha256']) from exc
        if (done['completion_sha256'] != row['completion_sha256'] or done['status'] != 'COMPLETED'
                or done.get('evidence_status') != 'VERIFIED'):
            raise sub.SubmissionError('PREREQUISITE_NOT_VERIFIED', row['completion_sha256'])
        verified.append(dict(row))
    return verified


def verify(contract, result, output):
    science = contract.get('claim') == 'SCIENCE'
    if science:
        try: _validate_science(contract)
        except (sub.SubmissionError, KeyError, TypeError, ValueError) as exc:
            return _science_report(contract, 'REJECTED', None, [], [{'reason': 'SCIENTIFIC_CONTRACT_POLICY', 'detail': str(exc)}])
    output = Path(output).resolve(); failures = evaluate_checks(result, contract['result_checks']); artifacts = []
    decoded = {}
    for row in contract['required_artifacts']:
        target = output / sub._relative(row['path'])
        if (not target.is_file() or not target.resolve().is_relative_to(output)
                or any(p.is_symlink() for p in (target, *target.parents) if p != output.parent)):
            failures.append({'path': row['path'], 'reason': 'ARTIFACT_MISSING_OR_UNSAFE'}); continue
        raw = target.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        artifacts.append({'path': row['path'], 'sha256': sha, 'size_bytes': len(raw)})
        if 'sha256' in row and sha != row['sha256']: failures.append({'path': row['path'], 'reason': 'ARTIFACT_HASH'})
        if row.get('json_checks') or contract.get('outcome', {}).get('artifact') == row['path']:
            try:
                decoded[row['path']] = _strict_json(raw) if science else json.loads(raw.decode('utf-8'))
                failures.extend(dict(x, path=row['path']) for x in evaluate_checks(decoded[row['path']], row.get('json_checks', [])))
            except Exception: failures.append({'path': row['path'], 'reason': 'ARTIFACT_JSON'})
    outcome=None
    if contract['claim']=='SCIENCE':
        try:
            if 'outcome' in contract:
                rule=contract['outcome']
                outcome=pointer(decoded[rule['artifact']],rule['pointer'])
            else:outcome=result.get('outcome')
            if not isinstance(outcome,str) or outcome in _PROCESS_OUTCOMES or outcome not in contract['declared_outcomes']:raise ValueError('unregistered')
        except Exception:failures.append({'reason':'SCIENTIFIC_OUTCOME_UNRESOLVED_OR_UNREGISTERED'})
    declared = bool(contract['required_artifacts'] or contract['result_checks'])
    if science:
        return _science_report(contract, 'REJECTED' if failures else 'VERIFIED', outcome if not failures else None, artifacts, failures)
    return {'schema_id': 'IG_DECODER_RESULT_VERIFICATION_V1', 'contract_sha256': canonical_sha256(contract),
            'claim': contract['claim'], 'status': 'REJECTED' if failures else 'VERIFIED' if declared else 'NOT_DECLARED',
            'scientific_outcome': outcome if contract['claim'] == 'SCIENCE' and not failures else None,
            'artifacts': artifacts, 'failures': failures}


def _science_report(contract, status, outcome, artifacts, failures):
    content = contract.get('scientific_content', [])
    return {'schema_id': 'IG_DECODER_RESULT_VERIFICATION_V2',
            'contract_sha256': canonical_sha256(contract), 'claim': 'SCIENCE',
            'status': status, 'scientific_outcome': outcome, 'artifacts': artifacts,
            'failures': failures, 'content_policy': SCIENCE_POLICY,
            'conformance_scope': 'DECLARED_CONTENT_ONLY', 'science_qualification_authority': 'NONE',
            'artifact_roles': [{'path': r['path'], 'role': 'SCIENTIFIC_CONTENT' if r['path'] in content else 'SUPPORTING'}
                               for r in contract.get('required_artifacts', []) if isinstance(r, dict) and 'path' in r]}
