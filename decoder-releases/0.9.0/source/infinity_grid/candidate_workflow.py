"""Small, non-authoritarian workflow for isolated Decoder candidates.

Any chat may record, test, preserve, and select a candidate for one task-local
project.  This module does not alter the shared Decoder pointer and does not
grant runtime security exceptions.  It records enough identity to make later
merges reproducible and refuses only objective overlap: a changed file or
declared contract changed by more than one candidate.

Shared release promotion remains a separate operation outside this module.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

from .canon import canonical_sha256

SCHEMA = 'IG_DECODER_CODE_CANDIDATE_V1'
SELECTION_SCHEMA = 'IG_DECODER_TASK_LOCAL_SELECTION_V1'
HASH = re.compile(r'^[0-9a-f]{64}$')
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$')
# Preserve historical candidate labels; additionally accept the supported local
# package suffix without relaxing project, contract, or test names.
VERSION = re.compile(r'^(?:[A-Za-z0-9][A-Za-z0-9._:-]{0,199}|0\.8\.[0-9]+(?:\.dev[0-9]+)?\+lib)$')
RELATIVE_PATH = re.compile(r'^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$')


class CandidateError(RuntimeError):
    pass


def _check_hash(value, field):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise CandidateError('CANDIDATE_HASH_REQUIRED:'+field)
    return value


def _names(values, field, pattern=NAME):
    if not isinstance(values, list) or not values:
        raise CandidateError('CANDIDATE_LIST_REQUIRED:'+field)
    result = sorted(set(values))
    if len(result) != len(values) or any(not isinstance(v, str) or not pattern.fullmatch(v) for v in result):
        raise CandidateError('CANDIDATE_NAME_INVALID:'+field)
    return result


def _sealed(record):
    row = dict(record)
    row['record_sha256'] = canonical_sha256(row)
    return row


def make_candidate(*, version, parent_source_sha256, candidate_source_sha256,
                   changed_files, contracts, test_results, reason):
    """Create a complete immutable candidate record; no owner token required."""
    if not isinstance(version, str) or not VERSION.fullmatch(version):
        raise CandidateError('CANDIDATE_VERSION_INVALID')
    if not isinstance(reason, str) or not reason.strip():
        raise CandidateError('CANDIDATE_REASON_REQUIRED')
    tests = []
    if not isinstance(test_results, list) or not test_results:
        raise CandidateError('CANDIDATE_TEST_RESULTS_REQUIRED')
    for result in test_results:
        if (not isinstance(result, dict) or result.get('outcome') not in {'PASS', 'FAIL'} or
                not isinstance(result.get('name'), str) or not NAME.fullmatch(result['name'])):
            raise CandidateError('CANDIDATE_TEST_RESULT_INVALID')
        tests.append({'name': result['name'], 'outcome': result['outcome']})
    return _sealed({
        'schema_id': SCHEMA,
        'version': version,
        'parent_source_sha256': _check_hash(parent_source_sha256, 'parent_source_sha256'),
        'candidate_source_sha256': _check_hash(candidate_source_sha256, 'candidate_source_sha256'),
        'changed_files': _names(changed_files, 'changed_files', RELATIVE_PATH),
        'affected_contracts': _names(contracts, 'affected_contracts'),
        'test_results': sorted(tests, key=lambda row: row['name']),
        'reason': reason.strip(),
        'scope': 'ISOLATED_TASK_LOCAL_CANDIDATE_NOT_SHARED_RELEASE',
        'authorization_required': False,
    })


def _write_immutable(path, row):
    raw = (json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n').encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise CandidateError('CANDIDATE_IMMUTABLE_CONFLICT:'+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw)


def save_candidate(store_root, record):
    if record.get('schema_id') != SCHEMA or record.get('record_sha256') != canonical_sha256({k:v for k,v in record.items() if k != 'record_sha256'}):
        raise CandidateError('CANDIDATE_RECORD_INVALID')
    path = Path(store_root)/'candidates'/(record['record_sha256']+'.json')
    _write_immutable(path, record)
    return {'status': 'CANDIDATE_SAVED', 'candidate_id': record['record_sha256'], 'path': str(path)}


def _load(store_root, candidate_id):
    _check_hash(candidate_id, 'candidate_id')
    path = Path(store_root)/'candidates'/(candidate_id+'.json')
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateError('CANDIDATE_NOT_FOUND:'+candidate_id) from exc
    if row.get('record_sha256') != candidate_id or canonical_sha256({k:v for k,v in row.items() if k != 'record_sha256'}) != candidate_id:
        raise CandidateError('CANDIDATE_RECORD_INVALID')
    return row


def merge_plan(store_root, candidate_ids):
    """Return a deterministic merge order or refuse an objective overlap."""
    if not isinstance(candidate_ids, list) or len(candidate_ids) < 2:
        raise CandidateError('MERGE_REQUIRES_MULTIPLE_CANDIDATES')
    rows = [_load(store_root, cid) for cid in sorted(set(candidate_ids))]
    parents = {row['parent_source_sha256'] for row in rows}
    if len(parents) != 1:
        raise CandidateError('CANDIDATE_PARENT_CONFLICT')
    seen_files = {}; seen_contracts = {}
    for row in rows:
        cid = row['record_sha256']
        for name in row['changed_files']:
            if name in seen_files:
                raise CandidateError('CANDIDATE_FILE_CONFLICT:'+name+':'+seen_files[name]+':'+cid)
            seen_files[name] = cid
        for name in row['affected_contracts']:
            if name in seen_contracts:
                raise CandidateError('CANDIDATE_CONTRACT_CONFLICT:'+name+':'+seen_contracts[name]+':'+cid)
            seen_contracts[name] = cid
    return _sealed({'schema_id': 'IG_DECODER_CANDIDATE_MERGE_PLAN_V1',
        'parent_source_sha256': rows[0]['parent_source_sha256'],
        'candidate_ids': [row['record_sha256'] for row in rows],
        'changed_files': sorted(seen_files), 'affected_contracts': sorted(seen_contracts),
        'status': 'MERGEABLE_NONCONFLICTING'})


def select_task_local(store_root, project_id, candidate_id):
    """Select a qualified candidate for one project without shared promotion."""
    if not isinstance(project_id, str) or not NAME.fullmatch(project_id):
        raise CandidateError('PROJECT_ID_INVALID')
    row = _load(store_root, candidate_id)
    if any(test['outcome'] != 'PASS' for test in row['test_results']):
        raise CandidateError('CANDIDATE_TESTS_NOT_PASSING')
    selection = _sealed({'schema_id': SELECTION_SCHEMA, 'project_id': project_id,
        'candidate_id': candidate_id, 'candidate_source_sha256': row['candidate_source_sha256'],
        'scope': 'TASK_LOCAL_ONLY_NOT_SHARED_POINTER', 'authorization_required': False})
    path = Path(store_root)/'projects'/project_id/'CANDIDATE_SELECTION.json'
    _write_immutable(path, selection)
    return {'status': 'TASK_LOCAL_SELECTED', 'selection': selection, 'path': str(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Task-local Decoder candidate records; never promotes the shared pointer.')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('merge-plan'); p.add_argument('store'); p.add_argument('candidate_ids', nargs='+')
    p = commands.add_parser('select'); p.add_argument('store'); p.add_argument('project_id'); p.add_argument('candidate_id')
    args = parser.parse_args(argv)
    result = merge_plan(args.store, args.candidate_ids) if args.command == 'merge-plan' else select_task_local(args.store, args.project_id, args.candidate_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
