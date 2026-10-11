"""Durable local control primitives; never generates or admits science."""
import hashlib
import ast
import json
import os
from pathlib import Path
import sys
import tempfile


class ControlError(RuntimeError):
    pass


def require(condition, reason='CONTROL_INTEGRITY'):
    if not condition:
        raise ControlError(reason)


def require_unoptimized():
    require(not sys.flags.optimize, 'OPTIMIZED_CONTROLLER_RUNTIME_REFUSED')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def ensure_bytes(path, data):
    path = Path(path)
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, 'RESUME_FILE_IDENTITY:' + str(path))
    else:
        atomic_bytes(path, data)


def explicit_assertions(source):
    """Keep generated operator checks effective under-O, preserving messages."""
    raw = source.encode()
    tree = ast.parse(raw)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    if not nodes:
        return source
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    def span(node):
        return offsets[node.lineno - 1] + node.col_offset, offsets[node.end_lineno - 1] + node.end_col_offset
    replacements = []
    for node in nodes:
        a, b = span(node); x, y = span(node.test)
        call = b'_require_integrity(' + raw[x:y]
        if node.msg:
            x, y = span(node.msg); call += b', lambda: ' + raw[x:y]
        replacements.append((a, b, call + b')'))
    position = 0
    for node in tree.body:
        if ((isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)) or
                (isinstance(node, ast.ImportFrom) and node.module == '__future__')):
            position = offsets[node.end_lineno]
        else:
            break
    helper = b'\n\ndef _require_integrity(condition, message=None):\n    if not condition:\n        if message is None: raise AssertionError()\n        raise AssertionError(message())\n\n'
    replacements.append((position, position, helper))
    for a, b, replacement in sorted(replacements, reverse=True):
        raw = raw[:a] + replacement + raw[b:]
    ast.parse(raw)
    return raw.decode()


def stage(path, name, action, artifacts, binding):
    """Reuse completed stages; pending actions require native missing-only semantics."""
    path = Path(path)
    state = read(path) if path.exists() else {'schema_id': 'IG_OBSERVER_STAGE_JOURNAL_V1', 'binding': binding, 'stages': {}}
    require(state['schema_id'] == 'IG_OBSERVER_STAGE_JOURNAL_V1' and state['binding'] == binding, 'STAGE_SOURCE_OR_INPUT_CHANGED')
    old = state['stages'].get(name)
    if old and old['phase'] == 'COMPLETED':
        for row in old['artifacts']:
            require(sha(row['path']) == row['sha256'], 'STAGE_OUTPUT_CHANGED')
        return False
    state['stages'][name] = {'phase': 'RUNNING'}; write(path, state)
    action()
    rows = [{'path': str(p), 'sha256': sha(p)} for p in artifacts]
    state['stages'][name] = {'phase': 'COMPLETED', 'artifacts': rows}; write(path, state)
    return True


class SaveJournal:
    """Persist intent, provider ID and byte-verified receipt per content object.

    An upload whose outcome is unknown is reconciled before another upload.
    No visible matching object is not proof that a provider write failed: that
    case stays blocked and is never blindly retried. Provider URLs are transient
    callback inputs and are deliberately excluded from this journal.
    """
    def __init__(self, path, upload, download, reconcile):
        self.path = Path(path)
        self.upload, self.download, self.reconcile = upload, download, reconcile
        self.state = read(self.path) if self.path.exists() else {'schema_id': 'IG_EXACT_SAVE_JOURNAL_V1', 'objects': {}}
        require(self.state.get('schema_id') == 'IG_EXACT_SAVE_JOURNAL_V1', 'SAVE_JOURNAL_SCHEMA')

    def save(self, path):
        path = Path(path)
        identity = {'sha256': sha(path), 'bytes': path.stat().st_size, 'basename': path.name}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        row = self.state['objects'].get(key)
        provider_name = 'IG_EXACT_' + key + '_' + path.name
        if row is None:
            row = {'identity': identity, 'provider_name': provider_name, 'phase': 'UPLOAD_INTENT'}
            self.state['objects'][key] = row; write(self.path, self.state)
            provider_id = self.upload(path, provider_name)
            require(isinstance(provider_id, str) and provider_id, 'UPLOAD_ID_MISSING')
            row.update(phase='ID_RECORDED', provider_id=provider_id); write(self.path, self.state)
        elif row['phase'] == 'UPLOAD_INTENT':
            candidates = self.reconcile(provider_name)
            require(candidates is not None, 'UPLOAD_OUTCOME_RECONCILIATION_INCOMPLETE')
            matches = []
            for provider_id in candidates:
                data = self.download(provider_id)
                if len(data) == identity['bytes'] and hashlib.sha256(data).hexdigest() == identity['sha256']:
                    matches.append(provider_id)
            require(matches, 'UPLOAD_OUTCOME_UNKNOWN_NO_RETRY')
            row.update(phase='ID_RECORDED', provider_id=sorted(matches)[0], reconciled_ids=sorted(matches))
            write(self.path, self.state)
        require(row['identity'] == identity, 'SAVE_LOCAL_IDENTITY_CHANGED')
        if row['phase'] != 'EXACT':
            data = self.download(row['provider_id'])
            require(len(data) == identity['bytes'] and hashlib.sha256(data).hexdigest() == identity['sha256'], 'SAVE_READBACK_MISMATCH')
            require(sha(path) == identity['sha256'], 'SAVE_LOCAL_MUTATED_DURING_UPLOAD')
            row['phase'] = 'EXACT'; write(self.path, self.state)
        return {'path': str(path), 'drive_file_id': row['provider_id'],
                'sha256': identity['sha256'], 'bytes': identity['bytes'],
                'drive_exact_readback_verified': True}


def publish_activation(work, batch, ledger, status, fault=None):
    """One durable decision, two recoverable derived files; commit is authority."""
    work, batch = Path(work), Path(batch)
    commit_path = batch / 'ACTIVATION_COMMIT.json'
    checkpoint = ledger['checkpoint']
    if not commit_path.exists():
        require(read(work / 'CURRENT_STATUS.json')['latest_checkpoint'] == checkpoint - 1, 'ACTIVATION_PREDECESSOR')
        commit = {'schema_id': 'IG_OBSERVER_ACTIVATION_COMMIT_V1', 'checkpoint': checkpoint,
                  'prior_checkpoint': checkpoint - 1, 'ledger': ledger, 'status': status,
                  'receipt_bindings': ledger['save_receipts']}
        commit['payload_sha256'] = hashlib.sha256(json.dumps(commit, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        write(commit_path, commit)
    replay_activation(work, batch, fault=fault)


def replay_activation(work, batch, fault=None):
    work, batch = Path(work), Path(batch)
    path = batch / 'ACTIVATION_COMMIT.json'
    commit = read(path)
    body = {k: v for k, v in commit.items() if k != 'payload_sha256'}
    require(hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == commit['payload_sha256'], 'ACTIVATION_COMMIT_HASH')
    require(commit['schema_id'] == 'IG_OBSERVER_ACTIVATION_COMMIT_V1', 'ACTIVATION_COMMIT_SCHEMA')
    require(commit['ledger']['checkpoint'] == commit['checkpoint'] == commit['status']['latest_checkpoint'], 'ACTIVATION_COMMIT_IDENTITY')
    for row in commit['receipt_bindings']:
        require(sha(row['path']) == row['sha256'], 'ACTIVATION_RECEIPT_CHANGED')
    current = read(work / 'CURRENT_STATUS.json')
    require(current['latest_checkpoint'] in (commit['prior_checkpoint'], commit['checkpoint']), 'ACTIVATION_STALE_COMMIT')
    if current['latest_checkpoint'] == commit['checkpoint']:
        require(current == commit['status'], 'ACTIVATION_STATUS_CONFLICT')
    if fault:
        fault('after_commit')
    write(batch / 'EARNED_LEDGER.json', commit['ledger'])
    if fault:
        fault('after_ledger')
    write(work / 'CURRENT_STATUS.json', commit['status'])
    if fault:
        fault('after_status')
    write(batch / 'ACTIVATION_APPLIED.json', {'status': 'APPLIED', 'commit_sha256': sha(path)})


def recover_next_activation(work):
    work = Path(work)
    checkpoint = read(work / 'CURRENT_STATUS.json')['latest_checkpoint']
    # A commit may be interrupted before or after CURRENT_STATUS publication.
    for number in (checkpoint, checkpoint + 1):
        batch = work / f'continuation{number:04d}'
        if (batch / 'ACTIVATION_COMMIT.json').exists():
            applied = batch / 'ACTIVATION_APPLIED.json'
            if applied.exists():
                require(read(applied)['commit_sha256'] == sha(batch / 'ACTIVATION_COMMIT.json'), 'ACTIVATION_APPLIED_IDENTITY')
                require(read(batch / 'EARNED_LEDGER.json') == read(batch / 'ACTIVATION_COMMIT.json')['ledger'], 'ACTIVATION_LEDGER_CHANGED')
            else:
                replay_activation(work, batch)
