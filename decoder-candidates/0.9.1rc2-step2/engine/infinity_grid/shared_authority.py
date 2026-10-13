"""Step 5 transactional coordination core, for a service-owned local database.

This module is not an access-control boundary. Production must expose operations
through an authenticated service whose database is inaccessible to chat clients.
It does not yet replace the registered controller's local scheduling decisions.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import sqlite3
from pathlib import Path


class CoordinationRefused(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise CoordinationRefused('SHA256_REQUIRED')
    return value


def work_identity(spec):
    """Labels are deliberately outside this strictly specified semantic object."""
    fields = {'engine', 'implementation', 'inputs', 'parameters', 'contract',
              'environment', 'resources', 'repeat'}
    if set(spec) != fields:
        raise CoordinationRefused('WORK_IDENTITY_FIELDS')
    _hash(spec['engine']); _hash(spec['implementation']); _hash(spec['contract'])
    if not isinstance(spec['inputs'], dict):
        raise CoordinationRefused('NAMED_INPUT_HASHES_REQUIRED')
    for name, value in spec['inputs'].items():
        if not name: raise CoordinationRefused('INPUT_NAME_REQUIRED')
        _hash(value)
    for field in ('parameters', 'environment', 'resources'):
        if not isinstance(spec[field], dict) or (field != 'parameters' and not spec[field]):
            raise CoordinationRefused('PROVENANCE_REQUIRED:' + field)
    if spec['repeat'] is not None and (not isinstance(spec['repeat'], str) or not spec['repeat'].strip()):
        raise CoordinationRefused('EXPLICIT_REPEAT_ID_REQUIRED')
    return digest(spec)


class Authority:
    """Short SQLite transactions serialize mutations; work runs outside them.

    Use a host-local disk, not a Drive-synchronized directory or network mount.
    All successful mutations and their event records commit together.
    """
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS release (singleton INTEGER PRIMARY KEY CHECK(singleton=1), generation INTEGER NOT NULL, engine TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS work (id TEXT PRIMARY KEY, spec TEXT NOT NULL, owner TEXT NOT NULL, token TEXT NOT NULL, state TEXT NOT NULL, evidence TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS labels (work_id TEXT NOT NULL, label TEXT NOT NULL, PRIMARY KEY(work_id,label))')
            db.execute('CREATE TABLE IF NOT EXISTS providers (role TEXT PRIMARY KEY, provider TEXT NOT NULL, checks TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL)')

    @contextlib.contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _event(self, db, operation, **payload):
        db.execute('INSERT INTO events(body) VALUES (?)', (canonical(dict(operation=operation, **payload)),))

    def initialize(self, engine):
        _hash(engine)
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM release').fetchone():
                raise CoordinationRefused('ALREADY_INITIALIZED')
            db.execute('INSERT INTO release VALUES (1,0,?)', (engine,))
            self._event(db, 'INITIALIZE', engine=engine)

    def current(self):
        with self._transaction() as db:
            row = db.execute('SELECT generation,engine FROM release').fetchone()
            if not row: raise CoordinationRefused('AUTHORITY_NOT_INITIALIZED')
            return dict(row)

    def activate(self, expected, engine, change, validation_evidence):
        """Service must verify the referenced registered validation before calling."""
        _hash(engine); _hash(change); _hash(validation_evidence)
        with self._transaction() as db:
            row = db.execute('SELECT generation,engine FROM release').fetchone()
            if not row or dict(row) != expected:
                raise CoordinationRefused('STALE_PARENT_RECONCILE')
            generation = row['generation'] + 1
            db.execute('UPDATE release SET generation=?,engine=? WHERE singleton=1', (generation, engine))
            self._event(db, 'ACTIVATE', parent=expected, engine=engine, generation=generation,
                        change=change, validation_evidence=validation_evidence)
            return {'generation': generation, 'engine': engine}

    def claim(self, spec, owner, label):
        wid = work_identity(spec)
        if not isinstance(owner, str) or not owner.strip() or not isinstance(label, str):
            raise CoordinationRefused('OWNER_AND_LABEL_REQUIRED')
        with self._transaction() as db:
            current = db.execute('SELECT engine FROM release').fetchone()
            if not current: raise CoordinationRefused('AUTHORITY_NOT_INITIALIZED')
            row = db.execute('SELECT * FROM work WHERE id=?', (wid,)).fetchone()
            if row:
                db.execute('INSERT OR IGNORE INTO labels VALUES (?,?)', (wid, label))
                self._event(db, 'DISCOVER', work_id=wid, owner=owner, label=label)
                # Never reveal the execution owner's token to a discovering client.
                return {'work_id': wid, 'status': 'REUSE' if row['state'] == 'COMPLETED' else row['state'],
                        'evidence': json.loads(row['evidence']) if row['evidence'] else None}
            if current['engine'] != spec['engine']:
                raise CoordinationRefused('STALE_ENGINE_NEW_WORK_REFUSED')
            token = secrets.token_hex(32)
            db.execute('INSERT INTO work VALUES (?,?,?,?,?,NULL)', (wid, canonical(spec), owner, token, 'RUNNING'))
            db.execute('INSERT INTO labels VALUES (?,?)', (wid, label))
            self._event(db, 'CLAIM', work_id=wid, owner=owner, label=label)
            return {'work_id': wid, 'status': 'CLAIMED', 'token': token}

    def finish(self, wid, owner, token, evidence):
        """Commit references after the service verifies archived completion bytes."""
        if set(evidence) != {'completion', 'snapshot'}:
            raise CoordinationRefused('COMPLETION_AND_SNAPSHOT_REQUIRED')
        for value in evidence.values(): _hash(value)
        with self._transaction() as db:
            row = db.execute('SELECT * FROM work WHERE id=?', (wid,)).fetchone()
            if not row or row['owner'] != owner or not secrets.compare_digest(row['token'], token):
                raise CoordinationRefused('CLAIM_OWNERSHIP_REQUIRED')
            if row['state'] == 'COMPLETED' and row['evidence'] == canonical(evidence):
                return {'work_id': wid, 'status': 'COMPLETED'}
            if row['state'] != 'RUNNING': raise CoordinationRefused('ATTEMPT_ALREADY_CLOSED')
            db.execute('UPDATE work SET state=?,evidence=? WHERE id=?', ('COMPLETED', canonical(evidence), wid))
            self._event(db, 'FINISH', work_id=wid, evidence=evidence)
            return {'work_id': wid, 'status': 'COMPLETED'}

    def replace_provider(self, role, expected, replacement, checks):
        if not isinstance(role, str) or not role.strip(): raise CoordinationRefused('SEMANTIC_ROLE_REQUIRED')
        if expected is not None: _hash(expected)
        _hash(replacement)
        if set(checks) != {'correctness', 'performance'}: raise CoordinationRefused('PROVIDER_CHECKS_REQUIRED')
        for value in checks.values(): _hash(value)
        with self._transaction() as db:
            row = db.execute('SELECT provider FROM providers WHERE role=?', (role,)).fetchone()
            if (row['provider'] if row else None) != expected:
                raise CoordinationRefused('PROVIDER_CONFLICT_RECONCILE')
            db.execute('INSERT INTO providers VALUES (?,?,?) ON CONFLICT(role) DO UPDATE SET provider=excluded.provider,checks=excluded.checks',
                       (role, replacement, canonical(checks)))
            self._event(db, 'REPLACE_PROVIDER', role=role, old=expected, replacement=replacement, checks=checks)

    def inspect_work(self, wid):
        with self._transaction() as db:
            row = db.execute('SELECT id,spec,owner,state,evidence FROM work WHERE id=?', (wid,)).fetchone()
            return dict(row) if row else None
