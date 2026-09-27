"""Durable reviews of server-owned, immutable task versions. Never executes science."""
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import uuid

from .models import AcceptedRequest
from .native import AdapterError, canonical_hash, contained


def review(settings, task, catalog):
    path = contained(Path(settings.specification_root), task.specification, directory=False)
    raw = path.read_bytes()
    if len(raw) > 2097152 or hashlib.sha256(raw).hexdigest() != task.specification_sha256:
        raise AdapterError('SPECIFICATION_CHANGED',409)
    try:
        spec = json.loads(raw)
        question = spec['question']
        resources = spec['resources']
        contract = spec['output_contract']
        if spec['schema_id'] != 'IG_DECODER_CAPTURE_SPEC_V1': raise ValueError()
        if not all(isinstance(question[k],str) for k in ('description','stopping_rule')): raise ValueError()
        if not isinstance(question['outcomes'],list): raise ValueError()
        inputs = []
        for item in spec.get('inputs',[]):
            matches = [i for i in catalog.inputs if i.sha256 == item['sha256']]
            approved = next((i for i in matches if i.available),None)
            verified = False
            if approved:
                try:
                    p = Path(item['path'])
                    if p.is_absolute() and not any(x.is_symlink() for x in (p,*p.parents)) and p.is_file():
                        with p.open('rb') as f: verified = hashlib.file_digest(f,'sha256').hexdigest() == item['sha256']
                except OSError: pass
            inputs.append({'role':item['logical_name'],'name':approved.name if approved else 'Required input unavailable',
                           'sha256':item['sha256'],'verified':verified})
        return {'task_id':task.id,'name':task.name,'specification_sha256':task.specification_sha256,
                'question':{k:question[k] for k in ('description','outcomes','stopping_rule')},
                'resources':{k:resources[k] for k in ('workers','start_method','memory_budget_bytes','workspace_budget_bytes','execution_policy') if k in resources},
                'expected_outputs':contract.get('required_artifacts',[]),'inputs':inputs,
                'editable':False,'can_prepare':all(i['verified'] for i in inputs),
                'configuration_note':'This task version fixes its inputs, parameters and resources. Choose another server-defined version to change them.'}
    except (KeyError,TypeError,ValueError) as exc:
        raise AdapterError('TASK_REVIEW_UNAVAILABLE') from exc


class Drafts:
    def __init__(self, queue):
        if not hasattr(queue,'transaction'): raise AdapterError('BACKGROUND_WORKER_NOT_CONFIGURED')
        self.queue = queue
        with queue.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS drafts(id TEXT PRIMARY KEY, creation_key TEXT UNIQUE NOT NULL, task_id TEXT NOT NULL, review TEXT NOT NULL, command TEXT NOT NULL, digest TEXT NOT NULL, created REAL NOT NULL)')

    def create(self, task_id, key, build):
        # Idempotency lookup precedes mutable catalogue reads, including after restart.
        with self.queue.transaction() as db:
            old = db.execute('SELECT * FROM drafts WHERE creation_key=?',(key,)).fetchone()
            if old:
                if old['task_id'] != task_id: raise AdapterError('IDEMPOTENCY_CONFLICT',409)
                draft_id = old['id']
            else:
                view,command = build()
                draft_id = 'draft-'+uuid.uuid4().hex
                db.execute('INSERT INTO drafts VALUES (?,?,?,?,?,?,?)',
                           (draft_id,key,task_id,json.dumps(view),json.dumps(asdict(command)),canonical_hash(asdict(command)),time.time()))
        return self.get(draft_id)

    def row(self, draft_id):
        with closing(self.queue.connect()) as db:
            row = db.execute('SELECT * FROM drafts WHERE id=?',(draft_id,)).fetchone()
        if row is None: raise AdapterError('DRAFT_NOT_FOUND',404)
        return row

    def accepted(self, draft_id):
        with closing(self.queue.connect()) as db:
            row = db.execute('SELECT id,status FROM requests WHERE key=?',('draft-capture:'+draft_id,)).fetchone()
        return AcceptedRequest(request_id=row['id'],status=row['status']) if row else None

    def get(self, draft_id):
        row = self.row(draft_id); accepted = self.accepted(draft_id)
        return {'id':draft_id,'created':row['created'],'review':json.loads(row['review']),
                'review_sha256':canonical_hash(json.loads(row['review'])),
                'request':accepted.model_dump() if accepted else None}

    def capture(self, draft_id, review_sha256, build):
        row = self.row(draft_id)
        if canonical_hash(json.loads(row['review'])) != review_sha256: raise AdapterError('REVIEW_CHANGED',409)
        accepted = self.accepted(draft_id)
        if accepted: return accepted
        view,command = build(row['task_id'])
        if canonical_hash(view) != review_sha256 or canonical_hash(asdict(command)) != row['digest']:
            raise AdapterError('TASK_CHANGED_REVIEW_REQUIRED',409)
        if not view['can_prepare']: raise AdapterError('REQUIRED_INPUT_UNAVAILABLE',409)
        return self.queue.submit(command,'draft-capture:'+draft_id,row['digest'])
