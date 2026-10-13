"""Actionable refusals for supported Decoder operations.

These contracts prevent accidental unsupported API use. They are not an access
boundary against code that can replace Decoder or freely execute on the host.
"""
from __future__ import annotations
import json
from pathlib import Path
from .v05_execution_authority import ExecutionAuthorityError


class InvocationRefused(ExecutionAuthorityError):
    def __init__(self, code, operation, *, workspace=None, job_id=None, checks=None,
                 next_operation=None, required_arguments=None):
        super().__init__(code, operation)
        self.operation = operation
        self.workspace = workspace
        self.job_id = job_id
        self.checks = checks or [code]
        self.next_operation = next_operation
        self.required_arguments = required_arguments

    def as_dict(self):
        return refusal_details(self, self.operation, self.workspace, self.job_id)


def refusal_details(exc, operation, workspace=None, job_id=None):
    cause = exc
    while not hasattr(cause, 'code') and cause.__cause__ is not None:
        cause = cause.__cause__
    if hasattr(cause, 'code'): exc = cause
    code = getattr(exc, 'code', str(exc).split(':', 1)[0] or type(exc).__name__)
    known = workspace is not None and job_id is not None
    repair = any(s in code for s in ('SOURCE', 'REGISTRATION', 'PREFLIGHT', 'PRIVATE_EXECUTION', 'HELPER'))
    next_op = ('capture' if repair or not known else 'run')
    if code == 'SAVE_REQUIRED': next_op = 'pending-saves'
    args = ({'store': None, 'specification': None} if next_op == 'capture' else
            {'workspace': str(workspace)} if next_op == 'pending-saves' else
            {'workspace': str(workspace), 'job_id': job_id})
    ids = []
    if workspace is not None:
        try:
            from .submission import capture_record, required_objects
            capture_record(Path(workspace))
            ids = [r['sha256'] for r in required_objects(Path(workspace))]
        except Exception:
            # A refusal must never imply that unsaved or missing bytes survived.
            ids = []
    actual_args = getattr(exc, 'required_arguments', None) or args
    result = {'schema_id': 'IG_DECODER_REFUSAL_V1', 'status': 'REFUSED',
            'reason_code': code, 'operation_attempted': operation,
            'preserved_artifact_ids': ids,
            'unmet_checks': getattr(exc, 'checks', [str(exc)]),
            'next_supported_operation': getattr(exc, 'next_operation', None) or next_op,
            'required_arguments': actual_args,
            'retry_is_safe': False,
            'retry_condition': 'Apply the listed correction and complete the save gate; completed jobs are verified and reused.',
            'missing_decisions': [k for k, v in actual_args.items() if v is None]}

    diagnostic = getattr(exc, 'evidence_diagnostic', None)
    if diagnostic is not None:
        result['evidence_diagnostic'] = diagnostic
    return result

def retain_refusal(workspace, job_id, exc, operation='run'):
    """Keep refusal separately from scientific evidence; never overwrite an attempt."""
    from .canon import canonical_sha256, write_json_atomic
    obj = refusal_details(exc, operation, workspace, job_id)
    root = Path(workspace).resolve()
    if root.is_dir():
        write_json_atomic(root/'runtime/refusals'/(canonical_sha256(obj)+'.json'), obj)
    return obj
