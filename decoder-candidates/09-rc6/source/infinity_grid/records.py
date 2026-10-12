from __future__ import annotations
import hashlib, json, platform
from pathlib import Path
from typing import Any
from . import __version__, build_meta
from .canon import canonical_sha256
from .schema import validate


def utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def runtime_descriptor() -> dict[str,Any]:
    return {'schema_id':'IG_RUNTIME_IDENTITY_V0_17_1','python':platform.python_version(),'implementation':platform.python_implementation(),'system':platform.system(),'machine':platform.machine(),'package_version':__version__}

def runtime_sha256() -> str: return canonical_sha256(runtime_descriptor())

def live_source_manifest() -> dict:
    pkg=Path(__file__).resolve().parent; out={}
    for p in sorted(x for x in pkg.rglob('*') if x.is_file() and x.name!='_build_meta.json' and '__pycache__' not in x.parts and not x.name.endswith('.pyc')):
        rel=p.relative_to(pkg).as_posix(); b=p.read_bytes(); out[rel]={'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
    return out

def live_source_sha256() -> str: return canonical_sha256(live_source_manifest())

def source_sha256(*,verify_build:bool=True) -> str:
    live=live_source_sha256()
    if verify_build:
        expected=str(build_meta().get('source_sha256','0'*64))
        if expected!=live: raise RuntimeError(f'live executable source identity mismatch: expected {expected}, observed {live}')
    return live

def evidence(origin,scope,disposition,authority,protocol_label): return {'origin':origin,'scope':scope,'disposition':disposition,'authority':authority,'protocol_label':protocol_label}

def new_run_record(*,run_id,protocol,subject,question_sha256,plan_sha256,input_artifacts,evidence_record,run_core_sha256,execution_mode,code_sha256,environment_sha256):
    return {
      'schema_id':'IG_UNIVERSAL_RUN_RECORD_V0_17','run_id':run_id,'protocol':protocol,'subject':subject,
      'question_sha256':question_sha256,'plan_sha256':plan_sha256,'run_core_sha256':run_core_sha256,
      'execution_mode':execution_mode,'code_identity':{'source_sha256':code_sha256},
      'environment_identity':{'runtime_sha256':environment_sha256,'descriptor':runtime_descriptor()},'input_artifacts':input_artifacts,
      'evidence':evidence_record,'lifecycle':'PLANNED','stages':[],'result_artifacts':[],'claim_refs':[],
      'started_utc':None,'finished_utc':None,'resource_summary':{},
    }

def validate_run(record: dict, schema: dict) -> list[str]: return validate(schema,record,raise_on_error=False)
def validate_checkpoint(record: dict, schema: dict) -> list[str]: return validate(schema,record,raise_on_error=False)
def validate_claim(record: dict, schema: dict) -> list[str]: return validate(schema,record,raise_on_error=False)
