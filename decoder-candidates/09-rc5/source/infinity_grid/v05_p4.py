from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .canon import canonical_sha256, write_json_atomic

PROJECTION_SPEC_SCHEMA = "IG_DECODER_V05_SCIENTIFIC_PROJECTION_SPEC_V1"
PROJECTION_RECORD_SCHEMA = "IG_DECODER_V05_SCIENTIFIC_PROJECTION_RECORD_V1"
PROVENANCE_SCHEMA = "IG_DECODER_V05_PROVENANCE_ENVELOPE_V1"
REPLAY_SCHEMA = "IG_DECODER_V05_SAME_SOURCE_COLD_REPLAY_V1"
MIGRATION_SCHEMA = "IG_DECODER_V05_MIGRATION_EQUIVALENCE_V1"
RESOLVER_SCHEMA = "IG_DECODER_V05_CONTENT_RESOLVER_INDEX_V1"
CLAIM_GRAPH_SCHEMA = "IG_DECODER_V05_CLAIM_EVIDENCE_GRAPH_V1"
CLOSEOUT_SCHEMA = "IG_DECODER_V05_PORTABLE_CLOSEOUT_V1"

class P4Error(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _require_sha(v: Any, field: str) -> str:
    if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):
        raise P4Error(f'{field} must be lowercase sha256')
    return v


def _safe_rel(v: str, field: str='path') -> str:
    p=Path(v)
    if not isinstance(v,str) or not v or p.is_absolute() or '..' in p.parts or v.startswith('~'):
        raise P4Error(f'{field} must be a safe relative path')
    return p.as_posix()


def seal_projection_spec(*, projection_id: str, runner: str, selectors: list[dict[str,str]]) -> dict:
    if not projection_id or not runner or not selectors:
        raise P4Error('projection spec requires id, runner and selectors')
    names=set(); normalized=[]
    for s in selectors:
        if set(s)!={'name','path'} or not s['name'] or not s['path']:
            raise P4Error('projection selector must contain name/path')
        if s['name'] in names: raise P4Error('duplicate projection field')
        names.add(s['name'])
        normalized.append({'name':s['name'],'path':s['path']})
    base={'schema_id':PROJECTION_SPEC_SCHEMA,'projection_id':projection_id,'runner':runner,'selectors':normalized}
    return dict(base, projection_spec_sha256=canonical_sha256(base))


def _get_path(obj: Any, dotted: str) -> Any:
    cur=obj
    for part in dotted.split('.'):
        if isinstance(cur,dict) and part in cur:
            cur=cur[part]
        else:
            raise P4Error(f'projection path missing: {dotted}')
    return cur


def project(result: dict[str,Any], spec: dict[str,Any]) -> dict[str,Any]:
    if spec.get('schema_id')!=PROJECTION_SPEC_SCHEMA:
        raise P4Error('unsupported projection spec schema')
    expect=canonical_sha256({k:v for k,v in spec.items() if k!='projection_spec_sha256'})
    if spec.get('projection_spec_sha256')!=expect: raise P4Error('projection spec hash mismatch')
    return {s['name']:_get_path(result,s['path']) for s in spec['selectors']}


def seal_projection(*, role: str, source_sha256: str, result: dict[str,Any], spec: dict[str,Any]) -> dict:
    _require_sha(source_sha256,'source_sha256')
    projection=project(result,spec)
    base={'schema_id':PROJECTION_RECORD_SCHEMA,'role':role,'source_sha256':source_sha256,'projection_spec_sha256':spec['projection_spec_sha256'],'projection':projection}
    return dict(base, projection_sha256=canonical_sha256(base))


def seal_same_source_replay(*, primary: dict, cold: dict) -> dict:
    failures=[]
    if primary.get('projection_spec_sha256')!=cold.get('projection_spec_sha256'): failures.append('PROJECTION_SPEC_MISMATCH')
    if primary.get('source_sha256')!=cold.get('source_sha256'): failures.append('SOURCE_IDENTITY_MISMATCH')
    if primary.get('projection')!=cold.get('projection'): failures.append('SCIENTIFIC_PROJECTION_MISMATCH')
    base={'schema_id':REPLAY_SCHEMA,'mode':'SAME_RELEASE_COLD_REPLAY','status':'PASS' if not failures else 'FAIL','failures':failures,
          'primary_source_sha256':primary.get('source_sha256'),'cold_source_sha256':cold.get('source_sha256'),
          'projection_spec_sha256':primary.get('projection_spec_sha256'),'primary_projection_sha256':primary.get('projection_sha256'),
          'cold_projection_sha256':cold.get('projection_sha256'),'projection_equal':primary.get('projection')==cold.get('projection')}
    return dict(base,replay_sha256=canonical_sha256(base))


def seal_migration_equivalence(*, baseline: dict, candidate: dict) -> dict:
    failures=[]
    if baseline.get('projection_spec_sha256')!=candidate.get('projection_spec_sha256'): failures.append('PROJECTION_SPEC_MISMATCH')
    if baseline.get('source_sha256')==candidate.get('source_sha256'): failures.append('SOURCE_IDENTITIES_NOT_DISTINCT')
    if baseline.get('projection')!=candidate.get('projection'): failures.append('SCIENTIFIC_PROJECTION_MISMATCH')
    base={'schema_id':MIGRATION_SCHEMA,'mode':'OLD_NEW_MIGRATION_EQUIVALENCE','status':'PASS' if not failures else 'FAIL','failures':failures,
          'baseline_source_sha256':baseline.get('source_sha256'),'candidate_source_sha256':candidate.get('source_sha256'),
          'projection_spec_sha256':baseline.get('projection_spec_sha256'),'baseline_projection_sha256':baseline.get('projection_sha256'),
          'candidate_projection_sha256':candidate.get('projection_sha256'),'projection_equal':baseline.get('projection')==candidate.get('projection')}
    return dict(base,migration_sha256=canonical_sha256(base))


def build_source_manifest(package_root: Path) -> dict:
    package_root=Path(package_root)
    files=[]
    for p in sorted(x for x in package_root.rglob('*') if x.is_file() and x.name!='_build_meta.json' and '__pycache__' not in x.parts and not x.name.endswith('.pyc')):
        rel=p.relative_to(package_root).as_posix(); files.append({'path':rel,'sha256':_sha_file(p),'size_bytes':p.stat().st_size})
    base={'schema_id':'IG_DECODER_V05_SOURCE_CLOSURE_MANIFEST_V1','scope':'src/infinity_grid excluding _build_meta.json, __pycache__, pyc','files':files}
    return dict(base,source_closure_sha256=canonical_sha256({x['path']:{'sha256':x['sha256'],'size_bytes':x['size_bytes']} for x in files}))


def build_resolver(*, object_root: Path, bindings: list[dict[str,Any]]) -> dict:
    object_root=Path(object_root); object_root.mkdir(parents=True,exist_ok=True)
    entries=[]; names=set()
    for b in bindings:
        logical=b['logical_name']; src=Path(b['path']); role=b['role']
        if logical in names: raise P4Error('duplicate resolver logical name')
        names.add(logical); sha=_sha_file(src); size=src.stat().st_size
        rel=f'objects/sha256/{sha[:2]}/{sha}'
        dst=object_root.parent/rel; dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists(): shutil.copy2(src,dst)
        elif _sha_file(dst)!=sha: raise P4Error('resolver object collision')
        entries.append({'logical_name':logical,'role':role,'sha256':sha,'size_bytes':size,'object_path':rel})
    base={'schema_id':RESOLVER_SCHEMA,'mode':'OFFLINE_COMPLETE_CONTENT_ADDRESSED','entries':sorted(entries,key=lambda x:x['logical_name'])}
    return dict(base,resolver_sha256=canonical_sha256(base))


def validate_resolver(portable_root: Path, index: dict) -> dict:
    failures=[]; portable_root=Path(portable_root)
    if index.get('schema_id')!=RESOLVER_SCHEMA: failures.append('SCHEMA')
    obs=canonical_sha256({k:v for k,v in index.items() if k!='resolver_sha256'})
    if index.get('resolver_sha256')!=obs: failures.append('INDEX_HASH')
    seen=set()
    for e in index.get('entries',[]):
        try:
            if e['logical_name'] in seen: raise P4Error('duplicate logical name')
            seen.add(e['logical_name']); rel=_safe_rel(e['object_path'],'object_path'); p=portable_root/rel
            if not p.is_file(): raise P4Error('missing object')
            if _sha_file(p)!=e['sha256'] or p.stat().st_size!=e['size_bytes']: raise P4Error('object identity mismatch')
        except Exception as exc: failures.append(f"{e.get('logical_name')}:{type(exc).__name__}:{exc}")
    return {'status':'PASS' if not failures else 'FAIL','failures':failures,'objects':len(index.get('entries',[]))}


def resolve(portable_root: Path, index: dict, logical_name: str, destination: Path) -> Path:
    vr=validate_resolver(portable_root,index)
    if vr['status']!='PASS': raise P4Error(f'resolver invalid: {vr}')
    e=next((x for x in index['entries'] if x['logical_name']==logical_name),None)
    if e is None: raise P4Error(f'unknown resolver logical name {logical_name}')
    src=Path(portable_root)/_safe_rel(e['object_path']); destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,destination)
    if _sha_file(destination)!=e['sha256']: raise P4Error('materialized object mismatch')
    return destination


def seal_provenance(*, run_id: str, source_manifest: dict, bindings: list[dict[str,Any]], component_hashes: dict[str,str], execution_state: str, publication_state: str) -> dict:
    if execution_state!='COMPLETE_VALID': raise P4Error('cannot seal incomplete execution provenance')
    norm=[]
    for b in bindings:
        norm.append({'role':b['role'],'logical_name':b['logical_name'],'sha256':_require_sha(b['sha256'],f"binding {b['logical_name']}"),'size_bytes':int(b['size_bytes'])})
    for k,v in component_hashes.items(): _require_sha(v,k)
    base={'schema_id':PROVENANCE_SCHEMA,'run_id':run_id,'execution_state':execution_state,'publication_state':publication_state,
          'source_closure_sha256':source_manifest['source_closure_sha256'],'source_file_count':len(source_manifest['files']),
          'component_hashes':dict(sorted(component_hashes.items())),'bindings':sorted(norm,key=lambda x:(x['role'],x['logical_name'])),
          'separation':'SCIENTIFIC_PROJECTION_HASH_IS_INDEPENDENT_OF_PROVENANCE_ENVELOPE_HASH'}
    return dict(base,provenance_sha256=canonical_sha256(base))


def validate_claim_graph(graph: dict) -> dict:
    failures=[]
    if graph.get('schema_id')!=CLAIM_GRAPH_SCHEMA: failures.append('SCHEMA')
    obs=canonical_sha256({k:v for k,v in graph.items() if k!='graph_sha256'})
    if graph.get('graph_sha256')!=obs: failures.append('GRAPH_HASH')
    nodes={n.get('node_id'):n for n in graph.get('nodes',[])}
    if None in nodes or len(nodes)!=len(graph.get('nodes',[])): failures.append('NODE_IDS')
    for n in graph.get('nodes',[]):
        for ref in n.get('depends_on',[])+n.get('evidence_refs',[])+n.get('supersedes',[]):
            if ref not in nodes: failures.append(f'UNRESOLVED:{n.get("node_id")}->{ref}')
    visiting=set(); done=set()
    def dfs(nid):
        if nid in done: return
        if nid in visiting: failures.append(f'CYCLE:{nid}'); return
        visiting.add(nid)
        for d in nodes[nid].get('depends_on',[]):
            if d in nodes: dfs(d)
        visiting.remove(nid); done.add(nid)
    for nid in list(nodes): dfs(nid)
    return {'status':'PASS' if not failures else 'FAIL','failures':sorted(set(failures)),'nodes':len(nodes)}


def reopen_projection(graph: dict, affected_assumptions: list[str]) -> dict:
    vg=validate_claim_graph(graph)
    if vg['status']!='PASS': raise P4Error(f'invalid claim graph: {vg}')
    nodes={n['node_id']:n for n in graph['nodes']}; affected=set()
    for nid,n in nodes.items():
        if set(n.get('assumption_tags',[])) & set(affected_assumptions): affected.add(nid)
    changed=True
    while changed:
        changed=False
        for nid,n in nodes.items():
            if nid not in affected and any(d in affected for d in n.get('depends_on',[])):
                affected.add(nid); changed=True
    return {'schema_id':'IG_DECODER_V05_CLAIM_REOPEN_AUDIT_V1','affected_assumptions':sorted(affected_assumptions),'affected_nodes':sorted(affected),'status':'REVIEW_REQUIRED' if affected else 'NO_CHANGE'}


def seal_claim_graph(nodes: list[dict]) -> dict:
    base={'schema_id':CLAIM_GRAPH_SCHEMA,'nodes':nodes}
    graph=dict(base,graph_sha256=canonical_sha256(base))
    vg=validate_claim_graph(graph)
    if vg['status']!='PASS': raise P4Error(f'claim graph invalid: {vg}')
    return graph


def scan_for_absolute_paths(obj: Any) -> list[str]:
    bad=[]
    def walk(v,path='root'):
        if isinstance(v,dict):
            for k,x in v.items(): walk(x,f'{path}.{k}')
        elif isinstance(v,list):
            for i,x in enumerate(v): walk(x,f'{path}[{i}]')
        elif isinstance(v,str) and (v.startswith('/') or '/mnt/data/' in v or '/home/' in v): bad.append(path)
    walk(obj); return bad
