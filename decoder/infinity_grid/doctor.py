from __future__ import annotations
import json
from .catalog import Catalogue
from .protocols import ProtocolRegistry
from .datasets import DatasetStore
from .store import ArtifactStore
from .releases import verify_release_archive
from .verification import verify_run
from .canon import canonical_sha256


def doctor(paths):
    paths.ensure(); checks={}; failures=[]; counts={}
    checks['directories']=all(p.is_dir() for p in [paths.app,paths.store,paths.runs,paths.catalog,paths.releases,paths.workspace])
    try: Catalogue(paths).initialize(); checks['catalogue']=True
    except Exception as e: checks['catalogue']=False; failures.append({'area':'catalogue','error':str(e)})
    # Protocol store: every executable descriptor must exist and match bytes/hash; unexpected stored records must self-hash.
    try:
        reg=ProtocolRegistry(); expected={d['descriptor_sha256'] for d in reg.list()}; observed=set()
        for p in sorted((paths.store/'protocols').glob('*.json')):
            observed.add(p.stem)
            try:
                o=json.loads(p.read_text()); base={k:v for k,v in o.items() if k!='descriptor_sha256'}
                if o.get('descriptor_sha256')!=p.stem or canonical_sha256(base)!=p.stem: raise RuntimeError('protocol hash mismatch')
                if o.get('protocol_id') in {d['protocol_id'] for d in reg.list()} and reg.get(o['protocol_id'], version=o.get('version'))!=o: raise RuntimeError('stored protocol differs from executable descriptor')
            except Exception as e: failures.append({'area':'protocol','path':str(p),'error':str(e)})
        missing=expected-observed
        if missing: failures.append({'area':'protocol','reason':'missing','sha256':sorted(missing)})
        checks['protocols']=not any(f['area']=='protocol' for f in failures); counts['protocols']=len(observed)
    except Exception as e: checks['protocols']=False; failures.append({'area':'protocol','error':str(e)})
    store=ArtifactStore(paths.store); ds=DatasetStore(store)
    # Dataset manifests/blobs.
    for p in sorted((paths.store/'datasets').glob('*.json')):
        counts['datasets']=counts.get('datasets',0)+1; v=ds.verify(p.stem)
        if v['status']!='PASS': failures.append({'area':'dataset','detail':v})
    checks['datasets']=not any(f['area']=='dataset' for f in failures)
    # Artifact metadata and every referenced blob.
    for p in sorted((paths.store/'artifacts').glob('*.json')):
        counts['artifacts']=counts.get('artifacts',0)+1
        try:
            o=json.loads(p.read_text());
            if o.get('sha256')!=p.stem: raise RuntimeError('artifact record filename/hash mismatch')
            v=store.verify(o['sha256'],o.get('size_bytes'))
            if v['status']!='PASS': raise RuntimeError(str(v))
        except Exception as e: failures.append({'area':'artifact','path':str(p),'error':str(e)})
    checks['artifacts']=not any(f['area']=='artifact' for f in failures)
    # Runs/checkpoints/full bindings.
    for p in sorted(paths.runs.glob('*/run.json')):
        counts['runs']=counts.get('runs',0)+1; v=verify_run(paths,p.parent.name)
        if v['status']!='PASS': failures.append({'area':'run','run_id':p.parent.name,'detail':v})
    checks['runs']=not any(f['area']=='run' for f in failures)
    # Claims: parse and ensure refs/dependencies resolve.
    for p in sorted((paths.store/'claims').glob('*/*.json')):
        counts['claims']=counts.get('claims',0)+1
        try:
            o=json.loads(p.read_text())
            for r in o.get('evidence_refs',[]):
                if verify_run(paths,r).get('status')!='PASS': raise RuntimeError(f'unresolved evidence run {r}')
            for d in o.get('dependencies',[]):
                if not (paths.store/'claims'/d).is_dir(): raise RuntimeError(f'unresolved dependency {d}')
        except Exception as e: failures.append({'area':'claim','path':str(p),'error':str(e)})
    checks['claims']=not any(f['area']=='claim' for f in failures)
    # Releases: external record binds archive and archive inventory/hash is exact.
    for p in sorted(paths.releases.glob('*/release_manifest.json')):
        counts['releases']=counts.get('releases',0)+1
        try:
            o=json.loads(p.read_text()); arc=p.parent/o['archive']; v=verify_release_archive(arc)
            if v['status']!='PASS': raise RuntimeError(str(v))
            from .hashing import sha256_file
            if sha256_file(arc)!=o.get('archive_sha256'): raise RuntimeError('external archive hash mismatch')
            if o.get('release_id')!=p.parent.name: raise RuntimeError('release directory identity mismatch')
        except Exception as e: failures.append({'area':'release','path':str(p),'error':str(e)})
    checks['releases']=not any(f['area']=='release' for f in failures)
    status='PASS' if all(checks.values()) and not failures else 'FAIL'
    return {'status':status,'checks':checks,'counts':counts,'failures':failures}
