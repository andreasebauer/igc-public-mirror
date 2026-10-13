from __future__ import annotations
from pathlib import Path
from .canon import canonical_sha256, write_json_atomic
from .store import ArtifactStore
from .safety import validate_relative_path

class DatasetStore:
    def __init__(self, store: ArtifactStore): self.store=store
    @property
    def manifest_dir(self):
        p=self.store.root/'datasets'; p.mkdir(parents=True,exist_ok=True); return p
    def import_directory(self, root: str|Path, *, logical_role: str, canonicalization_version='FILESET_V1') -> dict:
        root=Path(root).resolve(); shards=[]; files=sorted(p for p in root.rglob('*') if p.is_file())
        for ordinal,p in enumerate(files):
            rel=p.relative_to(root).as_posix(); validate_relative_path(rel)
            rec=self.store.put_file(p,logical_role=logical_role,source_name=rel); shards.append({'sha256':rec['sha256'],'size_bytes':rec['size_bytes'],'row_count':1,'ordinal':ordinal,'relative_path':rel})
        man={'schema_id':'IG_DATASET_MANIFEST_V0_17','logical_role':logical_role,'canonicalization_version':canonicalization_version,'format':'FILE_SET','compression':'MIXED_OR_NONE','sort_order':'relative_path_lexicographic','row_count':len(shards),'shards':shards}; dsid=canonical_sha256(man); out=dict(man,dataset_sha256=dsid); mp=self.manifest_dir/f'{dsid}.json'
        if mp.exists():
            import json
            observed=json.loads(mp.read_text(encoding='utf-8'))
            if observed!=out: raise RuntimeError('dataset identity collision/mismatch')
        else: write_json_atomic(mp,out)
        return out
    def load(self, dataset_sha256: str) -> dict:
        import json
        if len(dataset_sha256)!=64 or any(c not in '0123456789abcdef' for c in dataset_sha256): raise ValueError('invalid dataset sha256')
        p=self.manifest_dir/f'{dataset_sha256}.json'
        if not p.is_file(): raise FileNotFoundError(p)
        obj=json.loads(p.read_text(encoding='utf-8')); base={k:v for k,v in obj.items() if k!='dataset_sha256'}
        if obj.get('dataset_sha256') != dataset_sha256:
            raise RuntimeError('embedded dataset_sha256 does not match requested identity')
        if canonical_sha256(base)!=dataset_sha256: raise RuntimeError('dataset manifest identity mismatch')
        shards=obj.get('shards')
        if not isinstance(shards,list): raise RuntimeError('dataset shards must be a list')
        if int(obj.get('row_count',-1)) != len(shards): raise RuntimeError('dataset row_count mismatch')
        seen=set(); paths=[]
        for i,s in enumerate(shards):
            if int(s.get('ordinal',-1)) != i: raise RuntimeError('dataset shard ordinal mismatch')
            rel=validate_relative_path(s['relative_path'])
            if rel in seen: raise RuntimeError('duplicate normalized dataset path')
            seen.add(rel); paths.append(rel)
        if obj.get('sort_order') == 'relative_path_lexicographic' and paths != sorted(paths):
            raise RuntimeError('dataset shard ordering mismatch')
        return obj
    def verify(self, dataset_sha256: str) -> dict:
        try: m=self.load(dataset_sha256)
        except Exception as e: return {'status':'FAIL','dataset_sha256':dataset_sha256,'error':str(e)}
        failures=[]
        for s in m['shards']:
            v=self.store.verify(s['sha256'],s['size_bytes'])
            if v['status']!='PASS': failures.append({'shard':s,'verification':v})
        return {'status':'PASS' if not failures else 'FAIL','dataset_sha256':dataset_sha256,'shards':len(m['shards']),'failures':failures}
    def materialize(self, dataset_sha256: str, destination: str|Path) -> Path:
        m=self.load(dataset_sha256); dst=Path(destination).resolve(strict=False); dst.mkdir(parents=True,exist_ok=True); base=dst.resolve()
        for s in m['shards']:
            rel=validate_relative_path(s['relative_path']); target=(base/rel).resolve(strict=False)
            try: target.relative_to(base)
            except ValueError: raise RuntimeError('dataset materialization path escape')
            self.store.materialize(s['sha256'],target,expected_size=s['size_bytes'])
        return dst
