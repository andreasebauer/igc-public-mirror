from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class IGPaths:
    root: Path
    @property
    def app(self): return self.root/"app"
    @property
    def store(self): return self.root/"store"
    @property
    def runs(self): return self.root/"runs"
    @property
    def catalog(self): return self.root/"catalog"
    @property
    def releases(self): return self.root/"releases"
    @property
    def workspace(self): return self.root/"workspace"
    @property
    def db(self): return self.catalog/"ig_catalog.sqlite"
    def ensure(self):
        for p in (self.app,self.store,self.runs,self.catalog,self.releases,self.workspace): p.mkdir(parents=True,exist_ok=True)
        for p in (self.store/"sha256", self.store/"datasets", self.store/"artifacts", self.store/"claims", self.store/"protocols"): p.mkdir(parents=True,exist_ok=True)
        return self

def resolve_root(value: str | Path | None) -> IGPaths:
    if value is None: value=os.environ.get("IG_ROOT", "/ig")
    return IGPaths(Path(value).expanduser().resolve())
