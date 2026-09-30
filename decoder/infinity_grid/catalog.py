from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .paths import IGPaths

SCHEMA_VERSION = 2
DDL = '''
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts(sha256 TEXT PRIMARY KEY,size_bytes INTEGER NOT NULL,media_type TEXT,logical_role TEXT,source_name TEXT,created_by_run_id TEXT,record_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS datasets(dataset_sha256 TEXT PRIMARY KEY,logical_role TEXT,row_count INTEGER,format TEXT,manifest_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS protocols(protocol_id TEXT,version TEXT,descriptor_sha256 TEXT PRIMARY KEY,purpose TEXT,descriptor_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,protocol_id TEXT,protocol_version TEXT,lifecycle TEXT,plan_sha256 TEXT,question_sha256 TEXT,evidence_authority TEXT,evidence_label TEXT,run_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS stages(run_id TEXT,stage_id TEXT,status TEXT,attempt INTEGER,checkpoint_sha256 TEXT,checkpoint_path TEXT NOT NULL,PRIMARY KEY(run_id,stage_id));
CREATE TABLE IF NOT EXISTS claims(claim_id TEXT,version INTEGER,status TEXT,claim_type TEXT,statement TEXT,record_path TEXT NOT NULL,PRIMARY KEY(claim_id,version));
CREATE TABLE IF NOT EXISTS releases(release_id TEXT PRIMARY KEY,run_id TEXT,mode TEXT,sha256 TEXT,manifest_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS graph_nodes(node_id TEXT PRIMARY KEY,node_type TEXT NOT NULL,schema_version TEXT NOT NULL,status TEXT NOT NULL,retention_class TEXT NOT NULL,semantic_role TEXT NOT NULL,record_path TEXT NOT NULL,content_sha256 TEXT,producer_node_id TEXT);
CREATE TABLE IF NOT EXISTS graph_edges(edge_id TEXT PRIMARY KEY,edge_type TEXT NOT NULL,source_node_id TEXT NOT NULL,target_node_id TEXT NOT NULL,record_path TEXT NOT NULL,mandatory INTEGER NOT NULL,scope TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS recipes(recipe_sha256 TEXT PRIMARY KEY,recipe_id TEXT NOT NULL,version TEXT NOT NULL,record_path TEXT NOT NULL,replay_class TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS aliases(alias TEXT NOT NULL,version TEXT NOT NULL,node_id TEXT NOT NULL,record_path TEXT NOT NULL,status TEXT NOT NULL,PRIMARY KEY(alias,version));
CREATE TABLE IF NOT EXISTS reconstruction_proofs(proof_id TEXT PRIMARY KEY,target_node_id TEXT NOT NULL,root_set_sha256 TEXT NOT NULL,recipe_closure_sha256 TEXT NOT NULL,status TEXT NOT NULL,result_sha256 TEXT NOT NULL,record_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS gc_journal(transaction_id TEXT NOT NULL,node_id TEXT NOT NULL,action TEXT NOT NULL,status TEXT NOT NULL,timestamp_utc TEXT NOT NULL,PRIMARY KEY(transaction_id,node_id));
'''


class Catalogue:
    def __init__(self, paths: IGPaths, *, snapshot_ref=None):
        self.paths = paths
        self.snapshot_ref = snapshot_ref

    def _open_and_migrate(self):
        self.paths.catalog.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.paths.db)
        con.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        row = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        before = int(row[0]) if row else 0
        if before > SCHEMA_VERSION:
            con.close(); raise RuntimeError(f'catalogue schema {before} is newer than runtime {SCHEMA_VERSION}')
        con.executescript(DDL)
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))
        con.commit()
        return con, before

    def connect(self):
        con, _ = self._open_and_migrate(); return con

    def migrate(self):
        con, before = self._open_and_migrate(); con.close()
        return {"status": "PASS", "db": str(self.paths.db), "from_version": before, "to_version": SCHEMA_VERSION}

    def initialize(self):
        out = self.migrate(); out['schema_version'] = SCHEMA_VERSION; return out

    def rebuild(self):
        """Explicit writer: build and seal a new snapshot, never unlink the live DB."""
        from .catalog_read import rebuild_catalogue
        return rebuild_catalogue(self)

    def _populate_snapshot(self, con, read_record, logical_path):
        """Legacy table projection, with exact input bytes supplied by the builder."""
        names = ['artifacts', 'datasets', 'protocols', 'runs', 'stages', 'claims', 'releases', 'graph_nodes', 'graph_edges', 'recipes', 'aliases', 'reconstruction_proofs']
        counts = {k: 0 for k in names}
        for p in sorted((self.paths.store / 'artifacts').glob('*.json')):
            o = read_record(p); con.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)", (o['sha256'], o['size_bytes'], o.get('media_type'), o.get('logical_role'), o.get('source_name'), o.get('created_by_run_id'), logical_path(p))); counts['artifacts'] += 1
        for p in sorted((self.paths.store / 'datasets').glob('*.json')):
            o = read_record(p); con.execute("INSERT INTO datasets VALUES(?,?,?,?,?)", (o['dataset_sha256'], o.get('logical_role'), o.get('row_count'), o.get('format'), logical_path(p))); counts['datasets'] += 1
        for p in sorted((self.paths.store / 'protocols').glob('*.json')):
            o = read_record(p); con.execute("INSERT INTO protocols VALUES(?,?,?,?,?)", (o['protocol_id'], o['version'], o['descriptor_sha256'], o.get('purpose'), logical_path(p))); counts['protocols'] += 1
        for rp in sorted(self.paths.runs.glob('*/run.json')):
            o = read_record(rp); pr = o['protocol']; ev = o.get('evidence', {})
            con.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)", (o['run_id'], pr['protocol_id'], pr['version'], o['lifecycle'], o['plan_sha256'], o['question_sha256'], ev.get('authority'), ev.get('protocol_label'), logical_path(rp))); counts['runs'] += 1
            for cp in sorted((rp.parent / 'stages').glob('*/current.json')):
                ptr = read_record(cp); ap = cp.parent / 'attempts' / f"{int(ptr['attempt']):06d}.json"; co = read_record(ap)
                con.execute("INSERT INTO stages VALUES(?,?,?,?,?,?)", (o['run_id'], co['stage_id'], co['status'], co['attempt'], ptr.get('checkpoint_sha256'), logical_path(ap))); counts['stages'] += 1
        for p in sorted((self.paths.store / 'claims').glob('**/*.json')):
            o = read_record(p); con.execute("INSERT INTO claims VALUES(?,?,?,?,?,?)", (o['claim_id'], o['version'], o['status'], o['claim_type'], o['statement'], logical_path(p))); counts['claims'] += 1
        for p in sorted(self.paths.releases.glob('*/release_manifest.json')):
            o = read_record(p); con.execute("INSERT INTO releases VALUES(?,?,?,?,?)", (o['release_id'], o['run_id'], o['mode'], o.get('archive_sha256'), logical_path(p))); counts['releases'] += 1
        graph_root = self.paths.store / 'graph'
        for p in sorted((graph_root / 'nodes').glob('*.json')):
            o = read_record(p); refs = [r.get('sha256') for r in o.get('content_refs', []) if r.get('sha256')]
            con.execute("INSERT INTO graph_nodes VALUES(?,?,?,?,?,?,?,?,?)", (o['node_id'], o['node_type'], o['schema_version'], o['status'], o['retention_class'], o['semantic_role'], logical_path(p), refs[0] if len(refs) == 1 else None, o.get('producer_ref'))); counts['graph_nodes'] += 1
        for p in sorted((graph_root / 'edges').glob('*.json')):
            o = read_record(p); con.execute("INSERT INTO graph_edges VALUES(?,?,?,?,?,?,?)", (o['edge_id'], o['edge_type'], o['source_node_id'], o['target_node_id'], logical_path(p), int(o['mandatory']), o['scope'])); counts['graph_edges'] += 1
        for p in sorted((graph_root / 'recipes').glob('*.json')):
            o = read_record(p); con.execute("INSERT INTO recipes VALUES(?,?,?,?,?)", (o['recipe_sha256'], o['recipe_id'], o['recipe_version'], logical_path(p), o['replay_class'])); counts['recipes'] += 1
        for p in sorted((graph_root / 'aliases').glob('*.json')):
            o = read_record(p); con.execute("INSERT INTO aliases VALUES(?,?,?,?,?)", (o['alias'], o['version'], o['node_id'], logical_path(p), o['status'])); counts['aliases'] += 1
        for p in sorted((graph_root / 'reconstruction_proofs').glob('*.json')):
            o = read_record(p); con.execute("INSERT INTO reconstruction_proofs VALUES(?,?,?,?,?,?,?)", (o['proof_id'], o['target_node_id'], o['root_set_sha256'], o['recipe_closure_sha256'], o['status'], o['result_sha256'], logical_path(p))); counts['reconstruction_proofs'] += 1
        return counts

    def open_snapshot(self):
        """Open a pinned snapshot or the current browsing alias, without creating files."""
        from .catalog_read import CatalogueReader
        return CatalogueReader(self.paths, snapshot_ref=self.snapshot_ref)

    def query(self, sql, args=(), *, max_rows=1024, max_bytes=1048576, max_vm_steps=500000):
        """Bounded compatibility API; never silently returns an incomplete list.

        Unpinned instances follow the browsing alias once per query. Scientific
        consumers must provide snapshot_ref; this alias grants no acceptance.
        """
        with self.open_snapshot() as reader:
            return reader.query(sql, args, max_rows=max_rows, max_bytes=max_bytes,
                                max_vm_steps=max_vm_steps)

    def build_read_index(self, records, destination, *, release_root, source_collections, expected_inventory):
        """Explicit new immutable projection; never deletes/rebuilds the live catalog."""
        from .storage_catalog import build_index
        return build_index(records, destination, release_root=release_root,
            source_collections=source_collections, expected_inventory=expected_inventory)

    def open_read_index(self, directory, *, expected_manifest_ref, release_root):
        """Pure read path; does not call connect(), initialize() or migrate()."""
        from .storage_catalog import ReadIndex
        return ReadIndex(directory, expected_manifest_ref=expected_manifest_ref, release_root=release_root)
