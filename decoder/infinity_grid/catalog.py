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
    def __init__(self, paths: IGPaths): self.paths = paths

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
        for suffix in ['', '-wal', '-shm']:
            Path(str(self.paths.db) + suffix).unlink(missing_ok=True)
        con = self.connect()
        names = ['artifacts', 'datasets', 'protocols', 'runs', 'stages', 'claims', 'releases', 'graph_nodes', 'graph_edges', 'recipes', 'aliases', 'reconstruction_proofs']
        counts = {k: 0 for k in names}
        try:
            for p in sorted((self.paths.store / 'artifacts').glob('*.json')):
                o = json.loads(p.read_text()); con.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?,?,?,?,?,?)", (o['sha256'], o['size_bytes'], o.get('media_type'), o.get('logical_role'), o.get('source_name'), o.get('created_by_run_id'), str(p))); counts['artifacts'] += 1
            for p in sorted((self.paths.store / 'datasets').glob('*.json')):
                o = json.loads(p.read_text()); con.execute("INSERT OR REPLACE INTO datasets VALUES(?,?,?,?,?)", (o['dataset_sha256'], o.get('logical_role'), o.get('row_count'), o.get('format'), str(p))); counts['datasets'] += 1
            for p in sorted((self.paths.store / 'protocols').glob('*.json')):
                o = json.loads(p.read_text()); con.execute("INSERT OR REPLACE INTO protocols VALUES(?,?,?,?,?)", (o['protocol_id'], o['version'], o['descriptor_sha256'], o.get('purpose'), str(p))); counts['protocols'] += 1
            for rp in sorted(self.paths.runs.glob('*/run.json')):
                o = json.loads(rp.read_text()); pr = o['protocol']; ev = o.get('evidence', {})
                con.execute("INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?,?,?)", (o['run_id'], pr['protocol_id'], pr['version'], o['lifecycle'], o['plan_sha256'], o['question_sha256'], ev.get('authority'), ev.get('protocol_label'), str(rp))); counts['runs'] += 1
                for cp in sorted((rp.parent / 'stages').glob('*/current.json')):
                    ptr = json.loads(cp.read_text()); ap = cp.parent / 'attempts' / f"{int(ptr['attempt']):06d}.json"; co = json.loads(ap.read_text())
                    con.execute("INSERT OR REPLACE INTO stages VALUES(?,?,?,?,?,?)", (o['run_id'], co['stage_id'], co['status'], co['attempt'], ptr.get('checkpoint_sha256'), str(ap))); counts['stages'] += 1
            for p in sorted((self.paths.store / 'claims').glob('**/*.json')):
                o = json.loads(p.read_text()); con.execute("INSERT OR REPLACE INTO claims VALUES(?,?,?,?,?,?)", (o['claim_id'], o['version'], o['status'], o['claim_type'], o['statement'], str(p))); counts['claims'] += 1
            for p in sorted(self.paths.releases.glob('*/release_manifest.json')):
                o = json.loads(p.read_text()); con.execute("INSERT OR REPLACE INTO releases VALUES(?,?,?,?,?)", (o['release_id'], o['run_id'], o['mode'], o.get('archive_sha256'), str(p))); counts['releases'] += 1
            graph_root = self.paths.store / 'graph'
            for p in sorted((graph_root / 'nodes').glob('*.json')):
                o = json.loads(p.read_text()); refs = [r.get('sha256') for r in o.get('content_refs', []) if r.get('sha256')]
                con.execute("INSERT OR REPLACE INTO graph_nodes VALUES(?,?,?,?,?,?,?,?,?)", (o['node_id'], o['node_type'], o['schema_version'], o['status'], o['retention_class'], o['semantic_role'], str(p), refs[0] if len(refs) == 1 else None, o.get('producer_ref'))); counts['graph_nodes'] += 1
            for p in sorted((graph_root / 'edges').glob('*.json')):
                o = json.loads(p.read_text()); con.execute("INSERT OR REPLACE INTO graph_edges VALUES(?,?,?,?,?,?,?)", (o['edge_id'], o['edge_type'], o['source_node_id'], o['target_node_id'], str(p), int(o['mandatory']), o['scope'])); counts['graph_edges'] += 1
            for p in sorted((graph_root / 'recipes').glob('*.json')):
                o = json.loads(p.read_text()); con.execute("INSERT OR REPLACE INTO recipes VALUES(?,?,?,?,?)", (o['recipe_sha256'], o['recipe_id'], o['recipe_version'], str(p), o['replay_class'])); counts['recipes'] += 1
            for p in sorted((graph_root / 'aliases').glob('*.json')):
                o = json.loads(p.read_text()); con.execute("INSERT OR REPLACE INTO aliases VALUES(?,?,?,?,?)", (o['alias'], o['version'], o['node_id'], str(p), o['status'])); counts['aliases'] += 1
            for p in sorted((graph_root / 'reconstruction_proofs').glob('*.json')):
                o = json.loads(p.read_text()); con.execute("INSERT OR REPLACE INTO reconstruction_proofs VALUES(?,?,?,?,?,?,?)", (o['proof_id'], o['target_node_id'], o['root_set_sha256'], o['recipe_closure_sha256'], o['status'], o['result_sha256'], str(p))); counts['reconstruction_proofs'] += 1
            con.commit()
        finally:
            con.close()
        return {"status": "PASS", "counts": counts, "db": str(self.paths.db), "schema_version": SCHEMA_VERSION}

    def query(self, sql, args=()):
        con = self.connect(); con.row_factory = sqlite3.Row
        try: return [dict(r) for r in con.execute(sql, args).fetchall()]
        finally: con.close()
