"""Captured R3 engineering tests; no scientific replay or release acceptance."""
from pathlib import Path
import hashlib
import json
import shutil
import sqlite3
import pytest
from infinity_grid.catalog import Catalogue
from infinity_grid.catalog_read import ALIAS, CatalogueReader
from infinity_grid.storage_catalog import ReadIndexError, file_ref
from infinity_grid.paths import IGPaths
from infinity_grid.store import ArtifactStore
from infinity_grid.graph_store import GraphStore, build_edge


def seed(tmp_path,n=3):
    paths=IGPaths(tmp_path/'ig').ensure()
    store=ArtifactStore(paths.store)
    for i in range(n): store.put_bytes(('v'+str(i)).encode(),logical_role='ENGINEERING_FIXTURE')
    return paths,Catalogue(paths)


def tree_bytes(root):
    return {p.relative_to(root).as_posix():(file_ref(p),p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_query_no_longer_calls_live_connection_or_migration(tmp_path,monkeypatch):
    paths,cat=seed(tmp_path);built=cat.rebuild();before=tree_bytes(paths.root)
    def reject(*a,**kw):raise AssertionError('READ_CALLED_LIVE_WRITER')
    monkeypatch.setattr(Catalogue,'connect',reject)
    monkeypatch.setattr(Catalogue,'_open_and_migrate',reject)
    assert cat.query('SELECT count(*) AS n FROM artifacts')==[{'n':3}]
    assert tree_bytes(paths.root)==before
    assert built['scientific_acceptance']=='NOT_GRANTED'


def test_missing_snapshot_does_not_create_a_workspace(tmp_path):
    root=tmp_path/'missing';cat=Catalogue(IGPaths(root))
    with pytest.raises(ReadIndexError,match='SNAPSHOT_REQUIRED'):cat.query('SELECT 1')
    assert not root.exists()


def test_rebuild_does_not_unlink_live_wal_database(tmp_path):
    paths,cat=seed(tmp_path)
    con=cat.connect();con.execute("INSERT INTO meta VALUES('sentinel','keep')");con.commit()
    before={str(p):p.read_bytes() for p in paths.catalog.glob('ig_catalog.sqlite*')}
    built=cat.rebuild()
    after={str(p):p.read_bytes() for p in paths.catalog.glob('ig_catalog.sqlite*')}
    assert before==after
    assert con.execute("SELECT value FROM meta WHERE key='sentinel'").fetchone()==('keep',)
    assert built['db']!=str(paths.db)
    con.close()


def test_failed_rebuild_retains_prior_alias_and_exact_index(tmp_path):
    paths,cat=seed(tmp_path);built=cat.rebuild()
    before=file_ref(Path(built['db']));alias=(paths.catalog/ALIAS).read_bytes()
    (paths.store/'artifacts/bad.json').write_bytes(b'{broken')
    with pytest.raises(Exception):cat.rebuild()
    assert (paths.catalog/ALIAS).read_bytes()==alias
    assert file_ref(Path(built['db']))==before
    assert cat.query('SELECT count(*) AS n FROM artifacts')==[{'n':3}]
    assert list(paths.catalog.glob('.catalogue-build-*'))


def test_alias_advances_but_open_and_explicit_pinned_readers_stay_fixed(tmp_path):
    paths,cat=seed(tmp_path);one=cat.rebuild()
    with cat.open_snapshot() as old:
        ArtifactStore(paths.store).put_bytes(b'new')
        two=cat.rebuild();assert one['snapshot_ref']!=two['snapshot_ref']
        assert old.query('SELECT count(*) AS n FROM artifacts')==[{'n':3}]
        assert cat.query('SELECT count(*) AS n FROM artifacts')==[{'n':4}]
        assert Catalogue(paths,snapshot_ref=one['snapshot_ref']).query('SELECT count(*) AS n FROM artifacts')==[{'n':3}]


def test_exact_raw_sources_and_relative_paths_survive_relocation(tmp_path):
    paths,cat=seed(tmp_path);one=cat.rebuild()
    expected={p.relative_to(paths.root).as_posix():p.read_bytes() for p in (paths.store/'artifacts').glob('*.json')}
    with cat.open_snapshot() as reader:
        rows=reader.query('SELECT logical_path,raw,raw_sha256 FROM source_records ORDER BY logical_path')
        assert {x['logical_path']:x['raw'] for x in rows}==expected
        assert all(hashlib.sha256(x['raw']).hexdigest()==x['raw_sha256'] for x in rows)
        assert all(not Path(x['record_path']).is_absolute() for x in reader.query('SELECT record_path FROM artifacts'))
    relocated=tmp_path/'relocated';shutil.copytree(paths.root,relocated)
    moved=Catalogue(IGPaths(relocated),snapshot_ref=one['snapshot_ref'])
    assert moved.query('SELECT count(*) AS n FROM artifacts')==[{'n':3}]


@pytest.mark.parametrize('sql',[
    'DELETE FROM artifacts',"UPDATE meta SET value='bad'",'PRAGMA user_version=8',
    "ATTACH DATABASE ':memory:' AS extra", "SELECT load_extension('anything')",
    "SELECT randomblob(1000000000)", "SELECT 1; SELECT 2"])
def test_raw_sql_cannot_escape_read_only_budgeted_api(tmp_path,sql):
    paths,cat=seed(tmp_path);cat.rebuild();before=tree_bytes(paths.root)
    with pytest.raises(ReadIndexError):cat.query(sql)
    assert tree_bytes(paths.root)==before


def test_row_and_byte_budget_refuse_instead_of_returning_partial_lists(tmp_path):
    _,cat=seed(tmp_path);cat.rebuild()
    with pytest.raises(ReadIndexError,match='ROW_BUDGET_EXCEEDED'):cat.query('SELECT * FROM artifacts',max_rows=1)
    with pytest.raises(ReadIndexError,match='BYTE_BUDGET_EXCEEDED'):cat.query('SELECT * FROM artifacts',max_bytes=1)
    assert len(cat.query('SELECT * FROM artifacts'))==3


def test_vm_work_budget_and_duplicate_columns(tmp_path):
    _,cat=seed(tmp_path,n=30);cat.rebuild()
    with pytest.raises(ReadIndexError,match='WORK_BUDGET'):
        cat.query('SELECT count(*) AS n FROM artifacts a, artifacts b, artifacts c, artifacts d',max_vm_steps=1000)
    with pytest.raises(ReadIndexError,match='DUPLICATE_QUERY_COLUMNS'):cat.query('SELECT 1 AS n, 2 AS n')


@pytest.mark.parametrize('suffix',['-wal','-shm','-journal'])
def test_sidecars_are_refused_not_deleted(tmp_path,suffix):
    _,cat=seed(tmp_path);built=cat.rebuild();side=Path(built['db']+suffix);side.write_bytes(b'keep')
    with pytest.raises(ReadIndexError,match='INDEX_NOT_CLOSED'):cat.query('SELECT 1')
    assert side.read_bytes()==b'keep'


def test_changed_manifest_or_index_fails_closed(tmp_path):
    _,cat=seed(tmp_path);built=cat.rebuild()
    p=Path(built['db']);p.chmod(0o644)
    with p.open('ab') as f:f.write(b'x')
    with pytest.raises(ReadIndexError,match='BYTES_MISMATCH'):cat.query('SELECT 1')


def test_source_symlink_and_oversize_rejected_before_publication(tmp_path):
    paths,cat=seed(tmp_path);cat.rebuild();alias=(paths.catalog/ALIAS).read_bytes()
    p=paths.store/'artifacts/oversize.json';p.write_bytes(b'x'*65537)
    with pytest.raises(ReadIndexError,match='BUDGET_EXCEEDED'):cat.rebuild()
    assert (paths.catalog/ALIAS).read_bytes()==alias
    p.unlink();target=tmp_path/'outside.json';target.write_text('{}');p.symlink_to(target)
    with pytest.raises(ReadIndexError,match='SYMLINK'):cat.rebuild()
    assert (paths.catalog/ALIAS).read_bytes()==alias


def test_source_change_and_membership_change_leave_old_snapshot(tmp_path,monkeypatch):
    paths,cat=seed(tmp_path);cat.rebuild();alias=(paths.catalog/ALIAS).read_bytes()
    original=Catalogue._populate_snapshot
    def altered(self,con,read_record,logical_path):
        counts=original(self,con,read_record,logical_path)
        p=next((paths.store/'artifacts').glob('*.json'));p.write_bytes(p.read_bytes()+b' ')
        return counts
    monkeypatch.setattr(Catalogue,'_populate_snapshot',altered)
    with pytest.raises(ReadIndexError,match='SOURCE_CHANGED'):cat.rebuild()
    assert (paths.catalog/ALIAS).read_bytes()==alias


def test_duplicate_native_keys_do_not_disappear_via_replace(tmp_path):
    paths,cat=seed(tmp_path);cat.rebuild();alias=(paths.catalog/ALIAS).read_bytes()
    p=next((paths.store/'artifacts').glob('*.json'))
    (p.parent/'duplicate.json').write_bytes(p.read_bytes())
    with pytest.raises(sqlite3.IntegrityError):cat.rebuild()
    assert (paths.catalog/ALIAS).read_bytes()==alias


def test_existing_graph_methods_explicit_snapshot_avoid_global_files(tmp_path,monkeypatch):
    paths,cat=seed(tmp_path);graph=GraphStore(paths)
    for n in range(1200):
        graph.put_edge(build_edge(edge_type='REQUIRES',source_node_id='S'+str(n),target_node_id='T'+str(n)),require_endpoints=False)
    expected_out=graph.outgoing('S1199');expected_in=graph.incoming('T1199')
    cat.rebuild()
    def no_scan(*a,**kw):raise AssertionError('GLOBAL_SCAN')
    monkeypatch.setattr(GraphStore,'list_edges',no_scan)
    with cat.open_snapshot() as reader:
        assert graph.outgoing('S1199',snapshot=reader)==expected_out
        assert graph.incoming('T1199',snapshot=reader)==expected_in
        assert graph.outgoing('S1199',[],snapshot=reader)==[]
        assert reader.graph_page('S1199',direction='OUT',max_vm_steps=1000)['rows']==expected_out
        plan=reader._con.execute('EXPLAIN QUERY PLAN SELECT record_path FROM graph_edges WHERE source_node_id=? AND edge_type=? AND record_path>? ORDER BY record_path LIMIT ?',('S1199','REQUIRES','',2)).fetchall()
        assert any('SEARCH' in str(tuple(x)) and 'legacy_edge_out_type_path' in str(tuple(x)) for x in plan)


def test_graph_cursor_binds_snapshot_query_and_preserves_all_edges(tmp_path):
    paths,cat=seed(tmp_path);graph=GraphStore(paths)
    for n in range(7):
        graph.put_edge(build_edge(edge_type='REQUIRES',source_node_id='S',target_node_id='T'+str(n)),require_endpoints=False)
    expected=graph.outgoing('S');one=cat.rebuild()
    with cat.open_snapshot() as reader:
        out=[];cursor=None
        while True:
            page=reader.graph_page('S',direction='OUT',limit=2,after=cursor);out.extend(page['rows']);cursor=page['next_cursor']
            if not page['truncated']:break
        assert out==expected
        cursor=reader.graph_page('S',direction='OUT',limit=2)['next_cursor']
        with pytest.raises(ReadIndexError,match='CURSOR_BINDING'):reader.graph_page('other',direction='OUT',after=cursor)
        with pytest.raises(ReadIndexError,match='GRAPH_LIST_BUDGET'):reader.graph_list('S',direction='OUT',max_rows=2)
        with pytest.raises(ReadIndexError,match='BUDGET'):reader.graph_page('S',direction='OUT',max_bytes=1)
    graph.put_edge(build_edge(edge_type='REQUIRES',source_node_id='S',target_node_id='NEW'),require_endpoints=False)
    cat.rebuild()
    with cat.open_snapshot() as reader:
        with pytest.raises(ReadIndexError,match='CURSOR_BINDING'):reader.graph_page('S',direction='OUT',after=cursor)


def test_offline_runtime_recipe_rejects_unverified_source_without_compiling(tmp_path,monkeypatch):
    import importlib.util,zipfile
    path=Path(__file__).parents[1]/'tools/provision_sqlite_3510300.py'
    spec=importlib.util.spec_from_file_location('sqlite_recipe',path)
    recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
    archive=tmp_path/'unverified.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('source/sqlite3.c','not upstream bytes')
    def forbidden(*a,**kw):raise AssertionError('UNVERIFIED_COMPILATION')
    monkeypatch.setattr(recipe.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='SOURCE_SHA3_MISMATCH'):recipe.provision(archive,tmp_path/'runtime')
    assert not (tmp_path/'runtime').exists()
