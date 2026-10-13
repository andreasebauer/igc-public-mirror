"""Adversarial contract regressions. No production census/adoption/remote ACK."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import py_compile
from types import SimpleNamespace
import zipfile

import pytest
from infinity_grid import bulk_submission as bulk
from infinity_grid import submission as sub
from infinity_grid.bulk_mapping_store import MappingStore,BulkMappingError,read_mapping_rows
from infinity_grid.canon import canonical_sha256 as sha
from infinity_grid.structural_encoding import structural_canonical_bytes as enc


def plan():
    return {'plan_sha256':'p','preserved_rows':10,'domain_rows':14,
            'prefix':{'records':10,'files':[{'sha256':'kept'}]},
            'windows':[{'start':10,'stop':12,'rows':2},{'start':12,'stop':14,'rows':2}]}


def receipt(index):
    return dict(plan()['windows'][index],window_index=index,
                status='VERIFIED_SAVED_WINDOW',source_sha256='s',package_sha256='k')


def progress(count=2):
    value={}
    for i in range(count):value=bulk.advance_progress(plan(),value,receipt(i))
    return value


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(plan_sha256='foreign'),
    lambda p:p.update(schema_id='foreign'),
    lambda p:p.update(inherited_prefix={'records':10,'files':[]}),
    lambda p:p.update(next_ordinal=True),
    lambda p:p.update(next_ordinal=12),
    lambda p:p.update(missing_rows=2),
    lambda p:p.update(scientific_adoption=True),
    lambda p:p.update(source_cutover_frozen=False),
    lambda p:p['completed_windows'].append(receipt(1)),
    lambda p:p['completed_windows'].__setitem__(1,receipt(0)),
    lambda p:p['completed_windows'][0].update(status='NOT_VERIFIED'),
    lambda p:p['completed_windows'][0].update(rows=0),
    lambda p:p['completed_windows'][0].update(window_index=False),
    lambda p:p['completed_windows'][1].update(source_sha256='changed'),
])
def test_complete_progress_cannot_suppress_unverified_work(mutation):
    value=progress();mutation(value)
    with pytest.raises(ValueError):bulk.validate_progress(plan(),value)


def test_valid_complete_and_partial_progress_are_reverified_and_identity_bound():
    observed=[]
    def verify(i,r):observed.append(i);return copy.deepcopy(r)
    assert len(bulk.validate_progress(plan(),progress(),source_sha256='s',package_sha256='k',verify_receipt=verify))==2
    assert observed==[0,1]
    assert len(bulk.validate_progress(plan(),progress(1),verify_receipt=verify))==1
    with pytest.raises(ValueError,match='SOURCE_CHANGED'):
        bulk.validate_progress(plan(),progress(),source_sha256='foreign',package_sha256='k')
    with pytest.raises(ValueError,match='SAVED_RECEIPT_CHANGED'):
        bulk.validate_progress(plan(),progress(),verify_receipt=lambda i,r:dict(r,rows=0))


def test_runner_verifies_full_progress_before_returning(tmp_path,monkeypatch):
    from infinity_grid import isolated_runtime as ir,v05_controller_event_loop as loop
    engine=tmp_path/'engine';engine.mkdir();project=tmp_path/'adapter';project.mkdir()
    root=tmp_path/'run';root.mkdir();(root/'PROGRESS.json').write_bytes(enc(progress()))
    monkeypatch.setattr(ir,'verified_runtime',lambda:tmp_path)
    monkeypatch.setattr(loop,'_source_ids',lambda p:('s','k'))
    monkeypatch.setattr(bulk,'build_plan',lambda *a,**k:plan())
    adoption={'schema_id':'IG_DATA_OWNER_DECODER_ADOPTION_V1','status':'ADOPTED',
              'source_sha256':'s','package_sha256':'k','runtime_manifest_sha256':ir.BUNDLED_MANIFEST_SHA256,
              'plan_sha256':'p','preserved_prefix_rows':10}
    seen=[]
    def verified(root,p,i,r):seen.append(i);return r
    monkeypatch.setattr(bulk,'_saved_progress_receipt',verified)
    assert bulk.execute_adopted_windows(root,engine,project,'raw','prefix','sqlite',adoption,object())==progress()
    assert seen==[0,1,0,1]
    (root/'PROGRESS.json').write_bytes(enc({'completed_windows':[{},{}]}))
    with pytest.raises(ValueError,match='PROGRESS_FIELDS'):
        bulk.execute_adopted_windows(root,engine,project,'raw','prefix','sqlite',adoption,object())


def task(i=1):
    return SimpleNamespace(task_id='t'+str(i),binding_sha256='a'*64,
        payload={'descriptors':[{'ordinal':i,'edge':[i],'literal_descriptor_sha256':'d'*64}]})


def result(i=1):
    witness={'nodes':[1,2,3]}
    row={'ordinal':i,'raw_edge':[i],'literal_descriptor_sha256':'d'*64,
         'relabeling_witness':witness,'relabeling_witness_sha256':sha(witness)}
    signature={'schema':'IG_HET4_FULL_WITNESS_BATCH_V1','rows':[row]}
    return {'signature_canonical_bytes':enc(signature),'signature_sha256':sha(signature),'outcome_count':1}


@pytest.mark.parametrize('variant',['extra_signature','boolean','float','count_bool','edge_bool','missing_literal'])
def test_bad_output_refused_before_any_durable_change(tmp_path,variant):
    store=MappingStore(tmp_path/'mapping.sqlite3',{},[task()]);item=result()
    signature=json.loads(item['signature_canonical_bytes'])
    if variant=='extra_signature':signature['extra']='poison'
    if variant=='boolean':signature['rows'][0]['ordinal']=True
    if variant=='float':signature['rows'][0]['ordinal']=1.0
    if variant=='count_bool':item['outcome_count']=True
    if variant=='edge_bool':signature['rows'][0]['raw_edge']=[True]
    if variant=='missing_literal':del signature['rows'][0]['literal_descriptor_sha256']
    item.update(signature_canonical_bytes=enc(signature),signature_sha256=sha(signature))
    with pytest.raises(BulkMappingError):store.commit('t1',item)
    assert all(store.conn.execute('select count(*) from '+name).fetchone()[0]==0 for name in ('rows','tasks','witnesses'))
    store.commit('t1',result());store.summary();store.close()
    restored=MappingStore(tmp_path/'mapping.sqlite3',{},[task()]);assert not restored.pending();restored.close()


@pytest.mark.parametrize('damage',['witness','task','row','core_bool','scope','task_hash'])
def test_reader_refuses_corrupt_or_incomplete_data_before_yield(tmp_path,damage):
    path=tmp_path/'mapping.sqlite3';store=MappingStore(path,{},[task()]);store.commit('t1',result())
    if damage=='witness':store.conn.execute('delete from witnesses')
    if damage=='task':store.conn.execute('delete from tasks')
    if damage=='row':store.conn.execute('delete from rows')
    if damage=='core_bool':
        core=json.loads(store.conn.execute('select core from rows').fetchone()[0]);core['ordinal']=True
        store.conn.execute('update rows set core=?',(enc(core),))
    if damage=='scope':store.conn.execute("update meta set value='{}'")
    if damage=='task_hash':store.conn.execute("update tasks set result_sha256=?",('f'*64,))
    store.conn.commit();store.close()
    with pytest.raises(BulkMappingError):next(read_mapping_rows(path))


def test_reader_snapshot_checks_pins_and_explicit_partial_mode(tmp_path):
    path=tmp_path/'mapping.sqlite3';store=MappingStore(path,{'source':'s'},[task(1),task(2)])
    store.commit('t1',result());store.close()
    with pytest.raises(BulkMappingError,match='INCOMPLETE'):list(read_mapping_rows(path))
    assert len(list(read_mapping_rows(path,require_complete=False,expected_ordinals=[1])))==1
    with pytest.raises(BulkMappingError,match='COVERAGE'):
        list(read_mapping_rows(path,require_complete=False,expected_ordinals=[1,2]))
    with pytest.raises(BulkMappingError,match='DATABASE_HASH'):
        list(read_mapping_rows(path,expected_database_sha256='0'*64))


def test_source_metadata_restores_supported_identity_and_mutation_refusal(tmp_path,monkeypatch):
    from infinity_grid import records,contracts
    from infinity_grid.run_operations import RunOperations
    assert records.source_sha256()==contracts.release_identity()['source_sha256']
    envelope=contracts.create_envelope({'schema_id':'IG_STEP1_ENGINEERING_EXAMPLE','value':1},
                                        semantic_profile_id='STEP1_ENGINEERING',strict_payload=False)
    from infinity_grid import __version__
    assert envelope['decoder_release']['package_version']==__version__
    ops=RunOperations(tmp_path/'run',run_id='STEP1',plan={},plan_science_sha256='a'*64,
                      frontier_science_sha256='b'*64,entity_class_ref='ENG',regime_ref='ENG',
                      selected_test_refs=[],start_depth=0,max_depth=1)
    assert ops.expected_identity().payload['decoder_source_sha256']==records.source_sha256()
    monkeypatch.setattr(records,'live_source_sha256',lambda:'0'*64)
    with pytest.raises(RuntimeError,match='source identity mismatch'):records.source_sha256()
    with pytest.raises(RuntimeError,match='source identity mismatch'):contracts.release_identity()


def test_cached_project_code_requires_load_provenance(tmp_path,monkeypatch):
    from infinity_grid.project_stage import bind_callables,resolve_callable,ProjectStageError
    from infinity_grid.v05_engineering_worker import engineering_source_tree_digest as digest
    for name in list(sys.modules):
        if name=='project' or name.startswith('project.'):monkeypatch.delitem(sys.modules,name)
    source=tmp_path/'source';project=source/'project';project.mkdir(parents=True)
    path=project/'sample.py';path.write_text('def handler():\n    return "OLD"\n')
    monkeypatch.syspath_prepend(str(source));old=importlib.import_module('project.sample')
    path.write_text('def handler():\n    return "NEW"\n')
    binding=bind_callables(source,['project.sample:handler'],source_sha256=digest(source))['project.sample:handler']
    with pytest.raises(ProjectStageError,match='CACHED_MODULE_UNBOUND'):resolve_callable('project.sample:handler',binding)
    monkeypatch.delitem(sys.modules,'project.sample')
    fresh=resolve_callable('project.sample:handler',binding);assert fresh()=='NEW'
    assert resolve_callable('project.sample:handler',binding) is fresh
    monkeypatch.delitem(sys.modules,'project.sample');monkeypatch.delitem(sys.modules,'project')


def test_first_project_import_ignores_stale_bytecode_for_handler_and_helper(tmp_path,monkeypatch):
    from infinity_grid.project_stage import bind_callables,resolve_callable
    from infinity_grid.v05_engineering_worker import engineering_source_tree_digest as digest
    import os
    for name in list(sys.modules):
        if name=='project' or name.startswith('project.'):monkeypatch.delitem(sys.modules,name)
    source=tmp_path/'source';project=source/'project';project.mkdir(parents=True)
    helper=project/'helper.py';helper.write_text('value = "OLD"\n')
    stamp=helper.stat();py_compile.compile(str(helper),doraise=True)
    helper.write_text('value = "NEW"\n');os.utime(helper,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    handler=project/'fixture.py';handler.write_text('from .helper import value\ndef handler():\n    return value\n')
    monkeypatch.syspath_prepend(str(source))
    binding=bind_callables(source,['project.fixture:handler'],source_sha256=digest(source))['project.fixture:handler']
    assert resolve_callable('project.fixture:handler',binding)()=='NEW'
    for name in list(sys.modules):
        if name=='project' or name.startswith('project.'):monkeypatch.delitem(sys.modules,name)


@pytest.mark.parametrize('damage',['id','logical','size_bool','extra','timestamp_bool'])
def test_v2_receipt_requires_full_logical_and_provider_binding(tmp_path,damage):
    obj={'sha256':'a'*64,'size_bytes':1,'obligation_id':'b'*64,'role':'source',
         'logical_name':'source.zip','obligation_scope':'CAPTURE_SAVE'}
    row=dict(obj,schema_id='IG_DECODER_EXPLICIT_SAVE_RECEIPT_V2',provider='google_drive',
             drive_file_id='test_readback_12345',raw_readback_verified=True,recorded_ns=1,
             scope='EXPLICIT_CONNECTOR_OPERATOR_ATTESTATION_NOT_REMOTE_AUTHENTICATION')
    path=tmp_path/'receipt.json'
    def save():
        body={k:v for k,v in row.items() if k!='receipt_sha256'}
        row['receipt_sha256']=sha(body);path.write_bytes(enc(row))
    save();assert sub._valid_receipt(path,obj)
    if damage=='id':row['drive_file_id']=''
    if damage=='logical':row['logical_name']='foreign'
    if damage=='size_bool':row['size_bytes']=True
    if damage=='extra':row['unknown']='extra'
    if damage=='timestamp_bool':row['recorded_ns']=True
    save();assert not sub._valid_receipt(path,obj)


def test_archive_verifies_and_reads_one_handle(tmp_path,monkeypatch):
    from infinity_grid import bulk_descriptor_plan as module
    original=tmp_path/'raw.zip';replacement=tmp_path/'replacement.zip'
    for path,value in [(original,'PINNED'),(replacement,'UNPINNED')]:
        with zipfile.ZipFile(path,'w') as z:z.writestr('value',value)
    expected=hashlib.sha256(original.read_bytes()).hexdigest();real=module.hashlib.file_digest
    def replace(stream,algorithm):
        value=real(stream,algorithm);replacement.replace(original);return value
    monkeypatch.setattr(module.hashlib,'file_digest',replace)
    with pytest.raises(module.DescriptorPlanError,match='ARCHIVE_CHANGED'):
        with module.checked_archive(original,expected) as z:
            assert z.read('value')==b'PINNED'


def test_archive_normal_wrong_and_duplicate_bytes(tmp_path):
    from infinity_grid.bulk_descriptor_plan import checked_archive,DescriptorPlanError
    path=tmp_path/'raw.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('value','ok')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with checked_archive(path,digest) as z:assert z.read('value')==b'ok'
    with pytest.raises(DescriptorPlanError,match='MISMATCH'):
        with checked_archive(path,'0'*64):pass
    with zipfile.ZipFile(path,'a') as z:
        with pytest.warns(UserWarning):z.writestr('value','duplicate')
    with pytest.raises(DescriptorPlanError,match='DUPLICATE'):
        with checked_archive(path,hashlib.sha256(path.read_bytes()).hexdigest()):pass


@pytest.mark.parametrize('ordinal',[652366,-1,True,1.0])
def test_qualification_descriptor_domain_refuses_before_dispatch(tmp_path,monkeypatch,ordinal):
    from infinity_grid import bulk_descriptor_plan as planning
    adapter=Path(__file__).parents[2]/'adapter/bulk_adapter.py'
    if not adapter.exists():
        from infinity_grid import submission as submission_module
        enclosing=Path(__file__).parents[2]
        capture=submission_module.capture_record(enclosing)
        artifact=next(r for r in capture['environment']['artifacts'] if r['logical_name']=='candidate_adapter_archive')
        archive=enclosing/'runtime/intake/artifacts'/(artifact['sha256']+'.bin')
        raw=archive.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=artifact['sha256']:raise ValueError('ADAPTER_FIXTURE_HASH')
        import io,zipfile
        with zipfile.ZipFile(io.BytesIO(raw)) as z:body=z.read('bulk_adapter.py')
        adapter=tmp_path/'adapter/bulk_adapter.py';adapter.parent.mkdir();adapter.write_bytes(body)
    spec=importlib.util.spec_from_file_location('project.step1_adapter',adapter)
    # Fixture supplies only the historical helper; evaluator is never invoked.
    historical=SimpleNamespace(old_edge_alias=None,context=None)
    monkeypatch.setitem(sys.modules,'project.historical_alias',historical)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    saved=tmp_path/'plan.json';saved.write_bytes(enc({'domain_rows':652366,'windows':[]}))
    descriptors=tmp_path/'rows.json';descriptors.write_bytes(enc([{'ordinal':ordinal,'edge':[],'literal_descriptor_sha256':'a'*64}]))
    monkeypatch.setattr(planning,'build_plan',lambda *a,**k:json.loads(saved.read_bytes()))
    monkeypatch.setattr(planning,'iter_descriptors',lambda p:iter([]))
    stage={'execution':{'parameters':{'window_rows':16384,'registered_rows':1,'batch_size':16,'workers':1}},
           'input_artifacts':{'descriptor_sample':str(descriptors),'full_domain_plan':str(saved),
                              'raw_descriptor_archive':'raw','preserved_prefix_archive':'prefix'}}
    with pytest.raises(ValueError,match='QUALIFICATION_DESCRIPTOR_SCOPE'):
        module.handler(stage,object())
