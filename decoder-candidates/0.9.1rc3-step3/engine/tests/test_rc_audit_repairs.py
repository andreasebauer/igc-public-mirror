import os,stat
import pytest
from infinity_grid import immutable_publication as pub,submission,portable_registry
from infinity_grid.v05_stage_architecture import audit_module_source
from infinity_grid.graph_library_backend import automorphisms,supported
from infinity_grid.regime_scanner import _automorphisms_legacy


def audit(tmp_path,text):
    p=tmp_path/'candidate.py';p.write_text(text);return audit_module_source(p)


def test_platform_system_direct_and_module_alias_allowed(tmp_path):
    for src in ['import platform\nplatform.system()','import platform as env\nenv.system()']:
        assert audit(tmp_path,src)['status']=='PASS'


def test_platform_system_function_alias_allowed(tmp_path):
    for src in ['from platform import system\nsystem()','from platform import system as host_os\nhost_os()']:
        assert audit(tmp_path,src)['status']=='PASS'


def test_process_launch_aliases_still_refused(tmp_path):
    for src in ['import os\nos.system("x")','import os as platform\nplatform.system("x")','from os import system as query\nquery("x")','from subprocess import run as query\nquery([])','mystery.system()']:
        assert audit(tmp_path,src)['status']=='FAIL'


def test_shadowed_and_ambiguous_platform_query_refused(tmp_path):
    for src in ['import platform\nplatform=other\nplatform.system()','import platform\ndef f(platform):\n platform.system()','import os as p\np.system("x")\nimport platform as p','import platform\nplatform.system=other\nplatform.system()']:
        assert audit(tmp_path,src)['status']=='FAIL'


def test_submission_publication_syncs_file_then_directory(tmp_path,monkeypatch):
    events=[];real=pub.os.fsync
    def sync(fd):events.append('DIR' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'FILE');return real(fd)
    monkeypatch.setattr(pub.os,'fsync',sync)
    out=submission._put_object(tmp_path,b'payload','.bin','input')
    assert (tmp_path/'objects'/out['object_name']).read_bytes()==b'payload'
    assert 'FILE' in events and events[-1]=='DIR' and events.index('FILE')<len(events)-1
    assert not list((tmp_path/'objects').glob('.object-*'))


def test_registry_publication_and_existing_retry_sync(tmp_path,monkeypatch):
    events=[];real=pub.sync_directory
    def sync(p):events.append(p);return real(p)
    monkeypatch.setattr(pub,'sync_directory',sync)
    digest=portable_registry.put(tmp_path,b'payload');events.clear()
    assert portable_registry.put(tmp_path,b'payload')==digest and events[-1]==tmp_path/'objects'
    assert (tmp_path/'objects'/digest).read_bytes()==b'payload'


def test_directory_sync_failure_propagates_then_retry_recovers(tmp_path,monkeypatch):
    real=pub.sync_directory;failed=[]
    def sync(p):
        if p==tmp_path/'objects' and not failed:failed.append(True);raise OSError('injected barrier failure')
        return real(p)
    monkeypatch.setattr(pub,'sync_directory',sync)
    with pytest.raises(OSError,match='barrier failure'):submission._put_object(tmp_path,b'payload','.bin','input')
    assert list((tmp_path/'objects').glob('*.bin'))
    monkeypatch.setattr(pub.os,'link',lambda *args:pytest.fail('retry must reuse verified published object'))
    out=submission._put_object(tmp_path,b'payload','.bin','input')
    assert (tmp_path/'objects'/out['object_name']).read_bytes()==b'payload'
    monkeypatch.undo()
    events=[];target=tmp_path/'nested'/'store'/'objects'/'item';failed=[]
    def sync_nested(p):
        events.append(p)
        if p==tmp_path/'nested' and not failed:failed.append(True);raise OSError('ancestor barrier failure')
        return real(p)
    monkeypatch.setattr(pub,'sync_directory',sync_nested)
    with pytest.raises(OSError,match='ancestor barrier failure'):pub.publish_bytes(target,b'x',lambda:pytest.fail('mismatch'))
    events.clear();pub.publish_bytes(target,b'x',lambda:pytest.fail('mismatch'))
    assert tmp_path in events and tmp_path/'nested' in events and events[-1]==target.parent


def test_existing_object_mismatch_refused_by_both_stores(tmp_path):
    out=submission._put_object(tmp_path,b'payload','.bin','input');(tmp_path/'objects'/out['object_name']).write_bytes(b'changed')
    with pytest.raises(submission.SubmissionError):submission._put_object(tmp_path,b'payload','.bin','input')
    digest=portable_registry.put(tmp_path,b'payload');(tmp_path/'objects'/digest).write_bytes(b'changed')
    with pytest.raises(portable_registry.InvocationRefused):portable_registry.put(tmp_path,b'payload')


def test_concurrent_same_content_publication_collision(tmp_path,monkeypatch):
    target=tmp_path/'objects'/'item';real=pub.os.link
    def link(src,dst):real(src,dst);raise FileExistsError('simulated competing successful publisher')
    monkeypatch.setattr(pub.os,'link',link)
    pub.publish_bytes(target,b'payload',lambda:pytest.fail('same bytes are not a collision'))
    assert target.read_bytes()==b'payload' and not list(target.parent.glob('.object-*'))


def test_mixed_color_representations_agree_across_backends():
    color=(0,0,0,0,0,0,0);mixed=[list(color),color]
    expected=[(0,1),(1,0)]
    for mode in ['auto','igraph_bliss','legacy']:
        assert automorphisms(2,mixed,[],_automorphisms_legacy,backend=mode)==expected
    assert isinstance(mixed[0],list) and isinstance(mixed[1],tuple)


def test_color_normalization_preserves_distinct_colors_and_typed_edges():
    for colors,edges in [([[0]*7,(1,0,0,0,0,0,0)],[]),([[0]*7,tuple([0]*7)],[(0,1,2,2)])]:
        canonical=tuple(tuple(c) for c in colors)
        expected=_automorphisms_legacy(2,canonical,edges)
        assert automorphisms(2,colors,edges,_automorphisms_legacy,backend='igraph_bliss')==expected


def test_unsupported_color_domain_retains_fallback():
    colors=[['invalid']];seen=[]
    def fallback(n,c,e):seen.append(c);return ['fallback']
    assert not supported(1,colors,[])
    assert automorphisms(1,colors,[],fallback)==['fallback'] and seen[0] is colors
