import json
from importlib.resources import files
from infinity_grid.canon import canonical_sha256
from infinity_grid.g6_s2_repaired import (
    _all_contexts,_independent_sig_for_context,_primary_sig_for_context,
    _tree_from_record,_tree_to_record,_universe_from_s1,_universe_index,
)

def test_question_hash_and_stage():
    q=json.loads(files('infinity_grid').joinpath('resources/g6/G6_S2_REPAIRED_PREREGISTRATION_V1.json').read_text())
    assert q['stage_id']=='G6:S2R'
    assert q['question_sha256']==canonical_sha256({k:v for k,v in q.items() if k!='question_sha256'})
    assert q['authority']['G6:S1R']['selected_candidate']=='C3_ROOTED_OWNER_RESPONSE_BAG'

def test_context_ladder_sizes_and_binding():
    l1,l2,l3=_all_contexts()
    assert l1==[('D2_PATH',(0,0),'LEFT')]
    assert len(l2)==62 and len(l3)==248
    assert set(l1)<=set(l2)<=set(l3)

def test_s1r_authority_commit_integrity():
    d=json.loads(files('infinity_grid').joinpath('resources/g6/authority/G6__S1R.json').read_text())
    assert d['commit_sha256']==canonical_sha256({k:v for k,v in d.items() if k!='commit_sha256'})
    assert d['outcome']=='PASS_ROOTED_OWNER_RESPONSE_BAG_EXPANDED_UNIVERSE'
    assert d['result']['exact_s1_child_universe_count']==4520

def test_universe_rebuild_binding():
    u=_universe_from_s1(); idx=_universe_index(u)
    assert len(u)==4520
    assert canonical_sha256(idx)=='2c1ffee315adada202ec472acb06f9fef9871ebbb273075b0dd0ef2a7d523f30'

def test_primary_independent_first_context_one_state():
    u=_universe_from_s1(); can,t=min(u.items(),key=lambda kv:canonical_sha256(kv[0]))
    ctx=('D2_PATH',(0,0),'LEFT')
    assert _primary_sig_for_context(t,ctx)==_independent_sig_for_context(t,ctx)
