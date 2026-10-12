import json
from importlib.resources import files
from infinity_grid.canon import canonical_sha256
from infinity_grid.g6_s3_repaired import _axis_b_contexts
from infinity_grid.g6_s2_repaired import _all_contexts

def test_s3r_question_hash_and_authority():
    q=json.loads(files('infinity_grid').joinpath('resources/g6/G6_S3_REPAIRED_PREREGISTRATION_V1.json').read_text())
    assert q['stage_id']=='G6:S3R'
    assert q['question_sha256']==canonical_sha256({k:v for k,v in q.items() if k!='question_sha256'})
    assert q['authority']['G6:S2R']['outcome']=='PASS_DISCRETE_KERNEL_L1'
    assert q['authority']['G6:S2R']['exact_s1_child_universe_count']==4520

def test_higher_panel_frozen_sizes():
    q=json.loads(files('infinity_grid').joinpath('resources/g6/G6_S3_REPAIRED_PREREGISTRATION_V1.json').read_text())
    assert q['higher_panel']['axis_A_complete_00_motifs']['motif_context_count']==128
    assert q['higher_panel']['axis_B_adversarial_h4']['sample_size']==128
    assert q['higher_panel']['axis_B_adversarial_h4']['relation_context_count']==512
    assert len(_axis_b_contexts())==4

def test_observer_ladder_still_contains_s2r_context():
    l1,l2,l3=_all_contexts()
    assert l1==[('D2_PATH',(0,0),'LEFT')]
    assert len(l2)==62 and len(l3)==248
    assert set(l1)<=set(l2)<=set(l3)

def test_s2r_authority_commit_integrity():
    d=json.loads(files('infinity_grid').joinpath('resources/g6/authority/G6__S2R.json').read_text())
    assert d['commit_sha256']==canonical_sha256({k:v for k,v in d.items() if k!='commit_sha256'})
    assert d['outcome']=='PASS_DISCRETE_KERNEL_L1'
