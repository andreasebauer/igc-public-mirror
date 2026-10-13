from __future__ import annotations

import json
from pathlib import Path

from infinity_grid.adapters.g4_accepted import DecoratedG4Tree, G4AcceptedAdapter
from infinity_grid.exact_tree_relation_kernel import exact_graft_relation, prepare_g4_relation_tree
from infinity_grid.g6_stage_executors import _basis, _join_exact_relation
from infinity_grid.g6_s3_repaired import _axis_a_task


def _tree(obj):
    return DecoratedG4Tree(
        int(obj["n"]),
        tuple(tuple(map(int,e)) for e in obj["edges"]),
        tuple(obj["H_classes"]),
        tuple(tuple(map(int,x)) for x in obj["edge_operators"]),
    )


def test_exact_relation_kernel_matches_historical_on_operator_basis_panel():
    ad=G4AcceptedAdapter(); b=_basis(); ops=ad.operator_basis()
    pairs=[
        ("D2_PATH","D2_PATH"),
        ("D2_PATH","D2_BROOM"),
        ("D4_PATH","D4_BROOM"),
        ("D4_BROOM","D4_BROOM"),
    ]
    for lk,rk in pairs:
        lp=prepare_g4_relation_tree(b[lk],adapter=ad)
        rp=prepare_g4_relation_tree(b[rk],adapter=ad)
        for op in ops:
            _trees, old_canons, old_legal, old_attempted=_join_exact_relation(b[lk],b[rk],op)
            new=exact_graft_relation(lp,rp,op,adapter=ad)
            assert new.canons == tuple(old_canons)
            assert new.legal_owner_pairs == old_legal
            assert new.attempted_owner_pairs == old_attempted
            assert new.rooted_owner_pair_candidates <= new.legal_owner_pairs


def test_exact_relation_kernel_matches_historical_on_reconstructed_s3r_sample():
    states=[]; seen=set(); refs=sorted(_basis())
    for a in refs:
        for b in refs:
            for c in refs:
                sh=_axis_a_task((a,b,c))
                for row in sh['records']:
                    if row['state_id'] not in seen:
                        seen.add(row['state_id']); states.append(_tree(row['tree']))
                    if len(states)>=24: break
                if len(states)>=24: break
            if len(states)>=24: break
        if len(states)>=24: break
    assert len(states)==24
    probe=_basis()['D2_PATH']; ad=G4AcceptedAdapter(); pp=prepare_g4_relation_tree(probe,adapter=ad)
    strict_pruning_seen=False
    for state in states:
        _trees,old_canons,old_legal,old_attempted=_join_exact_relation(state,probe,(0,0))
        new=exact_graft_relation(prepare_g4_relation_tree(state,adapter=ad),pp,(0,0),adapter=ad)
        assert new.canons==tuple(old_canons)
        assert new.legal_owner_pairs==old_legal
        assert new.attempted_owner_pairs==old_attempted
        if new.rooted_owner_pair_candidates < new.legal_owner_pairs: strict_pruning_seen=True
    assert strict_pruning_seen
