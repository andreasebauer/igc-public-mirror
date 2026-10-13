from __future__ import annotations

import inspect
import pytest


def _valid_member():
    # Small exact tree using a real accepted repaired-H class and frozen operator.
    from infinity_grid.adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
    from infinity_grid.structural_encoding import structural_canonical_sha256
    ad=G4AcceptedAdapter(); klass=next(iter(sorted(ad._rows)))
    tree=DecoratedG4Tree(n=1,edges=tuple(),H_classes=(klass,),edge_operators=tuple())
    digest=structural_canonical_sha256(ad.unrooted_canon(tree))
    return {
        'identity_sha256':digest,
        'identity_canonical_bytes_sha256':digest,
        'state_token':digest+':0',
        'state':{'n':1,'edges':[],'H_classes':[klass],'edge_operators':[]},
    }


def test_a20_recomputes_exact_structural_member_identity():
    from infinity_grid.g6_s7_a6_reuse_provenance import recompute_member_identity
    row=_valid_member()
    assert recompute_member_identity(row)==row['identity_sha256']


def test_a20_mutated_state_fails_closed():
    from infinity_grid.g6_s7_a6_reuse_provenance import recompute_member_identity, G6S7A6ReuseProvenanceError
    row=_valid_member(); row['state']['H_classes'][0]='NOT_A_REAL_CLASS'
    with pytest.raises(Exception):
        recompute_member_identity(row)


def test_a20_mutated_state_token_fails_closed():
    from infinity_grid.g6_s7_a6_reuse_provenance import recompute_member_identity, G6S7A6ReuseProvenanceError
    row=_valid_member(); row['state_token']='0'*64+':0'
    with pytest.raises(G6S7A6ReuseProvenanceError,match='TOKEN'):
        recompute_member_identity(row)


def test_a20_receipt_source_cannot_self_certify_independent_membership():
    import infinity_grid.g6_s7_a6_reuse_provenance as p
    src=inspect.getsource(p.verify_reuse_provenance)
    assert '"original_partition_store_bytes_verified": False' in src
    assert '"independent_member_to_class_verification": False' in src
    assert 'PAUSE_PENDING_VERIFICATION' in src


def test_a20_phase_schedule_is_exact_inherited_plus_31_batches():
    from infinity_grid.g6_s7_a6_reuse_provenance import _expected_phase_ids
    for panel in ('s1','higher'):
        rows=_expected_phase_ids(panel)
        assert len(rows)==32
        assert rows[0].endswith('INHERITED_PUBLIC_PARTITION')
        assert rows[1].endswith('BATCH_00') and rows[-1].endswith('BATCH_30')


def test_a20_controller_owns_provenance_operation():
    import inspect
    from pathlib import Path
    import infinity_grid.v05_controller_event_loop as loop
    source=Path(loop.__file__).resolve().parents[1]
    reg=loop._registry(source)
    assert 'G6_S7_A6_REUSE_PROVENANCE_VERIFY' in reg[loop.SCIENCE_JOB]['allowed_operations']
    handler=inspect.getsource(loop._handle_g6_science)
    event=inspect.getsource(loop.controller_child_main)
    assert "op=='G6_S7_A6_REUSE_PROVENANCE_VERIFY'" in handler
    assert 'G6_S7_A6_REUSE_PROVENANCE_VERIFY' in event
    assert "runtime/'validation'" in handler
