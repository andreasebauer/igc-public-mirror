from infinity_grid.g6_r0_post_graduation_fiber import derive_singleton_fiber


def test_r0_singleton_fiber_direct_corollary():
    grad={
      "status":"PASS","decision":"G6_GRADUATED","g6_graduated":True,
      "authorizes":"G6:R0_POST_GRADUATION_STRUCTURAL_RECONNAISSANCE",
      "public_descriptor":{"id":"Q_D_MARKER_RELATION","decoder":"ANY_CHILD_UNIQUE_D_DELETION"},
    }
    s5={
      "status":"PASS","classification":"PASS_S5_MARKER_OBSERVER_STATE_AND_GLOBAL_WRITE_LAW_EARNED",
      "global_marker_observer_injectivity_earned":True,"q_D_state_earned":True,
      "candidate":{"read_state":"Q_D_MARKER_RELATION","observer_decoder":"ANY_CHILD_UNIQUE_D_DELETION"},
    }
    s6={
      "status":"PASS","classification":"RECURSIVE_CLOSURE_CANDIDATE_MARKER_STATE","recursive_closure_candidate_earned":True,
      "proof_obligations":[
        {"id":"R3_QD_GLOBAL_INJECTIVITY","status":"PASS"},
        {"id":"R4_WRITE_FACTORISATION","status":"PASS"},
        {"id":"R6_STRUCTURAL_INDUCTION_ALL_FINITE_TERMS","status":"PASS"},
      ],
    }
    out=derive_singleton_fiber(graduation=grad,s5=s5,s6=s6)
    assert out["status"]=="PASS"
    assert out["fiber_cardinality"]==1
