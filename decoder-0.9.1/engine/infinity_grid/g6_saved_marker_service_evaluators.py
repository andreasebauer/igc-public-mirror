"""Thin saved-case comparators; all marker semantics come from KernelView."""
from .v05_kernel_services import current_kernel_view, KernelServiceViolation

def _tuples(value):
    return tuple(_tuples(x) for x in value) if type(value) is list else value

def _scope(view):
    _wanted = {'EXACT_IDENTITY','EXACT_RELATION','MARKER_Q','MARKER_DECODE','MARKER_WRITE'}
    snap = view.snapshot()
    assert snap['role'] == 'PRIMARY'
    assert set(snap['allowed_kernel_services']) == set(snap['bound_provider_services']) == _wanted
    refused = 0
    for service in ('PREPARE_TREE','LEGACY_MARKER_DECODE_ORACLE'):
        try:
            view.call(service)
        except KernelServiceViolation:
            refused += 1
        else:
            raise AssertionError('UNDECLARED_SERVICE_EXECUTED')
    return refused

def saved_marker_q_evaluator(payload):
    view = current_kernel_view()
    refused = _scope(view)
    q = view.call('MARKER_Q',payload['tree'])
    assert q == _tuples(payload['marker_state'])
    assert view.call('EXACT_IDENTITY',payload['tree']) == _tuples(payload['identity'])
    dec = view.call('MARKER_DECODE',q)
    assert dec['status'] == 'PASS' and dec['unrooted_canon'] == _tuples(payload['identity'])
    return {'signature':('PASS',payload['occurrence'],len(q)),
            'metrics':{'marker_q_matches':1,'warm_decode_matches':1,'forbidden_service_refusals':refused}}

def saved_marker_cold_decode_evaluator(payload):
    view = current_kernel_view()
    refused = _scope(view)
    # No MARKER_Q call in this phase. Decoder owns the fresh per-phase kernel.
    dec = view.call('MARKER_DECODE',_tuples(payload['marker_state']))
    assert dec['status'] == 'PASS' and dec['candidate_count'] == 1
    assert dec['unrooted_canon'] == _tuples(payload['identity'])
    assert not dec['stats'].get('cache_seeded_from_exact_marker_observation',False)
    assert view.call('EXACT_IDENTITY',dec['tree']) == _tuples(payload['identity'])
    return {'signature':('PASS',payload['occurrence'],dec['candidate_count']),
            'metrics':{'cold_decode_matches':1,'forbidden_service_refusals':refused}}

def saved_marker_write_evaluator(payload):
    view = current_kernel_view()
    refused = _scope(view)
    left = view.call('MARKER_Q',payload['left'])
    right = view.call('MARKER_Q',payload['right'])
    assert left == _tuples(payload['left_marker_state']) and right == _tuples(payload['right_marker_state'])
    written = view.call('MARKER_WRITE',left,right,tuple(payload['operator']))
    assert written['states'] == _tuples(payload['states'])
    diag = written['diagnostics']
    assert diag['fused_raw_projection'] is True
    assert diag['raw_child_count'] == payload['raw_child_count'] == 18
    assert diag['raw_child_canons'] == _tuples(payload['raw_child_canons'])
    for record,q,can in diag['raw_child_rows']:
        dec = view.call('MARKER_DECODE',q)
        assert dec['status'] == 'PASS' and dec['unrooted_canon'] == can == view.call('EXACT_IDENTITY',record)
    return {'signature':('PASS',18,len(written['states'])),
            'metrics':{'marker_write_matches':1,'written_child_decode_matches':18,'forbidden_service_refusals':refused}}
