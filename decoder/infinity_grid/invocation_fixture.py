"""Step4 saved engineering fixture for the native STAGE protocol."""
from types import SimpleNamespace
from .invocation import InvocationRefused
from .v05_origin_guard import current_execution_context_snapshot, registered_workspace_scope


def stage_handler(stage, runtime):
    snapshot = current_execution_context_snapshot()
    assert snapshot and snapshot['attempt']['attempt_id']
    refused = []
    try:
        with registered_workspace_scope(snapshot['attempt']['workspace'], snapshot['attempt']['job_id']):
            raise AssertionError('scope should refuse even inside a handler')
    except InvocationRefused as issue:
        refused.append(issue.as_dict())
    original = runtime._root
    runtime._root = original.parents[len(original.parents)-1]/'decoder-outside-attempt'
    try:
        runtime.publish_json('unexpected', {'bad':True})
        raise AssertionError('outside output should refuse')
    except InvocationRefused as issue:
        refused.append(issue.as_dict())
    finally:
        runtime._root = original
    publication = runtime.publish_json('recorded-result', {'fixture':True, 'attempt':snapshot['attempt']})
    return SimpleNamespace(result={'outcome':'PASS', 'refusals':refused, 'publication':publication})
