from __future__ import annotations
from typing import Any
from .graph_replay import minimum_preservation_set
from .graph_store import GraphStore


def gc_dry_run(paths, targets: list[str]) -> dict[str, Any]:
    graph=GraphStore(paths); roots=minimum_preservation_set(graph,targets)
    required=set(roots['required_roots']); unresolved=roots['unresolved']; candidates=[]; protected=[]
    for node in graph.list_nodes():
        nid=node['node_id']; ret=node['retention_class']
        if ret=='PINNED' or nid in required:
            protected.append({'node_id':nid,'reason':'PINNED_OR_REQUIRED_ROOT','retention_class':ret}); continue
        if unresolved:
            protected.append({'node_id':nid,'reason':'UNRESOLVED_TARGET_CLOSURE','retention_class':ret}); continue
        candidates.append({'node_id':nid,'retention_class':ret,'action':'WOULD_DELETE_CONTENT_ONLY','destructive_enabled':False})
    return {'schema_id':'IG_GRAPH_GC_DRY_RUN_V0_1','status':'PASS' if not unresolved else 'REFUSED_UNRESOLVED','mode':'DRY_RUN_ONLY','destructive_gc_enabled':False,'targets':roots['targets'],'minimum_preservation':roots,'candidates':candidates,'protected':protected}


def destructive_gc(*args,**kwargs):
    raise RuntimeError('Destructive graph garbage collection is disabled in Phase 1')
