from __future__ import annotations
from typing import Any
from .graph_replay import ReplayPlanner, minimum_preservation_set
from .graph_store import GraphStore


def graph_doctor(paths, targets: list[str]|None=None) -> dict[str,Any]:
    graph=GraphStore(paths); validation=graph.validate_graph(targets=targets)
    failures=list(validation.get('failures',[])); reconstructability=[]
    for ref in targets or []:
        try:
            node=graph.resolve(ref); plan=ReplayPlanner(graph).plan(node['node_id'])
            reconstructability.append({'target':ref,'node_id':node['node_id'],'status':plan['status'],'unresolved':plan['unresolved']})
            if plan['status']!='PASS': failures.append({'reason':'unreconstructable_target','target':ref,'unresolved':plan['unresolved']})
        except Exception as e:
            failures.append({'reason':'target_resolution','target':ref,'error':str(e)})
    roots=minimum_preservation_set(graph,targets or []) if targets else None
    return {'schema_id':'IG_GRAPH_DOCTOR_RESULT_V0_1','status':'PASS' if not failures else 'FAIL','graph_validation':validation,'reconstructability':reconstructability,'minimum_preservation':roots,'failures':failures}
