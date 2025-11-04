from typing import Dict, List, Set
from igc.db.pg import cx, fetchall_dict

def list_metric_groups() -> List[Dict]:
    """Return six fixed groups with counts from view metgroup_count."""
    with cx() as conn:
        rows = fetchall_dict(conn, """
  SELECT g.id, g.name, COALESCE(COUNT(m.id),0) AS count
  FROM metgroup g
  LEFT JOIN metrics m ON m.group_id = g.id
  GROUP BY g.id, g.name
  ORDER BY g.id
""")
    return rows

def list_metrics_by_group() -> Dict[int, List[Dict]]:
    """Return metrics per group from view metrics_group."""
    with cx() as conn:
        rows = fetchall_dict(conn, 'SELECT group_id, metric_id, metric_name, metric_description FROM metrics_group ORDER BY group_name, metric_name')
    groups: Dict[int, List[Dict]] = {}
    for r in rows:
        g = groups.setdefault(r['group_id'], [])
        g.append({'id': r['metric_id'], 'name': r['metric_name'], 'desc': r.get('metric_description') or '', 'out': ''})
    return groups

def list_assigned_metrics(sim_id: int) -> Set[int]:
    """Return currently enabled metric IDs for given simulation."""
    with cx() as conn:
        rows = fetchall_dict(conn, 'SELECT metric_id FROM simmetricmatcher WHERE sim_id=%s AND enabled=true', (sim_id,))
    return {r['metric_id'] for r in rows}
