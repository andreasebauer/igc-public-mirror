"""Exact V2 DAG boundary checks; no state generation or process ownership."""
from .canon import canonical_sha256


def require(condition, message):
    if not condition:
        raise ValueError('MATURATION_DAG_' + message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name + '_EXACT_INTEGER')
    return value


def text(value, name):
    require(type(value) is str and bool(value), name + '_NONEMPTY_STRING')
    return value


def exact_data(value):
    if type(value) in (list, tuple):
        return tuple(exact_data(x) for x in value)
    if type(value) is int:
        return integer(value, 'SCALAR')
    if type(value) is str:
        return value
    if type(value) is dict:
        require(all(type(k) is str for k in value), 'STRING_KEYS')
        return {k: exact_data(v) for k, v in value.items()}
    raise ValueError('MATURATION_DAG_INEXACT_VALUE_' + type(value).__name__)


def o7_source_coordinates(state):
    """Complete source coordinates, not a digest used as equality authority."""
    ctx = state.ctx
    integer(ctx.n, 'O7_OWNER_COUNT', 1)
    require(len(ctx.parents) == len(ctx.source_order) == ctx.n, 'O7_CONTEXT_ARITY')
    source_order = tuple(text(x, 'O7_PARENT_SOURCE') for x in ctx.source_order)
    parents = tuple((text(p.exact_color, 'O7_PARENT_COLOR'), exact_data(p.h6)) for p in ctx.parents)
    edges = exact_data(state.edges)
    for edge in edges:
        require(len(edge) == 14 and all(type(x) is int for x in edge), 'O7_EDGE_COORDINATES')
    return source_order, parents, edges


def validate_state_dag(obj):
    """Reject malformed inputs before authority loading or scientific reconstruction."""
    require(type(obj) is dict and set(obj) == {'schema_id', 'execution_policy', 'roots', 'nodes', 'science_sha256'}, 'TOP_FIELDS')
    require(obj['schema_id'] == 'IG_MATURATION_STATE_DAG_V2', 'EXPLICIT_POLICY_V2_REQUIRED')
    policy = obj['execution_policy']
    require(policy in {'NATIVE_V3', 'HISTORICAL_COORDINATE_V1'}, 'EXECUTION_POLICY')
    exact_data(obj)
    raw = {k: v for k, v in obj.items() if k != 'science_sha256'}
    require(obj['science_sha256'] == canonical_sha256(raw), 'CONTENT_SEAL')
    nodes, roots = obj['nodes'], obj['roots']
    require(type(nodes) is dict and nodes and type(roots) is list and roots, 'NONEMPTY_ROOTS_NODES')
    require(all(type(r) is str and r in nodes for r in roots), 'ROOT_REFERENCE')
    common = {'kind', 'construction_digest', 'execution_policy', 'execution_order_key', 'lane'}
    for key, row in nodes.items():
        text(key, 'NODE_ALIAS')
        require(type(row) is dict and row.get('construction_digest') == key, 'NODE_ALIAS_BINDING')
        require(row.get('execution_policy') == policy, 'NODE_POLICY')
        text(row.get('execution_order_key'), 'EXECUTION_ORDER_KEY')
        text(row.get('lane'), 'LANE')
        if row.get('kind') == 'O7':
            require(set(row) == common | {'source_id', 'reserve_counts'}, 'O7_FIELDS')
            text(row['source_id'], 'O7_SOURCE')
            counts = row['reserve_counts']
            require(type(counts) is list and len(counts) == 7, 'O7_RESERVATION_ARITY')
            for value in counts:
                integer(value, 'O7_RESERVATION')
        else:
            require(row.get('kind') == 'LIFT' and set(row) == common | {'level', 'children', 'top_edges_full', 'motif_id'}, 'LIFT_FIELDS')
            integer(row['level'], 'LIFT_LEVEL', 8)
            text(row['motif_id'], 'MOTIF')
            require(type(row['children']) is list and row['children'], 'ORDERED_CHILDREN')
            require(all(type(x) is str and x in nodes for x in row['children']), 'CHILD_REFERENCE')
            require(type(row['top_edges_full']) is list, 'FULL_EDGE_LIST')
    active, visited = set(), set()
    def visit(key):
        if key in visited:
            return
        require(key not in active, 'CYCLE')
        active.add(key)
        row = nodes[key]
        for child in row.get('children', []):
            visit(child)
            level = 7 if nodes[child]['kind'] == 'O7' else nodes[child]['level']
            require(level == row['level'] - 1, 'CHILD_LEVEL')
        active.remove(key); visited.add(key)
    for root in roots:
        visit(root)
    require(visited == set(nodes), 'UNREACHABLE_NODE')
    def witness(key, value, endpoint):
        row = nodes[key]
        require(type(value) is list, 'WITNESS_SEQUENCE')
        if row['kind'] == 'O7':
            require(len(value) == 2, 'LEAF_WITNESS_ARITY')
            t, ordinal = value
            integer(t, 'LEAF_ENDPOINT'); integer(ordinal, 'LEAF_ORDINAL')
            require(t == endpoint and ordinal < row['reserve_counts'][endpoint], 'LEAF_RESERVATION')
        else:
            require(len(value) == 3, 'OWNER_WITNESS_ARITY')
            slot, local, t = value
            integer(slot, 'WITNESS_SLOT'); integer(t, 'WITNESS_ENDPOINT')
            require(slot < len(row['children']) and t == endpoint, 'WITNESS_OWNER_TYPE')
            witness(row['children'][slot], local, endpoint)
    for row in nodes.values():
        if row['kind'] != 'LIFT':
            continue
        for edge in row['top_edges_full']:
            require(type(edge) is list and len(edge) == 6, 'EDGE_ARITY')
            u, v, a, b, wu, wv = edge
            for value in (u, v, a, b):
                integer(value, 'EDGE_COORDINATE')
            require(u != v and max(u, v) < len(row['children']) and max(a, b) < 7, 'EDGE_RANGE')
            witness(row['children'][u], wu, a); witness(row['children'][v], wv, b)
    return obj
