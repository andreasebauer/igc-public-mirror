"""Independent, bounded interpreter of the frozen O6 resource-action grammar.

Nested containers are unordered bags. This implementation builds an immutable
indexed tree and recomputes only ancestors of changed sites for each action.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _site_hash(port, free) -> str:
    return _hash(b'SITE|' + json.dumps([list(port), list(free)], separators=(',', ':')).encode())


def _bag_hash(tag, children) -> str:
    return _hash((tag + '|' + '|'.join(sorted(children))).encode())


class ResourceMerkle:
    """Indexed nested-bag signatures, pointed classes and local successors."""

    def __init__(self, h6):
        self.nodes = []  # tag, children, parent, digest
        self.sites = {}  # path -> (port, free, leaf index)

        def add(tag, children=(), site=None):
            index = len(self.nodes)
            digest = (_site_hash(*site) if site is not None else
                      _bag_hash(tag, [self.nodes[c][3] for c in children]))
            self.nodes.append([tag, tuple(children), None, digest])
            for child in children:
                self.nodes[child][2] = index
            return index

        o5s = []
        for a, o5 in enumerate(h6):
            o4s = []
            for j, o4 in enumerate(o5):
                o3s = []
                for k, o3 in enumerate(o4):
                    o2s = []
                    for v, block in enumerate(o3):
                        leaves = []
                        for i, (port, free) in enumerate(block):
                            leaf = add('SITE', site=(port, free))
                            self.sites[(a, j, k, v, i)] = (tuple(port), tuple(free), leaf)
                            leaves.append(leaf)
                        o2s.append(add('O2', leaves))
                    o3s.append(add('O3', o2s))
                o4s.append(add('O4', o3s))
            o5s.append(add('O5', o4s))
        self.root = add('R6', o5s)
        self.digest = self.nodes[self.root][3]

    def pointed(self, leaf):
        nodes = self.nodes
        sig = _hash(('POINTSITE|' + nodes[leaf][3]).encode())
        node = leaf
        while nodes[node][2] is not None:
            parent = nodes[node][2]
            siblings = [nodes[c][3] for c in nodes[parent][1] if c != node]
            sig = _hash((nodes[parent][0] + 'P|' + sig + '|' +
                         '|'.join(sorted(siblings))).encode())
            node = parent
        return sig

    def successor(self, replacements):
        nodes = self.nodes
        affected = set()
        for leaf in replacements:
            while leaf is not None:
                affected.add(leaf)
                leaf = nodes[leaf][2]
        memo = {}

        def visit(node):
            if node not in affected:
                return nodes[node][3]
            if node not in memo:
                if node in replacements:
                    memo[node] = _site_hash(*replacements[node])
                else:
                    memo[node] = _bag_hash(nodes[node][0],
                                           [visit(c) for c in nodes[node][1]])
            return memo[node]
        return visit(self.root)


def resource_profile(h6, templates, pairs):
    """Return exact action labels, successor digests and occurrence counts."""
    nodes = []  # (tag, child IDs, parent ID, current digest)
    sites = []  # (path, port, free, leaf ID)

    def add(tag, child_ids=(), site=None):
        index = len(nodes)
        signature = _site_hash(*site) if site is not None else _bag_hash(
            tag, [nodes[c][3] for c in child_ids])
        nodes.append([tag, tuple(child_ids), None, signature])
        for child in child_ids:
            nodes[child][2] = index
        return index

    o5s = []
    for a, o5 in enumerate(h6):
        o4s = []
        for j, o4 in enumerate(o5):
            o3s = []
            for k, o3 in enumerate(o4):
                o2s = []
                for v, block in enumerate(o3):
                    leaves = []
                    for i, (port, free) in enumerate(block):
                        leaf = add('SITE', site=(port, free))
                        sites.append(((a, j, k, v, i), tuple(port), tuple(free), leaf))
                        leaves.append(leaf)
                    o2s.append(add('O2', leaves))
                o3s.append(add('O3', o2s))
            o4s.append(add('O4', o3s))
        o5s.append(add('O5', o4s))
    root = add('R6', o5s)

    def successor(changes):
        affected = set()
        for leaf in changes:
            while leaf is not None:
                affected.add(leaf)
                leaf = nodes[leaf][2]
        cached = {}

        def signature(node):
            if node not in affected:
                return nodes[node][3]
            if node not in cached:
                if node in changes:
                    cached[node] = _site_hash(*changes[node])
                else:
                    cached[node] = _bag_hash(nodes[node][0],
                                             [signature(child) for child in nodes[node][1]])
            return cached[node]
        return signature(root)

    counts = Counter()
    for _path, port, free, leaf in sites:
        reserved = tuple(p - f for p, f in zip(port, free))
        for template in templates:
            typ = template['source']
            next_port = tuple(p + d for p, d in zip(port, template['dp']))
            if port[typ] <= 0 or min(next_port) < 0:
                continue
            if free[typ] > 0:
                mode = 'STRICT'
            elif all(p >= used for p, used in zip(next_port, reserved)):
                mode = 'CROSS'
            else:
                continue
            next_free = tuple(p - used for p, used in zip(next_port, reserved))
            counts[(('LIFT', mode, template['index']),
                    successor({leaf: (next_port, next_free)}))] += 1

    for left_index, (path_l, port_l, free_l, leaf_l) in enumerate(sites):
        for path_r, port_r, free_r, leaf_r in sites[left_index + 1:]:
            first_difference = next((pos for pos, (l, r) in enumerate(zip(path_l, path_r))
                                     if l != r), None)
            if first_difference is None:
                continue
            level = ('R6', 'R5', 'R4', 'R3', 'R2')[first_difference]
            for typ_l, typ_r in pairs:
                if free_l[typ_l] <= 0 or free_r[typ_r] <= 0:
                    continue
                next_l, next_r = list(free_l), list(free_r)
                next_l[typ_l] -= 1
                next_r[typ_r] -= 1
                digest = successor({leaf_l: (port_l, tuple(next_l)),
                                    leaf_r: (port_r, tuple(next_r))})
                counts[((level, min(typ_l, typ_r), max(typ_l, typ_r)), digest)] += 1
    return counts
