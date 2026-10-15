"""Lossless unordered typed-resource trees; no exact-H or transport claims.

Counter recursion is on explicit values, not cryptographic hash equality.
One endpoint choice is one positive-capacity site/type occurrence.
"""
from collections import Counter
import hashlib
import json


def site(p, f):
    return ('SITE', tuple(p), tuple(f))


def container(tag, children):
    return (tag, tuple(sorted(children, key=repr)))


def from_h6(h):
    return container('R6', [container('O5', [container('O4', [container('O3', [
        container('O2', [site(p, f) for p, f in block]) for block in o3])
        for o3 in o4]) for o4 in o5]) for o5 in h])


def native_digest(tree):
    if tree[0] == 'SITE':
        raw = b'SITE|' + json.dumps([list(tree[1]), list(tree[2])], separators=(',', ':')).encode()
    else:
        raw = (tree[0] + '|' + '|'.join(sorted(native_digest(c) for c in tree[1]))).encode()
    return hashlib.sha256(raw).hexdigest()


def reserve(tree, t):
    """Structural recursion over child OCCURRENCES, accumulating collisions."""
    out = Counter()
    if tree[0] == 'SITE':
        p, f = tree[1:]
        if f[t] > 0:
            new = list(f)
            new[t] -= 1
            out[site(p, new)] += 1
    else:
        tag, children = tree
        for i, child in enumerate(children):
            for successor, count in reserve(child, t).items():
                post = list(children)
                post[i] = successor
                out[container(tag, post)] += count
    return out


def profile(tree, pairs):
    tag, children = tree
    out = Counter()
    for i in range(len(children)):
        for j in range(i + 1, len(children)):
            for a, b in pairs:
                for left, ml in reserve(children[i], a).items():
                    for right, mr in reserve(children[j], b).items():
                        post = list(children)
                        post[i], post[j] = left, right
                        out[((tag, min(a, b), max(a, b)), container(tag, post))] += ml * mr
    return out


def pushed(counter):
    out = Counter()
    for (label, tree), count in counter.items():
        out[label, native_digest(tree)] += count
    return out


def to_state_tree(state):
    # Caller supplies exact O7 or ExactLift from the qualified mapper.
    if state.level == 7:
        return container('R7', [from_h6(h) for h in state.owner_h6s])
    return container('R' + str(state.level), [to_state_tree(c) for c in state.children])
