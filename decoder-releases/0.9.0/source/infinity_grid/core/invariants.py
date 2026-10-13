from __future__ import annotations


def weakest_link_score(*scores: int) -> int:
    if not scores:
        raise ValueError("at least one score is required")
    return min(int(s) for s in scores)


def joined_label(*labels: int) -> int:
    out = 0
    for label in labels:
        out |= int(label)
    return out


def retained_port_count(n: int, m: int) -> int:
    n = int(n); m = int(m)
    if n < 1 or m < 1:
        raise ValueError("single-bridge composition requires at least one exposed port on each side")
    return n + m - 2


def selected_tied_max(records, score_index: int = 2):
    records = tuple(records)
    if not records:
        return tuple()
    best = max(int(r[score_index]) for r in records)
    return tuple(r for r in records if int(r[score_index]) == best)
