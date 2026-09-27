"""Independent base site-orbit maps for pinned O6 parent resources."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .o6_resource_observer import ResourceMerkle


def base_orbits(h6):
    tree = ResourceMerkle(h6)
    groups = defaultdict(list)
    for path, (_port, _free, leaf) in tree.sites.items():
        groups[tree.pointed(leaf)].append(path)
    path_to_orbit = {path: orbit for orbit, paths in groups.items() for path in paths}
    orbit_to_paths = {orbit: tuple(sorted(paths)) for orbit, paths in groups.items()}
    return path_to_orbit, orbit_to_paths


def rebuild_parent_orbits(parents):
    """Retain pinned parent inputs while replacing their derived orbit maps."""
    result = {}
    for key, parent in parents.items():
        by_path, by_orbit = base_orbits(parent.h6)
        result[key] = replace(parent, base_p2k=by_path, base_k2p=by_orbit)
    return result
