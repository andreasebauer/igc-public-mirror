"""Pure semantic core for Infinity Grid Algebra Decoder v0.26.5.

The core is deliberately unable to perform filesystem I/O, inspect clocks or
processes, consult environment/global runtime state, or use randomness.  It
contains only exact deterministic value transformations extracted from the
v0.26 implementation without changing algebra semantics.
"""
from .canonical import CANONICALIZER_ID, CANONICALIZER_VERSION, CanonicalEncodingError, canonical_bytes, canonical_text, canonical_sha256, normalize_json
from .boundary import destination_tuple, canonical_boundary_record, record_sha256, bridge_ports, compose_boundary_records
from .invariants import weakest_link_score, joined_label, retained_port_count, selected_tied_max

__all__ = (
    'CANONICALIZER_ID','CANONICALIZER_VERSION','CanonicalEncodingError','canonical_bytes','canonical_text','canonical_sha256','normalize_json',
    'destination_tuple','canonical_boundary_record','record_sha256','bridge_ports','compose_boundary_records',
    'weakest_link_score','joined_label','retained_port_count','selected_tied_max',
)
