from __future__ import annotations

import random
import string
import pytest

from infinity_grid.canon import canonical_bytes, canonical_sha256
from infinity_grid.core.canonical import CanonicalEncodingError
from infinity_grid.structural_encoding import structural_canonical_bytes, structural_canonical_sha256


def _values():
    return [
        None, True, False, 0, -17, 2**80, 0.0, -0.0, 1.25, 1e-30,
        "", "ascii", "quote\\\"slash\\\\", "snowman ☃ / ไทย",
        (), [], (1, 2, ("x", False)), [1, (2, 3), [4]],
        {"b": 2, "a": (1, "x"), "unicode": "é"},
        ("V", ("seq", (1, "A")), (((0, 1), ("V", "x", ())),)),
    ]


@pytest.mark.parametrize("value", _values())
def test_fast_structural_encoder_is_byte_identical_to_authoritative_canonical_json(value):
    assert structural_canonical_bytes(value) == canonical_bytes(value)
    assert structural_canonical_sha256(value) == canonical_sha256(value)


def test_fast_structural_encoder_random_nested_equivalence():
    rng = random.Random(20260907)
    atoms = [None, True, False, -3, 0, 7, 1.5, "x", "é", "☃"]
    def build(depth=0):
        if depth >= 5 or rng.random() < 0.42:
            return rng.choice(atoms)
        kind = rng.choice(["tuple", "list", "dict"])
        if kind == "tuple":
            return tuple(build(depth+1) for _ in range(rng.randrange(5)))
        if kind == "list":
            return [build(depth+1) for _ in range(rng.randrange(5))]
        keys = rng.sample(list(string.ascii_lowercase), rng.randrange(5))
        return {k: build(depth+1) for k in keys}
    for _ in range(500):
        value = build()
        assert structural_canonical_bytes(value) == canonical_bytes(value)


def test_fast_structural_encoder_rejects_same_noncanonical_values():
    for bad in [float("nan"), float("inf"), {1: "bad"}, object()]:
        with pytest.raises(CanonicalEncodingError):
            structural_canonical_bytes(bad)
