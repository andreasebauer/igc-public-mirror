"""Infinity Grid Decoder: isolated 0.8lib development branch."""
from ._version import __version__
from .platform_support import require_supported_platform

# Fail before imports of Linux-only runtime modules produce opaque errors.
require_supported_platform()

def build_meta():
    import json
    from importlib.resources import files
    meta = json.loads(files("infinity_grid").joinpath("_build_meta.json").read_text(encoding="utf-8"))
    return dict(meta, version=__version__)
