from __future__ import annotations
import json
from importlib.resources import files

def load_json(rel: str):
    return json.loads(files("infinity_grid").joinpath("resources").joinpath(rel).read_text(encoding="utf-8"))
def schema(name: str): return load_json(f"schemas/{name}")
def protocol_descriptor_file(name: str): return load_json(f"protocols/{name}")
def build_meta():
    from . import build_meta as active_build_meta
    return active_build_meta()
