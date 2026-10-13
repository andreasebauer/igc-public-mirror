"""Retain historical physical source pins without granting current authority."""
from pathlib import Path
import hashlib
import json

OLD_UPLIFT_SHA='3eb7e414c918e76deb176979abc8cfec1e278f90f70273bbeda6da71f68c13e6'

def preserved_p9_source(root, ref, expected=None):
    root=Path(root)
    folder=root/'tests/fixtures/historical_source_bindings'
    pins=json.loads((folder/'P9_PROVENANCE.json').read_text())['pins']
    if ref=='infinity_grid/uplift_structural.py':
        raw=(folder/'uplift_structural.py').read_bytes()
        assert hashlib.sha256(raw).hexdigest()==OLD_UPLIFT_SHA
    elif ref in pins:
        pin=pins[ref]
        raw=(folder/pin['fixture_file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==pin['sha256']
    else:
        raw=(root/ref).read_bytes()
    if expected is not None:assert hashlib.sha256(raw).hexdigest()==expected
    return raw

def preserved_binding_root(source, root, destination):
    root=Path(root).resolve()
    destination=Path(destination)/'historical-source-pins'
    pins={}
    def collect(value):
        if isinstance(value,dict):
            if isinstance(value.get('ref'),str) and isinstance(value.get('sha256'),str):
                pins.setdefault(value['ref'],set()).add(value['sha256'])
            for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(source)
    for ref, expected in pins.items():
        path=(root/ref).resolve()
        if not path.is_relative_to(root) or not path.is_file():continue
        raw=path.read_bytes()
        if ref=='infinity_grid/uplift_structural.py' and expected=={OLD_UPLIFT_SHA}:
            retained=root/'tests/fixtures/historical_source_bindings/uplift_structural.py'
            raw=retained.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==OLD_UPLIFT_SHA
        target=destination/ref
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    return destination
