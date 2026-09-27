"""Stage existing bytes for standard S3 clients; no execution or remote mutation."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[1]


def prepare(output):
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Stage outside the Git checkout")
    # Refuse accidental reuse of an unrelated or incomplete staging area.
    output.mkdir(parents=True, exist_ok=False)
    runpy.run_path(str(ROOT / "decoder-import/verify_source.py"))["verify"]()
    entries = []

    def put(raw, role, source_path):
        sha = hashlib.sha256(raw).hexdigest()
        key = f"objects/sha256/{sha[:2]}/{sha}"
        path = output / key
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError("Content-address collision or corrupt staging file")
        else:
            with path.open("xb") as stream:
                stream.write(raw)
        if len(path.read_bytes()) != len(raw) or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("Local staging readback mismatch")
        row = {"key": key, "sha256": sha, "bytes": len(raw), "role": role,
               "source_path": source_path}
        entries.append(row)
        return row

    for directory, role in [("decoder/infinity_grid/resources", "decoder-resource"),
                            ("decoder/tests/fixtures", "decoder-test-fixture")]:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_symlink():
                raise ValueError("Symlink in source resources")
            if path.is_file():
                put(path.read_bytes(), role, path.relative_to(ROOT).as_posix())
    paths = [ROOT / f"microscope/payload/part-{i:02d}.b64" for i in range(20)]
    html = gzip.decompress(base64.b64decode("".join(p.read_text() for p in paths))).decode("utf-8")
    matches = re.findall(r'<script type="application/json" id="initial-data">(.*?)</script>', html, re.S)
    if len(matches) != 1:
        raise ValueError("Expected exactly one existing Microscope data pack")
    raw = matches[0].encode("utf-8")
    pack = json.loads(raw)
    if pack.get("schema") != "ig.reverse-microscope/1.1":
        raise ValueError("Unexpected existing Microscope schema")
    view = put(raw, "microscope-view", "microscope/payload/part-00.b64..part-19.b64#initial-data")
    view.update({"schema": pack["schema"], "dataset_id": pack["dataset"]["id"]})
    manifest = {"schema": "ig.shared-data.catalogue/1", "status": "LOCAL_BYTE_COPY_ONLY",
                "source_sha256": json.loads((ROOT / "decoder-import/source-manifest.json").read_text())["source_sha256"],
                "microscope": view, "objects": entries}
    (output / "catalogue.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    unique = {r["sha256"]: r["bytes"] for r in entries}
    summary = {"references": len(entries), "unique_objects": len(unique),
               "unique_bytes": sum(unique.values()), "microscope_sha256": view["sha256"],
               "microscope_bytes": view["bytes"]}
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args().output)
