"""Deterministic small-part packaging and raw-readback reconstruction.

Drive upload remains an external connector action.  This module makes that
action reliable: it emits immutable parts plus a manifest, lists only missing
or unverified parts on retry, verifies every downloaded part, and reconstructs
the original byte-for-byte.  Default parts are 16 MiB; callers may retry with
8 or 4 MiB when a connector rejects a large transfer.
"""
from __future__ import annotations

import argparse
import lzma
from pathlib import Path

from . import submission as sub
from .change_sessions import _immutable, _sealed

SCHEMA = "IG_DECODER_CHUNKED_SAVE_MANIFEST_V1"
RECEIPT_SCHEMA = "IG_DECODER_CHUNK_READBACK_RECEIPTS_V1"
DEFAULT_PART_BYTES = 16 * 1024 * 1024
ALLOWED_STANDARD_SIZES = (16 * 1024 * 1024, 8 * 1024 * 1024, 4 * 1024 * 1024)


def _error(code, detail=""):
    raise sub.SubmissionError(code, str(detail))


def pack(source, output_dir, part_bytes=DEFAULT_PART_BYTES):
    source = Path(source).resolve(strict=True); output = Path(output_dir).resolve()
    if not source.is_file() or type(part_bytes) is not int or part_bytes <= 0:
        _error("CHUNK_SAVE_ARGUMENTS")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with source.open("rb") as handle:
        index = 0
        while True:
            raw = handle.read(part_bytes)
            if not raw:
                break
            name = f"{source.name}.part-{index:05d}"
            path = output / name
            if path.exists() and path.read_bytes() != raw:
                _error("CHUNK_SAVE_EXISTING_PART_CONFLICT", name)
            if not path.exists():
                path.write_bytes(raw)
            rows.append({"index": index, "name": name, "sha256": sub._sha(raw), "size_bytes": len(raw)})
            index += 1
    if not rows:
        name = f"{source.name}.part-00000"; (output / name).write_bytes(b"")
        rows.append({"index": 0, "name": name, "sha256": sub._sha(b""), "size_bytes": 0})
    manifest = _sealed({"schema_id": SCHEMA, "source_name": source.name,
        "source_sha256": sub._sha(source.read_bytes()), "source_size_bytes": source.stat().st_size,
        "part_bytes": part_bytes, "parts": rows}, "manifest_sha256")
    _immutable(output / "CHUNK_MANIFEST.json", manifest)
    return manifest


def pending(manifest, readback_dir):
    readback = Path(readback_dir)
    missing = []
    for row in manifest["parts"]:
        path = readback / row["name"]
        if not path.is_file() or path.stat().st_size != row["size_bytes"] or sub._sha(path.read_bytes()) != row["sha256"]:
            missing.append(row)
    return missing


def verify_and_reconstruct(manifest, readback_dir, output):
    missing = pending(manifest, readback_dir)
    if missing:
        _error("CHUNK_SAVE_PARTS_PENDING", ",".join(r["name"] for r in missing))
    data = b"".join((Path(readback_dir) / row["name"]).read_bytes() for row in manifest["parts"])
    if len(data) != manifest["source_size_bytes"] or sub._sha(data) != manifest["source_sha256"]:
        _error("CHUNK_SAVE_RECONSTRUCTION_MISMATCH")
    target = Path(output)
    if target.exists() and target.read_bytes() != data:
        _error("CHUNK_SAVE_OUTPUT_CONFLICT")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    receipts = _sealed({"schema_id": RECEIPT_SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
        "reconstructed_sha256": sub._sha(data),
        "parts": [{"name": r["name"], "sha256": r["sha256"], "size_bytes": r["size_bytes"]}
                  for r in manifest["parts"]]})
    return receipts


def verify_xz_transport(transport_manifest, readback_dir, output, *, name_template="part-{index:05d}"):
    """Verify the historical XZ_CHUNKS transport and reconstruct its object."""
    if (transport_manifest.get("schema_id") != "IG_DECODER_TRANSPORT_MANIFEST_V1"
            or transport_manifest.get("encoding") != "XZ_CHUNKS"
            or not isinstance(transport_manifest.get("parts"), list)):
        _error("CHUNK_SAVE_TRANSPORT_MANIFEST")
    compressed = bytearray(); verified = []
    for index, row in enumerate(transport_manifest["parts"]):
        path = Path(readback_dir) / name_template.format(index=index)
        if (not path.is_file() or path.stat().st_size != row.get("size_bytes")
                or sub._sha(path.read_bytes()) != row.get("sha256")):
            _error("CHUNK_SAVE_PARTS_PENDING", path.name)
        raw = path.read_bytes(); compressed.extend(raw)
        verified.append({"name": path.name, "sha256": row["sha256"], "size_bytes": len(raw)})
    try: data = lzma.decompress(bytes(compressed))
    except lzma.LZMAError: _error("CHUNK_SAVE_XZ_RECONSTRUCTION")
    obj = transport_manifest.get("object", {})
    if len(data) != obj.get("size_bytes") or sub._sha(data) != obj.get("sha256"):
        _error("CHUNK_SAVE_RECONSTRUCTION_MISMATCH")
    target = Path(output)
    if target.exists() and target.read_bytes() != data: _error("CHUNK_SAVE_OUTPUT_CONFLICT")
    if not target.exists(): target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    return _sealed({"schema_id": RECEIPT_SCHEMA,
        "transport_schema_id": transport_manifest["schema_id"],
        "reconstructed_sha256": sub._sha(data), "parts": verified})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare or verify deterministic small Drive-upload parts.")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("pack"); p.add_argument("source"); p.add_argument("output_dir"); p.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    p = commands.add_parser("pending"); p.add_argument("manifest"); p.add_argument("readback_dir")
    p = commands.add_parser("reconstruct"); p.add_argument("manifest"); p.add_argument("readback_dir"); p.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "pack": result = pack(args.source, args.output_dir, args.part_bytes)
    else:
        manifest = sub._read(args.manifest)
        result = pending(manifest, args.readback_dir) if args.command == "pending" else verify_and_reconstruct(manifest, args.readback_dir, args.output)
    print(__import__("json").dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
