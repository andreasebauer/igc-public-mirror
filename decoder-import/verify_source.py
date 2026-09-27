"""Read-only byte integrity check for imported source; no scientific execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def verify():
    manifest = json.loads((ROOT / "decoder-import/source-manifest.json").read_text())
    source = ROOT / "decoder"
    rows = []
    skip = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".engineering_tmp"}
    for p in sorted(source.rglob("*")):
        rel = p.relative_to(source)
        if any(x in skip for x in rel.parts) or p.suffix in {".pyc", ".pyo"}:
            continue
        if p.is_symlink():
            raise ValueError(f"Unexpected symlink: {rel}")
        if p.is_file():
            rows.append([rel.as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()])
    if rows != manifest["files"]:
        raise ValueError("Source file inventory or bytes differ from the imported archive")
    source_id = digest({"schema_id": "IG_DECODER_ENGINEERING_SOURCE_TREE_V1", "files": rows})
    package_rows = [[n[len("infinity_grid/"):], h] for n, h in rows if n.startswith("infinity_grid/")]
    package_id = digest({"schema_id": "IG_DECODER_EXECUTABLE_SOURCE_TREE_V1", "files": package_rows})
    if (source_id, package_id) != (manifest["source_sha256"], manifest["package_sha256"]):
        raise ValueError("Source/package identity mismatch")
    print(json.dumps({"status": "BYTE_EXACT_SOURCE_PASS", "files": len(rows),
                      "source_sha256": source_id, "package_sha256": package_id}, indent=2))


if __name__ == "__main__":
    verify()
