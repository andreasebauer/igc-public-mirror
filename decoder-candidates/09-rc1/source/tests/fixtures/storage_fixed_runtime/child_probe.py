"""Bounded subprocess fixture for the registered storage runtime test."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

if sys.argv[1] == "probe":
    con = sqlite3.connect(":memory:")
    version, source = con.execute("select sqlite_version(), sqlite_source_id()").fetchone()
    con.close()
    names = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/") and "libsqlite3" in parts[5]:
            names.add(parts[5])
    libs = [hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sorted(names)]
    print(json.dumps({"version": version, "source_id": source, "library_hashes": libs}))
elif sys.argv[1] == "crash":
    con = sqlite3.connect(sys.argv[2], isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("CREATE TABLE t(k INTEGER PRIMARY KEY, v TEXT NOT NULL)")
    con.execute("INSERT INTO t VALUES(1, 'committed')")
    con.execute("BEGIN IMMEDIATE")
    con.execute("INSERT INTO t VALUES(2, 'uncommitted')")
    os._exit(73)
else:
    raise SystemExit("unknown fixture mode")
