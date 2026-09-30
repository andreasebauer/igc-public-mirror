"""Offline, no-system-replacement SQLite build recipe. NOT YET BUILD-QUALIFIED.

Usage: python tools/provision_sqlite_3510300.py AMALGAMATION_ZIP NEW_DIRECTORY
The original upstream sqlite3.c bytes must match the official release digest.
This is environment provisioning only, not a scientific test runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

SOURCE_SHA3 = '32d5424f97e0a7fc5ed2f6335afbb58be4e0298bd7117a34e39d345ff13d859e'
SOURCE_ID = '2026-03-13 10:38:09 737ae4a34738ffa0c3ff7f9bb18df914dd1cad163f28fd6b6e114a344fe6d618'


def provision(archive: Path, destination: Path) -> dict:
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_size>32*1024*1024:
        raise ValueError('SOURCE_ARCHIVE_MISSING_OR_BUDGET')
    if destination.exists() or destination.is_symlink(): raise ValueError('NEW_DESTINATION_REQUIRED')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)): raise ValueError('DUPLICATE_ARCHIVE_MEMBERS')
        names=[n for n in names if Path(n).name=='sqlite3.c']
        if len(names)!=1: raise ValueError('EXACTLY_ONE_AMALGAMATION_REQUIRED')
        member=z.getinfo(names[0])
        if member.file_size>16*1024*1024: raise ValueError('SOURCE_SIZE_BUDGET')
        raw=z.read(member)
    if hashlib.sha3_256(raw).hexdigest()!=SOURCE_SHA3:
        raise ValueError('UPSTREAM_SOURCE_SHA3_MISMATCH')
    compiler=shutil.which('gcc')
    if compiler is None: raise RuntimeError('GCC_REQUIRED')
    destination.mkdir(parents=True)
    source=destination/'sqlite3.c';source.write_bytes(raw)
    libdir=destination/'lib';libdir.mkdir()
    library=libdir/'libsqlite3.so.0'
    command=[compiler,'-O2','-fPIC','-shared','-DSQLITE_THREADSAFE=1','-DSQLITE_USE_URI=1',
        '-DSQLITE_ENABLE_COLUMN_METADATA','-DSQLITE_ENABLE_FTS5','-DSQLITE_ENABLE_RTREE',
        '-DSQLITE_ENABLE_DBSTAT_VTAB','-DSQLITE_ENABLE_MATH_FUNCTIONS',
        '-Wl,-soname,libsqlite3.so.0',str(source),'-o',str(library),'-lpthread','-ldl','-lm']
    with (destination/'BUILD_LOG.txt').open('w') as out:
        out.write(json.dumps(command)+'\n');out.flush()
        subprocess.run(command,check=True,stdout=out,stderr=subprocess.STDOUT)
    # New processes only. No ctypes hot-swap and no system library overwrite.
    wrapper=destination/'python-sqlite-fixed'
    wrapper.write_text('#!/bin/sh\nset -eu\nHERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        +'export LD_LIBRARY_PATH="$HERE/lib"\nexec '+__import__('shlex').quote(str(Path(sys.executable).resolve()))+' "$@"\n')
    wrapper.chmod(0o755)
    probe='import sqlite3,json; c=sqlite3.connect(":memory:");print(json.dumps(c.execute("select sqlite_version(),sqlite_source_id()").fetchone()))'
    observed=json.loads(subprocess.check_output([str(wrapper.resolve()),'-c',probe],text=True))
    if observed!=['3.51.3',SOURCE_ID]: raise RuntimeError('FIXED_LIBRARY_NOT_LOADED')
    report={'schema_id':'IG_SQLITE_OFFLINE_BUILD_RECEIPT_V1',
        'upstream_release':'https://sqlite.org/releaselog/3_51_3.html',
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'sqlite3_c_sha3_256':SOURCE_SHA3,'sqlite_source_id':SOURCE_ID,
        'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        'compiler_sha256':hashlib.sha256(Path(compiler).resolve().read_bytes()).hexdigest(),
        'command':command,'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
        'python_executable_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        'native_qualification':'REQUIRED_SEPARATELY','scientific_acceptance':'NOT_GRANTED'}
    (destination/'BUILD_RECEIPT.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('destination',type=Path)
    args=p.parse_args();print(json.dumps(provision(args.archive,args.destination),indent=2))
