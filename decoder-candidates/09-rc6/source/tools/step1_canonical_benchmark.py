#!/usr/bin/env python3
"""Bounded same-input labeling microbenchmark; not a master-runtime forecast."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infinity_grid.o7_edge_canonicalization import canonicalize_edges
from infinity_grid.o7_legacy_library_comparison import ARCHIVE


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=a.output/'native_inputs';os.environ['OSCOUT_DATA_ROOT']=str(r.resolve());spec=importlib.util.spec_from_file_location('_step1_benchmark',r/'02_CODE/o7_live_engine.py');engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine);engine.O6=engine.import_o6();parents=engine.load_parent_records()
    with zipfile.ZipFile(ARCHIVE) as z:rows=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/graduation_compact/07_INPUT_SNAPSHOTS/O7_IMMUTABLE_SURVIVORS.json'))))['records']
    old=[];new=[]
    for row in rows[::6]:
        ctx=engine._profile_row_context(row,parents);edges=tuple(tuple(e) for e in row['edges'])
        start=time.perf_counter();engine.canonicalize_edges(ctx,edges);old.append(time.perf_counter()-start)
        start=time.perf_counter();canonicalize_edges(ctx,edges);new.append(time.perf_counter()-start)
    def stat(values):return {'calls':len(values),'total_seconds':sum(values),'median_seconds':statistics.median(values),'max_seconds':max(values)}
    result={'schema_id':'IG_STEP1_CANONICAL_MICROBENCHMARK_V1','legacy':stat(old),'corrected':stat(new),'sample':'every sixth row of retained 205-row O7 panel','speed_ratio_total':sum(new)/sum(old),'warning':'Different algorithms and output contracts; legacy labeling is incorrect on demonstrated states. Canonicalizer-only timing, not full-master forecast; no save or profile calculation cost.'}
    (a.output/'CANONICAL_MICROBENCHMARK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
