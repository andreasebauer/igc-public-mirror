"""Reproduce the single non-scientific runtime policy change on the pinned source."""
from pathlib import Path
import sys
p=Path(sys.argv[1])/"infinity_grid/v05_stage_runtime.py"
s=p.read_text()
old="or type(max_result_bytes) is not int or not 1<=max_result_bytes<=8*1024*1024)"
new="or type(max_result_bytes) is not int or max_result_bytes < 1)"
assert s.count(old)==1
p.write_text(s.replace(old,new))
