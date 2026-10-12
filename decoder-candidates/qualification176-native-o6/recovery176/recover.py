from pathlib import Path
import hashlib, io, json, sys, zipfile

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'successor175/source'
sys.path.insert(0, str(SOURCE))
from infinity_grid.frontier import _load_engine
from infinity_grid.canon import canonical_sha256

outer = SOURCE / 'infinity_grid/resources/decoder/O7_MATERIAL_ROOT_cb6f48641eb9.zip'
raw = outer.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'cb6f48641eb99d374d19d5dc8d5bfada13e12cf1ecb20ed0e66958629ec08f1b'
z = zipfile.ZipFile(io.BytesIO(raw))
member, = [n for n in z.namelist() if n.endswith('/replay/Infinity_Grid_O7_COMPACT_REPLAY_ROOT_v1_2026-08-29.zip')]
inner = z.read(member)
q = zipfile.ZipFile(io.BytesIO(inner))
assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in q.namelist())
dest = Path(__file__).resolve().parent / 'retained_native'
assert not dest.exists()
q.extractall(dest)
root = dest / 'Infinity_Grid_O7_COMPACT_REPLAY_ROOT_v1_2026-08-29'
engine = _load_engine(root)
records = engine.load_parent_records()
parents = {p.pid:p for p in records.values()}
ids = ['O6P:66a20d2196504084a036c4f7399cf1655359b13af5337287f2081a409bc3bf8d:3,3:d665ae55aa52fb6c',
       'O6P:e3759f2aafd495565e5a69cfb59a97370e357d82275da802ce9f9ab2beb86c60:3,2:22bfb0feba1417ad']
assert all(i in parents for i in ids)
proof = {'status':'PASS_RETAINED_NATIVE_PARENTS_RECOVERED',
 'outer_sha256':hashlib.sha256(raw).hexdigest(), 'outer_bytes':len(raw),
 'inner_sha256':hashlib.sha256(inner).hexdigest(), 'inner_bytes':len(inner),
 'inner_member':member, 'parent_count':len(parents),
 'parents':[{'pid':i,'h6_sha256':canonical_sha256(parents[i].h6),'exact_color':parents[i].exact_color} for i in ids],
 'members':{n:{'bytes':len(q.read(n)),'sha256':hashlib.sha256(q.read(n)).hexdigest()} for n in q.namelist()},
 'source_sha256':'fa582a47df9aea0ee40645cfffb44ffc41b53aa4e56a7bec5ef3f6a7d48ecc9b',
 'observer_execution':False, 'scientific_promotion':False}
(BASE/'recovery176/PROOF.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:v for k,v in proof.items() if k!='members'},indent=2))
