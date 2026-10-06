from pathlib import Path
import json,hashlib,zipfile,io
B=Path(__file__).resolve().parent;R=B.parent;inv=json.load(open(R/'o2_recovery0154/ARCHIVE_SEARCH.json'));sha=lambda b:hashlib.sha256(b).hexdigest();refs=json.load(open(B/'RECOVERED_SOURCE_REFS.json'))
want=['v2.4_PHASE0_O4','v2.4_PHASE1_BOUNDED_O4','v2.5_PHASE1_BOUNDED_O5','v2.6_PHASE2_O6','v0.1_O7_LIVE_RECONNAISSANCE_PREREGISTRATION_COMPLETE']
for name in want:
 hits=[r for r in inv['archives'] if name in Path(r['chain'][-1]).name and '__MACOSX' not in '|'.join(r['chain'])];chain=min(hits,key=lambda r:len(r['chain']))['chain'];raw=(R/'project_sources'/chain[0]).read_bytes()
 for member in chain[1:]:raw=zipfile.ZipFile(io.BytesIO(raw)).read(member)
 dest=B/'sources'/Path(chain[-1]).stem;dest.mkdir(exist_ok=True);members=[]
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  for n in z.namelist():
   if n.endswith('/') or n.startswith('__MACOSX/') or Path(n).suffix.lower()=='.zip':continue
   assert not Path(n).is_absolute() and '..' not in Path(n).parts
   b=z.read(n);p=dest/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);members.append({'name':n,'sha256':sha(b),'bytes':len(b)})
 refs.append({'chain':chain,'archive_sha256':sha(raw),'archive_bytes':len(raw),'recovered_members':members,'nested_zip_entries_not_extracted':True})
(B/'RECOVERED_SOURCE_REFS.json').write_text(json.dumps(refs,indent=2));print(json.dumps([{'source':r['chain'][-1],'files':len(r['recovered_members']),'bytes':sum(e['bytes'] for e in r['recovered_members'])} for r in refs]))
