from pathlib import Path
import json,hashlib,zipfile,io
B=Path(__file__).resolve().parent;R=B.parent;inv=json.load(open(R/'readiness0143/ARCHIVE_SOURCE_INVENTORY.json'));sha=lambda b:hashlib.sha256(b).hexdigest();refs=[]
want=['o6_phase0_reference_audit.py','o6_phase1_bounded_generation_audit.py','o5_phase2_graduation_classification_audit.py','o7_live_engine_v0.2.4_CLOSEOUT.py']
for name in want:
 hit=next(r for r in inv['python_sources'] if Path(r['chain'][-1]).name==name);chain=hit['chain'];raw=(R/'project_sources'/chain[0]).read_bytes()
 for member in chain[1:-1]:raw=zipfile.ZipFile(io.BytesIO(raw)).read(member)
 assert sha(zipfile.ZipFile(io.BytesIO(raw)).read(chain[-1]))==hit['sha256'];tag=Path(chain[-2]).stem;dest=B/'sources'/tag;dest.mkdir(exist_ok=True);members=[]
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  for n in z.namelist():
   if n.endswith('/') or n.startswith('__MACOSX/') or Path(n).suffix.lower()=='.zip':continue
   assert not Path(n).is_absolute() and '..' not in Path(n).parts
   b=z.read(n);p=dest/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);members.append({'name':n,'sha256':sha(b),'bytes':len(b)})
 refs.append({'chain':chain[:-1],'archive_sha256':sha(raw),'archive_bytes':len(raw),'recovered_members':members,'nested_zip_entries_not_extracted':True,'selected_code':hit})
(B/'RECOVERED_SOURCE_REFS.json').write_text(json.dumps(refs,indent=2));print(json.dumps([{'source':r['chain'][-1],'recovered_files':len(r['recovered_members']),'unpacked_bytes':sum(e['bytes'] for e in r['recovered_members'])} for r in refs]))
