from pathlib import Path
import json,zipfile,hashlib
h=Path(__file__).resolve().parent;r=h.parent;out=r/'deliverables/native169';out.mkdir(parents=True,exist_ok=True)
p=json.loads((h/'results/CAPTURE_PENDING.json').read_text());w=Path(p['workspace']);native=json.loads((h/'results/NATIVE_RESULT.json').read_text());n=native['result']['result']['nodes'];assert len(n)==64 and all(x['status']=='PASS' for x in n)
cold=json.loads((h/'results/FINAL_COLD.json').read_text());assert cold['status']=='PASS'
save=json.loads((h/'results/FINAL_PRESERVED.json').read_text());assert save['status']=='PRESERVED' and not save['pending_objects']
summary={'status':'PASS_NATIVE_CANONICAL_EXTERNALIZED_CHECKPOINT_AND_SAME_HOST_COLD','tests':64,'capture_id':p['capture']['capture_id'],'source_sha256':native['source_sha256'],'cold':cold,'preservation':save,'full_p0':False,'independent_host':False,'live_controller_changed':False}
(h/'results/SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
report=out/'IG_08LIB_NATIVE_EXTERNALIZATION169_2026-10-12.txt'
report.write_text('NATIVE CANONICAL EXTERNALIZATION QUALIFICATION\n\nSame source digest d2c3c694ed7a5184d9c9b40e6ccc47dd57cbe02d57acfdd0ba35f0529cd4faa0. Restored pinned runtime archive SHA256 25f76fa883c03b2ccc8a902d0aca799abd167279dbc6108ee2dad8095ac90222; all 4721 manifest files exact. Python 3.13.5 and SQLite 3.51.3.\n\n64/64 registered native validation checks passed under canonical contract with project_evidence_externalization=1 verified before capture. Native post-completion checkpoint succeeded; real Drive raw readbacks drained all pending obligations. PRESERVED, zero pending bytes. Repeated full exports identical; cold restored '+str(cold['state_files_exact'])+' state files and modes exactly. Full portable project snapshot matched exactly. Same-host directory qualification only.\n\nRetained earlier capture168 failure: its descriptive legacy contract dropped preservation options during normalization, so it used the inline path and hit CHECKPOINT_TREE_RAW_LIMIT:33762780 after 64 passing tests. Corrected registration is a new capture; no old identity or result was overwritten. The earlier closure167 prospective contract is superseded. No core source edits or raised size limits.\n\nFull P0, broader captured regression and independent-host qualification remain OPEN. Live controller unchanged; no master admission or science regeneration. Next: expand canonical captured regression in bounded registered waves while preserving source and capture evidence, then independent-host qualification.\n')
files=[]
for directory in [h,r/'capture168']:
 for f in directory.rglob('*'):
  if f.is_file() and 'PRIVATE' not in f.name and '__pycache__' not in f.parts:files.append((f,f.relative_to(r).as_posix()))
for f in (w/'runtime').rglob('*'):
 if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith(('-wal','-shm')):files.append((f,'native_workspace/'+f.relative_to(w).as_posix()))
for folder in ['durability']:
 for f in (w/folder).rglob('*.json'):files.append((f,'native_workspace/'+f.relative_to(w).as_posix()))
for name in ['CAPTURE.json','WORKSPACE.json','PROJECT_BINDING.json']:files.append((w/name,'native_workspace/'+name))
files.append((r/'capture168/results/RUNTIME_VERIFIED.json','runtime/RUNTIME_VERIFIED.json'))
files.append((r/'runtime168/pointers/restored/matched_protocol/SOURCE_POINTERS.json','runtime/PREDECESSOR_SOURCE_RUNTIME_POINTERS.json'))
archive=out/'IG_08LIB_NATIVE_EXTERNALIZATION169_HANDOFF_2026-10-12.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 z.write(report,'READ_FIRST.txt');manifest=[];seen=set()
 for f,name in files:
  if name in seen:continue
  seen.add(name);b=f.read_bytes();z.writestr(name,b);manifest.append({'path':name,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
print(json.dumps({'report':str(report),'handoff':str(archive),'bytes':archive.stat().st_size,'cold_files':cold['state_files_exact']}))
