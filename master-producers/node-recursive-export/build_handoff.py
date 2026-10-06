from pathlib import Path
import json,hashlib,zipfile,sys
R=Path(__file__).resolve().parent;B=R.parents[1]
def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0,str(B/'engine'))
from infinity_grid.v05_engineering_worker import engineering_source_tree_digest
master='4753cee3dd1adb0b963f10ae11d98fb0c35640b43f3706f1d90b836150458102'
assert sha(B/'unified139/CATALOG_0139.json')==master
engine=engineering_source_tree_digest(B/'engine');assert engine=='bb65b91085c86d6aee1deab5a43300dfec6e8b225370b4df8f11f016e8c11952'
n=load(R/'NATIVE_RESULT.json');q=load(R/'SCIENTIFIC_READBACK_QUALIFICATION.json');cold=load(R/'COLD_REUSE_RESULT.json');pres=load(R/'PRESERVATION_STATUS.json');cap=load(R/'CAPTURE_SAVE_CONFIRMED.json');packing=load(R/'SCIENTIFIC_PACKAGING.json');transport=load(R/'SCIENTIFIC_TRANSPORT_SAVED.json')
assert n['evidence_status']=='VERIFIED' and n['result']['outcome']=='PASS' and q['status']=='PASS_DOWNLOADED_SCIENTIFIC_EXPORT'
assert cap['status']=='SAVED' and not cap['pending_objects'] and pres['pending_bytes']==cold['pending_bytes']==0 and cold['reused'] and not cold['generation_reexecuted']
assert q['all_rows_and_dependency_fields_checked']==199579 and q['generator_calls']==n['result']['generator_calls']==0
mapping=load(R/'CHECKPOINT_READBACK_MAPPING.json');plan=load(R/'CHECKPOINT_PENDING.json');needed={x['sha256'] for x in plan['physical_objects']}
slim={h:mapping[h] for h in sorted(needed)}
for h,x in slim.items():assert sha(Path(x['path']))==h and Path(x['path']).stat().st_size==x['size_bytes']
write(R/'RESTORE_OBJECTS.json',slim)
rootref=n['result']['export_root'];raw=Path(slim[rootref['sha256']]['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==rootref['sha256'];(R/'SCIENTIFIC_ROOT.json').write_bytes(raw)
checks={'status':'PASS_ALL_EXPORT_AND_COMPARISON_GATES_SCOPED_ADMISSION_NEXT','master_catalog_unchanged_sha256':master,'engine_unchanged_digest':engine,'capture_save':'PASS_ZERO_PENDING','registered_export':'PASS_EVIDENCE_VERIFIED','exact_member_comparison':'PASS_182_MEMBERS_286694082_BYTES','raw_drive_transport':'PASS_PART_HASHES_AND_CONCATENATED_ARCHIVE_HASH','checkpoint_preservation':'PASS_ZERO_PENDING_BYTES','cold_restore_completion_reuse':'PASS_NO_REGENERATION','export_reader':'PASS_ALL_199579_ROWS_AND_FIELDS','depth2_dry_comparison_sha256':q['full_depth2_stream_sha256'],'primitive_authority':'All13 complete event/witness records equal previous admitted-J3-verified packet at pinned SHA; catalog/foundation references retained','scoped_master_admission':'NOT_ISSUED','catalog_reader_integration':'NEXT_REQUIRED','generator_calls':0,'historical_selection_weights_qualified':False}
write(R/'EXPORT_ACCEPTANCE_CHECKS.json',checks)
cursor={'date':'2026-10-06','completed':'Registered recursive scientific export, saved transport and checkpoints, cold reuse and full downloaded-byte reader comparison','master':'MASTER_DATA_V1_0139','master_catalog_sha256':master,'scientific_export_root_sha256':rootref['sha256'],'scientific_archive_sha256':packing['archive_sha256'],'next':'Prepare and register scoped admission for NODE_NATIVE_ROOTED_DEPTH1 and NODE_NATIVE_ROOTED_DEPTH2, then a catalog extension and unified reader routing','next_steps':['Pin this exact scientific root and byte-verified archive in a read-only admission request under the unchanged common runtime','Bind accepted J3/foundation dependencies, frozen identity recipes and all comparison/closure evidence; refuse any mismatch','Issue a distinct scoped admission only after registered admission evidence passes; retain historical-weight and census exclusions','Extend a new catalog from immutable0139 and add unified construction, formation, projection-preimage and lineage routes for the admitted slice','Verify existing0139 families unchanged and all199579 exported rows reachable through the new unified reader; save checkpoint, cold reuse and slim handoff'],'do_not_regenerate':['Depth1 constructions','Six completed recursive depth2 tranches','Completed export V2'],'other_open':['Historical D/B/S/O/L selection weights','Original historical L13 recursive lineage','CORE_ROLE_ENVELOPE source gap','Mature NODE_IN/SCOUT census','O/G population through G8'],'superseded_capture':'V1 unsaved/unrun, transport-size issue only; record retained; never acknowledge it as saved or executed'}
write(R/'CONTINUATION_CURSOR.json',cursor)
basename='IG_NODE_RECURSIVE_SCIENTIFIC_EXPORT_VERIFIED_2026-10-06'
text=f'''INFINITY GRID — RECURSIVE SCIENTIFIC EXPORT VERIFIED — 2026-10-06

Completed: registered export-only job MASTER.EXPORT.NODE.NATIVE.DEPTH1_DEPTH2.V2.
Captured source, inputs and environment were saved to Drive and raw-readback verified before execution. Completion evidence VERIFIED. Generator calls: 0.

Scientific scope: the frozen13-event native J3 alphabet; exhaustive lawful rooted attachments at depths1 and2. Depth1:1,391 constructions /108 projected boundaries. Depth2:198,188 constructions /802 projected boundaries. Total:199,579 full native rows, retaining construction IDs, formation IDs, bridges, ports, origins, parent/primitive references, projection maps/preimages and microscopic realization counts. This is a scoped recursive slice.

Registered scientific root SHA256: {rootref['sha256']}
Qualified view SHA256: 3ad27f29023db1b39442d07383d08df0d2829819a014203c6cd10a79112bfc9b
Exact scientific archive SHA256: {packing['archive_sha256']}
Archive bytes: {packing['archive_size_bytes']}; scientific payload:286,694,082 bytes in182 content-addressed members, plus the scientific root. Members are179 exact original data shards, the complete13-event primitive packet, exact1,391-row parent packet, and qualified view. Operational/proof metadata is outside this scientific archive.
Depth2 ordered row stream SHA256: {q['full_depth2_stream_sha256']}

Transport: raw ordered parts, each <=24MiB. Download the two observed Drive files below as raw bytes, concatenate in index order, then verify the whole archive SHA above before opening. SCIENTIFIC_TRANSPORT_SAVED.json records exact part sizes/hashes and observed IDs. Individual parts are not standalone ZIP archives.
Part00: {transport['parts'][0]['url']}
Part01: {transport['parts'][1]['url']}

Comparison: all182 scientific members equal captured saved bytes. Fresh scientific reader used ONLY the reassembled raw Drive scientific export, checked every199,579 row and dependency field, exhaustive lawful endpoint coverage, IDs, references, constructors, projections, microscopic counts, all source streams and the independent full depth2 dry stream. Lookup, formation lookup, lineage and copy isolation passed. Seven reader rejection checks passed; the earlier qualified view's16 negative checks remain bound in the included prior report. All13 primitive event records and realization witnesses match the previously verified admitted-J3 packet. Catalog and foundation references are pinned; this export does not re-admit those existing dependencies.

Preservation: capture SAVED. Registered checkpoint PRESERVED with0 pending bytes. Slim checkpoint raw Drive readback verified. Cold restore/reuse PASSED with identical completion/result hashes, VERIFIED evidence,0 pending bytes and no generation reexecution.
Completion SHA256: {n['completion_sha256']}
Result SHA256: {n['result_sha256']}
Checkpoint SHA256: {load(R/'CHECKPOINT_EXPORT.json')['sha256']}
Checkpoint Drive: {load(R/'CHECKPOINT_ARCHIVE_SAVED.json')['url']}

Current scientific master remains MASTER_DATA_V1_0139, unchanged SHA256: {master}
Engine unchanged digest: {engine}
NO new scoped admission or catalog extension has been issued by this export. Its captured contract explicitly reports scientific_master_admission=false. Export completion is distinct from admission.

Next: follow CONTINUATION_CURSOR.json: registered read-only scoped admission for the two native rooted families, then new catalog and unified reader routes; compare prior0139 families and all exported rows; preserve/cold-reuse and save the next slim handoff. Do not regenerate completed rows. Current export gates PASS; admission and catalog routing are the remaining separate steps.

Limits: historical D/B/S/O/L weights remain unqualified. Native microscopic realization counts cannot replace them. Historical L13 reproduction, mature NODE_IN/SCOUT census, universal NODE census, O/G population and full L0-to-G8 completion are not claimed. Original historical recursive lineage and CORE_ROLE_ENVELOPE source gaps remain open.

Transport adaptations: relative project import and project initialization path were corrected before successful capture. A182-input proposal exceeded the128-input limit before capture. First successful single-bundle V1 capture was unsaved and unrun because its48,484,626-byte input exceeded raw single-file download capability. Its complete capture metadata remains in SUPERSEDED_UNRUN_CAPTURE.json and the original capture remains intact locally. V2 recaptured the exact frozen ZIP bytes in two parts and alone passed save/execution gates. No science ran in V1. Runtime publishes one selected scientific root; transport packages the182 exact selected existing members, rather than republishing179 shards through its64-shard publication limit. No engine changes.

Continuation recovery: use FINAL_CHECKPOINT_SLIM.zip and all dependencies in RESTORE_OBJECTS.json, downloading each observed Drive ID and checking its SHA/size, then restore through the pinned engine. cold_ops.py documents restore/reuse. Source code is mirrored with exact committed-source readback verification in CODE_MIRROR.json and SCIENTIFIC_READER_CODE_MIRROR.json. scientific_reader.py reads the pinned scientific archive; qualify_scientific_export.py repeats downloaded-byte checks using updated local part readback paths. Restore path metadata is a previous-local-path hint, not authority; hashes and observed Drive IDs are authoritative.

This slim handoff includes scientific root, capture/registration results, checks, cursor, source, dependency maps and slim checkpoint. It excludes bulk science; retrieve it using the multipart manifest. Prior reader/profile and full six-tranche campaign references remain included.
'''
(B/(basename+'.txt')).write_text(text)
files={}
for p in R.glob('*.json'):
    if p.name not in {'CHECKPOINT_READBACK_MAPPING.json','CHECKPOINT_UPLOAD_QUEUE.json','CAPTURE_UPLOAD_QUEUE.json','HANDOFF_CONTENTS.json','HANDOFF_PACKAGE.json','HANDOFF_DRIVE_SAVED.json','RECEIPT_DRIVE_SAVED.json'}:files[str(p.relative_to(B))]=p
for p in [*R.glob('*.py'),*R.joinpath('project').glob('*.py'),R/'FINAL_CHECKPOINT_SLIM.zip',B/(basename+'.txt'),B/'unified139/CATALOG_0139.json']:
    files[str(p.relative_to(B))]=p
for n in ['ROOT.json','READER_QUALIFICATION.json','ADMITTED_PRIMITIVE_CLOSURE.json','SCOPED_ADMISSION_CONTRACT.json','EXPORT_PROFILE.json','EXTERNAL_OBJECTS.json','CODE_MIRROR.json','HANDOFF_DRIVE_SAVED.json']:
    p=R.parent/n;files[str(p.relative_to(B))]=p
for n in ['CAMPAIGN_SAVED_UNION_VERIFICATION.json','COMMON_RUNTIME_PROGRESS_AND_NEXT.json','HANDOFF_DRIVE_SAVED.json']:
    p=B/'recursive_native'/n
    if p.exists():files[str(p.relative_to(B))]=p
manifest={name:{'sha256':sha(p),'size_bytes':p.stat().st_size} for name,p in sorted(files.items())}
write(R/'HANDOFF_CONTENTS.json',manifest);files[str((R/'HANDOFF_CONTENTS.json').relative_to(B))]=R/'HANDOFF_CONTENTS.json'
archive=B/(basename+'.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for name,p in sorted(files.items()):z.write(p,name)
with zipfile.ZipFile(archive) as z:
    assert set(z.namelist())==set(files)
    for name,ref in manifest.items():raw=z.read(name);assert len(raw)==ref['size_bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
write(R/'HANDOFF_PACKAGE.json',{'path':str(archive),'sha256':sha(archive),'size_bytes':archive.stat().st_size,'members':len(files),'report_path':str(B/(basename+'.txt'))})
print(json.dumps(load(R/'HANDOFF_PACKAGE.json')))
