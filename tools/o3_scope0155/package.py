from pathlib import Path
import json,hashlib,zipfile,shutil,datetime
B=Path('o3_scope0155');v=json.load(open(B/'READINESS_VALIDATION.json'));assert v['status']=='PASS_WITH_EXPLICIT_EXTERNAL_QBANK_BOUNDARY'
sha=lambda b:hashlib.sha256(b).hexdigest()
contract={'schema':'IG_O3_SAVED_TYPED_INCIDENCE_SCOPE_V1','source_scope':'Saved r1..r64 selected O3 panels; r0 is an external seed panel.','entity_boundary':'Each entity binds a full frozen anonymous O2 Q-state (local p,u2 multiset). Microscopic O2 interiors and source-bank selection are external unresolved provenance.','root_preservation':['All selected row metadata and lane memberships','Typed six-endpoint edge multiset, including parallel multiplicity','Original entity sequence and Q-hash references','Original parent labels with adjacent-panel closure','Frozen bank/spec/primitives byte hashes','Connected component restrictions plus original full forest root'],'identity':'Original producer exact_key plus full literal payload SHA256. Source-bound symmetric fallback labels are not independently canonicalized.','registration':'Readiness only. No export, registered job, admission, or master mutation yet.','next':'Export saved typed incidence and nested Q states with a cold reusable reader under this explicit boundary; retain missing microscopic O2 recovery as a separate blocked lane.','generator_calls':0,'missing_exact_O2_panels':76,'master_catalog_sha256':'f39144bb1ee906f2a7ff1ae9d866aa99f20b9fb118bcf25d9006447e583bcead'}
(B/'SCOPE_CONTRACT.json').write_text(json.dumps(contract,indent=2))
report=f'''INFINITY GRID SAVED O3 SCOPE READINESS — 2026-10-06

PASS within explicit external O2 Q-bank boundary. All65 saved panels r0..r64 agree byte-for-byte across4 original audit bundles. Verified{v['summary']['states']} carrier occurrences,{v['summary']['edges']} typed O3 edges,{v['parent_links']} adjacent-panel parent references, and{v['component_rows']} connected component occurrences. Nested256 frozen Q states, compatibility, endpoint capacity, component topology and rho identities checked; zero failures. Original producer keys recomputed where its finite canonical branch applies; fallback labels remain source-bound. No generation or new graduation claim.

The76 missing microscopic O2 panels remain unresolved. Q states preserve anonymous local(p,u2) resources; they do not reconstruct microscopic O2 incidence. This readiness step authorizes only the separately stated saved O3 scope. Master0143/143 slices is unchanged.

NEXT:Export saved O3 typed carriers/components and nested Q states with a reusable reader under SCOPE_CONTRACT.json. Register and verify before any scoped admission. Do not regenerate completed constructions.
'''
(B/'REPORT.txt').write_text(report)
D=Path('handoff0155_o3_scope');D.mkdir(exist_ok=True);dest=D/B;dest.mkdir(exist_ok=True)
for n in ['SOURCE_ARCHIVE_REFS.json','READINESS_VALIDATION.json','PAYLOAD_SHA256.json','COMPONENT_ROWS.json','SCOPE_CONTRACT.json','REPORT.txt','validate.py','verify_saved.py','package.py','CODE_MIRROR.json']:
 p=B/n
 if p.exists():shutil.copy2(p,dest/n)
primary=next((B/'sources').glob('*V1_3*/*'));shutil.copytree(primary,dest/'sources'/primary.parent.name/primary.name,dirs_exist_ok=True)
orig=next((B/'sources').glob('*V1_1*/*/parent_v1_1_snapshot/code/scout3_o3_roadmap_v1_0.py'));target=dest/orig.relative_to(B);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(orig,target)
# The other original bundles are external audit witnesses, not duplicate payloads in this compact delta.
refs={'prior_readiness_bundle':{'name':'IG_O2_O3_READINESS_RECOVERY_HANDOFF_2026-10-06.zip','sha256':'23483fe980ba90fd908397dc3eee0e50ba025a6404d476c2cbaee254f5c6476f','drive_file_id':'1qYKA1xE-GNR2iIT9zeI4ze2uMoB2uMn9'},'master_restore':{'name':'IG_MASTER_DATA_V1_0143_O2_HANDOFF_2026-10-06.zip','sha256':'625133b8b4cadbcf2df083d4930fe4b7762dba3dedb4644e925c0096316effcb','drive_file_id':'15pHttKvMJmR7NjqRK1Ji-zEVwMBG4FtY'},'original_cross_audit_sources':json.load(open(B/'SOURCE_ARCHIVE_REFS.json')),'fresh_unpack_check':'Verify every MANIFEST.json byte hash; primary frozen integrity sweep can run without external sources. Cross-bundle comparison requires original08-O3.zip; bank match requires prior readiness bundle.'}
(dest/'RECOVERY_DEPENDENCIES.json').write_text(json.dumps(refs,indent=2));(D/'READ_FIRST.txt').write_text(report)
s=json.load(open('CURRENT_STATUS.json'));s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope='SAVED_O3_TYPED_INCIDENCE_EXPORT_WITH_EXTERNAL_QBANK',next_scope_status='READY_WITH_EXPLICIT_EXTERNAL_QBANK_BOUNDARY',o3_scope_readiness='o3_scope0155/READINESS_VALIDATION.json',o3_missing_panel_count=76);Path('CURRENT_STATUS.json').write_text(json.dumps(s,indent=2));(D/'STATUS.json').write_text(json.dumps(s,indent=2));(D/'CONTINUATION_CURSOR.json').write_text(json.dumps({'completed_action':'0155_SAVED_O3_SCOPE_READINESS','next_scope':s['next_scope'],'master_release':'MASTER_DATA_V1_0143','scientific_slices':143,'pending_bytes':0,'generator_calls':0,'new_admissions':0},indent=2));(D/'MANIFEST.json').unlink(missing_ok=True);m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file()};(D/'MANIFEST.json').write_text(json.dumps(m,indent=2))
p=Path('IG_O3_SAVED_SCOPE_READINESS_HANDOFF_2026-10-06.zip')
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for n in [*m,'MANIFEST.json']:z.write(D/n,n)
with zipfile.ZipFile(p) as z:
 for n,x in m.items():assert sha(z.read(n))==x['sha256']
Path('CURRENT_START.txt').write_text(str(D/'READ_FIRST.txt')+'\n');Path('IG_O3_SAVED_SCOPE_READINESS_START_2026-10-06.txt').write_text(report+'\nBundle:'+p.name+'\nSHA256:'+sha(p.read_bytes())+'\n');meta={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'manifest_files':len(m)};(B/'DELIVERY_BINDINGS.json').write_text(json.dumps(meta));print(meta)
