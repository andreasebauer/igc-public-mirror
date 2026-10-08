from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent;seed=json.loads((B/'SEED_ANCHOR.json').read_text());early=json.loads((B/'EARLY_DEPTH14_ANCHOR.json').read_text());remap=json.loads((B/'IDENTITY_REMAP.json').read_text());assert not seed['historical_first_seed_in_replay_selected_sources'] and seed['all24_selected_construction_identities_changed'] and early['historical_formula_rehashed_root_overlap']==0 and remap['terminal_legacy_identity_remapped_overlap']==0
report=f'''CHECKPOINT0210 — HISTORICAL G1 REPLAY DIVERGENCE DIAGNOSED
2026-10-08

Cause established at level7: historical0.28.8 O7State.construction_digest hashes source ID, ordered parent IDs, exact edges and reservation counts. The native replay's0.8.0.dev151+lib O7State uses IG_O7_SCANNER_CONSTRUCTION_V2 canonical object identity plus reservation projection. Those are different scientific identities, despite byte-identical seed/resource controls.

The original0190binding checked frozen resource equality but pinned current scientific modules without proving their compatibility with the historical terminal reference. The seed selector sorts construction digests, deduplicates by those digests, and starts with the minimum digest. Thus identity policy affects the seed panel itself and propagates into higher-state ordering/tie decisions.

READ-ONLY PROOF
Recovered the saved7–24 replay checkpoint state object through Drive and verified its exact SHA256/length. Selected early artifacts match the prior native audit snapshot hashes.
Historical seed selection MUST include source {seed['historical_first_required_seed']['source_id']} with legacy minimum digest {seed['historical_first_required_seed']['digest']}. That source is absent from the saved replay's24selected seeds. All24saved seed construction identities differ from their historical formula identities. Therefore the historical replay diverges at seed selection7, before depth8 generation.

State builder, recipe enumerator, beam selector, feature vector routine and warm seed-kernel callable are AST-equal between historical and replay source. O7 construction identity and base cache key differ. Candidate descriptor worker has an added execution-only total_caps field. Full source hashes/AST fingerprints/diffs are preserved. This diagnosis does not revert or invalidate the current canonical O7 identity globally.

Rehashing the existing saved witnessed DAG with the historical digest formula yields0of193terminal reference roots. The saved depth14panel similarly gives0of24historical selected roots after rehashing. This excludes a repair that merely relabels the saved root digests. Rehashing is administrative evidence, not a fresh historical state replay or physical-interface equivalence proof. No states were built and no candidates generated in this diagnosis.

Separate operational failure: native compact terminal state JSON was18,853,804bytes, while pretty JSON publication was1,190,033,706bytes and exceeded the frozen1GiB per-file checkpoint bound. The scientific mismatch and publication-size issue require separate corrections. Neither prior capture nor frozen limits were altered.

NEXT — REPAIR QUALIFICATION
REPAIR_PLAN.json specifies a separate historically faithful scientific helper closure hosted by the current native controller. Pin/attest pure historical helpers and their dependencies, retain current canonical O7 behavior elsewhere, preflight the native architecture and raw publication sizes, then register a fresh bounded historical seed/early-depth qualification. Its seed panel and historical level14probe/observation anchors must pass before longer tranches. Any generation requires that fresh saved registration. The wrong-namespace8–100states cannot be used as historical ancestry; preserve them as diagnostic provenance. Full193terminal equality remains mandatory before scoped admission/Q2.

Master151 unchanged; admissions0; generators0; native captures created0; engine revisions0; Q2 unavailable. The prepared restore-only0210plan from the predecessor remains unexecuted and blocked; this diagnostic checkpoint does not activate it. No terminal PASS, G2 graduation or full L0–G8 completion.

HANDOFF
Includes original190recipe, source-provenance snapshots/diffs, frozen seed authority, exact saved7/8/13/14artifacts, readback receipt, identity mapping, seed/depth14proofs and repair qualification plan. Included predecessor0209handoff locates the byte-verified full terminal forensic archive and prior recovery/runtime chain. MANIFEST.json verifies each bundled member. Absolute paths are locators; use hash-identical restored data. No scientific states in a public code mirror; CODE_MIRROR.json binds authored diagnostic scripts.
''';(B/'REPORT.txt').write_text(report)
s=json.loads((W/'CURRENT_STATUS.json').read_text());out={'status':'HISTORICAL_REPLAY_SEED_NAMESPACE_DIVERGENCE_DIAGNOSED','first_confirmed_divergence_level':7,'historical_required_first_seed_absent':True,'all24_seed_identities_changed':True,'historical_identity_rehashed_depth14_overlap':0,'historical_identity_rehashed_terminal_overlap':0,'diagnostic_generators':0,'diagnostic_states_constructed':0,'native_capture_created':False,'new_engine_revision_created':False,'master_slices':151,'new_admissions':0};s.update({'status_as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'latest_checkpoint':210,'current_phase':'G1_HISTORICAL_SOURCE_COMPATIBILITY_REPAIR_REQUIRED','in_flight_active':'NONE','generator_calls':0,'generator_calls_scope':'0210:read-only source/hash/selection-anchor diagnosis; no state construction or candidate generation','candidate_build_calls':0,'historical_reconstruction_acceptance':'FAIL_FROM_SEED_SELECTION7','prior_bounded_replay_claim_scope':'Internally native/cold verified under current identity; not matching historical193terminal population','next_scope':'WP6_G1_HISTORICAL_HELPER_SOURCE_COMPATIBILITY_AND_EARLY_ANCHOR_QUALIFICATION','next_scope_status':'SOURCE_MISMATCH_CAUSE_PROVEN;FRESH_CAPTURE_REQUIRED_BEFORE_ANY_CORRECTED_GENERATION','latest_diagnosis':out,'checkpoint0210':out,'code_mirror':json.loads((B/'CODE_MIRROR.json').read_text())});(B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2));(W/'CURRENT_STATUS.json').write_text(json.dumps(s,indent=2))
files={}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.suffix in ('.json','.txt','.zip','.log','.py') and '__pycache__' not in p.parts and not p.name.startswith('private_') and p.name not in ('DELIVERABLES.json','SAVE_RECEIPT.json'):files['diagnosis0210/'+str(p.relative_to(B))]=p
pred=W/'IG_MASTER151_G1_TERMINAL_0209_MISMATCH_STOP_HANDOFF_2026-10-08.zip';files['predecessor/'+pred.name]=pred
manifest={k:{'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size} for k,p in files.items()};out=W/'IG_MASTER151_G1_DIVERGENCE_DIAGNOSIS_0210_HANDOFF_2026-10-08.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,p in files.items():z.write(p,k)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
r={'path':str(out),'sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'bytes':out.stat().st_size,'checkpoint':210,'generator_calls':0,'new_admissions':0,'first_confirmed_divergence_level':7,'next_scope':s['next_scope']};(B/'DELIVERABLES.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
