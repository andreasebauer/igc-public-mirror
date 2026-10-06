from pathlib import Path
import hashlib,json,shutil,subprocess,sys
R=Path.cwd();B=R/'g_readiness0175';I=B/'inputs';I.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
paths={
 'CATALOG_0149.json':R/'o7_additional_integrate0174/CATALOG_0149.json',
 'SCOPED_ADMISSION.json':R/'o7_additional_integrate0174/SCOPED_ADMISSION.json',
 'RELEASE_VERIFICATION.json':R/'o7_additional_integrate0174/RELEASE_VERIFICATION.json',
 'additional_O7.zip':R/'o7_additional_export0173/SCIENTIFIC_EXPORT.zip',
 'base_O7.zip':R/'o7_export0169/SCIENTIFIC_EXPORT.zip',
 'O7_seed.zip':R/'engine/infinity_grid/resources/decoder/O7_MATERIAL_ROOT_cb6f48641eb9.zip',
 'ORIGINAL_POPULATION_PLAN.txt':R/'scope_reconcile0171/ORIGINAL_POPULATION_PLAN.txt',
 'PRIOR_FAMILY_REGISTER.json':R/'scope_reconcile0171/FAMILY_REGISTER.json'}
for f in ['materialized_discovery.py','uplift_campaign.py','g2_relation.py','regime_scanner.py','uplift_r0.py']:
 paths[f]=R/'engine/infinity_grid'/f
for f in ['O_REGIME_MATERIALIZED_DISCOVERY_SPEC_v1.json','O_REGIME_ADAPTIVE_SCANNER_SPEC_v1.json','O_REGIME_EARNED_LAW_REGISTRY_v1.json']:
 paths[f]=R/'engine/infinity_grid/resources/decoder'/f
for f in ['G2_RELATION_VALUED_RESERVATION_SPEC_V1.json','G_UPLIFT_EXPERIMENT_REGISTRY_V1.json']:
 paths[f]=R/'engine/infinity_grid/resources/uplift'/f
for n,pattern,h in [
 ('G1_saved.zip','*G1*.zip','a09e2610bd63af005861baf381d9fdbcaa9b1101633d29644ffeadb35796cc45'),
 ('G2_final.zip','*G2_V0306_FINAL*.zip','702a5cd3d73bc5db8721e252dbbc1f6e51825d8db6a2233d6ae471d9ae14d37c'),
 ('G2_R0.zip','*G2_R0*.zip','2d1b2c870c8367884271de59c0859ce4f2961130bdf0b2b0579d3b1863f9cb46')]:
 p=next((B/'recovered').rglob(pattern));assert sha(p.read_bytes())==h;paths[n]=p
pins={}
for n,p in paths.items():
 shutil.copy2(p,I/n);pins[n]={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'source_path':str(p.relative_to(R))}
dump(B/'INPUT_PINS.json',pins)
p=subprocess.run([sys.executable,str(B/'validate.py')],text=True,capture_output=True,check=True)
result=json.loads(p.stdout);dump(B/'READINESS_VALIDATION.json',result)
register=json.loads((I/'PRIOR_FAMILY_REGISTER.json').read_bytes())
register.update(master_release='MASTER_DATA_V1_0149',scientific_slices=149,catalog_sha256=pins['CATALOG_0149.json']['sha256'])
for f in register['families']:
 if f['family']=='O7 admitted0148':
  f.update(family='O7 admitted0149',status='COMPLETE_IN_DECLARED_SAVED_SCOPE',scope='98 whole root routes:96 nonseed+2 external rank0;205 component occurrences/164 contextual objects separately admitted',gap='68 source whole-root references,source owner index mapping,full Counter certification and full ancestry unavailable')
 if f['family']=='O7 additional saved roots/components':
  f.update(status='ADMITTED_AND_SEED_BINDING_VERIFIED',scope='24 HOM6 rank2/3 roots;205 saved component occurrences/164 exact contextual objects;533 component owners',gap='External adversarial O6 twins retain external role;no absent TWIN4 whole forests or full profile certification')
 if f['family']=='G1':
  f.update(status='ADMITTED_O7_SEED_AND_SAVED_R100_EVIDENCE_BOUND_EXACT_COHORT_PENDING',scope='205 O7 input rows and selected/twin bindings match master149 exports. Saved80 R21..R100 checkpoint file/chain checks;193 motif signatures and24 selected probes;current materialized_discovery bytes match saved producer',gap='Saved evidence is not193 exact carrier objects. Locate saved exact cohort/complete construction recipe closure before materialization;do not invoke ensure_g1_r100_population as read')
 if f['family']=='G2':
  f.update(status='HISTORICAL_CERTIFICATE_AND_RELATION_SPEC_BOUND_EXACT_INPUT_PENDING',scope='Recovered final and R0 closeouts manifest-verified;identical G2 certificate and current relation spec match historical bytes semantically',gap='Current exact G1 cohort still pending. Source revisions uplift_campaign/g2_relation differ;semantic compatibility and finite exact carrier/action/witness export binding required. Historical authority not fresh master admission')
register['work_packages']['WP5']='DECLARED_ADDITIONAL_SAVED_O7_ADMITTED_AND_CURRENT_G1_SEED_BOUND;global closure not complete'
register['work_packages']['WP6']='G1_SAVED_R100_EVIDENCE_AND_G2_HISTORICAL_CERTIFICATE_RECOVERED;exact G1 carrier/recipe closure pending'
register['new_saved_data']={'status':'MASTER149_ADMISSION_ALREADY_COMPLETE_REUSED_WITHOUT_GENERATION','master_release':'MASTER_DATA_V1_0149','new_in_action0175_admissions':0}
register['next_scope']=result['next_scope'];dump(B/'FAMILY_REGISTER.json',register)
print(p.stdout)
