"""Explicit Q47 source compatibility; protected old code and legacy dispatch stay pinned."""
from pathlib import Path
import ast,hashlib
ADDITIONS={'full_owner_export.py', 'master_g6_initial.py', 'master_g3_saved_storage.py', 'g4_term_admission.py', 'historical_g4_saved_terms.py', 'master_retained_storage_repair.py', 'saved_read_run_plan.py', 'saved_read_source_compatibility.py', 'g4_rebase_saved_pipeline.py', 'full_g3_export.py', 'master_existing_export_pipeline.py', 'master_full_export_services.py', 'full_marker_export.py', 'master_g6_initial_storage.py', 'master_scientific_storage.py', 'master_full_export_evaluators.py', 'combined_saved_master_pipeline.py', 'master_collection_writer.py', 'saved_g1_scope_compatibility.py', 'g4_rebase_saved_pins.json', 'saved_g2_g3_read_plan.py'}
CHANGED_DIGESTS={'materialized_discovery.py': '83b87f52c0d08d620ee27d744e44272088b8dfeb283d55803e9fc4f8010b69b9', 'maturation_parallel.py': 'f49560814f9c3d4b69080053800617089bdd29d8f4a4223af1871b5f8c71e300', 'storage_collections.py': '3ae04c7fd6e23c5bd9feb4f86cc8175b365958838e5b5e416b0d66908b9b20b9', 'v05_kernel_service_providers.py': '3e303671746802ec8a6519cc55bc0d0d31005d687a9b5745c9a40a3d3cdd05cb', 'v05_kernel_services.py': '6e91a1883599eaa15747a2bb9a4cd815246ad87e8fb461b62ba413cb1134cfce', 'v05_stage_registry.py': '3c6440b13909d231338d77d636318916b604ec4398bd391ea5591c9d99117f68'}
FUNCTIONS={'execute_combined_saved_master_pipeline_run_plan': 'f111fbe38d5a3b266cc7b7e171a89d3839041000117af339510251296cfca66c', 'execute_g4_rebase_saved_pipeline_run_plan': '01898c7ae87c642e04e167da3bc00468b55da44bbf08038884b8f17e1a25cf14', 'execute_historical_g4_saved_term_run_plan': '3768bb9d051dfad30649bb51438e61416c73901a5999c7db79e3723fd174462e', 'execute_existing_export_attach_run_plan': 'a24df1ba3744642006c5755c6d5a64041a09b312a4c554eef6c12e2f361b0499', 'execute_original_g2_g3_saved_read_run_plan': '91ee5d19031214d37e0ad2136ae0ba4fa5278177fb1fa521e8cd6ee58319ed90'}
COMPILE_PREFIX=['ed528d148d440dbde78fa00c5b46725c496abc1563f52bc711b0f7f2686f6a7e', 'b65ac07820c0a5da7fca61a9fba7ab24d1625559579c7c084b29fe31321a06b2', 'd479348d42eeb405899ec9107d89285a4f559b9359646b115aa56baa4b19cec6', '9f7f283c7c7f672df0e5fb93cc80e88f4e7f26c4f83f3c20215c494293a3170a', '54e5764efd82f2de09704ee3a69133e67f6dd326cb8f7ea104b759c23e9fa88b']
LEGACY_AST='ed49d0a1be38cc887db805322200f40757f4776acaa653986bc357ce70c0c4f3'
class SourceCompatibilityError(RuntimeError):pass
def need(ok,why):
 if not ok:raise SourceCompatibilityError(why)
def _sha(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def validate_runplan(text):
 tree=ast.parse(text);found={n.name:_sha(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS}
 need(found==FUNCTIONS,'ADDED_DISPATCH_CHANGED')
 tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name not in FUNCTIONS]
 f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compile_run_plan_v2')
 need([_sha(n) for n in f.body[:len(COMPILE_PREFIX)]]==COMPILE_PREFIX,'ADDED_COMPILER_CHANGED')
 f.body=f.body[len(COMPILE_PREFIX):]
 need(_sha(tree)==LEGACY_AST,'LEGACY_RUNPLAN_CHANGED')
 return True
def verify_source(old,current):
 need(type(old) is dict and type(current) is dict,'SOURCE_MANIFEST_TYPES')
 need(not set(old)-set(current) and set(current)-set(old)==ADDITIONS,'EXPLICIT_ADDITION_DOMAIN')
 modified={n for n in old if old[n]!=current[n]}
 need(modified==set(CHANGED_DIGESTS)|{'run_plan_v2.py'},'PROTECTED_SOURCE_CHANGED')
 need(all(current[n]==h for n,h in CHANGED_DIGESTS.items()),'QUALIFIED_CHANGED_SOURCE_CHANGED')
 validate_runplan(Path(__file__).with_name('run_plan_v2.py').read_text())
 return {'protected_old_files':len(old)-len(modified),'exact_pinned_changed_files':len(CHANGED_DIGESTS),'legacy_RunPlan_AST_unchanged':True,'explicit_additions':sorted(ADDITIONS),'production_authority':'NONE'}
