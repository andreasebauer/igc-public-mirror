"""Freeze source consumers, catalog dispositions and the remaining migration gates."""
import argparse
import json
from pathlib import Path
from map_identities import sha,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    items=[
      ('infinity_grid/o7_edge_canonicalization.py',['def state_identity','def bind_engine'], 'BASE_IDENTITY', 'V3 key plus explicit V2 alias; fixed-context V2 edge/witness interface unchanged'),
      ('infinity_grid/frontier.py',['def _load_engine','return bind_engine'], 'ENGINE_BINDING', 'Bound engine exposes V3 context identity; do not infer every old digest consumer calls it'),
      ('infinity_grid/regime_scanner.py',['class O7State','def construction_digest','IG_O7_SCANNER_CONSTRUCTION_V3','uniq={s.construction_digest'], 'FIRST_DEDUP_SELECTION', 'O7 construction key changes; dedup and hash-sorted farthest selection can change panel membership'),
      ('infinity_grid/regime_scanner.py',['class LiftState','"children":[c.construction_digest'], 'RECURSIVE_KEY_DEPENDENCY', 'Parent construction keys bind children; changes propagate to recursive keys and joins'),
      ('infinity_grid/materialized_discovery.py',['rs.O7State','s.construction_digest'], 'O7_AND_LIFT_CONSUMER', 'Uses new O7 construction namespace; reconcile saved selected panels and alternatives before successor runs'),
      ('infinity_grid/regime_raise_audit.py',['rs.O7State','rs._farthest_select'], 'PANEL_SELECTION_CONSUMER', 'Hash namespace affects deterministic selection even when the retained class partition is unchanged'),
      ('infinity_grid/regime_scanner_validation.py',['O7State('], 'SCANNER_VALIDATION_CONSUMER', 'Reference panels require explicit identity/schema bindings'),
      ('infinity_grid/maturation_parallel.py',['rs.O7State'], 'PARALLEL_STATE_SERIALIZATION', 'Worker reconstruction retains state type; source/capture must bind namespace'),
      ('infinity_grid/exact_carrier_unblinding.py',['class ExactO7State','Match the v0.26 identity','"parents": list(self.ctx.source_order)'], 'SEPARATE_FROZEN_IDENTITY', 'ExactO7State deliberately retains historical source/coordinates; not migrated merely by changing O7State'),
      ('infinity_grid/exact_carrier_acceleration.py',['ExactO7State('], 'SEPARATE_FROZEN_CONSUMER', 'Preserve its distinct identity contract; no blanket V3 substitution'),
      ('infinity_grid/carrier_reuse.py',['construction_digest'], 'REUSE_JOIN', 'Reused carrier references need schema-qualified mappings and all preimages'),
      ('infinity_grid/uplift_structural.py',['carrier.construction_digest','carrier construction_digest'], 'UPLIFT_PROVENANCE', 'Provenance references inherit carrier namespace; map matched scopes before comparing outputs'),
      ('tools/step1_native_identity_validation.py',['legacy_context_identity','global_identity'], 'VALIDATION_OUTPUT', 'Records both schemas; historical edge digest is a different namespace from V2 context identity')]
    entries=[]
    for filename,tokens,role,meaning in items:
        q=a.source/filename;lines=q.read_text().splitlines()
        hits=[{'line':i,'text':line.strip()} for i,line in enumerate(lines,1) if any(t in line for t in tokens)]
        entries.append({'file':filename,'source_sha256':sha(q),'role':role,'meaning':meaning,'evidence':hits})
    catalog=json.loads(a.catalog.read_text());scopes=[]
    for s in catalog['slices']:
        name=s['dataset_id']
        if 'J3' in name or name.startswith(('NODE','O2','O3','O4','O5','O6','IG_MASTER')):
            decision='PRESERVE_CURRENT_SCOPED_IDENTITY';reason='No direct global E7 V2 key dependency established for this scoped saved slice; no regeneration ordered'
        elif name.startswith('O7'):
            decision='AVAILABLE_SAVED_RECORDS_MAPPED_NO_MIGRATION';reason='Mapping registers join exact literal/context/occurrence IDs to computed V2 aliases and V3 keys; discarded pre-dedup states remain open'
        else:
            decision='PRESERVE_AND_RECONCILE_MATCHED_DESCENDANTS';reason='Historical identity contracts remain frozen; exact recipes/serializer and overlays require separate matched replay'
        scopes.append({'dataset_id':name,'scientific_root_sha256':s.get('scientific_root_sha256'),'format':s.get('format'),
            'identity_profile':s.get('identity_profile'),'decision':decision,'reason':reason,'counts':s.get('counts'),
            'archive':s.get('archive')})
    plan={'schema':'IG_P0_IDENTITY_TRANSITION_DECISION_V1','status':'MIGRATION_BLOCKED_PENDING_AFFECTED_SCOPE_CLOSURE',
        'source_candidate':'dev156','source_inventory_file':'SOURCE_MANIFEST.json','catalog_snapshot_sha256':sha(a.catalog),
        'historical_ids_are_immutable':True,'native_controller_takeover':False,
        'mapping_contract':{'lookup':'(schema, data scope, exact parent context, occurrence/literal binding, old identity) -> set of new identities',
            'old_v2_is_alias_only':True,'preserve_all_preimages_and_occurrences':True,
            'hash_namespace_change_is_not_a_scientific_difference':True,'identity_key_sorting_can_change_bounded_selection':True,
            'current_scanner_mapping_scope':'ZERO_RESERVATION_ONLY; nonzero overlay binds actual consumed resource and coordinate base'},
        'required_next_gates':[
            {'priority':1,'gate':'Bind affected finite pre-dedup scopes','method':'Recover source inputs/caps and first dedup; enumerate only affected raw candidates with V3, retain all occurrences and V1/V2 diagnostics','exit':'No unexplained exact object/resource/action mismatch; explicit missing/discarded-state boundary'},
            {'priority':2,'gate':'G1/Q2/observer matched recipe reconciliation','method':'Keep historical runs frozen; map O7 anchors and all nonzero reservation overlays, recursive DAG nodes, alternatives and construction-key selections','exit':'All required descendants matched or registered for affected/missing rerun; 7058 Q2 members and separate observer scope explicitly bound'},
            {'priority':3,'gate':'Fresh full-dev156 native and registered RC qualification','method':'Run complete source-bound candidate with native project guard, exact source/runtime/dependencies/modes; independent-host qualification under full release authority','exit':'Declared RC rows and restoration gates pass; bounded mapping is not substituted for RC'},
            {'priority':4,'gate':'Unknown provider outcome and outage qualification','method':'Exact canonical-ID reconciliation before retry; retain partial writes and failure evidence','exit':'No duplicate dispatch/credit or unverified save claim'},
            {'priority':5,'gate':'Explicit promotion and new controller boundary','method':'Only after relevant accepted gates; preserve old source/captures and attach versioned mappings','exit':'No in-flight work; exact saved boundary; source-bound new capture'}],
        'nonclaims':['NO_FULL_L0_G8_REGENERATION','NO_OLD_COUNTS_AUTOMATICALLY_TRANSFERRED','NO_MASTER_REWRITE','NO_G2_PROMOTION','NO_RELEASE_ACCEPTANCE']}
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'CONSUMER_REGISTER.json',{'schema':'IG_V2_V3_SOURCE_CONSUMERS_V1','scope':'Reviewed direct consumers and named recursive/join paths; not a proof every runtime call is covered','records':entries})
    write(a.output/'CATALOG_DISPOSITION.json',{'schema':'IG_CATALOG_IDENTITY_IMPACT_TRIAGE_V1','catalog_sha256':sha(a.catalog),'release_id':catalog['release_id'],'slice_count':len(scopes),'records':scopes})
    write(a.output/'TRANSITION_DECISION.json',plan)

if __name__=='__main__':main()
