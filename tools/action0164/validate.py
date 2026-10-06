from pathlib import Path
import json,hashlib,importlib.util,sys,collections
BASE=Path(__file__).resolve().parent;R=BASE.parent
sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.loads(p.read_bytes());dump=lambda n,x:(BASE/n).write_text(json.dumps(x,indent=2)+'\n')
refs=load(BASE/'RECOVERED_SOURCE_REFS.json');sourcechecks=[]
for ref in refs:
 root=BASE/'sources'/Path(ref['chain'][-1]).stem
 for e in ref['recovered_members']:
  p=root/e['name'];assert p.stat().st_size==e['bytes'] and sha(p.read_bytes())==e['sha256']
 sourcechecks.append({'archive_sha256':ref['archive_sha256'],'files':len(ref['recovered_members']),'nested_zip_entries_not_extracted':ref['nested_zip_entries_not_extracted']})
root=next((BASE/'sources').glob('*v2.6_PHASE1*/*'));code=root/'03_CODE/o6_phase1_bounded_generation_audit.py';sp=importlib.util.spec_from_file_location('saved_o6',code);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
manifestchecks=[]
for source in [root,next((BASE/'sources').glob('*v2.6_PHASE0*/*'))]:
 manifest=source/'07_PROVENANCE/SHA256_MANIFEST.txt';checked=0;external=[]
 for line in manifest.read_text().splitlines():
  h,n=line.split(maxsplit=1);n=n.strip().lstrip('*');p=source/n
  if p.exists():assert sha(p.read_bytes())==h,(n,h);checked+=1
  else:external.append({'name':n,'sha256':h})
 manifestchecks.append({'source':source.name,'checked':checked,'external_members':external})
sys.path.insert(0,str(R/'o5_export0162'));from project.reader import CarrierReader
pins=load(R/'o5_export0162/EXPORT_BINDINGS.json');reader=CarrierReader(R/'o5_export0162/SCIENTIFIC_EXPORT.zip',pins['archive_sha256'],pins['root_sha256'],R/'o4_export0159/SCIENTIFIC_EXPORT.zip',R/'o3_export0156/SCIENTIFIC_EXPORT.zip')
catpath=R/'o5_integrate0163/CATALOG_0146.json';assert sha(catpath.read_bytes())=='b6405e3e58c61a3f83fec8e7ed082fdcda55bb24ae40b452d063ee61b9abd2cd';assert load(catpath)['slices'][-1]['scientific_root_sha256']==pins['root_sha256']
pool,P,pin_alias=m.load_prototypes();pin=load(root/'08_INPUTS/PINNED_ACTUAL_O5_COMPONENT.json');selection=load(root/'08_INPUTS/O6_PHASE1_PROTOTYPE_SELECTION.json');selection_ok,selected=m.verify_prototype_selection(pool,selection);assert selection_ok
bindings=[]
for proto in [*pool['prototypes'],pin]:
 lane=proto.get('source_lane',proto.get('lane'));key=(lane,proto['source_state_m5'],proto['source_state_digest']);row=reader.lookup(*key)['record'];owners=proto['source_component_members'];matches=[c for c in reader.components(*key) if c['source_owners']==owners];assert len(matches)==1;c=matches[0];resources=reader.resources(*key);ordered=json.loads(json.dumps([resources[i] for i in owners]));assert ordered==proto['ordered_resource_o4_carriers'];assert c['edges']==proto['exact_E5_edges'] and c['proto_ids']==proto['source_o4_prototype_ids']
 assert proto['canonical_R5_sha256']==m.shaj(m.canon_o5(m.nested_o5(ordered)))
 A=proto['accounting'];inv=c['accounting']
 for f in ['K4','N','U2','m3','m4','m5','d','r','beta5','P','F','g']:assert A[f]==inv[f],(proto['prototype_id'],f)
 assert A['beta_flat5']==inv['beta_flat']
 sites=[site for o4 in ordered for o3 in o4 for block in o3 for site in block];free=[sum(s[1][a] for s in sites) for a in range(7)];assert A['free7']==free and A['total_free']==sum(free) and A['min_site_free']==min(sum(s[1]) for s in sites) and A['distinct_site_states']==len({repr(s) for s in sites})
 if 'features' in proto:assert proto['features']==[A[f] for f in pool['feature_order'][:13]]+free
 bindings.append({'prototype_id':proto['prototype_id'],'runtime_alias':pin_alias if proto is pin else None,'source_lane':lane,'source_rank':key[1],'source_digest':key[2],'source_owners':owners,'root_payload_sha256':m.shaj(row),'component_E5_edges':c['edges'],'O4_prototype_ids':c['proto_ids'],'O4_bindings':[reader.owner(*key,i)['binding'] for i in owners],'ordered_resource_sha256':m.shaj(ordered),'canonical_R5_sha256':proto['canonical_R5_sha256'],'component_accounting':inv})
assert len(pool['prototypes'])==132 and len({p['prototype_id'] for p in pool['prototypes']})==132
classes=set();occurrences=0
for key in reader.records:
 resources=reader.resources(*key)
 for c in reader.components(*key):classes.add(m.shaj(m.canon_o5(tuple(resources[i] for i in c['source_owners']))));occurrences+=1
assert occurrences==169 and len(classes)==132 and classes=={p['canonical_R5_sha256'] for p in pool['prototypes']}
assert pin['prototype_id']==selection['pinned_prototype_id'] and pin['canonical_R5_sha256']==selection['pinned_R5_sha256'];assert pin['prototype_id'] in P
# Source pool's canonical ID and producer alias reference the same exact ordered pin.
assert P[pin_alias]==dict(P[pin['prototype_id']],prototype_id=pin_alias)
spec=load(root/'01_SPEC/O6_PHASE1_FORMAL_SPEC.json');states=load(root/'04_RESULTS/O6_PHASE1_GENERATED_SELECTED_STATES.json')['states'];ports,pairs=m.load_rules();counts=collections.Counter();ranks=collections.defaultdict(collections.Counter);payloads={};components=[]
for st in states:
 lane=st['lane'];L=spec['lanes'][lane];ids=tuple(st['proto_ids']);expected=tuple(L['prototype_ids']) if lane=='HET4' else tuple([pin_alias]*L['copies']);assert ids==expected
 edges=tuple(tuple(e) for e in st['edges']);assert len(edges)==st['m6'] and 0<=st['m6']<=L['max_m6'] and edges==tuple(sorted(edges)) and m.state_digest(edges)==st['digest']
 for e in edges:
  assert len(e)==12 and all(type(x)==int and 0<=x<2**32 for x in e);c,j,k,v,i,a,d,J,K,V,I,b=e;assert 0<=c<d<len(ids) and (a,b) in pairs
  for owner,o4,o3,block,site,port in [(c,j,k,v,i,a),(d,J,K,V,I,b)]:
   resource=P[ids[owner]]['o5'];assert 0<=o4<len(resource) and 0<=o3<len(resource[o4]) and 0<=block<len(resource[o4][o3]) and 0<=site<len(resource[o4][o3][block]) and 0<=port<7
 assert not m.validate_exact_state(ids,P,edges,pairs)
 label=f'{lane}:{st["m6"]}:{st["digest"]}';assert label not in payloads;payloads[label]=m.shaj(st)
 for comp in m.graph_components(len(ids),edges):
  if len(comp)>1:
   inv=m.component_invariants(ids,P,edges,comp);assert inv['ok'];pids,ee=m.normalize_component(ids,edges,comp);components.append({'state_key':label,'source_owners':list(comp),'proto_ids':list(pids),'edges':[list(e) for e in ee],'accounting':inv})
 counts['states']+=1;counts['typed_E6_edges']+=len(edges);counts['O5_owner_occurrences']+=len(ids);ranks[lane][st['m6']]+=1
assert counts['states']==134
historical={}
for name in ['O6_PHASE0_REFERENCE_AUDIT_RESULT.json','O5_PHASE2_GRADUATION_AUDIT_RESULT.json']:
 p=root/'08_INPUTS'/name;d=load(p);historical[name]={'file_sha256':sha(p.read_bytes()),'science_sha256':d['science_sha256'],'role':'saved source evidence; no fresh graduation authority'}
assert historical['O6_PHASE0_REFERENCE_AUDIT_RESULT.json']['science_sha256']==spec['parent_phase0_science_sha256'];assert historical['O5_PHASE2_GRADUATION_AUDIT_RESULT.json']['science_sha256']==spec['parent_o5_science_sha256']
out={'schema':'IG_SAVED_O6_READINESS_V1','status':'SAVED_O6_SCOPE_SOURCE_BOUND_READY','master_release':'MASTER_DATA_V1_0146','scientific_slices':146,'catalog_sha256':sha(catpath.read_bytes()),'O5_archive_sha256':pins['archive_sha256'],'O5_root_sha256':pins['root_sha256'],'source_checks':sourcechecks,'manifest_checks':manifestchecks,'producer_sha256':sha(code.read_bytes()),'prototype_pool_bindings':bindings,'O5_component_occurrences_checked':occurrences,'unique_R5_resource_classes':len(classes),'prototype_selection_pass':True,'selected_prototype_ids':selected,'pinned_runtime_alias':pin_alias,'pinned_canonical_prototype_id':pin['prototype_id'],'counts':dict(counts,nontrivial_component_occurrences=len(components),nonseed_states=132,external_seed_states=2),'rank_counts':{k:dict(v) for k,v in ranks.items()},'historical_reference_evidence':historical,'generation_calls':0,'new_admissions':0,'complete_derivation_lineage':False,'microscopic_O2_source_closed':False,'missing_O2_panels':76,'full_l0_to_g8_complete':False}
dump('READINESS_VALIDATION.json',out);dump('O6_PAYLOAD_SHA256.json',payloads);dump('COMPONENTS.json',components);reader.close();print(json.dumps({k:out[k] for k in ['status','counts','prototype_selection_pass','unique_R5_resource_classes']}))
