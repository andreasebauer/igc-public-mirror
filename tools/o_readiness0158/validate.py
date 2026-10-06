from pathlib import Path
import json,hashlib,importlib.util,sys,collections
B=Path(__file__).resolve().parent;R=B.parent;sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.load(open(p));refs=load(B/'RECOVERED_SOURCE_REFS.json');source_checks=[]
for ref in refs:
 root=B/'sources'/Path(ref['chain'][-1]).stem
 for e in ref['recovered_members']:
  p=root/e['name'];assert p.stat().st_size==e['bytes'] and sha(p.read_bytes())==e['sha256']
 source_checks.append({'source':ref['chain'][-1],'files':len(ref['recovered_members']),'status':'RECOVERED_BYTE_HASH_PASS','original_archive_sha256':ref['archive_sha256']})
sys.path.insert(0,str(R/'o3_export0156'));from project.reader import CarrierReader
p=load(R/'o3_export0156/EXPORT_BINDINGS.json');reader=CarrierReader(R/'o3_export0156/SCIENTIFIC_EXPORT.zip',p['archive_sha256'],p['root_sha256']);cat=load(R/'o3_integrate0157/CATALOG_0144.json');assert cat['slices'][-1]['scientific_root_sha256']==p['root_sha256']
P0=next((B/'sources').glob('*v2.4_PHASE0*/*'));P1=next((B/'sources').glob('*v2.4_PHASE1*/*'));manifest_checks=[]
for root in [P0,P1]:
 checked=0;external=[]
 for line in (root/'SHA256_MANIFEST.txt').read_text().splitlines():
  parts=line.split();h,n=parts[:2];fp=root/n
  if fp.exists():assert sha(fp.read_bytes())==h,(n,h);checked+=1
  else:external.append({'name':n,'sha256':h})
 manifest_checks.append({'source':root.name,'checked_files':checked,'external_members':external,'status':'PRESENT_MEMBER_MANIFEST_PASS'})
code=P1/'03_CODE/o4_phase1_bounded_generation_audit.py';sp=importlib.util.spec_from_file_location('frozen_o4',code);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
protos=load(P1/'07_INPUTS/SELECTED_O3_PROTOTYPES.json')['selected'];pin=load(P1/'07_INPUTS/PHASE0_PINNED_ACTUAL_O3_COMPONENT.json');assert pin==load(P0/'07_INPUTS/PINNED_ACTUAL_O3_COMPONENT.json');bindings=[]
for proto in [*protos,pin]:
 rank=proto.get('r3_rank',proto.get('source_r3_rank'));h=proto['source_exact_key'];comp=proto['source_component_entities'];row=reader.lookup(rank,h)['record'];a=reader.audit;blocks=a.expanded(row);U=a.usage3(row,blocks);ordered=[[[list(p),[p[k]-u2[k]-U[v][i][k] for k in range(7)]] for i,(p,u2) in enumerate(blocks[v])] for v in comp];saved=[[[s['p'],s['f']] for s in block] for block in proto['ordered_resource_blocks']];assert ordered==saved
 assert sha(reader.content_bytes('panel'+str(rank)))==proto['source_panel_sha256'];cs=[c for c in reader.components(rank,h) if c['source_entities']==comp];assert len(cs)==1
 if 'canonical_R3_sha256' in proto:
  canonical=a.component_r3(row,comp,blocks,U);assert proto['canonical_R3_sha256']==sha(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode())
 if 'internal_o3_edges' in proto:assert proto['internal_o3_edges']==[e for e in row['edges'] if e[0] in comp and e[3] in comp]
 bindings.append({'prototype_id':proto.get('prototype_id','PHASE0_PINNED_ACTUAL_O3_COMPONENT'),'source_rank':rank,'source_exact_key':h,'source_component_entities':comp,'root_payload_sha256':load(R/'o3_scope0155/PAYLOAD_SHA256.json')[f'{rank}:{h}'],'R3_sha256':cs[0]['diagnostic']['R3_sha256'],'source_panel_sha256':proto['source_panel_sha256'],'base_E3_incidence_recoverable_from_admitted_O3_root':True,'resource_projection_is_not_the_full_root':True})
selection_ok,selected_ids=m.verify_selection();assert selection_ok
P={x['prototype_id']:x for x in protos};ordered=pin['ordered_resource_blocks'];N=sum(len(b) for b in ordered);d=sum(sum(s['p'])-2 for b in ordered for s in b);m3=len(pin['internal_o3_edges']);rho=sum(sum(s['p'][k]-s['f'][k] for k in range(7)) for b in ordered for s in b)//2
P['PHASE0_PINNED_ACTUAL_O3_COMPONENT']={'prototype_id':'PHASE0_PINNED_ACTUAL_O3_COMPONENT','ordered_resource_blocks':ordered,'N':N,'d':d,'m3':m3,'U2':rho-m3,'rho3':rho,'beta_flat3':rho-N+1,'beta3':m3-len(ordered)+1,'n3':len(ordered),'r3_rank':pin['source_r3_rank']}
ports,pairs=m.load_rules();spec=load(P1/'01_SPEC/O4_PHASE1_FORMAL_SPEC.json');states=load(P1/'04_RESULTS/O4_PHASE1_GENERATED_SELECTED_STATES.json')['states'];counts=collections.Counter();ranks=collections.defaultdict(collections.Counter);payloads={};components=0
for st in states:
 lane=st['lane'];L=spec['lanes'][lane];ids=tuple(st['proto_ids']);expected=tuple(L['prototype_ids']) if lane=='HET4' else tuple([L['prototype_id']]*L['copies']);assert ids==expected
 edges=tuple(tuple(e) for e in st['edges']);assert len(edges)==st['m4'] and 0<=st['m4']<=L['max_m4'];assert edges==tuple(sorted(edges)) and m.state_digest(edges)==st['digest']
 usage=m.usage(edges)
 for e in edges:
  assert len(e)==8 and all(type(x)==int for x in e)
  c,b,s,a,d,g,t,q=e;assert 0<=c<d<len(ids) and (a,q) in pairs
  assert 0<=b<len(P[ids[c]]['ordered_resource_blocks']) and 0<=g<len(P[ids[d]]['ordered_resource_blocks'])
  assert 0<=s<len(P[ids[c]]['ordered_resource_blocks'][b]) and 0<=t<len(P[ids[d]]['ordered_resource_blocks'][g])
 for (c,b,s,a),used in usage.items():assert 0<=used<=m.base_free(ids,P,c,b,s,a)
 for comp in m.graph_components(len(ids),edges):
  if len(comp)>1:assert m.comp_invariants(ids,P,edges,comp)['ok'];components+=1
 key=f'{lane}:{st["m4"]}:{st["digest"]}';assert key not in payloads;payloads[key]=m.shaj(st);counts['states']+=1;counts['typed_E4_edges']+=len(edges);counts['O3_owner_occurrences']+=len(ids);ranks[lane][st['m4']]+=1
out={'schema':'IG_REMAINING_O_PRODUCER_READINESS_V1','status':'SAVED_O4_SCOPE_READY_WITH_INHERITED_QBANK_BOUNDARY','master_release':'MASTER_DATA_V1_0144','scientific_slices':144,'catalog_sha256':sha((R/'o3_integrate0157/CATALOG_0144.json').read_bytes()),'recovered_source_checks':source_checks,'O4_manifest_checks':manifest_checks,'O4_producer_sha256':sha(code.read_bytes()),'O3_parent_bindings':bindings,'prototype_selection_pass':True,'selected_prototype_ids':selected_ids,'O4_saved_counts':dict(counts,nontrivial_component_occurrences=components),'O4_lanes':{k:{'rank_counts':dict(v),'spec':spec['lanes'][k]} for k,v in ranks.items()},'generation_calls':0,'new_admissions':0,'missing_microscopic_O2_panels':76,'limits':['Only354 saved O4 selected-state occurrences validated; no fresh generation or graduation.','Base O3 E3 incidence must be retained through explicit component/root references; ordered resource blocks alone lose it.','Lane/rank/digest plus full payload SHA256; edge-only digest alone does not identify different owner forests.','Saved rows lack parent/action occurrence records; no complete derivation lineage claimed.','O5/O6/O7 sources recovered and hashed only; nested producer bindings not yet validated.','O7 closeout archive labels execution incomplete; do not infer campaign completion.']}
(B/'READINESS_VALIDATION.json').write_text(json.dumps(out,indent=2));(B/'O4_PAYLOAD_SHA256.json').write_text(json.dumps(payloads,indent=2));reader.close();print(json.dumps({k:out[k] for k in ['status','O4_saved_counts','prototype_selection_pass']}))
