from pathlib import Path
import json,hashlib,importlib.util,sys,collections
BASE=Path(__file__).resolve().parent;R=BASE.parent
sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.loads(p.read_bytes());dump=lambda n,x:(BASE/n).write_text(json.dumps(x,indent=2)+'\n')
refs=load(BASE/'RECOVERED_SOURCE_REFS.json');checks=[]
for ref in refs:
 root=BASE/'sources'/Path(ref['chain'][-1]).stem
 for e in ref['recovered_members']:
  p=root/e['name'];assert p.stat().st_size==e['bytes'] and sha(p.read_bytes())==e['sha256']
 checks.append({'archive_sha256':ref['archive_sha256'],'files':len(ref['recovered_members'])})
root=next((BASE/'sources').glob('*v2.5_PHASE1*/*'));code=root/'03_CODE/o5_phase1_bounded_generation_audit.py'
sp=importlib.util.spec_from_file_location('saved_o5',code);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
manifest=[];external=[]
for line in (root/'SHA256_MANIFEST.txt').read_text().splitlines():
 h,n=line.split()[:2];p=root/n
 if p.exists():assert sha(p.read_bytes())==h;manifest.append(n)
 else:external.append({'name':n,'sha256':h})
sys.path.insert(0,str(R/'o4_export0159'));from project.reader import CarrierReader;from project import frozen_o4 as o4
pins=load(R/'o4_export0159/EXPORT_BINDINGS.json');reader=CarrierReader(R/'o4_export0159/SCIENTIFIC_EXPORT.zip',pins['archive_sha256'],pins['root_sha256'],R/'o3_export0156/SCIENTIFIC_EXPORT.zip')
catpath=R/'o4_integrate0160/CATALOG_0145.json';assert sha(catpath.read_bytes())=='69399952d69963c9ddbf8ba278fba3cb2052ff2725c0cb8155d286f9a24bd304';assert load(catpath)['slices'][-1]['scientific_root_sha256']==pins['root_sha256']
selected=load(root/'08_INPUTS/SELECTED_O4_PROTOTYPES.json')['selected'];pin=load(root/'08_INPUTS/PHASE0_PINNED_ACTUAL_O4_COMPONENT.json');pid='PHASE0_PINNED_ACTUAL_O4_COMPONENT';A=pin['accounting'];P={p['prototype_id']:p for p in selected};P[pid]=dict(prototype_id=pid,ordered_resource_o3_carriers=pin['ordered_resource_o3_carriers'],K3=A['K3'],N=A['N'],U2=A['U2'],m3=A['m3'],m4=A['m4'],d=A['d'],r4=A['r'],beta_flat4=A['beta_flat4'],beta4=A['beta4']);bindings=[]
for proto in [*selected,pin]:
 lane=proto['source_lane'];digest=proto.get('source_state_digest',proto.get('source_phase1_state_digest'));matches=[k for k in reader.records if k[0]==lane and k[2]==digest];assert len(matches)==1;key=matches[0];row=reader.lookup(*key)['record'];owners=proto['source_component_indices'];cs=[c for c in reader.components(*key) if c['source_owners']==owners];assert len(cs)==1;c=cs[0]
 resources=o4.materialize(tuple(row['proto_ids']),reader.P,tuple(tuple(e) for e in row['edges']));ordered=json.loads(json.dumps([resources[i] for i in owners]));assert ordered==proto['ordered_resource_o3_carriers'];assert c['edges']==proto['internal_e4_edges'] and c['proto_ids']==proto['o3_prototype_ids']
 pp=P[proto.get('prototype_id',pid)]
 for f in ['N','U2','m3','m4','d','beta4']:assert pp[f]==c['accounting'][f],f
 assert pp['r4']==c['accounting']['r'] and pp['beta_flat4']==c['accounting']['beta_flat'] and pp['K3']==len(owners)
 if 'source_state_m4' in proto:assert proto['source_state_m4']==key[1]
 if 'canonical_R4_sha256' in proto:assert proto['canonical_R4_sha256']==m.shaj(m.canon_o4(m.proto_o4(pp)))
 bindings.append({'prototype_id':proto.get('prototype_id',pid),'source_lane':lane,'source_rank':key[1],'source_digest':digest,'source_owners':owners,'root_payload_sha256':o4.shaj(row),'component_E4_edges':c['edges'],'O3_prototype_ids':c['proto_ids'],'O3_bindings':[reader.owner(*key,i)['binding'] for i in owners],'ordered_resource_sha256':m.shaj(ordered),'component_accounting':c['accounting']})
selok,ids=m.verify_selection();assert selok
spec=load(root/'01_SPEC/O5_PHASE1_FORMAL_SPEC.json');states=load(root/'04_RESULTS/O5_PHASE1_GENERATED_SELECTED_STATES.json')['states'];ports,pairs=m.load_rules();counts=collections.Counter();ranks=collections.defaultdict(collections.Counter);payloads={};components=[]
for st in states:
 lane=st['lane'];L=spec['lanes'][lane];ids=tuple(st['proto_ids']);expected=tuple(L['prototype_ids']) if lane=='HET4' else tuple([L['prototype_id']]*L['copies']);assert ids==expected
 edges=tuple(tuple(e) for e in st['edges']);assert len(edges)==st['m5'] and 0<=st['m5']<=L['max_m5'] and edges==tuple(sorted(edges)) and m.state_digest(edges)==st['digest']
 for e in edges:
  assert len(e)==10 and all(type(x)==int for x in e);c,g,b,s,a,d,G,B,S,q=e;assert 0<=c<d<len(ids) and (a,q) in pairs
  for owner,group,block,site,port in [(c,g,b,s,a),(d,G,B,S,q)]:
   resource=P[ids[owner]]['ordered_resource_o3_carriers'];assert 0<=group<len(resource) and 0<=block<len(resource[group]) and 0<=site<len(resource[group][block]) and 0<=port<7
 for (c,g,b,s,a),used in m.usage(edges).items():assert 0<=used<=m.base_free(ids,P,c,g,b,s,a)
 label=f'{lane}:{st["m5"]}:{st["digest"]}';assert label not in payloads;payloads[label]=m.shaj(st)
 for comp in m.graph_components(len(ids),edges):
  if len(comp)>1:
   inv=m.comp_invariants(ids,P,edges,comp);assert inv['ok'];pids,ee=m.normalize_component(ids,edges,comp);components.append({'state_key':label,'source_owners':list(comp),'proto_ids':list(pids),'edges':[list(e) for e in ee],'accounting':inv})
 counts['states']+=1;counts['typed_E5_edges']+=len(edges);counts['O4_owner_occurrences']+=len(ids);ranks[lane][st['m5']]+=1
assert counts['states']==134
out={'schema':'IG_SAVED_O5_READINESS_V1','status':'SAVED_O5_SCOPE_SOURCE_BOUND_READY','master_release':'MASTER_DATA_V1_0145','scientific_slices':145,'catalog_sha256':sha(catpath.read_bytes()),'O4_archive_sha256':pins['archive_sha256'],'O4_root_sha256':pins['root_sha256'],'source_checks':checks,'manifest_files_checked':len(manifest),'external_manifest_members':external,'producer_sha256':sha(code.read_bytes()),'prototype_bindings':bindings,'prototype_selection_pass':True,'counts':dict(counts,nontrivial_component_occurrences=len(components),nonseed_states=132,external_seed_states=2),'rank_counts':{k:dict(v) for k,v in ranks.items()},'generation_calls':0,'new_admissions':0,'complete_derivation_lineage':False,'microscopic_O2_source_closed':False,'missing_O2_panels':76,'full_l0_to_g8_complete':False}
dump('READINESS_VALIDATION.json',out);dump('O5_PAYLOAD_SHA256.json',payloads);dump('COMPONENTS.json',components);reader.close();print(json.dumps({k:out[k] for k in ['status','counts','prototype_selection_pass']}))
