from pathlib import Path
import json,hashlib,sys,struct,types,collections
R=Path.cwd();B=R/'scope_reconcile0171';sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.loads(p.read_bytes());shaj=lambda d:sha(json.dumps(d,sort_keys=True,separators=(',',':')).encode())
def dump(n,d):(B/n).write_text(json.dumps(d,indent=2)+'\n')
sys.path.insert(0,str(R/'o7_export0169'));from project.reader import CarrierReader
from project import frozen_o7 as m
pins=load(R/'o7_export0169/EXPORT_BINDINGS.json');r=CarrierReader(R/'o7_export0169/SCIENTIFIC_EXPORT.zip',pins['archive_sha256'],pins['root_sha256'],R/'o6_export0165/SCIENTIFIC_EXPORT.zip',R/'o5_export0162/SCIENTIFIC_EXPORT.zip',R/'o4_export0159/SCIENTIFIC_EXPORT.zip',R/'o3_export0156/SCIENTIFIC_EXPORT.zip')
s=load(B/'SOURCE_SEARCH.json');selected=load(R/'o7_readiness0168/SELECTED.json')['prototypes'];twins=load(R/'o7_readiness0168/TWINS.json')['twins'];parents=dict(r.P);colors={p['prototype_id']:p['exact_representative_sha256'] for p in selected}
for t in twins:parents[t['twin_id']]=types.SimpleNamespace(pid=t['twin_id'],h6=m.nested_h6(t['ordered_resource_O5_carriers']),accounting=t['accounting']);colors[t['twin_id']]=t['exact_representative_sha256']
alg=json.loads(r.o6.raw['algebra']);ports=[tuple(x) for x in alg['port_atoms']];ix={p:i for i,p in enumerate(ports)};pairs={(ix[tuple(a)],ix[tuple(b)]) for a,b in alg['ordered_compatible_port_pairs']}
def check(ids,edges,ac,digest):
 edges=tuple(tuple(e) for e in edges);ctx=types.SimpleNamespace(n=len(ids),parents=tuple(parents[x] for x in ids));used=collections.Counter()
 assert edges==tuple(sorted(edges)) and sha(b'IG-E7-STATE-v1|'+b''.join(struct.pack('>14I',*e) for e in edges))==digest
 for e in edges:
  assert len(e)==14 and all(type(x)==int and 0<=x<2**32 for x in e)
  c,p,a,d,q,b=m.eparts(e);assert 0<=c<d<ctx.n and (a,b) in pairs
  for owner,path,typ in [(c,p,a),(d,q,b)]:
   cap,free=m.site_at(ctx.parents[owner].h6,path);used[(owner,path,typ)]+=1;assert used[(owner,path,typ)]<=free[typ]
 assert m.accounting(ctx,edges)==ac and ac['ok']
 return tuple(m.materialize_owner(p.h6,edges,i) for i,p in enumerate(ctx.parents))
new={};dup=0;checkpoint_hashes=set();rootrefs=[]
for hit in s['hits']:
 if not hit.get('beam_rows') or hit['sha256'] in checkpoint_hashes:continue
 checkpoint_hashes.add(hit['sha256']);p=R/hit['recovered_path'];raw=p.read_bytes();assert sha(raw)==hit['sha256'];d=json.loads(raw);assert shaj({k:v for k,v in d.items() if k!='payload_sha256'})==d['payload_sha256']
 assert d['prereg_science_sha256']=='1459e670df72028d4f9978549d2da0f179029079ae0769953eddcd19b426bbd8' and d['correction_science_sha256']=='e1ce490a8bbe29cc9d818a601539a5010f1ab9ada6a076b3e480c0b0a643741f' and d['parent_o6_science_sha256']=='7b9e3ec0fbabcdb0fa46c2d5d153ac186cdd2d2343275a407e82b6fd14d9da37'
 lane='HOM6' if 'HOM6' in p.name else 'HET4';rank=d['result'].get('m7',len(d['result']['beam'][0]['edges']));cfg=r.lanes[lane];ids=cfg.get('prototype_ids') or [cfg['prototype_id']]*cfg['owner_count']
 for row in d['result']['beam']:
  key=(lane,rank,row['digest']);check(ids,row['edges'],row['accounting'],row['digest'])
  if key in r.records:assert row==r.records[key];dup+=1
  elif key in new:assert new[key]['record']==row
  else:new[key]={'lane':lane,'rank':rank,'digest':row['digest'],'payload_sha256':shaj(row),'record':row,'checkpoint_sha256':hit['sha256'],'checkpoint_chain':hit['chain']}
 rootrefs.append(hit)
profilehit=next(x for x in s['hits'] if x['chain'][-1].endswith('O7_SELECTED_COMPONENT_PROFILES.json'));survivorhit=next(x for x in s['hits'] if x['chain'][-1].endswith('O7_IMMUTABLE_SURVIVORS.json'));profiles=load(R/profilehit['recovered_path']);survivors=load(R/survivorhit['recovered_path']);assert profiles['records']==survivors['records'];assert profilehit['sha256']=='77497fcd32952096bb13923d5cb8821a1725a8d8afad2eab7a18c2e95afb44a9';assert len(profiles['records'])==profiles['record_count']==205
# Exact source Merkle formula: SITE->[O2,O3,O4,O5,R6] containers; outer R7.
def container(tag,children):return sha((tag+'|'+'|'.join(sorted(children))).encode())
def sig6(h6):
 def leaf(site):return sha(b'SITE|'+json.dumps([list(site[0]),list(site[1])],separators=(',',':')).encode())
 return container('R6',[container('O5',[container('O4',[container('O3',[container('O2',[leaf(x) for x in block]) for block in o3]) for o3 in o4]) for o4 in o5]) for o5 in h6])
component_counts=collections.Counter();resources={}
for idx,row in enumerate(profiles['records']):
 ids=row['parent_ids'];assert row['component_owner_count']==len(ids) and row['parent_exact_colors']==[colors[x] for x in ids]
 h=check(ids,row['edges'],row['accounting'],row['state_digest']);assert container('R7',[sig6(x) for x in h])==row['R7_skin_sha256'];component_counts[row['lane']]+=1;resources[str(idx)]={'payload_sha256':shaj(row),'resource_sha256':shaj(h)}
dump('NEW_SAVED_ROOT_STATES.json',list(new.values()));dump('SURVIVOR_RESOURCE_HASHES.json',resources);dump('RECOVERED_PROFILE_BINDINGS.json',{'profiles':profilehit,'survivors':survivorhit,'records_equal':True,'profile_counter_details_available':False,'role':'205 literal component records and saved Counter summaries; full Counter entries not present','generation_calls':0})
out={'status':'ADDITIONAL_SAVED_O7_SCOPE_RECOVERED_AND_TYPED_BINDINGS_VERIFIED','master_release':'MASTER_DATA_V1_0148','new_root_states':len(new),'new_root_rank_counts':dict(collections.Counter(f'{k[0]}:{k[1]}' for k in new)),'new_root_E7_edges':sum(x['rank'] for x in new.values()),'duplicate_root_occurrences_agree':dup,'survivor_component_records':205,'component_lane_counts':dict(component_counts),'component_E7_edges':sum(len(x['edges']) for x in profiles['records']),'checks':['checkpoint payload hash and prereg/correction/O6 science pins','binary E7 state digests, typed coordinates, compatibility/capacity','saved accounting and exact remaining resource Merkle R7 hashes','selected O6 bindings and explicit external twin roots','205 profile/survivor rows agree byte-semantically'],'not_claimed':['Full root states at HOM6 ranks4..6 or TWIN4','Full direct Counter entries or fresh profile certification','Later graduation authority validation','New scientific admission'],'generation_calls':0,'new_admissions':0,'next_scope':'WP5_ADDITIONAL_SAVED_O7_ROOT_AND_COMPONENT_SCOPE_EXPORT_READINESS'};dump('RECOVERY_VALIDATION.json',out);r.close();print(json.dumps(out,indent=2))
