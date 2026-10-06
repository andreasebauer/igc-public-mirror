import pathlib,json,hashlib,zipfile,ast,sys
B=pathlib.Path(__file__).resolve().parent;R=B.parent;sha=lambda b:hashlib.sha256(b).hexdigest();catalog=(R/'pairs0142/CATALOG_0141.json').read_bytes();assert sha(catalog)=='dce298f06c046fecd0a201cd034be88b18556cc68fcab2d7e0d5fd4d3cb6b847'
sys.path.insert(0,str(R/'takeover/master_reader'));from ig_master.prefix_contract import event_id,j3_id
from ig_master.prefix_reader import PrefixReader
p=PrefixReader('/tmp/ig_takeover_20261006/DATA_SLICE','e1e2327070b2f9fc3bb548fc58a6961b70a0b885eb857d040ce0dcd5bfe4a351');p.verify()
slices=json.load(open(B/'ADMITTED_SOURCE_SLICES.json'));prev=json.load(open(R/'node_bindings0144/VALIDATION.json'));out=[];receipts=[]
for tid,s in zip([0,1830,3943],slices):
 path=next((R/'attachments').rglob(s['archive']['file_name']));raw=path.read_bytes();assert len(raw)==s['archive']['size_bytes'] and sha(raw)==s['archive']['sha256']
 z=zipfile.ZipFile(path);raw=z.read('ROOT.json');assert sha(raw)==s['scientific_root_sha256'];root=json.loads(raw);ref=next(x['content_ref'] for x in root['shards'] if x['first_tid']<=tid<=x['last_tid']);raw=z.read('content/'+ref['sha256']+'.blob');assert len(raw)==int(ref['size_bytes']) and sha(raw)==ref['sha256'];sh=json.loads(raw);row=next(x for x in sh['rows'] if x['exact_tid']==tid)
 receipts.append({'tid':tid,'archive':s['archive'],'root_sha256':s['scientific_root_sha256'],'selected_shard_ref':ref})
 (B/('ADMITTED_ROW_'+str(tid)+'.json')).write_text(json.dumps(row,indent=2))
 for old in [x for x in prev['primitive_bindings'] if x['historical_tid']==tid]:
  want=old['historical_record_sha256'];hits=[]
  for e in row['events']:
   rec=e['record'];norm=(rec[0],tuple(sorted(tuple(x) for x in rec[1])),rec[2],(rec[3],),rec[4])
   if sha(repr(norm).encode())!=want:continue
   assert event_id(rec)==e['event_id']
   for w in e['realizations']:
    v=p._choice_record(row['source_sids'],w['option_indices']);assert v and v[0]==rec and v[1]==w['target_sids'] and v[2]==w['target_tid']
   hits.append(e)
  assert hits
  assert sorted(w['option_indices'] for e in hits for w in e['realizations'])==sorted(x['option_indices'] for x in old['exact_choice_witnesses'])
  out.append({'historical_tid':tid,'historical_rank':old['historical_rank'],'historical_record_sha256':want,'admitted_j3_id':row['j3_id'],'scientific_root_sha256':s['scientific_root_sha256'],'events':hits,'status':'EXACT_ADMITTED_EVENT_AND_ALL_WITNESSES_BOUND'})
assert len(out)==9
res={'schema':'IG_NODE_PRIMITIVE_ADMITTED_EVENT_BINDINGS_V1','master_release':'MASTER_DATA_V1_0141','catalog_sha256':sha(catalog),'primitives_bound':len(out),'ordered_events_bound':sum(len(x['events']) for x in out),'option_witnesses_bound':sum(len(e['realizations']) for x in out for e in x['events']),'bindings':out,'archive_readbacks':receipts,'new_generation':False,'new_admission':False}
(B/'EVENT_BINDINGS.json').write_text(json.dumps(res,indent=2));print({k:v for k,v in res.items() if k not in ['bindings','archive_readbacks']})
