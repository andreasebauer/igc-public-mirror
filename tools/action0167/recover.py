from pathlib import Path
import zipfile,io,json,hashlib,struct,collections
R=Path.cwd(); B=R/'o7_readiness0167';B.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
outer=R/'project_sources/18-OScout-4-.zip';z=zipfile.ZipFile(outer)
markers=['CLO01_PARTIAL','POSTRUN_ANALYSIS','v0.2.4_C04','v0.2_PHASE_C_QUALIFICATION','FAST_PARALLEL_NEW_CHAT']
refs=[];rows={};occ=[]
for n in z.namelist():
 if '__MACOSX' in n or not n.endswith('.zip') or not any(s in n for s in markers):continue
 raw=z.read(n);q=zipfile.ZipFile(io.BytesIO(raw)); ref={'archive':n,'sha256':sha(raw),'recovered_members':{}}
 for info in q.infolist():
  if info.is_dir() or info.filename.endswith('.zip'):continue
  p=Path(info.filename);assert not p.is_absolute() and '..' not in p.parts
  b=q.read(info);dest=B/'sources'/Path(n).stem/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
  ref['recovered_members'][info.filename]={'sha256':sha(b),'bytes':len(b)}
  if p.suffix!='.json':continue
  d=json.loads(b)
  result=d.get('result',d) if isinstance(d,dict) else {}
  if not isinstance(result,dict) or 'beam' not in result:continue
  lane='HET4' if 'HET4_m7_' in p.name else 'HOM6' if 'HOM6_m7_' in p.name else None
  if not lane:continue
  rank=int(p.name.split('_m7_')[1].split('_')[0].split('.')[0])
  if 'payload_sha256' in d:
   core={k:v for k,v in d.items() if k!='payload_sha256'}
   assert sha(json.dumps(core,sort_keys=True,separators=(',',':')).encode())==d['payload_sha256'],p
  for row in result['beam']:
   edges=row['edges'];assert len(edges)==rank
   assert all(len(e)==14 and all(isinstance(a,int) and a>=0 for a in e) for e in edges)
   h=sha(b'IG-E7-STATE-v1|'+b''.join(struct.pack('>14I',*e) for e in sorted(edges)))
   assert h==row['digest'],(p,h,row['digest'])
   key=(lane,rank,h); literal=sha(json.dumps(row,sort_keys=True,separators=(',',':')).encode())
   if key in rows:assert rows[key]['payload_sha256']==literal
   else:rows[key]={'lane':lane,'rank':rank,'digest':h,'payload_sha256':literal,'record':row}
   occ.append({'key':list(key),'archive':n,'member':info.filename,'member_sha256':sha(b)})
 refs.append(ref)
dump(B/'RECOVERED_SOURCE_REFS.json',{'outer_archive':str(outer.relative_to(R)),'outer_sha256':sha(outer.read_bytes()),'archives':refs})
dump(B/'SAVED_STATE_ROWS.json',list(rows.values()));dump(B/'SAVED_STATE_OCCURRENCES.json',occ)
progress=next((B/'sources').rglob('CLO-01_PROGRESS.json'));pr=json.loads(progress.read_bytes())
selections=list((B/'sources').rglob('CLO-01_SELECTION_SOURCES/*.json'))
for p in selections:
 d=json.loads(p.read_bytes());core={k:v for k,v in d.items() if k!='payload_sha256'}
 assert sha(json.dumps(core,sort_keys=True,separators=(',',':')).encode())==d['payload_sha256'],p
summary={'status':'SAVED_O7_LITERAL_SCOPE_RECOVERED_BINDING_VALIDATION_PENDING','master_release':'MASTER_DATA_V1_0147','catalog_sha256':sha((R/'o6_integrate0166/CATALOG_0147.json').read_bytes()),'unique_states':len(rows),'state_occurrences':len(occ),'lane_rank_counts':dict(collections.Counter(f'{k[0]}:{k[1]}' for k in rows)),'typed_E7_edges':sum(len(d['record']['edges']) for d in rows.values()),'partial_selection_records':len(selections),'partial_progress_snapshot':pr,'verified':'recovered source bytes; checkpoint payload hashes; binary E7 state digests; duplicate literal row agreement; partial selection payload hashes','not_yet_verified':['O6 owner order and canonical resource bindings','E7 endpoint compatibility/capacity and component accounting','full final survivor/profile coverage','closeout certification debts D1-D6'],'generation_calls':0,'new_admissions':0,'o7_graduated':False,'automatic_o8_authorized':False,'full_L0_G8_complete':False,'next_scope':'WP5_SAVED_O7_RECOVERED_SCOPE_BINDING_VALIDATION'}
dump(B/'RECOVERY_VALIDATION.json',summary);print(json.dumps(summary,indent=2))
