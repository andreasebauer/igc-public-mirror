from pathlib import Path
import json,hashlib,struct,collections
R=Path.cwd();B=R/'o7_readiness0167';sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.loads(p.read_bytes())
refs=load(B/'RECOVERED_SOURCE_REFS.json')
for ref in refs['archives']:
 for n,x in ref['recovered_members'].items():
  b=(B/'sources'/Path(ref['archive']).stem/n).read_bytes();assert len(b)==x['bytes'] and sha(b)==x['sha256']
rows=load(B/'SAVED_STATE_ROWS.json');keys=set()
for row in rows:
 r=row['record'];e=r['edges'];assert len(e)==row['rank']
 assert sha(b'IG-E7-STATE-v1|'+b''.join(struct.pack('>14I',*x) for x in sorted(e)))==row['digest']
 assert sha(json.dumps(r,sort_keys=True,separators=(',',':')).encode())==row['payload_sha256']
 k=(row['lane'],row['rank'],row['digest']);assert k not in keys;keys.add(k)
v=load(B/'RECOVERY_VALIDATION.json');assert len(rows)==v['unique_states']==74
assert sum(len(r['record']['edges']) for r in rows)==v['typed_E7_edges']==192
assert sha((R/'o6_integrate0166/CATALOG_0147.json').read_bytes())==v['catalog_sha256']
assert not v['o7_graduated'] and not v['automatic_o8_authorized'] and not v['new_admissions']
print('PASS: source readback,74 literal O7 states,192 E7 edges,catalog0147; binding validation remains pending')
