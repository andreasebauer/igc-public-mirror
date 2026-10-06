from pathlib import Path
import json,hashlib,gzip,importlib.util
D=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest();m=json.load(open(D/'MANIFEST.json'))
for n,v in m.items():
 b=(D/n).read_bytes();assert len(b)==v['bytes'] and sha(b)==v['sha256']
B=D/'o3_readiness0153';L=B/'frozen_o3_launch';sp=importlib.util.spec_from_file_location('frozen_sc',L/'code/vendor/scout2_relational_longitudinal_v1_1.py');sc=importlib.util.module_from_spec(sp);sp.loader.exec_module(sc);dep=json.load(open(B/'RECOVERY_DEPENDENCIES.json'));count=0;prior={};links=0
for ref in dep['current_panel_payloads']:
 k=ref['k'];x=json.load(gzip.open(D/ref['path']));assert x['k']==k;rows={}
 for row in x['selected']:
  st=tuple(tuple(s) for s in row['states']);ed=tuple(tuple(e) for e in row['edges']);assert sc.capacity_valid(st,ed) and sc.exact_key(st,ed)==row['exact_key'] and sc.role_info(st,ed)[0]==row['role_hash'];assert all(len(s)==16 and all(type(a)==int and a>=0 for a in s) and sum(s[:7])==sum(s[7:])+2 for s in st);assert all(0<=u<len(st) and 0<=v<len(st) and u!=v and 0<=a<7 and 0<=b<7 and sc.bridge(sc.PORTS[a],sc.PORTS[b]) for u,a,v,b in ed);assert row['exact_key'] not in rows;rows[row['exact_key']]=row;count+=1
 if k>0 and k-1 in prior:
  for row in rows.values():
   for h in row['parents']:assert h in prior[k-1];links+=1
 prior[k]=rows
v=json.load(open(B/'READINESS_VALIDATION.json'));assert count==v['available_carrier_occurrences_verified'] and links==v['available_parent_links_resolved'];print({'status':'PASS_COMPACT_RECOVERY','manifest_files':len(m),'panels':len(prior),'carriers':count,'available_parent_links':links,'missing_panels':len(v['missing_k']),'full_readiness':False,'generation_calls':0})
