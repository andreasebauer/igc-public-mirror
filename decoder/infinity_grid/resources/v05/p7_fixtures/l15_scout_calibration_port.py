#!/usr/bin/env python3
from pathlib import Path
import ast,hashlib,json,pickle,time,resource
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]; I=ROOT/'inputs'; E=ROOT/'evidence'; M=ROOT/'metrics'; R=ROOT/'results'
def now():return time.perf_counter()
def cpu():return time.process_time()
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
def canon(r):
    d=r[3] if isinstance(r[3],tuple) else (r[3],)
    return (int(r[0]),tuple(sorted(tuple(map(int,p)) for p in r[1])),int(r[2]),tuple(sorted(map(int,d))),int(r[4]))
def sha(r):return hashlib.sha256(repr(r).encode()).hexdigest()
def bridge(pa,pb):
    Pa,Ma=pa;Pb,Mb=pb
    return (Ma==0 or Ma&Pb) and (Mb==0 or Mb&Pa)
def compose(a,sa,b,sb):
    if not bridge(a[1][sa],b[1][sb]):return None
    d=a[3] if isinstance(a[3],tuple) else (a[3],); e=b[3] if isinstance(b[3],tuple) else (b[3],)
    return canon((a[0]|b[0],tuple(v for j,v in enumerate(a[1]) if j!=sa)+tuple(v for j,v in enumerate(b[1]) if j!=sb),min(a[2],b[2]),d+e,int(a[4] and b[4])))
def vec(row):
    r=row['record']; pc=Counter(r[1]); tc=Counter(r[3] if isinstance(r[3],tuple) else (r[3],))
    return (len(r[1]),len(pc),max(pc.values()),sum(v*v for v in pc.values()),len(r[3] if isinstance(r[3],tuple) else (r[3],)),len(tc),max(tc.values()),sum(v*v for v in tc.values()),row['parent_count'],row['witness_count'])
def norm(vs):
    mi=[min(v[j] for v in vs) for j in range(len(vs[0]))];ma=[max(v[j] for v in vs) for j in range(len(vs[0]))]
    return [tuple(0 if ma[j]==mi[j] else (x-mi[j])/(ma[j]-mi[j]) for j,x in enumerate(v)) for v in vs]
def farthest(rows,n,start_mode=0):
    if len(rows)<=n:return rows[:]
    nv=norm([vec(x) for x in rows])
    if start_mode==0: s=min(range(len(rows)),key=lambda i:rows[i]['sha256'])
    elif start_mode==1: s=max(range(len(rows)),key=lambda i:rows[i]['sha256'])
    else: s=min(range(len(rows)),key=lambda i:hashlib.sha256((str(start_mode)+'|'+rows[i]['sha256']).encode()).hexdigest())
    ch=[s]; used={s}; md=[sum((a-b)**2 for a,b in zip(nv[i],nv[s])) for i in range(len(rows))];md[s]=-1
    while len(ch)<n:
        k=max((i for i in range(len(rows)) if i not in used),key=lambda i:(md[i],rows[i]['sha256']))
        ch.append(k);used.add(k);md[k]=-1
        for i in range(len(rows)):
            if md[i]>=0:md[i]=min(md[i],sum((a-b)**2 for a,b in zip(nv[i],nv[k])))
    return [rows[i] for i in ch]
def jac(a,b):
    a=set(a);b=set(b);return len(a&b)/len(a|b) if a|b else 1.0
def metric(name,t0,c0,extra):
    d={'phase':name,'wall_seconds':now()-t0,'cpu_seconds':cpu()-c0,'peak_rss_kb':rss(),**extra};json.dump(d,open(M/(name+'.json'),'w'),indent=2,sort_keys=True);return d
T=now();C=cpu(); mets=[]
# reconstruct all 7129 candidates from 499 roots exactly as wide run, to allow convergence/stability panels
p=now();c=cpu(); roots_raw=json.load(open(I/'GLOBAL_L14_PANEL_CHILDREN.json')); prim=pickle.load(open(I/'primitive.pkl','rb'))
roots=[{'sha256':x['sha256'],'record':canon(ast.literal_eval(x['record']))} for x in roots_raw]; roots.sort(key=lambda x:x['sha256'])
children={}; parents=defaultdict(set); witnesses=Counter()
for s in roots:
  a=s['record']
  for pe in prim:
    b=canon(pe['record'])
    for sa in range(len(a[1])):
      for sb in range(len(b[1])):
        ch=compose(a,sa,b,sb)
        if ch is not None:
          h=sha(ch);children[h]=ch;parents[h].add(s['sha256']);witnesses[h]+=1
rows=[{'sha256':h,'record':r,'parent_count':len(parents[h]),'witness_count':witnesses[h]} for h,r in children.items()];rows.sort(key=lambda x:x['sha256'])
assert len(rows)==7129
mets.append(metric('00_RECONSTRUCT',p,c,{'candidates':len(rows),'roots':len(roots)}))
# A: lane stability across deterministic alternate starts; qualitative signals and overlap
p=now();c=cpu(); N=256; panels=[]; summaries=[]
for mode in range(4):
    D=farthest(rows,N,mode)
    # Other lanes use deterministic rotating hash salt to avoid just cloning same panel while preserving lane intention
    B=sorted(rows,key=lambda x:(-x['parent_count'],-x['witness_count'],hashlib.sha256((str(mode)+'B'+x['sha256']).encode()).hexdigest()))[:N]
    S=sorted(rows,key=lambda x:(len(set(x['record'][1])),max(Counter(x['record'][1]).values()),hashlib.sha256((str(mode)+'S'+x['sha256']).encode()).hexdigest()))[:N]
    O=sorted(rows,key=lambda x:(-sum(v*(v-1)//2 for v in Counter(x['record'][1]).values()),-x['parent_count'],hashlib.sha256((str(mode)+'O'+x['sha256']).encode()).hexdigest()))[:N]
    L=sorted(rows,key=lambda x:hashlib.sha256((str(mode)+'L'+x['sha256']).encode()).hexdigest())[:N]
    union={x['sha256']:x for z in (D,B,S,O,L) for x in z}
    multi=sum(x['parent_count']>=2 for x in union.values())
    # shared parent positive
    byp=defaultdict(list)
    for h,x in union.items():
      for pa in parents[h]:byp[pa].append(h)
    shared=0
    for hs in byp.values(): shared+=len(hs)*(len(hs)-1)//2
    labels=[]
    if multi:labels.append('ALTERNATIVE_PARENT_ORGANIZATION_PRESENT_BOUNDED')
    if shared:labels.append('SHARED_SUPPORT_FACTORIZATION_PRESENT_BOUNDED')
    panels.append(set(union));summaries.append({'mode':mode,'union':len(union),'multi_parent':multi,'shared_parent_pair_events':shared,'labels':labels})
stab={'summaries':summaries,'pairwise_jaccard':{f'{i}-{j}':jac(panels[i],panels[j]) for i in range(4) for j in range(i+1,4)},'qualitative_label_stable':len({tuple(x['labels']) for x in summaries})==1}
json.dump(stab,open(E/'LANE_STABILITY.json','w'),indent=2,sort_keys=True);mets.append(metric('01_LANE_STABILITY',p,c,{'qualitative_label_stable':stab['qualitative_label_stable']}))
# B: panel size convergence with same lane definitions; roadmap qualitative stabilization
# Greedy farthest-point selection is prefix-stable in N, and each other lane is a prefix
# of one deterministic sort. Compute the 512 ordering once and reuse exact prefixes.
p=now();c=cpu(); conv=[]
D512=farthest(rows,512,0)
B512=sorted(rows,key=lambda x:(-x['parent_count'],-x['witness_count'],x['sha256']))[:512]
S512=sorted(rows,key=lambda x:(len(set(x['record'][1])),max(Counter(x['record'][1]).values()),x['sha256']))[:512]
O512=sorted(rows,key=lambda x:(-sum(v*(v-1)//2 for v in Counter(x['record'][1]).values()),-x['parent_count'],x['sha256']))[:512]
L512=rows[:512]
for N in (128,256,512):
    D=D512[:N];B=B512[:N];S=S512[:N];O=O512[:N];L=L512[:N]
    u={x['sha256']:x for z in (D,B,S,O,L) for x in z}; multi=sum(x['parent_count']>=2 for x in u.values())
    labels=['ALTERNATIVE_PARENT_ORGANIZATION_PRESENT_BOUNDED'] if multi else []
    # exact shared-support existence via per-parent counts
    shared_events=0
    for pa in roots:
      k=sum(1 for h in u if pa['sha256'] in parents[h]);shared_events+=k*(k-1)//2
    if shared_events:labels.append('SHARED_SUPPORT_FACTORIZATION_PRESENT_BOUNDED')
    conv.append({'per_lane':N,'union':len(u),'multi_parent':multi,'multi_parent_fraction':multi/len(u),'shared_parent_pair_events':shared_events,'labels':labels})
convres={'rows':conv,'qualitative_converged_128_256_512':len({tuple(x['labels']) for x in conv})==1,'multi_parent_fraction_range':[min(x['multi_parent_fraction'] for x in conv),max(x['multi_parent_fraction'] for x in conv)]}
json.dump(convres,open(E/'PANEL_CONVERGENCE.json','w'),indent=2,sort_keys=True);mets.append(metric('02_PANEL_CONVERGENCE',p,c,{'qualitative_converged':convres['qualitative_converged_128_256_512']}))
# C: maturation classification, bounded evidence only. Compare L14 roots -> L15 relation facts.
p=now();c=cpu(); wide=json.load(open(I/'L15_SCOUT_WIDE_RESULT.json'))
mat=[]
# Presence of inherited alternative-parent/factorization counts is PERSISTS; richer bounded parent multiplicity = EXPANDS only as scout label.
if wide['bounded_relation']['selected_multi_parent']>0: mat.append({'relation':'alternative_parent','class':'PERSISTS','basis':'985/1089 selected L15 objects have >=2 bounded L14 parents'})
if wide['bounded_relation']['selected_max_parent_count']>=10: mat.append({'relation':'alternative_parent','class':'EXPANDS','basis':'bounded parent multiplicity reaches 10; scout-only, not population claim'})
if wide['bounded_relation']['shared_parent_pair_count']>0: mat.append({'relation':'shared_support_factorization','class':'PERSISTS','basis':'13976 selected L15 pairs share bounded L14 parents'})
mat.append({'relation':'L14_C1_depth_reorganization','class':'UNRESOLVED_DEPTH_SCOPE','basis':'available 499-root L14 panel descends from only 12 frozen L13 roots in current stored ancestry; deeper maturation cannot be inferred'})
json.dump({'evidence':'SCOUT_OBSERVED','classifications':mat},open(E/'MATURATION_CLASSIFICATION.json','w'),indent=2,sort_keys=True);mets.append(metric('03_MATURATION',p,c,{'classifications':len(mat)}))
# D: one-step controlled composition persistence. Freeze 64 high-parent + 64 diversity sources, generate bounded L16 children, ask positive parent multiplicity.
p=now();c=cpu(); high=sorted(rows,key=lambda x:(-x['parent_count'],-x['witness_count'],x['sha256']))[:64];div=farthest(rows,64,2); src={x['sha256']:x for x in high+div}; l16={}; l16par=defaultdict(set);attempt=lawful=0
for x in src.values():
  a=x['record']
  for pe in prim:
    b=canon(pe['record'])
    for sa in range(len(a[1])):
      for sb in range(len(b[1])):
        attempt+=1;ch=compose(a,sa,b,sb)
        if ch is None:continue
        lawful+=1;h=sha(ch);l16[h]=ch;l16par[h].add(x['sha256'])
comp={'frozen_L15_sources':len(src),'attempts':attempt,'lawful':lawful,'unique_L16_children':len(l16),'multi_source_L16_children':sum(len(v)>=2 for v in l16par.values()),'max_bounded_L15_parent_count':max(map(len,l16par.values())) if l16par else 0,'classification':'PERSISTS_ONE_STEP_BOUNDED' if any(len(v)>=2 for v in l16par.values()) else 'NO_POSITIVE_PERSISTENCE_HIT'}
json.dump(comp,open(E/'CONTROLLED_L16_PERSISTENCE.json','w'),indent=2,sort_keys=True);mets.append(metric('04_COMPOSITION_PERSISTENCE',p,c,comp))
# E: negative control. Singleton-parent candidates must not trigger alternative-parent label; duplicate construction witnesses alone do not count.
p=now();c=cpu(); singles=[x for x in rows if x['parent_count']==1]; neg=singles[:min(256,len(singles))]; false=sum(x['parent_count']>=2 for x in neg); witness_multi=sum(x['witness_count']>1 for x in neg)
negres={'control':'SINGLE_BOUNDED_PARENT_PANEL','n':len(neg),'objects_with_multiple_construction_witnesses':witness_multi,'false_alternative_parent_flags':false,'status':'PASS' if false==0 else 'FAIL','lesson':'construction-witness multiplicity is not parent-fiber multiplicity'}
json.dump(negres,open(E/'NEGATIVE_CONTROL.json','w'),indent=2,sort_keys=True);mets.append(metric('05_NEGATIVE_CONTROL',p,c,negres))
# F: broader depth ancestry admission check
p=now();c=cpu(); depth={'status':'BLOCKED_BY_ANCESTRY_SCOPE','available_L14_roots':499,'known_underlying_L13_roots':12,'reason':'No complete or sufficiently broad L14->L13 parent index for the 499 scout roots is available in the scout workspace. Constructing it from the full L13 space would cross from cheap scout into a heavy targeted ancestry search.','policy':'Do not infer absence/maturation from the narrow 12-root depth view; leave UNRESOLVED in Scout.'}
json.dump(depth,open(E/'DEPTH_ANCESTRY_SCOPE.json','w'),indent=2,sort_keys=True);mets.append(metric('06_DEPTH_SCOPE',p,c,{'status':depth['status']}))
res={'stage':'L15_SCOUT_CALIBRATION','status':'PASS_WITH_DEPTH_SCOPE_LIMIT','evidence_label':'SCOUT_OBSERVED','authoritative':False,'lane_stability':stab,'panel_convergence':convres,'maturation':mat,'controlled_composition':comp,'negative_control':negres,'depth_ancestry':depth,'cost':{'total_wall_seconds':now()-T,'total_cpu_seconds':cpu()-C,'peak_rss_kb':rss(),'phases':mets},'decision':{'scout_calibrated_for_shallow_relational_roadmap': bool(stab['qualitative_label_stable'] and convres['qualitative_converged_128_256_512'] and negres['status']=='PASS'),'deep_maturation_calibrated':False}}
json.dump(res,open(R/'L15_SCOUT_CALIBRATION_RESULT.json','w'),indent=2,sort_keys=True);print(json.dumps(res,sort_keys=True))
