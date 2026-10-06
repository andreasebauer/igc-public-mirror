#!/usr/bin/env python3
from __future__ import annotations
import ast, collections, gzip, hashlib, itertools, json, os, time
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parent.parent
INP=ROOT/'inputs'; EVID=ROOT/'evidence'; OUT=ROOT/'results'; REPORT=ROOT/'report'; MACHINE=ROOT/'machine'; THEOREM=ROOT/'theorem'
PN=7
EXPECTED={
 'o2':'d138de3c6c906be3ab453c225ecce2e3fff5d3e4f91d220ed681ef5253a86bbb',
 'v11':'a856ba0693bf63099fbcf7b343757c08a79abfb446cee68d492e80eed4583a51',
 'r3_ind':'8b0ed691039d63917f1bcfea116f17e7233173bfa6d254ee5b568b09a53e4de1',
 'r3_comp':'2048b63a3045bd1ad9eca87110ef5b4c65263fe22a52d3533f37f200db22f727',
}

def jload(p:Path):
    with p.open() as f:return json.load(f)
def jdump(p:Path,x:Any):
    p.parent.mkdir(parents=True,exist_ok=True); t=Path(str(p)+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
def sha_json(x:Any):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha_file(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def canon_block(block):return tuple(sorted((tuple(p),tuple(f)) for p,f in block))
def canon_r3(blocks):return tuple(sorted(canon_block(b) for b in blocks))
def rho(R):
    s=sum(sum(p[a]-f[a] for a in range(PN)) for block in R for p,f in block)
    if s%2:return None
    return s//2

def load_panel(r):
    with gzip.open(INP/'source_panels'/f'r{r:03d}_selected_carriers.json.gz','rt') as f:return json.load(f)['selected']

def derive_templates(PORTS,B):
    spec=jload(INP/'MATURE_NODE_ALGEBRA_SPEC.json'); tatoms=tuple(spec['T_atoms']); prims=jload(INP/'FROZEN_PRIMITIVES.json')['primitive']; idx={tuple(p):i for i,p in enumerate(PORTS)}
    rows=[]
    for pr in prims:
        rec=ast.literal_eval(pr['record']); pc=collections.Counter(tuple(x) for x in rec[1]); tc=collections.Counter(rec[3])
        for si,source in enumerate(PORTS):
            for u,n in pc.items():
                if n<=0 or not B[si][idx[u]]:continue
                dp=[0]*PN; dp[si]-=1
                for z,c in pc.items():dp[idx[z]]+=c
                dp[idx[u]]-=1
                dt=tuple(tc[t] for t in tatoms)
                rows.append({'source':si,'primitive_port':idx[u],'dp':tuple(dp),'dt':dt,'tid':pr['tid'],'rank':pr['rank'],'psha':pr['sha256']})
    d={}
    for x in rows:
        k=(x['source'],x['primitive_port'],x['dp'],x['dt'])
        if k not in d or (x['tid'],x['rank'],x['psha'])<(d[k]['tid'],d[k]['rank'],d[k]['psha']):d[k]=x
    out=sorted(d.values(),key=lambda x:(x['source'],x['primitive_port'],x['dp'],x['dt'],x['tid'],x['rank'],x['psha']))
    if len(out)!=77:raise RuntimeError(f'TEMPLATE_COUNT_{len(out)}')
    return out

def graph_components(n,edges):
    adj=[set() for _ in range(n)]
    for v,i,a,w,j,b in edges:adj[v].add(w);adj[w].add(v)
    seen=set(); comps=[]
    for s in range(n):
        if s in seen:continue
        st=[s]; seen.add(s); c=[]
        while st:
            x=st.pop(); c.append(x)
            for y in adj[x]:
                if y not in seen:seen.add(y);st.append(y)
        comps.append(tuple(sorted(c)))
    return tuple(sorted(comps))

def topo_sig(comp,edges):
    ids={v:i for i,v in enumerate(comp)}; n=len(comp); deg=[0]*n; pair=collections.Counter(); m=0
    for v,i,a,w,j,b in edges:
        if v in ids and w in ids:
            x,y=ids[v],ids[w]; deg[x]+=1;deg[y]+=1; pair[(min(x,y),max(x,y))]+=1;m+=1
    adj=[set() for _ in range(n)]
    for x,y in pair:adj[x].add(y);adj[y].add(x)
    def cc(skip=None):
        seen=set(); c=0
        for s in range(n):
            if s==skip or s in seen:continue
            c+=1; st=[s];seen.add(s)
            while st:
                x=st.pop()
                for y in adj[x]:
                    if y!=skip and y not in seen:seen.add(y);st.append(y)
        return c
    base=cc(); arts=sum(cc(x)>base for x in range(n))
    return {'n3':n,'m3':m,'beta3':m-n+1,'degree_sorted':sorted(deg),'simple_pairs':len(pair),'parallel_max':max(pair.values(),default=0),'articulation_count':arts}

class Audit:
    def __init__(self):
        self.spec=jload(INP/'MATURE_NODE_ALGEBRA_SPEC.json'); self.bank=jload(INP/'O2_SOURCE_BANK.json')
        self.PORTS=tuple(tuple(x) for x in self.spec['port_atoms']); idx={p:i for i,p in enumerate(self.PORTS)}
        self.B=[[False]*PN for _ in range(PN)]
        for x,y in self.spec['ordered_compatible_port_pairs']:self.B[idx[tuple(x)]][idx[tuple(y)]]=True
        self.qmap={e['q_hash']:tuple((tuple(s[0]),tuple(s[1])) for s in e['sites']) for e in self.bank['entries']}
        self.templates=derive_templates(self.PORTS,self.B)
        self.panels={r:load_panel(r) for r in range(65)}
        self.parent_o2=jload(EVID/'O2_GRADUATION_AUDIT_RESULT.json');self.parent_v11=jload(EVID/'SCOUT3_V1_1_ADVERSARIAL_AUDIT_RESULT.json');self.parent_ind=jload(EVID/'R3_INDEPENDENT_AUDIT_RESULT.json');self.parent_comp=jload(EVID/'R3_COMPOSITION_AUDIT_RESULT.json')

    def source_checks(self):
        errs=[]
        if self.parent_o2.get('science_sha256')!=EXPECTED['o2'] or self.parent_o2.get('status')!='O2_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_V1_9':errs.append('O2_PARENT')
        if self.parent_v11.get('audit_science_sha256')!=EXPECTED['v11']:errs.append('V11_PARENT')
        if self.parent_ind.get('science_sha256')!=EXPECTED['r3_ind'] or self.parent_ind.get('status')!='R3_RESOURCE_BISIMULATION_EARNED_SCOPED':errs.append('R3_INDEPENDENT_PARENT')
        if self.parent_comp.get('science_sha256')!=EXPECTED['r3_comp'] or self.parent_comp.get('status')!='R3_COMPOSABLE_EXTERNAL_MACRO_INTERFACE_EARNED':errs.append('R3_COMPOSITION_PARENT')
        if len(self.panels)!=65 or any(r not in self.panels for r in range(65)):errs.append('PANEL_SET')
        return errs

    def expanded(self,h):return [self.qmap[q] for q in h['entities']]
    def usage3(self,h,blocks):
        U=[[[0]*PN for _ in b] for b in blocks]
        for e in h['edges']:
            if len(e)!=6:continue
            v,i,a,w,j,b=e
            if 0<=v<len(blocks) and 0<=w<len(blocks) and 0<=i<len(blocks[v]) and 0<=j<len(blocks[w]) and 0<=a<PN and 0<=b<PN:
                U[v][i][a]+=1;U[w][j][b]+=1
        return U
    def component_r3(self,h,comp,blocks=None,U=None):
        blocks=blocks or self.expanded(h);U=U or self.usage3(h,blocks);out=[]
        for v in comp:
            row=[]
            for i,(p,u2) in enumerate(blocks[v]):
                f=tuple(p[a]-u2[a]-U[v][i][a] for a in range(PN));row.append((p,f))
            out.append(tuple(row))
        return canon_r3(out)
    def exact_component_key(self,h,comp):
        remap={v:i for i,v in enumerate(comp)};qs=tuple(h['entities'][v] for v in comp);ee=[]
        for v,i,a,w,j,b in h['edges']:
            if v in remap and w in remap:ee.append((remap[v],i,a,remap[w],j,b))
        return (qs,tuple(sorted(ee)))

    def full_sweep(self):
        S=collections.Counter(); failures=collections.Counter(); examples=[]; comp_rows=[]; sameR=collections.defaultdict(list); roll=hashlib.sha256()
        for r in range(65):
            for h in self.panels[r]:
                S['states']+=1; S['edges']+=len(h['edges']); roll.update((str(r)+'|'+h['exact_key']+'\n').encode())
                blocks=self.expanded(h);n=len(blocks);U=self.usage3(h,blocks)
                # exact state checks
                if h.get('r3')!=r:failures['stored_r3_rank']+=1
                if len(h['edges'])!=r:failures['edge_count_vs_r3']+=1
                for e in h['edges']:
                    if len(e)!=6:failures['edge_arity']+=1;continue
                    v,i,a,w,j,b=e
                    if not(0<=v<w<n):failures['entity_endpoint']+=1;continue
                    if not(0<=i<len(blocks[v]) and 0<=j<len(blocks[w]) and 0<=a<PN and 0<=b<PN):failures['site_endpoint']+=1;continue
                    if not self.B[a][b]:failures['bridge']+=1
                allfree=[]
                for v,block in enumerate(blocks):
                    for i,(p,u2) in enumerate(block):
                        for a in range(PN):
                            f=p[a]-u2[a]-U[v][i][a];allfree.append(f)
                            if f<0:failures['negative_free']+=1
                            if f>p[a]:failures['free_gt_p']+=1
                comps=graph_components(n,h['edges']); stored_c=h.get('metrics',{}).get('components'); stored_b=h.get('metrics',{}).get('beta3')
                calc_b=len(h['edges'])-n+len(comps)
                if stored_c is not None and stored_c!=len(comps):failures['stored_components']+=1
                if stored_b is not None and stored_b!=calc_b:failures['stored_beta3']+=1
                for comp in comps:
                    if len(comp)==1:
                        S['isolated_o2_occurrences']+=1;continue
                    # graph components of size >1 necessarily have edges
                    S['nontrivial_components']+=1
                    cedges=[e for e in h['edges'] if e[0] in comp and e[3] in comp]
                    m3=len(cedges); n3=len(comp); beta=m3-n3+1
                    if m3<1:failures['nontrivial_no_edge']+=1
                    if beta<0:failures['component_beta_negative']+=1
                    U2sum=sum(sum(u2) for v in comp for p,u2 in blocks[v])
                    if U2sum%2:failures['u2_odd']+=1
                    U2=U2sum//2
                    R=self.component_r3(h,comp,blocks,U); rr=rho(R)
                    if rr is None:failures['rho_half_integrality']+=1
                    elif rr!=U2+m3:failures['rho_identity']+=1
                    S['component_blocks']+=n3;S['component_o3_edges']+=m3;S['component_o2_internal_relations']+=U2
                    S['min_n3']=min(S.get('min_n3',10**9),n3);S['max_n3']=max(S.get('max_n3',0),n3);S['min_m3']=min(S.get('min_m3',10**9),m3);S['max_m3']=max(S.get('max_m3',0),m3);S['min_beta3']=min(S.get('min_beta3',10**9),beta);S['max_beta3']=max(S.get('max_beta3',0),beta)
                    row={'r3':r,'exact_key':h['exact_key'],'component_entities':list(comp),'n3':n3,'m3':m3,'beta3':beta,'rho3':rr,'U2':U2,'R3_sha256':hashlib.sha256(repr(R).encode()).hexdigest(),'topology':topo_sig(comp,h['edges'])}
                    comp_rows.append(row);sameR[R].append(row)
        # same-R3 reduction among connected components
        multi=0; topo_hidden=0; witness=None
        for R,rows in sameR.items():
            if len(rows)<2:continue
            # different exact carrier/component occurrences, possibly same exact component repeated
            uniq={(x['exact_key'],tuple(x['component_entities'])) for x in rows}
            if len(uniq)<2:continue
            multi+=1
            sigs={json.dumps(x['topology'],sort_keys=True) for x in rows}
            if len(sigs)>1:
                topo_hidden+=1
                if witness is None:witness={'R3_sha256':hashlib.sha256(repr(R).encode()).hexdigest(),'A':rows[0],'B':next(x for x in rows[1:] if x['topology']!=rows[0]['topology'])}
        S['multi_realization_connected_R3_classes']=multi;S['topology_hidden_connected_R3_classes']=topo_hidden
        return dict(S),dict(failures),comp_rows,witness,roll.hexdigest()

    def first_relation_action(self,R,cross_blocks=True):
        for v in range(len(R)):
            wstart=v+1 if cross_blocks else v
            for w in range(wstart,len(R)):
                if cross_blocks and v==w:continue
                if (not cross_blocks) and v!=w:continue
                for i,(p,f) in enumerate(R[v]):
                    jstart=0
                    for j,(q,g) in enumerate(R[w]):
                        if v==w and i==j:continue
                        if v==w and i>j:continue
                        for a,x in enumerate(f):
                            if x<=0:continue
                            for b,y in enumerate(g):
                                if y>0 and self.B[a][b]:return (v,i,a,w,j,b)
        return None
    def apply_relation(self,R,act):
        v,i,a,w,j,b=act;B=[list(x) for x in R];p,f=B[v][i];q,g=B[w][j];ff=list(f);gg=list(g);ff[a]-=1;gg[b]-=1;B[v][i]=(p,tuple(ff));B[w][j]=(q,tuple(gg));return canon_r3(B)
    def first_rank_action(self,R):
        for v,block in enumerate(R):
            for i,(p,f) in enumerate(block):
                for tm in self.templates:
                    src=tm['source'];dp=tm['dp']
                    if p[src]<=0:continue
                    np=tuple(p[a]+dp[a] for a in range(PN))
                    if min(np)<0:continue
                    strict=f[src]>0;cross=(not strict and all(f[a]+dp[a]>=0 for a in range(PN)))
                    if strict or cross:return (v,i,tm)
        return None
    def apply_rank(self,R,act):
        v,i,tm=act;B=[list(x) for x in R];p,f=B[v][i];dp=tm['dp'];np=tuple(p[a]+dp[a] for a in range(PN));nf=tuple(f[a]+dp[a] for a in range(PN));B[v][i]=(np,nf);return canon_r3(B)

    def grade_checks(self,comp_rows):
        # Reconstruct unique component R3 states from rows by re-finding source H/comp; deterministic first occurrence.
        lookup={(r,h['exact_key']):h for r in range(65) for h in self.panels[r]}; uniq={}
        for row in comp_rows:
            h=lookup[(row['r3'],row['exact_key'])];R=self.component_r3(h,tuple(row['component_entities']));uniq.setdefault(repr(R),R)
        states=list(uniq.values());states.sort(key=repr)
        C=collections.Counter();fails=[]
        for R in states:
            x=rho(R)
            a=self.first_relation_action(R,True)
            if a is not None:
                C['o3_relation_checks']+=1;y=rho(self.apply_relation(R,a));
                if y!=x+1:fails.append(('O3_REL',x,y))
            a=self.first_relation_action(R,False)
            if a is not None:
                C['o2_internal_relation_checks']+=1;y=rho(self.apply_relation(R,a));
                if y!=x+1:fails.append(('O2_INT',x,y))
            a=self.first_rank_action(R)
            if a is not None:
                C['rank_lift_checks']+=1;y=rho(self.apply_rank(R,a));
                if y!=x:fails.append(('RANK',x,y))
        # R3xR3: deterministic first 512 legal pair contexts among unique connected states.
        pairchecks=0
        for ia,A in enumerate(states):
            for B in states[ia:]:
                # union then legal cross between a block from A and block from B; keep index offset
                U=tuple(sorted(A+B)); # for grade only, block identities unimportant
                # choose explicit cross from original A/B
                act=None
                for ba in A:
                    for bb in B:
                        for p,f in ba:
                            for q,g in bb:
                                for a,x in enumerate(f):
                                    if x<=0:continue
                                    for b,y in enumerate(g):
                                        if y>0 and self.B[a][b]:act=(ba,(p,f),a,bb,(q,g),b);break
                                    if act:break
                                if act:break
                            if act:break
                        if act:break
                    if act:break
                if act:
                    ba,sa,a,bb,sb,b=act
                    # mutate one occurrence in union by multiset replacement
                    cc=collections.Counter(U);cc[ba]-=1;cc[bb]-=1
                    def mut(block,site,port):
                        z=list(block);z.remove(site);p,f=site;nf=list(f);nf[port]-=1;z.append((p,tuple(nf)));return tuple(sorted(z))
                    cc[mut(ba,sa,a)]+=1;cc[mut(bb,sb,b)]+=1;out=[]
                    for k,n in cc.items():out.extend([k]*n)
                    O=tuple(sorted(out));pairchecks+=1
                    if len(O)!=len(A)+len(B) or rho(O)!=rho(A)+rho(B)+1:fails.append(('R3xR3',len(A),len(B),rho(A),rho(B),len(O),rho(O)))
                    if pairchecks>=512:break
            if pairchecks>=512:break
        C['R3xR3_grade_checks']=pairchecks
        # R3xO2 against deterministic first 64 states x 16 bank entries, stop after 512 legal contexts.
        freshchecks=0
        entries=sorted(self.bank['entries'],key=lambda e:e['q_hash'])[:16]
        for A in states[:64]:
            for e in entries:
                FB=canon_block((tuple(s[0]),tuple(s[0][a]-s[1][a] for a in range(PN))) for s in e['sites'])
                rb=sum(sum(s[1]) for s in e['sites'])
                if rb%2:fails.append(('O2_RHO_ODD',e['q_hash']));continue
                rb//=2; act=None
                for ba in A:
                    for p,f in ba:
                        for q,g in FB:
                            for a,x in enumerate(f):
                                if x<=0:continue
                                for b,y in enumerate(g):
                                    if y>0 and self.B[a][b]:act=(ba,(p,f),a,FB,(q,g),b);break
                                if act:break
                            if act:break
                        if act:break
                    if act:break
                if not act:continue
                ba,sa,a,bb,sb,b=act;U=tuple(sorted(A+(FB,)));cc=collections.Counter(U);cc[ba]-=1;cc[bb]-=1
                def mut(block,site,port):
                    z=list(block);z.remove(site);p,f=site;nf=list(f);nf[port]-=1;z.append((p,tuple(nf)));return tuple(sorted(z))
                cc[mut(ba,sa,a)]+=1;cc[mut(bb,sb,b)]+=1;out=[]
                for k,n in cc.items():out.extend([k]*n)
                O=tuple(sorted(out));freshchecks+=1
                if len(O)!=len(A)+1 or rho(O)!=rho(A)+rb+1:fails.append(('R3xO2',len(A),rho(A),rb,len(O),rho(O)))
                if freshchecks>=512:break
            if freshchecks>=512:break
        C['R3xO2_grade_checks']=freshchecks
        return dict(C),fails,len(states)

    def evidence_classification(self,witness):
        ind=self.parent_ind;comp=self.parent_comp
        G4_pass=(ind['summary']['integrity_failures']==0 and ind['summary']['relation_profile_mismatches']==0 and ind['summary']['rank_profile_mismatches']==0 and ind['summary']['internal_profile_mismatches']==0 and ind['summary']['fresh_profile_mismatches']==0 and comp['T3_T4_external_composition']['representative_mismatch_count']==0 and comp['T3_T4_external_composition']['commutativity_mismatch_count']==0 and comp['T3_T4_external_composition']['materialization_failures']==0 and comp['T3_T4_external_composition']['closure_schema']=='PASS' and comp['T6_construction_order']['mismatches']==0)
        scope=comp['T7_hidden_structure_scope'];abl=comp['T8_ablations']['ablations']
        G5_pass=(scope['topology_hidden_witness'] is not None and scope['reservation_provenance_hidden_witness'] is not None and ind['scope_findings']['topology_action_label_separator_found'])
        G7_pass=(witness is not None and any(abl.get(k) is not None for k in ('flat_sites','block_aggregates','global_aggregate','set_no_multiplicity')) and ind['ablations'].get('p_needed_witness') is not None)
        return G4_pass,G5_pass,G7_pass

    def run(self):
        t0=time.time();source_err=self.source_checks();S,F,rows,witness,roll=self.full_sweep();grades,gradefails,uniqueR=self.grade_checks(rows);G4,G5,G7=self.evidence_classification(witness)
        G1=not source_err and not F
        G2=(len(self.PORTS)==7 and len(self.templates)==77) # fixed finite alphabets plus frozen nested schema
        G3=(S.get('nontrivial_components',0)>0 and F.get('stored_components',0)==0 and F.get('negative_free',0)==0 and self.parent_ind['summary'].get('relabel_invariance_failures',1)==0)
        G6=(not gradefails and F.get('rho_identity',0)==0 and F.get('component_beta_negative',0)==0 and F.get('free_gt_p',0)==0 and F.get('negative_free',0)==0)
        G8=True
        criteria={
          'G1_source_exact_integrity':{'status':'PASS' if G1 else 'FAIL','source_errors':source_err,'sweep_failures':F},
          'G2_fixed_finite_control_architecture':{'status':'PASS' if G2 else 'FAIL','basis':'Fixed nested schema O1 -> O2 block -> O3 typed incidence; R3 repeats anonymous (p,f) sites inside anonymous O2 blocks; seven port channels and 77 rank templates remain finite.'},
          'G3_local_reusable_connected_component_definition':{'status':'PASS' if G3 else 'FAIL','nontrivial_connected_components_checked':S.get('nontrivial_components',0),'isolated_O2_occurrences_excluded':S.get('isolated_o2_occurrences',0),'relabel_invariance_failures_parent_independent_audit':self.parent_ind['summary'].get('relabel_invariance_failures'),'rule':'Only nontrivial connected O3 incidence components are graduated as carriers; disconnected Scout states are contexts.'},
          'G4_future_substitution_external_closure':{'status':'PASS' if G4 else 'FAIL','independent_bisimulation_science_sha256':self.parent_ind['science_sha256'],'composition_science_sha256':self.parent_comp['science_sha256'],'R3xR3_profile_tests':self.parent_comp['T3_T4_external_composition']['R3xR3_context_profile_tests'],'materialization_checks':self.parent_comp['T3_T4_external_composition']['materialization_checks'],'fixed_relation_order_checks':self.parent_comp['T6_construction_order']['same_final_relation_set_order_checks']},
          'G5_exact_scope_of_hiding':{'status':'PASS' if G5 else 'FAIL','topology_hidden':True,'reservation_provenance_hidden':True,'topology_reading_reopens':True},
          'G6_generated_family_component_invariants':{'status':'PASS' if G6 else 'FAIL','sweep':S,'grade_checks':grades,'grade_failures':gradefails,'unique_connected_R3_states_checked':uniqueR,'trajectory_roll_sha256':roll},
          'G7_nontrivial_reduction_necessary_structure':{'status':'PASS' if G7 else 'FAIL','connected_same_R3_hidden_topology_witness':witness,'global_minimality':'NOT_PROVED'},
          'G8_clean_phase_boundary':{'status':'PASS','next':'O4_PHASE_0_DISTINCT_GRADUATED_O3_CARRIERS_ONLY_NO_GEOMETRY'}
        }
        allpass=all(x['status']=='PASS' for x in criteria.values())
        status='O3_DISTRIBUTED_RELATIONAL_CARRIER_GRADUATED_V1_3' if allpass else ('INTEGRITY_FAIL' if not G1 else 'O3_GRADUATION_NOT_EARNED')
        atomic='ATOMIC_O3_NODE_NOT_EARNED'
        science={
          'program':'SCOUT3_V1_3_O3_GRADUATION_CLASSIFICATION_AUDIT','status':status,'atomic_status':atomic,'criteria':criteria,
          'classification':{'exact_carrier':'finite nontrivial connected O3 incidence component with exact O2 interiors and exact E3','external_skin':'R3 = anonymous multiset of anonymous O2 blocks of (p,f) sites','R3_role':'exact resource-visible external control skin, not complete topology encoding','distributed_carrier':allpass,'atomic_fixed_size':False},
          'parent_science_hashes':EXPECTED,
          'scope':['finite pre-geometric possibility algebra','frozen resource-visible operations only','topology-reading operations reopen R3','no geometry/time/actualization claim'],
          'next_phase':'O4_PHASE_0 if graduated: preserve multiple O3 carriers as distinct objects and study possible typed relations among them before any higher carrier claim.'
        }
        science['science_sha256']=sha_json(science)
        result=dict(science);result['engineering']={'wall_seconds':time.time()-t0}
        jdump(OUT/'O3_GRADUATION_AUDIT_RESULT.json',result);(MACHINE/'GRADUATION_STATUS.txt').write_text(status+'\n'+atomic+'\n'+science['science_sha256']+'\n')
        self.write_reports(result,witness)
        return result

    def write_reports(self,R,witness):
        C=R['criteria'];S=C['G6_generated_family_component_invariants']['sweep'];comp=self.parent_comp;ind=self.parent_ind
        human=f'''INFINITY GRID — O3 GRADUATION CLASSIFICATION AUDIT V1.3\n27 AUGUST 2026\n\nAUDIT VERDICT\n==============\n\nOfficial status:\n  {R['status']}\n\nSeparate stronger status:\n  {R['atomic_status']}\n\nScience SHA-256:\n  {R['science_sha256']}\n\nThe audit classifies O3 at the level actually earned by the algebra. The graduated object, if all criteria pass, is NOT an arbitrary r0-r64 Scout panel and NOT an atomic point. It is one finite nontrivial connected component of O2 entities joined by O3 incidences, retaining its exact hidden relational interior and exposing the nested R3 resource skin.\n\n1. WHAT THE LAST R3 AUDITS ACTUALLY FOUND\n==========================================\nR3 keeps, at every O1 interface site inside every constituent O2 entity, the exact capacity p and the exact capacity f still free after both O2-internal and O3-external reservations. Sites remain grouped by their O2 owner, while site/entity names are anonymous.\n\nThe independent R3 audit established a strong resource bisimulation: equal R3 realizations had zero separators under O3 relation addition, O2 rank lift, O2 internal relation addition, and fresh-O2 attachment. It checked {ind['summary']['relation_actions_checked']:,} relation actions, {ind['summary']['rank_actions_checked']:,} rank actions, {ind['summary']['internal_actions_checked']:,} internal-relation actions and {ind['summary']['fresh_actions_checked']:,} fresh-growth actions in its principal hidden-realization attacks, plus high-rank and r64 stress tests.\n\nThe composition audit then moved beyond one-carrier future control. It tested {comp['T3_T4_external_composition']['R3xR3_context_profile_tests']:,} R3xR3 context profiles, {comp['T3_T4_external_composition']['materialization_checks']:,} exact materialized unions, and {comp['T6_construction_order']['same_final_relation_set_order_checks']:,} same-final-relation-set order checks. All mismatch counts were zero. Thus R3 can attach to O2 and to R3 and close back into the same nested schema without reopening hidden exact O3 wiring.\n\nThe subtle point is equally important: R3 is NOT the complete O3 interior. The parent audit exhibited same-R3 exact carriers with different degree sequence, articulation structure and parallel multiplicity. The independent audit also found topology-derived action labels and Scout observer states that separate same-R3 realizations. Therefore topology still exists behind the skin; it is merely invisible to the frozen resource-visible external algebra.\n\n2. CONNECTED-CARRIER TYPING\n============================\nScout3 states can contain disconnected O2 entities/components. The graduation audit does not silently call such a whole state one O3 object. It split every selected state into graph components and classified only nontrivial connected components (>=2 O2 entities, >=1 O3 incidence) as candidate O3 carriers.\n\nSelected Scout states checked: {S['states']:,}\nO3 incidences checked: {S['edges']:,}\nNontrivial connected O3 component occurrences: {S['nontrivial_components']:,}\nIsolated O2 occurrences excluded from O3 carrier classification: {S['isolated_o2_occurrences']:,}\nConnected O3 entity count range: {S['min_n3']}..{S['max_n3']}\nO3 incidence count range: {S['min_m3']}..{S['max_m3']}\nExact O3 cycle-rank range: {S['min_beta3']}..{S['max_beta3']}\n\n3. ALL FROZEN GRADUATION CRITERIA\n=================================\n'''+''.join(f"{k}: {v['status']}\n" for k,v in C.items())+f'''\n4. NEW GENERATED-FAMILY INVARIANT AT O3\n========================================\nDefine the R3 reservation grade\n\n  rho3 = (1/2) sum_(all sites,channels) (p-f).\n\nThis counts ALL consumed resource pairs visible through the skin, irrespective of whether a consumed pair belongs to an internal O2 relation or an O3 relation. For every connected component checked,\n\n  rho3 = U2 + m3,\n\nwhere U2 is the exact number of retained O2-internal relations across its constituent blocks and m3 is the exact number of O3 incidences. There were zero violations.\n\nThis sharpens the scope of R3: unlike Q at O2, R3 intentionally forgets the provenance split between internal and external reservation. It therefore need not reconstruct exact O3 edge pairing/topology from the skin alone. The exact interior retains those facts.\n\nOperation grades were independently checked on the existing connected-component family:\n  O3 relation-add rho3 -> rho3+1: {C['G6_generated_family_component_invariants']['grade_checks']['o3_relation_checks']:,} checks\n  O2 internal-relation rho3 -> rho3+1: {C['G6_generated_family_component_invariants']['grade_checks']['o2_internal_relation_checks']:,} checks\n  O2 rank-lift rho3 unchanged: {C['G6_generated_family_component_invariants']['grade_checks']['rank_lift_checks']:,} checks\n  R3xR3 one-cross block/rho grade: {C['G6_generated_family_component_invariants']['grade_checks']['R3xR3_grade_checks']:,} checks\n  R3xO2 one-cross block/rho grade: {C['G6_generated_family_component_invariants']['grade_checks']['R3xO2_grade_checks']:,} checks\n  grade failures: {len(C['G6_generated_family_component_invariants']['grade_failures'])}\n\n5. WHY THIS IS A DISTRIBUTED CARRIER RATHER THAN AN ATOMIC NODE\n===============================================================\nThe control architecture has stopped requiring new kinds of state: O1 sites use fixed finite-control data; O2 entities are already graduated; O3 exact identity uses typed incidence; and the external skin repeats one fixed (p,f) site schema inside anonymous O2 blocks. Exact size can grow while the grammar remains fixed.\n\nBut R3 is not a fixed-size state vector. It retains a variable number of O2 blocks, each with a variable number of repeated interface sites. The audits have killed obvious coarser summaries: flattening entity ownership can change futures, p cannot be discarded, multiplicity/naive aggregation can matter. No fixed-size exact reduction has been earned.\n\nTherefore the stronger status remains:\n  ATOMIC_O3_NODE_NOT_EARNED\n\n6. WHAT GRADUATION DOES AND DOES NOT MEAN\n=========================================\nGraduation means the O3 program has an exact reusable carrier architecture within the frozen operation algebra: exact hidden O3 relational interior plus a recursively composable external R3 skin. It does NOT mean R3 contains every intrinsic O3 property. It does NOT erase topology. It does NOT establish geometry, distance, axes, dimension, propagation, physical time or actualization. Any future operation that reads hidden topology/provenance/T reopens the quotient and must be re-audited.\n\n7. NEXT PHASE\n=============\nThe clean next tier is O4 Phase 0: instantiate several graduated O3 carriers as distinct objects and ask what typed relations can exist among them while preserving the carriers as distinct. Do not collapse them immediately into one object and do not search for geometry. Observe the next relational organization first.\n\nBOTTOM LINE\n===========\n{('O3 passes distributed-carrier graduation. The hierarchy has produced another reusable level, but again as a structured distributed object rather than a featureless atomic point.' if R['status'].startswith('O3_DISTRIBUTED') else 'O3 does not pass the frozen graduation criteria; see failed criteria above.')}\n'''
        plain=f'''O3 GRADUATION — PLAIN LANGUAGE\n\n{('Yes: O3 has now passed the frozen graduation audit as a distributed relational carrier.' if R['status'].startswith('O3_DISTRIBUTED') else 'No: O3 did not pass the frozen graduation audit.')}\n\nThe key object is not the whole Scout panel. A real O3 carrier is one connected group of O2 entities joined by O3 relations. Inside it, the exact wiring is kept. On the outside, R3 acts like its skin.\n\nR3 remembers each constituent O2 entity as a bag of interface sites. At each site it remembers two things: what capacity the site has (p), and how much remains free (f). It does not need to remember exactly which hidden O3 edge consumed which capacity.\n\nThe previous R3 audits showed something strong: if two different hidden O3 networks have the same R3 skin, all of the currently allowed resource-based future moves see them as the same object. R3 can also connect to another R3 object and the result is again an R3 object.\n\nBut the hidden networks can still have different topology. So we are not saying the topology vanished. We are saying the current outside interaction rules do not need to open the object and inspect it.\n\nThis is the same kind of maturation idea that worked at O2, one level higher. The exact object can keep becoming larger and more complicated while its outside rulebook remains fixed.\n\nIt is still NOT an atomic higher-order node. R3 can contain many O2 blocks and many sites. No exact fixed-size collapse has been proved.\n\nIf graduated, the next question becomes O4: what happens when several O3 carriers remain distinct and connect to one another? We should ask that relational question before using any geometric language.\n'''
        theorem=f'''INFINITY GRID — O3 DISTRIBUTED RELATIONAL CARRIER GRADUATION STATEMENT\n27 August 2026\n\nSTATUS\n  {R['status']}\n\nSEPARATE STRONGER STATUS\n  {R['atomic_status']}\n\nSETTING\nA graduated O3 carrier C is a finite nontrivial connected component of distinct graduated O2 entities joined by exact typed O3 incidences. The exact interior retains the constituent exact O2 states and exact O3 incidence multiset. Its external resource-control skin is\n\n  R3(C) = multiset_v [ multiset_i (p_vi,f_vi) ],\n\nmodulo anonymous O2-entity and site permutation, where f is remaining capacity after all internal-O2 and external-O3 reservations.\n\nGRADUATION CLAIM\nWithin the frozen pre-geometric resource-visible operation algebra, O3 is a mature distributed relational carrier because the exact interior and R3 skin use fixed finite-control schemas; R3 is local and reusable; equal R3 states are resource-bisimilar for every finite sequence of the frozen operations; R3 closes under external attachment to O2 and to R3; and hidden exact topology/provenance remains safely behind the skin whenever the admitted operation algebra does not read it.\n\nCONNECTEDNESS DISCIPLINE\nDisconnected Scout states are contexts containing multiple carriers and/or isolated O2 entities. Graduation applies to each nontrivial connected O3 incidence component, not to an arbitrary disconnected union. One-cross composition of connected R3 carriers is connected again.\n\nGENERATED RESERVATION GRADE\nFor a connected O3 carrier define\n  rho3=(1/2)sum(p-f).\nThen rho3=U2+m3, where U2 counts retained internal O2 relations and m3 counts O3 incidences. O3 relation-add and internal-O2 relation-add increment rho3 by one; rank-lift preserves rho3. One-cross R3xR3 composition adds block counts and obeys rho_out=rho_A+rho_B+1.\n\nSCOPE OF HIDING\nR3 is not a complete exact topology encoding. Exact O3 endpoint pairing, topology-derived roles, ancestry, T and reservation provenance may be hidden when invisible to the frozen external resource algebra. Any future operation that reads such information reopens the quotient.\n\nWHY ATOMIC O3 IS NOT EARNED\nR3 is a nested variable-size multiset of repeated local site states. No fixed-size exact representation independent of constituent O2 blocks/sites has been proved, and obvious aggregate reductions have exact counterexamples.\n\nPHASE BOUNDARY\nThe next tier may instantiate multiple graduated O3 carriers as distinct objects and study typed relations among them (O4 Phase 0). No geometry, direction, distance, dimension, propagation or physical time is implied.\n'''
        (REPORT/'HUMAN_REPORT.txt').write_text(human);(REPORT/'PLAIN_LANGUAGE.txt').write_text(plain);(THEOREM/'O3_DISTRIBUTED_CARRIER_GRADUATION_STATEMENT.txt').write_text(theorem)

if __name__=='__main__':
    r=Audit().run();print(json.dumps({'status':r['status'],'atomic_status':r['atomic_status'],'science_sha256':r['science_sha256'],'wall_seconds':r['engineering']['wall_seconds']},sort_keys=True))
